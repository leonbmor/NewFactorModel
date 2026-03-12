"""
ic_study.py
===========
Information Coefficient (IC) study for valuation metrics.

Measures within-sector Spearman IC between each valuation metric rank
and forward idiosyncratic returns (residualized vs market + sector).

Usage:
    from ic_study import run_ic_study
    ic_ts, ic_summary = run_ic_study(Pxs_df, sectors_s)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text

# ==============================================================================
# CONFIG
# ==============================================================================
VALUATION_TABLE = 'valuation_consolidated'   # swap to 'valuation_metrics_anchors' when ready
RESIDUAL_SOURCE = 'factor_residuals_joint'   # 'factor_residuals_joint' or 'prices'
HORIZONS        = [21, 63]                   # forward return horizons in trading days
MIN_STOCKS      = 10                         # min stocks per sector per date to include
WINSOR          = (0.01, 0.99)               # winsorization percentiles per metric per sector
METRICS         = ['sP/S', 'sP/E', 'sP/GP', 'P/S', 'P/Ee', 'P/GP']

DB_URL = 'postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db'
ENGINE = create_engine(DB_URL)


# ==============================================================================
# HELPERS
# ==============================================================================

def normalize_ticker(t: str) -> str:
    return str(t).split(' ')[0].strip().upper()


def normalize_index(idx: pd.Index) -> pd.Index:
    return pd.DatetimeIndex([pd.Timestamp(d) for d in idx])


def normalize_df_tickers(df: pd.DataFrame, col: str = 'ticker') -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].apply(normalize_ticker)
    return df


def normalize_series_index(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = [normalize_ticker(t) for t in s.index]
    return s


def winsorize_series(s: pd.Series,
                     lower: float = WINSOR[0],
                     upper: float = WINSOR[1]) -> pd.Series:
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_valuation_dates() -> list:
    """Return sorted list of distinct dates in VALUATION_TABLE."""
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT DISTINCT date FROM {VALUATION_TABLE}
            ORDER BY date
        """)).fetchall()
    return [pd.Timestamp(r[0]) for r in rows]


def load_valuation_snapshot(date: pd.Timestamp) -> pd.DataFrame:
    """
    Load all metric rows for a given date from VALUATION_TABLE.
    Returns DataFrame with bare tickers as index, metric columns.
    """
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT ticker, {', '.join(f'"{m}"' for m in METRICS)}
            FROM {VALUATION_TABLE}
            WHERE date = :d
              AND ticker IS NOT NULL
        """), conn, params={"d": date})

    if df.empty:
        return pd.DataFrame()

    df = normalize_df_tickers(df, 'ticker')
    df = df.drop_duplicates(subset='ticker').set_index('ticker')
    df.index = [normalize_ticker(t) for t in df.index]
    for col in METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ==============================================================================
# RESIDUALIZATION
# ==============================================================================

def compute_residual_returns(Pxs_df: pd.DataFrame,
                              sectors_s: pd.Series,
                              anchor_date: pd.Timestamp,
                              horizon: int) -> pd.Series:
    """
    Dispatcher — routes to the correct residual computation based on RESIDUAL_SOURCE.

    'factor_residuals_joint' : compound daily residuals from DB over forward window
    'prices'                 : OLS market beta + sector dummies from price history
    """
    if RESIDUAL_SOURCE == 'factor_residuals_joint':
        return _residuals_from_db(anchor_date, horizon)
    else:
        return _residuals_from_prices(Pxs_df, sectors_s, anchor_date, horizon)


def _residuals_from_db(anchor_date: pd.Timestamp,
                        horizon: int) -> pd.Series:
    """
    Load daily residuals from factor_residuals_joint for the forward window
    starting strictly after anchor_date, then compound over horizon trading days.

    Returns a Series of compounded residual returns indexed by bare ticker.
    """
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT date, ticker, residual
            FROM {RESIDUAL_SOURCE}
            WHERE date > :anchor
            ORDER BY date
        """), conn, params={"anchor": anchor_date})

    if df.empty:
        return pd.Series(dtype=float)

    df['date']   = pd.to_datetime(df['date'])
    df['ticker'] = df['ticker'].apply(normalize_ticker)

    # Keep only the first `horizon` distinct trading dates after anchor
    trading_dates = sorted(df['date'].unique())
    if len(trading_dates) < horizon:
        return pd.Series(dtype=float)
    window_dates = trading_dates[:horizon]

    df = df[df['date'].isin(window_dates)]

    # Pivot to (date x ticker), compound across dates
    pivot = df.pivot(index='date', columns='ticker', values='residual')
    pivot = pivot.sort_index()

    compounded = (1 + pivot).prod(axis=0) - 1
    return compounded.dropna()


def _residuals_from_prices(Pxs_df: pd.DataFrame,
                            sectors_s: pd.Series,
                            anchor_date: pd.Timestamp,
                            horizon: int) -> pd.Series:
    """
    Compute forward returns from prices then residualize vs
    trailing market beta + sector dummies via OLS.

    Returns a Series of residual returns indexed by bare ticker.
    """
    # --- forward return window ---
    future_dates = Pxs_df.index[Pxs_df.index > anchor_date]
    if len(future_dates) < horizon:
        return pd.Series(dtype=float)

    end_date   = future_dates[horizon - 1]
    past_dates = Pxs_df.index[Pxs_df.index <= anchor_date]
    if past_dates.empty:
        return pd.Series(dtype=float)
    start_date = past_dates[-1]

    px_start = Pxs_df.loc[start_date].dropna()
    px_end   = Pxs_df.loc[end_date].reindex(px_start.index).dropna()
    common   = px_start.index.intersection(px_end.index)
    if common.empty:
        return pd.Series(dtype=float)

    fwd_ret       = (px_end[common] / px_start[common] - 1)
    fwd_ret.index = [normalize_ticker(t) for t in fwd_ret.index]
    mkt_ret       = fwd_ret.mean()

    sec   = normalize_series_index(sectors_s).reindex(fwd_ret.index)
    valid = fwd_ret.index[sec.notna()]
    if len(valid) < 30:
        return pd.Series(dtype=float)

    sec_val = sec[valid]
    sectors = sorted(sec_val.unique())
    n       = len(valid)

    # --- trailing beta estimation ---
    trail_start_idx = max(0, len(past_dates) - 63)
    trail_dates     = past_dates[trail_start_idx:]

    if len(trail_dates) < 20:
        betas = pd.Series(1.0, index=valid)
    else:
        cols     = [t + ' US' if t + ' US' in Pxs_df.columns else t for t in valid]
        px_trail = Pxs_df.loc[trail_dates, cols].copy()
        px_trail.columns = [normalize_ticker(c) for c in px_trail.columns]
        r_trail  = px_trail.pct_change().dropna()
        mkt_r    = r_trail.mean(axis=1)

        betas = {}
        for t in valid:
            if t not in r_trail.columns:
                betas[t] = 1.0
                continue
            sr  = r_trail[t].dropna()
            mr  = mkt_r.reindex(sr.index)
            if len(sr) < 10:
                betas[t] = 1.0
                continue
            cov       = np.cov(sr.values, mr.values)
            betas[t]  = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 1.0
        betas = pd.Series(betas)

    # --- subtract market contribution ---
    y_adj = fwd_ret[valid] - betas.reindex(valid).fillna(1.0) * mkt_ret

    # --- OLS on sector dummies ---
    X_sec = np.hstack([
        np.ones((n, 1)),
        np.column_stack([(sec_val == s).astype(float).values
                         for s in sectors[:-1]])   # drop last sector (reference)
    ])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X_sec, y_adj.values, rcond=None)
        residuals        = y_adj.values - X_sec @ coeffs
    except Exception:
        residuals = y_adj.values - y_adj.values.mean()

    return pd.Series(residuals, index=valid)


# ==============================================================================
# IC COMPUTATION
# ==============================================================================

def compute_ic_for_date(valuation_snap: pd.DataFrame,
                         residual_rets: pd.Series,
                         sectors_s: pd.Series,
                         metric: str) -> float:
    """
    Compute weighted-average within-sector Spearman IC for one metric on one date.

    Steps:
    1. Align metric values, residual returns, and sectors
    2. Within each sector: drop NaN + negatives, winsorize, rank both sides
    3. Spearman IC = corr(metric_rank, resid_rank) per sector
    4. Weighted average by stock count across sectors with >= MIN_STOCKS stocks

    Returns scalar IC, or np.nan if insufficient data.
    """
    if metric not in valuation_snap.columns:
        return np.nan

    df = pd.DataFrame({
        'metric': valuation_snap[metric],
        'resid':  residual_rets,
        'sector': normalize_series_index(sectors_s),
    }).dropna()

    if df.empty:
        return np.nan

    # Drop non-positive metric values (P/E, P/S etc. are undefined when negative)
    df = df[df['metric'] > 0]
    if df.empty:
        return np.nan

    sector_ics     = []
    sector_weights = []

    for sec, grp in df.groupby('sector'):
        if len(grp) < MIN_STOCKS:
            continue

        # Winsorize metric within sector
        m_wins = winsorize_series(grp['metric'])

        # Rank both sides (average method handles ties)
        m_rank = m_wins.rank(method='average')
        r_rank = grp['resid'].rank(method='average')

        # Spearman IC
        ic, _ = stats.spearmanr(m_rank, r_rank)
        if np.isnan(ic):
            continue

        sector_ics.append(ic)
        sector_weights.append(len(grp))

    if not sector_ics:
        return np.nan

    weights    = np.array(sector_weights, dtype=float)
    weights   /= weights.sum()
    return float(np.dot(weights, sector_ics))


# ==============================================================================
# MAIN RUN
# ==============================================================================

def run_ic_study(Pxs_df: pd.DataFrame,
                 sectors_s: pd.Series) -> tuple:
    """
    Run full IC study across all anchor dates and horizons.

    Args:
        Pxs_df    : Price DataFrame (dates as index, tickers as columns).
                    Tickers may be bare or with ' US' extension.
        sectors_s : Series mapping bare ticker -> sector string.

    Returns:
        ic_ts      : DataFrame of IC values.
                     MultiIndex columns: (metric, horizon).
                     Index: anchor dates.
        ic_summary : DataFrame of summary stats per (metric, horizon).
                     Columns: mean_IC, t_stat, pct_positive, IC_IR, n_dates.
    """
    # --- normalize inputs ---
    Pxs_df            = Pxs_df.copy()
    Pxs_df.index      = normalize_index(Pxs_df.index)
    sectors_s         = normalize_series_index(sectors_s)

    anchor_dates = load_valuation_dates()
    print(f"  {len(anchor_dates)} anchor dates in {VALUATION_TABLE}")
    print(f"  Metrics         : {METRICS}")
    print(f"  Horizons        : {HORIZONS} trading days")
    print(f"  Residual source : {RESIDUAL_SOURCE}")
    print(f"  Valuation table : {VALUATION_TABLE}\n")

    # ic_ts[date][(metric, horizon)] = IC value
    records = []

    for d_idx, anchor in enumerate(anchor_dates, 1):
        print(f"[{d_idx:>3}/{len(anchor_dates)}] {anchor.date()}", end='')

        snap = load_valuation_snapshot(anchor)
        if snap.empty:
            print("  — no valuation data, skipping")
            continue

        row = {'date': anchor}

        for horizon in HORIZONS:
            resid = compute_residual_returns(Pxs_df, sectors_s, anchor, horizon)
            if resid.empty:
                for m in METRICS:
                    row[(m, horizon)] = np.nan
                continue

            for m in METRICS:
                ic = compute_ic_for_date(snap, resid, sectors_s, m)
                row[(m, horizon)] = ic

        # Quick progress summary
        sample_ics = [row[(m, HORIZONS[0])] for m in METRICS
                      if (m, HORIZONS[0]) in row and not np.isnan(row[(m, HORIZONS[0])])]
        if sample_ics:
            print(f"  IC({HORIZONS[0]}d) range [{min(sample_ics):.3f}, {max(sample_ics):.3f}]")
        else:
            print("  — no IC computed")

        records.append(row)

    if not records:
        print("No results.")
        return pd.DataFrame(), pd.DataFrame()

    # --- build IC time series ---
    ic_ts       = pd.DataFrame(records).set_index('date')
    ic_ts.index = normalize_index(ic_ts.index)
    ic_ts.columns = pd.MultiIndex.from_tuples(
        [(m, h) for m, h in ic_ts.columns],
        names=['metric', 'horizon']
    )

    # --- summary stats ---
    summary_rows = []
    for m in METRICS:
        for h in HORIZONS:
            if (m, h) not in ic_ts.columns:
                continue
            s       = ic_ts[(m, h)].dropna()
            n       = len(s)
            if n < 2:
                continue
            mean_ic = s.mean()
            t_stat  = float(stats.ttest_1samp(s, 0).statistic)
            pct_pos = (s > 0).mean() * 100
            ic_ir   = mean_ic / s.std() if s.std() > 0 else np.nan
            summary_rows.append({
                'metric':       m,
                'horizon':      h,
                'mean_IC':      round(mean_ic, 4),
                't_stat':       round(t_stat,  3),
                'pct_positive': round(pct_pos, 1),
                'IC_IR':        round(ic_ir,   3) if not np.isnan(ic_ir) else np.nan,
                'n_dates':      n,
            })

    ic_summary = pd.DataFrame(summary_rows).set_index(['metric', 'horizon'])

    # --- annual breakdown ---
    ic_annual = _compute_annual_breakdown(ic_ts)

    # --- print summary ---
    print("\n" + "=" * 65)
    print("  IC STUDY SUMMARY")
    print("=" * 65)
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.width', 120)
    print(ic_summary.to_string())
    print("=" * 65)

    print("\n" + "=" * 65)
    print("  IC ANNUAL BREAKDOWN  (mean_IC  |  t_stat  |  n)")
    print("=" * 65)
    _print_annual_breakdown(ic_annual)
    print("=" * 65)

    return ic_ts, ic_summary, ic_annual


def _compute_annual_breakdown(ic_ts: pd.DataFrame) -> dict:
    """
    For each (metric, horizon), compute per-year mean_IC, t_stat, n_dates.
    Returns nested dict: {year: {(metric, horizon): {mean_IC, t_stat, n}}}
    """
    result = {}
    years  = sorted(ic_ts.index.year.unique())

    for year in years:
        mask      = ic_ts.index.year == year
        ic_year   = ic_ts[mask]
        result[year] = {}
        for col in ic_ts.columns:
            s = ic_year[col].dropna()
            n = len(s)
            if n < 2:
                result[year][col] = (np.nan, np.nan, n)
                continue
            mean_ic = s.mean()
            t_stat  = float(stats.ttest_1samp(s, 0).statistic)
            result[year][col] = (round(mean_ic, 4), round(t_stat, 3), n)

    return result


def _print_annual_breakdown(ic_annual: dict):
    """
    Print annual breakdown grouped by horizon, one metric per column.
    Format: year | metric1 (IC / t / n) | metric2 ...
    """
    years   = sorted(ic_annual.keys())
    metrics = METRICS

    for horizon in HORIZONS:
        print(f"\n  Horizon: {horizon}d")
        # Header
        header = f"  {'Year':<6}" + "".join(f"  {m:>20}" for m in metrics)
        print(header)
        print("  " + "-" * (6 + 22 * len(metrics)))

        for year in years:
            row = f"  {year:<6}"
            for m in metrics:
                key  = (m, horizon)
                vals = ic_annual[year].get(key, (np.nan, np.nan, 0))
                ic, t, n = vals
                if np.isnan(ic):
                    cell = f"{'—':>20}"
                else:
                    cell = f"{ic:+.3f} / {t:+.2f} / {n:>2}"
                    cell = f"{cell:>20}"
                row += f"  {cell}"
            print(row)


if __name__ == "__main__":
    print("Usage: from ic_study import run_ic_study")
    print("       ic_ts, ic_summary, ic_annual = run_ic_study(Pxs_df, sectors_s)")
    print(f"\nCurrent VALUATION_TABLE = '{VALUATION_TABLE}'")
    print(f"Metrics  : {METRICS}")
    print(f"Horizons : {HORIZONS}")
