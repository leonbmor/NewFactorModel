"""
quality_factor_diag.py
======================
Quality factor diagnostic study.

For each quality metric, identifies the top and bottom 10% performing stocks
(by forward idiosyncratic return) on each anchor date, then reports median and
z-score of each metric for those deciles — revealing which metrics consistently
separate winners from losers vs which are cyclical or noisy.

Residualization: OLS with market beta + sector dummies + log(size).

Usage:
    from quality_factor_diag import run_quality_diag
    summary, annual, raw = run_quality_diag(Pxs_df, sectors_s)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text

# ==============================================================================
# CONFIG
# ==============================================================================
QUALITY_TABLE   = 'valuation_metrics_anchors'   # swap to 'valuation_consolidated' if needed
HORIZONS        = [21, 63]                       # forward return horizons in trading days
TOP_PCTILE      = 0.10                           # top/bottom decile threshold
MIN_STOCKS      = 20                             # min stocks per date for date to be included
WINSOR          = (0.01, 0.99)                   # winsorization per metric per date

# Quality metrics to test — grouped by dimension for readability
QUALITY_METRICS = [
    # Growth — standalone
    'HSG', 'GS', 'GE', 'GGP', 'SGD', 'LastSGD', 'PIG', 'PSG',
    # Profitability level
    'OM', 'ROI', 'ROE', 'FCF_PG',
    # Profitability trend
    'OMd', 'ROId', 'ROEd', 'ISGD',
    # R&D intensity
    'r&d',
    # Vol-adjusted growth (derived: base / max(Vol, VOL_MIN))
    'GS/S_Vol', 'HSG/S_Vol', 'PSG/S_Vol',
    'GE/E_Vol', 'PIG/E_Vol',
    'GGP/GP_Vol',
    # R²-weighted growth/trend (derived: base * r2)
    'GS*r2_S', 'SGD*r2_S', 'OMd*r2_S',
    'GE*r2_E', 'PIG*r2_E',
    'GGP*r2_GP',
]

# Raw columns to fetch from DB (includes modifiers needed for derived metrics)
RAW_DB_COLS = [
    'Size',
    'HSG', 'GS', 'GE', 'GGP', 'SGD', 'LastSGD', 'PIG', 'PSG',
    'OM', 'ROI', 'ROE', 'FCF_PG',
    'OMd', 'ROId', 'ROEd', 'ISGD', 'r&d',
    'S Vol', 'E Vol', 'GP Vol',
    'r2 S', 'r2 E', 'r2 GP',
]

VOL_MIN = 1.0   # floor for Vol divisor

DB_URL = 'postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db'
ENGINE = create_engine(DB_URL)


# ==============================================================================
# HELPERS
# ==============================================================================

def normalize_ticker(t: str) -> str:
    return str(t).split(' ')[0].strip().upper()


def normalize_index(idx: pd.Index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(d) for d in idx])


def normalize_df(df: pd.DataFrame, ticker_col: str = 'ticker') -> pd.DataFrame:
    df = df.copy()
    df[ticker_col] = df[ticker_col].apply(normalize_ticker)
    return df


def normalize_series(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = [normalize_ticker(t) for t in s.index]
    return s


def winsorize(s: pd.Series) -> pd.Series:
    lo = s.quantile(WINSOR[0])
    hi = s.quantile(WINSOR[1])
    return s.clip(lower=lo, upper=hi)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_anchor_dates() -> list:
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT DISTINCT date FROM {QUALITY_TABLE}
            ORDER BY date
        """)).fetchall()
    return [pd.Timestamp(r[0]) for r in rows]


def build_derived_metrics(snap: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived Vol-adjusted and R²-weighted columns to snapshot.
    Raw modifier columns (S Vol, r2 S etc.) remain present but are
    not in QUALITY_METRICS so are never ranked directly.
    """
    s = snap.copy()

    def col(name):
        return s[name] if name in s.columns else pd.Series(np.nan, index=s.index)

    def safe_div(num, denom):
        d = col(denom).clip(lower=VOL_MIN)
        return col(num) / d

    def safe_mul(base, r2):
        return col(base) * col(r2)

    s['GS/S_Vol']   = safe_div('GS',  'S Vol')
    s['HSG/S_Vol']  = safe_div('HSG', 'S Vol')
    s['PSG/S_Vol']  = safe_div('PSG', 'S Vol')
    s['GE/E_Vol']   = safe_div('GE',  'E Vol')
    s['PIG/E_Vol']  = safe_div('PIG', 'E Vol')
    s['GGP/GP_Vol'] = safe_div('GGP', 'GP Vol')

    s['GS*r2_S']   = safe_mul('GS',  'r2 S')
    s['SGD*r2_S']  = safe_mul('SGD', 'r2 S')
    s['OMd*r2_S']  = safe_mul('OMd', 'r2 S')
    s['GE*r2_E']   = safe_mul('GE',  'r2 E')
    s['PIG*r2_E']  = safe_mul('PIG', 'r2 E')
    s['GGP*r2_GP'] = safe_mul('GGP', 'r2 GP')

    return s


def load_quality_snapshot(date: pd.Timestamp) -> pd.DataFrame:
    """
    Load all raw quality metric rows for a given date, then build derived metrics.
    Returns DataFrame with bare tickers as index and all QUALITY_METRICS columns.
    """
    cols_sql = ', '.join([f'"{m}"' for m in RAW_DB_COLS])
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT ticker, {cols_sql}
            FROM {QUALITY_TABLE}
            WHERE date = :d AND ticker IS NOT NULL
        """), conn, params={"d": date})

    if df.empty:
        return pd.DataFrame()

    df = normalize_df(df, 'ticker')
    df = df.drop_duplicates(subset='ticker').set_index('ticker')
    for col in RAW_DB_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = build_derived_metrics(df)
    return df


# ==============================================================================
# RESIDUALIZATION: mkt beta + sector dummies + log(size)
# ==============================================================================

def compute_residual_returns(Pxs_df: pd.DataFrame,
                              sectors_s: pd.Series,
                              size_s: pd.Series,
                              anchor_date: pd.Timestamp,
                              horizon: int) -> pd.Series:
    """
    Compute forward returns from prices then residualize vs:
      - trailing market beta
      - sector dummies
      - log(size)

    Returns a Series of residual returns indexed by bare ticker.
    """
    eq_cols      = [c for c in Pxs_df.columns if c != 'USGG10YR']
    Pxs_df       = Pxs_df[eq_cols]
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

    # Align sectors and size
    sec  = normalize_series(sectors_s).reindex(fwd_ret.index)
    sz   = normalize_series(size_s).reindex(fwd_ret.index)
    valid = fwd_ret.index[sec.notna() & sz.notna() & (sz > 0)]

    if len(valid) < MIN_STOCKS:
        return pd.Series(dtype=float)

    sec_val  = sec[valid]
    sz_valid = sz[valid]
    sectors  = sorted(sec_val.unique())
    n        = len(valid)

    # Trailing beta estimation (63d)
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
            cov      = np.cov(sr.values, mr.values)
            betas[t] = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 1.0
        betas = pd.Series(betas)

    # Subtract market contribution
    y_adj = fwd_ret[valid] - betas.reindex(valid).fillna(1.0) * mkt_ret

    # Design matrix: intercept + (S-1) sector dummies + log(size)
    log_size = np.log(sz_valid.values)
    log_size = (log_size - log_size.mean()) / log_size.std()   # z-score for stability

    sec_dummies = np.column_stack([
        (sec_val == s).astype(float).values for s in sectors[:-1]
    ]) if len(sectors) > 1 else np.zeros((n, 0))

    X = np.hstack([
        np.ones((n, 1)),
        sec_dummies,
        log_size.reshape(-1, 1)
    ])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y_adj.values, rcond=None)
        residuals        = y_adj.values - X @ coeffs
    except Exception:
        residuals = y_adj.values - y_adj.values.mean()

    return pd.Series(residuals, index=valid)


# ==============================================================================
# WITHIN-SECTOR RANKING (0-1)
# ==============================================================================

def rank_within_sector(df: pd.DataFrame,
                        sectors_s: pd.Series,
                        metrics: list) -> pd.DataFrame:
    """
    For each metric, rank stocks 0-1 within their sector.
    Missing values are left as NaN.
    Returns DataFrame of same shape with ranked values.
    """
    ranked = pd.DataFrame(index=df.index)
    sec    = normalize_series(sectors_s).reindex(df.index)

    for m in metrics:
        if m not in df.columns:
            ranked[m] = np.nan
            continue

        col = df[m].copy()
        out = pd.Series(np.nan, index=df.index)

        for sector, grp_idx in sec.groupby(sec).groups.items():
            grp = col.reindex(grp_idx).dropna()
            if len(grp) < 3:
                continue
            # Winsorize before ranking
            grp_w = winsorize(grp)
            # Rank 0-1
            r = grp_w.rank(method='average', na_option='keep')
            out.loc[r.index] = (r - 1) / (len(r) - 1) if len(r) > 1 else 0.5

        ranked[m] = out

    return ranked


# ==============================================================================
# DECILE STATS
# ==============================================================================

def decile_stats(ranked_metrics: pd.DataFrame,
                 ranked_for_stats: pd.DataFrame,
                 resid_ret: pd.Series,
                 universe_stats: dict) -> dict:
    """
    Split stocks into top/bottom decile by residual return,
    then compute median and z-score of each metric for each decile.

    Both ranked_metrics and ranked_for_stats should be the sector-ranked (0-1)
    values — ensuring all stats are sector-neutral.

    universe_stats: {metric: (mean, std)} computed from sector-ranked values.

    Returns dict: {metric: {top: {median, z}, bottom: {median, z}, spread_z}}
    """
    n         = len(resid_ret)
    n_decile  = max(1, int(np.floor(n * TOP_PCTILE)))

    sorted_ret = resid_ret.sort_values(ascending=False)
    top_tickers    = sorted_ret.iloc[:n_decile].index
    bottom_tickers = sorted_ret.iloc[-n_decile:].index

    result = {}
    for m in QUALITY_METRICS:
        if m not in ranked_for_stats.columns:
            continue

        raw = ranked_for_stats[m]
        u_mean, u_std = universe_stats.get(m, (np.nan, np.nan))

        def stats_for(tickers):
            vals = raw.reindex(tickers).dropna()
            if len(vals) == 0:
                return {'median': np.nan, 'z': np.nan, 'n': 0}
            med = float(vals.median())
            z   = (med - u_mean) / u_std if u_std and u_std > 0 else np.nan
            return {'median': round(med, 4), 'z': round(z, 3), 'n': len(vals)}

        top_s    = stats_for(top_tickers)
        bottom_s = stats_for(bottom_tickers)

        # Normalized spread: (top_median - bottom_median) / universe_std
        if u_std and u_std > 0 and not np.isnan(top_s['median']) and not np.isnan(bottom_s['median']):
            spread_z = round((top_s['median'] - bottom_s['median']) / u_std, 3)
        else:
            spread_z = np.nan

        result[m] = {
            'top':      top_s,
            'bottom':   bottom_s,
            'spread_z': spread_z,
        }

    return result


# ==============================================================================
# MAIN RUN
# ==============================================================================

def run_quality_diag(Pxs_df: pd.DataFrame,
                     sectors_s: pd.Series) -> tuple:
    """
    Run quality factor diagnostic.

    Args:
        Pxs_df    : Price DataFrame (dates as index, tickers as columns).
        sectors_s : Series mapping bare ticker -> sector string.

    Returns:
        summary : DataFrame — overall stats per metric per horizon.
                  Columns: mean_spread_z, consistency (% dates spread_z > 0), n_dates.
        annual  : dict {year: {horizon: DataFrame of per-metric stats}}
        raw     : dict {(date, horizon): {metric: {top, bottom, spread_z}}}
                  Full raw results for custom analysis.
    """
    # Normalize inputs
    Pxs_df            = Pxs_df.copy()
    Pxs_df.index      = normalize_index(Pxs_df.index)
    sectors_s         = normalize_series(sectors_s)

    anchor_dates = load_anchor_dates()
    print(f"  {len(anchor_dates)} anchor dates in {QUALITY_TABLE}")
    print(f"  Metrics  : {len(QUALITY_METRICS)} quality metrics")
    print(f"  Horizons : {HORIZONS} trading days")
    print(f"  Decile   : top/bottom {int(TOP_PCTILE*100)}%\n")

    raw_results = {}   # (date, horizon) -> decile stats dict

    for d_idx, anchor in enumerate(anchor_dates, 1):
        print(f"[{d_idx:>3}/{len(anchor_dates)}] {anchor.date()}", end='')

        snap = load_quality_snapshot(anchor)
        if snap.empty:
            print("  — no data")
            continue

        # Size for residualization — always present via RAW_DB_COLS
        size_s = snap['Size'].dropna() if 'Size' in snap.columns else pd.Series(1.0, index=snap.index)
        size_s = size_s[size_s > 0]

        # Within-sector ranking
        ranked = rank_within_sector(snap, sectors_s, QUALITY_METRICS)

        # Universe stats from ranked metrics (sector-neutral, 0-1 scale)
        u_stats = {}
        for m in QUALITY_METRICS:
            if m in ranked.columns:
                s = ranked[m].dropna()
                u_stats[m] = (float(s.mean()), float(s.std())) if len(s) > 1 else (np.nan, np.nan)

        for horizon in HORIZONS:
            resid = compute_residual_returns(Pxs_df, sectors_s, size_s,
                                             anchor, horizon)
            if resid.empty or len(resid) < MIN_STOCKS:
                print(f"  h{horizon}:insufficient", end='')
                continue

            d_stats = decile_stats(ranked, ranked, resid, u_stats)
            raw_results[(anchor, horizon)] = d_stats

        n_computed = sum(1 for h in HORIZONS if (anchor, h) in raw_results)
        print(f"  {n_computed}/{len(HORIZONS)} horizons ok")

    if not raw_results:
        print("No results.")
        return pd.DataFrame(), {}, {}

    # ------------------------------------------------------------------
    # Build summary: mean spread_z and consistency per metric per horizon
    # ------------------------------------------------------------------
    summary_rows = []
    for m in QUALITY_METRICS:
        for h in HORIZONS:
            spread_zs = [
                raw_results[(d, h)][m]['spread_z']
                for d in anchor_dates
                if (d, h) in raw_results and m in raw_results[(d, h)]
                and not np.isnan(raw_results[(d, h)][m]['spread_z'])
            ]
            if not spread_zs:
                continue
            arr         = np.array(spread_zs)
            mean_sz     = float(arr.mean())
            t_stat      = float(stats.ttest_1samp(arr, 0).statistic) if len(arr) > 1 else np.nan
            consistency = float((arr > 0).mean() * 100)
            summary_rows.append({
                'metric':       m,
                'horizon':      h,
                'mean_spread_z': round(mean_sz, 3),
                't_stat':        round(t_stat, 3),
                'consistency':   round(consistency, 1),
                'n_dates':       len(arr),
            })

    summary = (pd.DataFrame(summary_rows)
                 .set_index(['metric', 'horizon'])
                 .sort_values('mean_spread_z', ascending=False))

    # ------------------------------------------------------------------
    # Annual breakdown
    # ------------------------------------------------------------------
    annual = {}
    all_years = sorted(set(d.year for d, h in raw_results.keys()))

    for year in all_years:
        annual[year] = {}
        for h in HORIZONS:
            year_dates = [d for d, hh in raw_results if hh == h and d.year == year]
            if not year_dates:
                continue

            rows = []
            for m in QUALITY_METRICS:
                spread_zs = [
                    raw_results[(d, h)][m]['spread_z']
                    for d in year_dates
                    if m in raw_results[(d, h)]
                    and not np.isnan(raw_results[(d, h)][m]['spread_z'])
                ]
                if not spread_zs:
                    continue
                arr = np.array(spread_zs)
                rows.append({
                    'metric':        m,
                    'mean_spread_z': round(float(arr.mean()), 3),
                    'consistency':   round(float((arr > 0).mean() * 100), 1),
                    'n_dates':       len(arr),
                })

            if rows:
                annual[year][h] = (pd.DataFrame(rows)
                                     .set_index('metric')
                                     .sort_values('mean_spread_z', ascending=False))

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    _print_summary(summary)
    _print_annual(annual)

    return summary, annual, raw_results


# ==============================================================================
# PRINTING
# ==============================================================================

def _print_summary(summary: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  QUALITY FACTOR DIAGNOSTIC — OVERALL SUMMARY")
    print("  (spread_z = (top10% median - bot10% median) / std, all sector-ranked 0-1)")
    print("=" * 70)
    pd.set_option('display.float_format', '{:.3f}'.format)
    pd.set_option('display.width', 120)

    for h in HORIZONS:
        print(f"\n  Horizon: {h}d")
        sub = summary.xs(h, level='horizon') if h in summary.index.get_level_values('horizon') else pd.DataFrame()
        if sub.empty:
            print("  No data.")
            continue
        print(f"  {'Metric':<12} {'mean_spread_z':>14} {'t_stat':>8} {'consistency%':>13} {'n_dates':>8}")
        print("  " + "-" * 57)
        for m, row in sub.iterrows():
            flag = ' *' if abs(row['t_stat']) > 2.0 else ''
            print(f"  {m:<12} {row['mean_spread_z']:>14.3f} {row['t_stat']:>8.3f} "
                  f"{row['consistency']:>13.1f} {int(row['n_dates']):>8}{flag}")
    print("=" * 70)


def _print_annual(annual: dict):
    print("\n" + "=" * 70)
    print("  QUALITY FACTOR DIAGNOSTIC — ANNUAL BREAKDOWN")
    print("  (mean_spread_z  |  consistency%  |  n)")
    print("=" * 70)

    years = sorted(annual.keys())

    for h in HORIZONS:
        print(f"\n  Horizon: {h}d\n")

        # Collect all metrics that appear in any year
        all_metrics = []
        for yr in years:
            if h in annual[yr]:
                for m in annual[yr][h].index:
                    if m not in all_metrics:
                        all_metrics.append(m)

        if not all_metrics:
            continue

        # Header
        yr_cols = "".join(f"  {y:>22}" for y in years)
        print(f"  {'Metric':<12}{yr_cols}")
        print("  " + "-" * (12 + 24 * len(years)))

        for m in all_metrics:
            row_str = f"  {m:<12}"
            for yr in years:
                if h not in annual[yr] or m not in annual[yr][h].index:
                    row_str += f"  {'—':>22}"
                else:
                    r   = annual[yr][h].loc[m]
                    sz  = r['mean_spread_z']
                    con = r['consistency']
                    n   = int(r['n_dates'])
                    cell = f"{sz:+.3f} / {con:.0f}% / {n}"
                    row_str += f"  {cell:>22}"
            print(row_str)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("Usage: from quality_factor_diag import run_quality_diag")
    print("       summary, annual, raw = run_quality_diag(Pxs_df, sectors_s)")
    print(f"\nCurrent QUALITY_TABLE = '{QUALITY_TABLE}'")
    print(f"Metrics  : {QUALITY_METRICS}")
    print(f"Horizons : {HORIZONS}")
