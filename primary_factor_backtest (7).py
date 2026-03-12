#!/usr/bin/env python
# coding: utf-8

"""
Primary Factor Backtest
========================
1. Recalculates key valuation metrics on a bi-monthly basis from 2019-01-01
   using 8 backward quarters from each stock's historical FEQ
2. Builds primary factor: GS_sn / median(P/S_q, P/Ee_q, P/GP_q)
3. Backtests long-N equal-weight portfolio, full rebalance every 2 months
4. Returns daily NAV series + portfolio holdings DataFrame

Assumptions:
  - All tickers in Pxs_df are bare (no ' US' extension)
  - sectors_s index is also bare tickers

Optional enhancements (vs baseline):
  - Vol filter      : exclude top volatility quintile at each rebalance
  - Momentum        : blend z-scored momentum with z-scored primary factor (50/50)
                      '12m1' = raw price momentum
                      'idio' = idiosyncratic residual momentum from factor model
  - Mkt cap floor   : exclude stocks below a minimum market cap
  - Sector cap      : max N stocks per sector, relaxed gradually if needed
  - Portfolio size  : configurable number of stocks (default 20)

Usage:
    from primary_factor_backtest import run
    nav_base, nav_alt, port_base, port_alt = run(Pxs_df, sectors_s)
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

ENGINE         = create_engine("postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db")
TARGET_TABLE   = 'key_valuation_metrics'
IDIO_MOM_TABLE = 'factor_residuals_joint'
START_DATE     = pd.Timestamp('2019-01-01')
STEP_DAYS      = 60
TOP_N          = 20
N_QUANTILES    = 5
MOM_LONG       = 252
MOM_SKIP       = 21


# ==============================================================================
# TABLE SETUP
# ==============================================================================

def create_target_table():
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
                date       DATE    NOT NULL,
                ticker     TEXT    NOT NULL,
                "GS"       NUMERIC,
                "P/S"      NUMERIC,
                "P/Ee"     NUMERIC,
                "P/GP"     NUMERIC,
                "mkt_cap"  NUMERIC,
                PRIMARY KEY (date, ticker)
            )
        """))
        # Add any columns introduced after initial table creation
        for col, dtype in [('mkt_cap', 'NUMERIC'), ('GS_vol', 'NUMERIC'), ('GS_adj', 'NUMERIC')]:
            conn.execute(text(f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{TARGET_TABLE}'
                          AND column_name = '{col}'
                    ) THEN
                        ALTER TABLE {TARGET_TABLE} ADD COLUMN "{col}" {dtype};
                    END IF;
                END $$;
            """))
    print(f"  Table '{TARGET_TABLE}' ready.")


# ==============================================================================
# HELPERS
# ==============================================================================

def clean_ticker(ticker: str) -> str:
    """Strip any ' US' suffix and whitespace, return uppercase bare ticker."""
    return ticker.strip().split(' ')[0].upper()


def add_quarters(quarter: str, n: int) -> str:
    year, q = int(quarter[:4]), int(quarter[5])
    q += n
    while q > 4: q -= 4; year += 1
    while q < 1: q += 4; year -= 1
    return f"{year}Q{q}"


def generate_calc_dates(Pxs_df: pd.DataFrame) -> list:
    end_date = Pxs_df.index.max()
    dates    = []
    current  = START_DATE
    while current <= end_date:
        available = Pxs_df.index[Pxs_df.index >= current]
        if available.empty:
            break
        dates.append(available[0])
        current += pd.Timedelta(days=STEP_DAYS)
    return sorted(set(dates))


def get_already_calculated_dates() -> set:
    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT DISTINCT date FROM {TARGET_TABLE}
            """)).fetchall()
        return {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        return set()


def save_metrics(metrics_df: pd.DataFrame):
    if metrics_df is None or metrics_df.empty:
        return
    calc_date = metrics_df['date'].iloc[0]
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            DELETE FROM {TARGET_TABLE} WHERE date = :d
        """), {"d": calc_date})
    metrics_df.to_sql(TARGET_TABLE, ENGINE, if_exists='append', index=False)


def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


# ==============================================================================
# HISTORICAL FEQ
# ==============================================================================

def get_historical_feq(ticker: str, calc_date: pd.Timestamp) -> str:
    """
    Estimate the first estimated quarter (FEQ) for a ticker at a given calc_date.

    Logic:
    1. Get cFEQ and last_fund_date from estimation_status (today's anchor)
    2. Get fund_date = most recent download date in income_data <= calc_date
    3. Compute i = round((last_fund_date - fund_date).days / 90)
       Candidates: cFEQ-(i-1), cFEQ-i, cFEQ-(i+1)
    4. Compare fund_date vs dl_after snapshots within those 3 candidates
       using >1% change threshold — first match wins
    5. Fallback if no match: cFEQ - i
    """
    t = clean_ticker(ticker)

    # Step 1: get cFEQ and last_fund_date from estimation_status
    with ENGINE.connect() as conn:
        row = conn.execute(text("""
            SELECT first_estimated_period, last_checked
            FROM estimation_status
            WHERE ticker = :t
              AND category = 'income'
            ORDER BY last_checked DESC
            LIMIT 1
        """), {"t": t}).fetchone()

    if not row or not row[0]:
        return None

    cFEQ           = row[0]   # e.g. '2026Q3'
    last_fund_date = pd.Timestamp(row[1])

    # Step 2: fund_date = most recent download date in income_data <= calc_date
    with ENGINE.connect() as conn:
        fd_row = conn.execute(text("""
            SELECT MAX(download_date) FROM income_data
            WHERE ticker = :t
              AND metric_name = 'totalRevenues'
              AND download_date <= :d
        """), {"t": t, "d": calc_date}).fetchone()

        dl_after_row = conn.execute(text("""
            SELECT MIN(download_date) FROM income_data
            WHERE ticker = :t
              AND metric_name = 'totalRevenues'
              AND download_date > :d
        """), {"t": t, "d": calc_date}).fetchone()

    fund_date = pd.Timestamp(fd_row[0])   if fd_row  and fd_row[0]       else None
    dl_after  = pd.Timestamp(dl_after_row[0]) if dl_after_row and dl_after_row[0] else None

    if not fund_date:
        return None

    # Step 3: compute i and candidate quarters
    i = int(round((last_fund_date - fund_date).days / 90, 0))

    def shift_quarter(q: str, n: int) -> str:
        return add_quarters(q, -n)

    candidates = [shift_quarter(cFEQ, i - 1),
                  shift_quarter(cFEQ, i),
                  shift_quarter(cFEQ, i + 1)]

    # Step 4: compare fund_date vs dl_after snapshots within candidates
    if dl_after:
        with ENGINE.connect() as conn:
            snap_before = pd.read_sql(text("""
                SELECT period, value FROM income_data
                WHERE ticker = :t
                  AND metric_name = 'totalRevenues'
                  AND download_date = :d
                  AND value IS NOT NULL
                ORDER BY period
            """), conn, params={"t": t, "d": fund_date})

            snap_after = pd.read_sql(text("""
                SELECT period, value FROM income_data
                WHERE ticker = :t
                  AND metric_name = 'totalRevenues'
                  AND download_date = :d
                  AND value IS NOT NULL
                ORDER BY period
            """), conn, params={"t": t, "d": dl_after})

        if not snap_before.empty and not snap_after.empty:
            sb = snap_before.set_index('period')['value']
            sa = snap_after.set_index('period')['value']
            common = sorted(set(sb.index) & set(sa.index))

            for period in common:
                if period not in candidates:
                    continue
                denom = max(abs(float(sa[period])), 1.0)
                if abs(float(sb[period]) - float(sa[period])) / denom > 0.01:
                    return period

    # Step 5: fallback
    return shift_quarter(cFEQ, i)


# ==============================================================================
# METRICS CALCULATION
# ==============================================================================

def get_8q_backward(ticker: str, feq: str, metric: str, table: str) -> pd.Series:
    t        = clean_ticker(ticker)
    start_q  = add_quarters(feq, -8)
    quarters = [add_quarters(start_q, i) for i in range(8)]

    rows = []
    with ENGINE.connect() as conn:
        for period in quarters:
            row = conn.execute(text(f"""
                SELECT value FROM {table}
                WHERE ticker = :t
                  AND metric_name = :m
                  AND period = :p
                  AND value IS NOT NULL
                ORDER BY download_date DESC
                LIMIT 1
            """), {"t": t, "m": metric, "p": period}).fetchone()
            rows.append((period, float(row[0]) if row else np.nan))

    return pd.Series(dict(rows)).sort_index().ffill()


def get_4q_forward(ticker: str, feq: str, calc_date: pd.Timestamp) -> pd.Series:
    """
    Fetch 4 estimated revenue quarters starting at FEQ, as of calc_date.
    Uses the most recent download_date <= calc_date for each period.
    Returns a Series indexed by quarter string, or empty Series if unavailable.
    """
    t        = clean_ticker(ticker)
    quarters = [add_quarters(feq, i) for i in range(4)]

    rows = []
    with ENGINE.connect() as conn:
        for period in quarters:
            row = conn.execute(text("""
                SELECT value FROM income_data
                WHERE ticker = :t
                  AND metric_name = 'totalRevenues'
                  AND period = :p
                  AND download_date <= :d
                  AND value IS NOT NULL
                ORDER BY download_date DESC
                LIMIT 1
            """), {"t": t, "p": period, "d": calc_date}).fetchone()
            if row:
                rows.append((period, float(row[0])))

    return pd.Series(dict(rows)).sort_index() if rows else pd.Series(dtype=float)


def sym_growth(s: pd.Series) -> pd.Series:
    """Symmetric YoY growth rate for a quarterly series."""
    shifted = s.shift(4)
    denom   = ((s + shifted) / 2).abs().replace(0, np.nan)
    return ((s - shifted) / denom).replace([np.inf, -np.inf], np.nan)


def calc_gs_blended(ticker: str, feq: str, calc_date: pd.Timestamp) -> float:
    """
    Blended sales growth: GS = (2 * FWD_G + ACT_G) / 3

    FWD_G = median symmetric YoY growth for 4 estimated quarters [FEQ : FEQ+4]
            vs their year-ago actuals
    ACT_G = median symmetric YoY growth for 4 actual quarters [LAQ-3 : LAQ]
            vs their year-ago actuals (i.e. 8q backward window, last 4 pairs)

    Falls back to ACT_G only if forward estimates are unavailable.
    Returns percentage (e.g. 12.5 means 12.5%).
    """
    t   = clean_ticker(ticker)
    laq = add_quarters(feq, -1)   # last actual quarter

    # 8 actual quarters ending at LAQ (covers year-ago base for both windows)
    act_s = get_8q_backward(t, feq, 'totalRevenues', 'income_data')

    # ACT_G: last 4 actual pairs (positions 4-7 in the 8q series)
    act_growth = sym_growth(act_s)
    act_pairs  = act_growth.iloc[4:].dropna()
    if len(act_pairs) == 0:
        return np.nan
    act_g = float(np.nanmedian(act_pairs))

    # FWD_G: 4 estimated quarters + their year-ago actuals from act_s
    fwd_s = get_4q_forward(t, feq, calc_date)
    if fwd_s.empty:
        # fallback: ACT_G only
        return round(act_g * 100, 2)

    # Build combined series for FWD window: need year-ago actuals (LAQ-3 : LAQ)
    # Those are the last 4 quarters of act_s
    laq_idx     = sorted(act_s.index).index(laq) if laq in act_s.index else -1
    if laq_idx < 3:
        return round(act_g * 100, 2)

    year_ago = act_s.iloc[laq_idx - 3 : laq_idx + 1]   # 4 year-ago actuals
    combined = pd.concat([year_ago, fwd_s]).sort_index()
    combined = combined[~combined.index.duplicated(keep='last')]

    fwd_growth = sym_growth(combined)
    # Keep only the 4 forward periods
    fwd_pairs  = fwd_growth.reindex(fwd_s.index).dropna()

    n_fwd = len(fwd_pairs)
    if n_fwd == 0:
        return round(act_g * 100, 2)

    fwd_g = float(np.nanmedian(fwd_pairs))

    # Blend: 2:1 toward forward, scaled by how many fwd quarters available
    fwd_weight = 2.0 * (n_fwd / 4.0)   # scale down if fewer than 4 fwd quarters
    gs = (fwd_weight * fwd_g + act_g) / (fwd_weight + 1.0)
    return round(gs * 100, 2)


def calc_gs(s: pd.Series) -> float:
    """Legacy ACT_G only — kept for show_top_stocks YoY breakdown."""
    shifted = s.shift(4)
    denom   = ((s + shifted) / 2).abs().replace(0, np.nan)
    growth  = (s - shifted) / denom
    median  = np.nanmedian(growth.replace([np.inf, -np.inf], np.nan).dropna())
    return round(float(median) * 100, 2) if not np.isnan(median) else np.nan


def safe_val(mkt_cap: float, med: float) -> float:
    if np.isnan(mkt_cap) or np.isnan(med) or med == 0:
        return np.nan
    return round(mkt_cap / (4 * med), 2)


def calc_metrics_for_date(calc_date: pd.Timestamp,
                           Pxs_df: pd.DataFrame) -> pd.DataFrame:
    prices_on_date = Pxs_df.loc[calc_date].dropna()
    tickers        = prices_on_date.index.tolist()
    print(f"    {len(tickers)} tickers with price data")

    results = []
    errors  = 0

    for ticker in tickers:
        t = clean_ticker(ticker)
        try:
            feq = get_historical_feq(t, calc_date)
            if not feq:
                continue

            rev_s    = get_8q_backward(t, feq, 'totalRevenues',        'income_data')
            ni_s     = get_8q_backward(t, feq, 'netIncome',            'income_data')
            gp_s     = get_8q_backward(t, feq, 'grossProfit',          'income_data')
            shares_s = get_8q_backward(t, feq, 'dilutedAverageShares', 'income_data')

            shares  = shares_s.dropna().iloc[-1] if not shares_s.dropna().empty else np.nan
            px      = float(prices_on_date[ticker])
            mkt_cap = shares * px if not np.isnan(shares) else np.nan

            med_rev = np.nanmedian(rev_s.replace(0, np.nan))
            med_ni  = np.nanmedian(ni_s.replace(0, np.nan))
            med_gp  = np.nanmedian(gp_s.replace(0, np.nan))

            # GS components — single fetch used for GS, GS_vol, GS_adj
            comp   = calc_gs_components(t, calc_date)
            gs     = comp['GS']
            g_vals = pd.Series([comp[k] for k in
                                 ['YoY_Q1','YoY_Q2','YoY_Q3','YoY_Q4',
                                  'FWD_Q1','FWD_Q2','FWD_Q3','FWD_Q4']]).dropna()
            if len(g_vals) >= 2:
                gs_vol = max(g_vals.std() / 10, 1.0)
            else:
                gs_vol = 1.0
            gs_adj = gs / gs_vol if not np.isnan(gs) else np.nan

            results.append({
                'ticker':  t,
                'date':    calc_date,
                'GS':      gs,
                'GS_vol':  round(gs_vol, 3),
                'GS_adj':  round(gs_adj, 2) if not np.isnan(gs_adj) else np.nan,
                'P/S':     safe_val(mkt_cap, med_rev),
                'P/Ee':    safe_val(mkt_cap, med_ni),
                'P/GP':    safe_val(mkt_cap, med_gp),
                'mkt_cap': mkt_cap,
            })

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    ERROR {ticker}: {type(e).__name__}: {e}")

    print(f"    -> {len(results)} ok, {errors} errors")
    return pd.DataFrame(results) if results else pd.DataFrame()


# ==============================================================================
# FACTOR CONSTRUCTION
# ==============================================================================

def build_factor(metrics_df: pd.DataFrame,
                 sectors_s: pd.Series) -> pd.DataFrame:
    df = metrics_df.copy()

    # Ensure bare tickers
    df['ticker'] = df['ticker'].apply(clean_ticker)
    df['Sector'] = df['ticker'].map(sectors_s)
    df = df.dropna(subset=['Sector'])

    if df.empty or 'GS_adj' not in df.columns:
        return pd.DataFrame()

    sm_gs       = df.groupby('Sector')['GS_adj'].transform('median')
    df['GS_sn'] = df['GS_adj'] - sm_gs

    for col in ['P/S', 'P/Ee', 'P/GP']:
        if col not in df.columns:
            df[f'{col}_q'] = np.nan
            continue
        df[f'{col}_q'] = pd.qcut(
            df[col].where(df[col] > 0).rank(method='first'),
            q=N_QUANTILES,
            labels=list(range(1, N_QUANTILES + 1)),
            duplicates='drop'
        ).astype(float)

    df['val_q']  = df[['P/S_q', 'P/Ee_q', 'P/GP_q']].median(axis=1)
    df['factor'] = np.where(df['val_q'] > 0, df['GS_sn'] / df['val_q'], np.nan)
    df = df.dropna(subset=['GS_sn', 'P/S_q', 'P/Ee_q', 'P/GP_q', 'val_q', 'factor'])
    df = df.drop_duplicates(subset='ticker', keep='last')

    result            = df.set_index('ticker')[['factor', 'GS_sn', 'GS', 'GS_adj', 'GS_vol',
                                                 'P/S_q', 'P/Ee_q', 'P/GP_q', 'val_q', 'Sector']]
    result['mkt_cap'] = df.set_index('ticker')['mkt_cap'] if 'mkt_cap' in df.columns else np.nan
    return result


# ==============================================================================
# SECTOR CAP SELECTION
# ==============================================================================

def select_with_sector_cap(ranked_df: pd.DataFrame,
                            sector_cap: int,
                            top_n: int) -> pd.DataFrame:
    """
    Greedily select top_n stocks walking down ranked_df,
    respecting sector_cap. Relax cap by 1 and retry if
    not enough stocks, up to top_n (no constraint).
    """
    cap = sector_cap
    while cap <= top_n:
        selected      = []
        sector_counts = {}

        for ticker, row in ranked_df.iterrows():
            sector = row['Sector']
            count  = sector_counts.get(sector, 0)
            if count < cap:
                selected.append(ticker)
                sector_counts[sector] = count + 1
            if len(selected) == top_n:
                break

        if len(selected) == top_n:
            if cap > sector_cap:
                print(f"    Sector cap relaxed to {cap} to fill {top_n} slots")
            return ranked_df.loc[selected]

        cap += 1

    return ranked_df.head(top_n)


# ==============================================================================
# MOMENTUM
# ==============================================================================

def calc_momentum_12m1(rebal_date: pd.Timestamp,
                        candidates: pd.Index,
                        Pxs_df: pd.DataFrame) -> pd.Series:
    """
    Raw 12M1 price momentum, z-scored.
    candidates: bare tickers present in Pxs_df.columns.
    """
    all_dates = Pxs_df.index[Pxs_df.index <= rebal_date]

    if len(all_dates) < MOM_LONG + 1:
        return pd.Series(np.nan, index=candidates)

    date_start = all_dates[-(MOM_LONG + 1)]
    date_end   = all_dates[-(MOM_SKIP + 1)]

    valid = [t for t in candidates if t in Pxs_df.columns]
    if not valid:
        return pd.Series(np.nan, index=candidates)

    px_start = Pxs_df.loc[date_start, valid] if date_start in Pxs_df.index \
               else Pxs_df.loc[Pxs_df.index[Pxs_df.index >= date_start][0], valid]
    px_end   = Pxs_df.loc[date_end,   valid] if date_end   in Pxs_df.index \
               else Pxs_df.loc[Pxs_df.index[Pxs_df.index >= date_end][0],   valid]

    mom = (px_end - px_start) / px_start.replace(0, np.nan)
    mom = mom.dropna()

    if len(mom) < 5:
        return pd.Series(np.nan, index=candidates)

    return zscore(mom)


def load_idio_momentum_db() -> pd.DataFrame:
    """
    Load full idiosyncratic residuals from factor model DB.
    Returns pivot: date x ticker (bare tickers).
    """
    print("  Loading idiosyncratic momentum from DB...")
    df = pd.read_sql(text(f"""
        SELECT date, ticker, resid FROM {IDIO_MOM_TABLE}
        ORDER BY date, ticker
    """), ENGINE)
    df['date']   = pd.to_datetime(df['date'])
    df['ticker'] = df['ticker'].apply(clean_ticker)
    pivot = df.pivot_table(index='date', columns='ticker',
                           values='resid', aggfunc='last')
    print(f"  Idio residuals loaded: {pivot.shape[0]} dates x {pivot.shape[1]} stocks")
    return pivot


def calc_idio_momentum_score(rebal_date: pd.Timestamp,
                              candidates: pd.Index,
                              resid_pivot: pd.DataFrame,
                              vol_pivot: pd.DataFrame = None) -> pd.Series:
    """
    Cumulative idio residuals over [t-MOM_LONG, t-MOM_SKIP],
    z-scored cross-sectionally. Mirrors factor_model_step1 logic.
    vol_pivot is optional — if provided, residuals are additionally weighted
    by precomputed volume scalars at rebalance time (use with care if residuals
    from factor model already incorporate vol scaling).
    candidates: bare tickers.
    """
    all_resid_dates = resid_pivot.index[resid_pivot.index < rebal_date]
    if len(all_resid_dates) < MOM_LONG + 1:
        return pd.Series(np.nan, index=candidates)

    window    = all_resid_dates[-MOM_LONG:-MOM_SKIP]
    valid_stk = [t for t in candidates if t in resid_pivot.columns]

    if not valid_stk or len(window) < MOM_LONG - MOM_SKIP - 10:
        return pd.Series(np.nan, index=candidates)

    resid_window = resid_pivot.loc[window, valid_stk]

    if vol_pivot is not None:
        common_days = window.intersection(vol_pivot.index)
        vol_cols    = [t for t in valid_stk if t in vol_pivot.columns]
        if len(common_days) >= MOM_LONG - MOM_SKIP - 10 and vol_cols:
            resid_window = (resid_window.loc[common_days, vol_cols] *
                            vol_pivot.loc[common_days, vol_cols])

    cum_resid = resid_window.sum(axis=0).dropna()
    if len(cum_resid) < 5:
        return pd.Series(np.nan, index=candidates)

    return zscore(cum_resid)


# ==============================================================================
# ENHANCEMENT FILTERS
# ==============================================================================

def apply_vol_filter(candidates: pd.Index, rebal_date: pd.Timestamp,
                     Pxs_df: pd.DataFrame) -> pd.Index:
    """candidates: bare tickers present in Pxs_df.columns."""
    all_dates = Pxs_df.index[Pxs_df.index <= rebal_date]
    if len(all_dates) < 20:
        return candidates

    lookback = all_dates[-126:]
    valid    = [t for t in candidates if t in Pxs_df.columns]
    px       = Pxs_df.loc[lookback, valid].ffill()
    vol      = px.pct_change().std() * np.sqrt(252)
    cutoff   = vol.quantile(0.8)
    return vol[vol <= cutoff].index


# ==============================================================================
# BACKTEST
# ==============================================================================

def run_backtest(factor_by_date: dict,
                 calc_dates: list,
                 Pxs_df: pd.DataFrame,
                 use_vol_filter: bool = False,
                 use_mom_12m1: bool = False,
                 use_mom_idio: bool = False,
                 resid_pivot: pd.DataFrame = None,
                 vol_pivot: pd.DataFrame = None,
                 mktcap_floor: float = None,
                 sector_cap: int = None,
                 top_n: int = TOP_N,
                 mom_weight: float = 1.0) -> tuple:
    """
    All tickers are bare (no ' US') throughout.
    Pxs_df columns are bare tickers.
    factor_by_date index is bare tickers.
    """
    nav          = 1.0
    nav_series   = {}
    portfolio    = []
    port_records = {}
    pxs_columns  = set(Pxs_df.columns)

    for i, rebal_date in enumerate(calc_dates):
        next_date = calc_dates[i + 1] if i + 1 < len(calc_dates) else Pxs_df.index.max()

        if rebal_date in factor_by_date:
            fdf = factor_by_date[rebal_date].copy()

            # Market cap floor
            if mktcap_floor is not None and 'mkt_cap' in fdf.columns:
                fdf = fdf[fdf['mkt_cap'].fillna(0) >= mktcap_floor]

            # Filter to tickers present in Pxs_df (bare tickers, direct match)
            fdf = fdf.loc[[t for t in fdf.index if t in pxs_columns]]

            # Vol filter
            if use_vol_filter and len(fdf) > top_n:
                surviving = apply_vol_filter(fdf.index, rebal_date, Pxs_df)
                fdf       = fdf.loc[fdf.index.intersection(surviving)]

            if len(fdf) < top_n:
                print(f"  Skipping {rebal_date.date()}: "
                      f"only {len(fdf)} stocks after filters (need {top_n})")
                portfolio = []
            else:
                # Momentum blend
                if use_mom_12m1 or use_mom_idio:
                    fdf['factor_z'] = zscore(fdf['factor'])

                    if use_mom_12m1:
                        mom_z = calc_momentum_12m1(rebal_date, fdf.index, Pxs_df)
                    else:
                        mom_z = calc_idio_momentum_score(rebal_date, fdf.index,
                                                          resid_pivot,
                                                          vol_pivot=vol_pivot)

                    fdf['mom_z']    = mom_z.reindex(fdf.index)
                    fdf['combined'] = (fdf['factor_z'] + mom_weight * fdf['mom_z']) / (1.0 + mom_weight)
                    fdf             = fdf.dropna(subset=['combined'])
                    rank_col        = 'combined'
                else:
                    rank_col = 'factor'

                if len(fdf) < top_n:
                    print(f"  Skipping {rebal_date.date()}: "
                          f"only {len(fdf)} stocks after momentum dropna "
                          f"(need {top_n})")
                    portfolio = []
                else:
                    ranked = fdf.sort_values(rank_col, ascending=False)

                    # Sector cap with gradual relaxation
                    if sector_cap is not None:
                        top = select_with_sector_cap(ranked, sector_cap, top_n)
                    else:
                        top = ranked.head(top_n)

                    # Portfolio is bare tickers directly usable in Pxs_df
                    portfolio = [t for t in top.index if t in pxs_columns]

                    port_records[rebal_date] = (
                        list(top.index) + [None] * (top_n - len(top.index))
                    )[:top_n]

        if not portfolio:
            period_dates = Pxs_df.index[
                (Pxs_df.index >= rebal_date) & (Pxs_df.index < next_date)
            ]
            for d in period_dates:
                nav_series[d] = nav
            continue

        period_dates = Pxs_df.index[
            (Pxs_df.index >= rebal_date) & (Pxs_df.index < next_date)
        ]

        if len(period_dates) < 2:
            nav_series[rebal_date] = nav
            continue

        px_period  = Pxs_df.loc[period_dates, portfolio].ffill()
        daily_rets = px_period.pct_change().fillna(0)
        port_rets  = daily_rets.mean(axis=1)

        for d, r in port_rets.items():
            nav *= (1 + r)
            nav_series[d] = nav

    nav_s = pd.Series(nav_series).sort_index()
    if START_DATE not in nav_s.index:
        nav_s[START_DATE] = 1.0
        nav_s = nav_s.sort_index()

    port_df = pd.DataFrame.from_dict(
        port_records, orient='index',
        columns=[f'Stock{i+1}' for i in range(top_n)]
    )

    return nav_s, port_df


# ==============================================================================
# PERFORMANCE SUMMARY
# ==============================================================================

def print_performance(nav_s: pd.Series, label: str = ''):
    total_ret  = nav_s.iloc[-1] / nav_s.iloc[0] - 1
    n_years    = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
    cagr       = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / n_years) - 1
    daily_rets = nav_s.pct_change().dropna()
    vol        = daily_rets.std() * np.sqrt(252)
    sharpe     = cagr / vol if vol > 0 else np.nan
    max_dd     = ((nav_s / nav_s.cummax()) - 1).min()

    print(f"\n  {'='*70}")
    print(f"  PERFORMANCE: {label}")
    print(f"  {'='*70}")
    print(f"  Period      : {nav_s.index[0].date()} -> {nav_s.index[-1].date()}")
    print(f"  Total return: {total_ret*100:+.1f}%")
    print(f"  CAGR        : {cagr*100:+.1f}%")
    print(f"  Annual vol  : {vol*100:.1f}%")
    print(f"  Sharpe      : {sharpe:.2f}")
    print(f"  Max drawdown: {max_dd*100:.1f}%")
    print(f"  Final NAV   : {nav_s.iloc[-1]:.4f}")


# ==============================================================================
# MAIN
# ==============================================================================

def run(Pxs_df: pd.DataFrame,
        sectors_s: pd.Series,
        force_recalc: bool = False,
        volumeTrd_df: pd.DataFrame = None):
    """
    Returns: nav_base, nav_alt, port_base, port_alt
    nav_alt and port_alt are None if no enhancements selected.
    All tickers assumed bare (no ' US' extension).
    """
    print("=" * 70)
    print("  PRIMARY FACTOR BACKTEST")
    print("=" * 70)

    # Deduplicate upfront
    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]

    print("\n  ENHANCEMENT OPTIONS (vs baseline):")
    vol_input    = input("  Apply vol filter? (y/n) [default=n]: ").strip().lower()
    mom_input    = input("  Momentum blend? (n / 12m1 / idio) [default=n]: ").strip().lower()
    mktcap_input = input("  Min market cap floor ($M, or Enter to skip): ").strip()
    sector_input = input("  Max stocks per sector (or Enter to skip): ").strip()
    topn_input   = input("  Number of stocks in portfolio (or Enter for default=20): ").strip()

    use_vol      = vol_input == 'y'
    use_mom_12m1 = mom_input == '12m1'
    use_mom_idio = mom_input == 'idio'
    use_mom      = use_mom_12m1 or use_mom_idio
    if use_mom:
        mom_w_input  = input("  Momentum weight (vs factor weight of 1.0) [default=1.0]: ").strip()
        mom_weight    = float(mom_w_input) if mom_w_input else 1.0
        vol_vs_input  = input("  Volume-scaled idio? (y/n) [default=n]: ").strip().lower()
        use_vol_scale = vol_vs_input == 'y'
    else:
        mom_weight    = 1.0
        use_vol_scale = False
    mktcap_floor = float(mktcap_input) * 1e6 if mktcap_input else None
    sector_cap   = int(sector_input)          if sector_input else None
    top_n        = int(topn_input)            if topn_input   else TOP_N
    run_alt      = use_vol or use_mom or mktcap_floor is not None \
                   or sector_cap is not None or top_n != TOP_N

    # Load idio residuals if needed
    resid_pivot = None
    vol_pivot   = None
    if use_mom_idio:
        resid_pivot = load_idio_momentum_db()
        if use_vol_scale:
            if volumeTrd_df is None:
                print("  WARNING: vol scaling requested but volumeTrd_df not provided — disabling")
                use_vol_scale = False
            else:
                vol_pivot = volumeTrd_df

    if run_alt:
        label = ' + '.join(filter(None, [
            'VolFilter'                     if use_vol               else '',
            f'Mom(12M1,w={mom_weight})'      if use_mom_12m1          else '',
            f'Mom(Idio,w={mom_weight}{"_VS" if use_vol_scale else ""})'  if use_mom_idio else '',
            f'MktCap>=${mktcap_input}M'     if mktcap_floor          else '',
            f'SectorCap={sector_cap}'       if sector_cap            else '',
            f'N={top_n}'                    if top_n != TOP_N        else '',
        ]))
        print(f"\n  Enhancements: {label}")
    else:
        print("\n  No enhancements selected -- baseline only")

    create_target_table()

    calc_dates    = generate_calc_dates(Pxs_df)
    already_done  = get_already_calculated_dates()
    dates_to_calc = calc_dates if force_recalc else [d for d in calc_dates
                                                      if d not in already_done]

    print(f"\n  Calculation dates : {len(calc_dates)} total")
    print(f"  Already in DB     : {len(already_done)}")
    print(f"  To calculate      : {len(dates_to_calc)}")

    confirm = input("\n  Proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("  Cancelled.")
        return None, None, None, None

    # Step 1: Calculate and store metrics
    if dates_to_calc:
        print(f"\n  Calculating metrics for {len(dates_to_calc)} dates...")
        for d_idx, calc_date in enumerate(dates_to_calc, 1):
            print(f"  [{d_idx}/{len(dates_to_calc)}] {calc_date.date()}...")
            metrics_df = calc_metrics_for_date(calc_date, Pxs_df)
            if not metrics_df.empty:
                save_metrics(metrics_df)

    # Step 2: Load metrics from DB
    print("\n  Loading metrics from DB...")
    all_metrics = pd.read_sql(text(f"""
        SELECT * FROM {TARGET_TABLE}
        WHERE date >= :start
        ORDER BY date, ticker
    """), ENGINE, params={"start": START_DATE})
    all_metrics['date']   = pd.to_datetime(all_metrics['date'])
    all_metrics['ticker'] = all_metrics['ticker'].apply(clean_ticker)
    print(f"  Loaded {len(all_metrics)} rows | "
          f"{all_metrics['date'].nunique()} dates | "
          f"{all_metrics['ticker'].nunique()} tickers")

    # Step 3: Build factor for each date
    print("\n  Building factor scores...")
    factor_by_date = {}
    for calc_date, grp in all_metrics.groupby('date'):
        fdf = build_factor(grp, sectors_s)
        if not fdf.empty:
            factor_by_date[calc_date] = fdf
    print(f"  Factor built for {len(factor_by_date)} dates")

    # Step 4: Baseline backtest (no filters, default 20 stocks)
    print("\n  Running baseline backtest...")
    nav_base, port_base = run_backtest(
        factor_by_date, calc_dates, Pxs_df,
        use_vol_filter=False,
        use_mom_12m1=False,
        use_mom_idio=False,
        resid_pivot=None,
        mktcap_floor=None,
        sector_cap=None,
        top_n=TOP_N,
    )
    print_performance(nav_base, "BASELINE")

    # Step 5: Alternative backtest
    nav_alt  = None
    port_alt = None
    if run_alt:
        print(f"\n  Running alternative backtest ({label})...")
        nav_alt, port_alt = run_backtest(
            factor_by_date, calc_dates, Pxs_df,
            use_vol_filter=use_vol,
            use_mom_12m1=use_mom_12m1,
            use_mom_idio=use_mom_idio,
            resid_pivot=resid_pivot,
            vol_pivot=vol_pivot,
            mktcap_floor=mktcap_floor,
            sector_cap=sector_cap,
            top_n=top_n,
            mom_weight=mom_weight,
        )
        print_performance(nav_alt, f"ALTERNATIVE ({label})")

        # Side-by-side comparison
        print(f"\n  {'='*70}")
        print(f"  COMPARISON")
        print(f"  {'='*70}")
        print(f"  {'':40} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'MDD':>8}")
        print(f"  {'-'*75}")
        for nav_s, lbl in [(nav_base, 'Baseline'), (nav_alt, f'Alt ({label})')]:
            n_yrs  = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
            cagr   = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1 / n_yrs) - 1
            vol    = nav_s.pct_change().dropna().std() * np.sqrt(252)
            sharpe = cagr / vol if vol > 0 else np.nan
            mdd    = ((nav_s / nav_s.cummax()) - 1).min()
            print(f"  {lbl:<40} {cagr*100:>7.1f}% {vol*100:>7.1f}% "
                  f"{sharpe:>8.2f} {mdd*100:>7.1f}%")

    print(f"\n  port_base: {len(port_base)} rebalance dates x {len(port_base.columns)} stocks")
    if port_alt is not None:
        print(f"  port_alt : {len(port_alt)} rebalance dates x {len(port_alt.columns)} stocks")

    return nav_base, nav_alt, port_base, port_alt, factor_by_date, all_metrics


# ==============================================================================
# TOP STOCKS DIAGNOSTIC
# ==============================================================================

def calc_gs_components(ticker: str, calc_date: pd.Timestamp) -> dict:
    """
    Re-fetch revenue series and return individual YoY pairs plus blended GS.
    Returns: {YoY_Q1..Q4 (ACT), FWD_Q1..Q4, ACT_G, FWD_G, GS}
    """
    result = {f'YoY_Q{i}': np.nan for i in range(1, 5)}
    result.update({f'FWD_Q{i}': np.nan for i in range(1, 5)})
    result.update({'ACT_G': np.nan, 'FWD_G': np.nan, 'GS': np.nan})
    try:
        feq = get_historical_feq(ticker, calc_date)
        if not feq:
            return result

        # Actual pairs
        act_s  = get_8q_backward(ticker, feq, 'totalRevenues', 'income_data')
        growth = sym_growth(act_s) * 100
        for i, v in enumerate(growth.iloc[4:].values[:4], 1):
            result[f'YoY_Q{i}'] = round(float(v), 1) if not np.isnan(v) else np.nan
        act_pairs = growth.iloc[4:].dropna()
        result['ACT_G'] = round(float(np.nanmedian(act_pairs)), 1) if len(act_pairs) else np.nan

        # Forward pairs
        fwd_s = get_4q_forward(ticker, feq, calc_date)
        if not fwd_s.empty:
            laq     = add_quarters(feq, -1)
            laq_idx = sorted(act_s.index).index(laq) if laq in act_s.index else -1
            if laq_idx >= 3:
                year_ago   = act_s.iloc[laq_idx - 3: laq_idx + 1]
                combined   = pd.concat([year_ago, fwd_s]).sort_index()
                combined   = combined[~combined.index.duplicated(keep='last')]
                fwd_growth = sym_growth(combined).reindex(fwd_s.index) * 100
                for i, v in enumerate(fwd_growth.values[:4], 1):
                    result[f'FWD_Q{i}'] = round(float(v), 1) if not np.isnan(v) else np.nan
                fwd_pairs = fwd_growth.dropna()
                result['FWD_G'] = round(float(np.nanmedian(fwd_pairs)), 1) if len(fwd_pairs) else np.nan

        result['GS'] = calc_gs_blended(ticker, feq, calc_date)
    except Exception:
        pass
    return result


def show_top_stocks(factor_by_date: dict,
                    all_metrics: pd.DataFrame,
                    sectors_s: pd.Series,
                    n: int = 50,
                    ref_date: str = None) -> pd.DataFrame:
    """
    Show top N ranked stocks with full scoring breakdown for diagnostic purposes.

    Parameters
    ----------
    factor_by_date : dict  — output from run()
    all_metrics    : df    — output from run(), raw metrics from DB
    sectors_s      : Series — sector mapping (bare tickers)
    n              : int   — number of stocks to show (default 50)
    ref_date       : str   — 'YYYY-MM-DD' or None for last rebalance date.
                             Uses last rebalance date <= ref_date if provided.

    Returns
    -------
    DataFrame with columns:
        Sector, mkt_cap($B), GS_sn, YoY_Q1..Q4, GS,
        P/S_q, P/Ee_q, P/GP_q, val_q, factor
    """
    rebal_dates = sorted(factor_by_date.keys())

    if ref_date is not None:
        ref_ts     = pd.Timestamp(ref_date)
        candidates = [d for d in rebal_dates if d <= ref_ts]
        if not candidates:
            print(f"  No rebalance date <= {ref_date}")
            return pd.DataFrame()
        chosen = max(candidates)
    else:
        chosen = rebal_dates[-1]

    print(f"\n  {'='*70}")
    print(f"  TOP {n} STOCKS — Rebalance date: {chosen.date()}")
    print(f"  {'='*70}")

    fdf = factor_by_date[chosen].copy()
    fdf = fdf.sort_values('factor', ascending=False)
    top = fdf.head(n).copy()

    # mkt_cap in $B
    top['mkt_cap($B)'] = (top['mkt_cap'] / 1e9).round(2)

    # Fetch individual YoY components for each stock
    print(f"  Fetching YoY growth components for {n} stocks...")
    comp_rows = {}
    for i, ticker in enumerate(top.index, 1):
        if i % 10 == 0:
            print(f"    {i}/{n}...")
        comp_rows[ticker] = calc_gs_components(ticker, chosen)

    comp_df = pd.DataFrame(comp_rows).T
    top     = top.join(comp_df, how='left')

    # Compute GS_vol and GS_adj fresh from components
    g_cols = ['YoY_Q1','YoY_Q2','YoY_Q3','YoY_Q4','FWD_Q1','FWD_Q2','FWD_Q3','FWD_Q4']
    top['GS_vol'] = (top[g_cols].std(axis=1) / 10).clip(lower=1.0)
    top['GS_adj'] = top['GS'] / top['GS_vol']

    # Recompute GS_sn on GS_adj, sector-neutralized within top-N sample
    sm_gs        = top.groupby('Sector')['GS_adj'].transform('median')
    top['GS_sn'] = (top['GS_adj'] - sm_gs).round(2)

    # Recompute factor with fresh GS_sn
    top['factor'] = np.where(top['val_q'] > 0,
                             top['GS_sn'] / top['val_q'],
                             np.nan)

    # Re-rank by fresh factor descending
    top = top.sort_values('factor', ascending=False)

    out = top[[
        'Sector', 'mkt_cap($B)',
        'GS_sn',
        'YoY_Q1', 'YoY_Q2', 'YoY_Q3', 'YoY_Q4', 'ACT_G',
        'FWD_Q1', 'FWD_Q2', 'FWD_Q3', 'FWD_Q4', 'FWD_G',
        'GS', 'GS_vol', 'GS_adj',
        'P/S_q', 'P/Ee_q', 'P/GP_q',
        'val_q', 'factor'
    ]].copy()

    pd.set_option('display.max_rows',     n + 5)
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.width',        240)
    print(out.to_string())
    print(f"\n  NOTE: GS_adj = GS / GS_vol, GS_sn sector-neutralized on GS_adj.")
    print(f"  All values recomputed fresh — stale DB values overridden.")

    return out


if __name__ == "__main__":
    print("Usage: from primary_factor_backtest import run, show_top_stocks")
    print("       nav_base, nav_alt, port_base, port_alt, factor_by_date, all_metrics = run(Pxs_df, sectors_s)")
    print("       top50 = show_top_stocks(factor_by_date, all_metrics, sectors_s)")
    print("       top50 = show_top_stocks(factor_by_date, all_metrics, sectors_s, ref_date='2024-06-01')")
