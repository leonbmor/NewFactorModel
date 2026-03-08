#!/usr/bin/env python
# coding: utf-8

"""
Factor Model - Step 1
======================
Sequential Fama-MacBeth factor residualization.
Factors added in order of relevance, validated by variance reduction.

Steps:
  1. UFV      : full variance of raw returns (baseline, from st_dt)
  2. mkt_UFV  : + market beta (EWMA, hl=126, window=252)
  3. size_UFV : + size (z-scored log market cap, cross-sectional)
  4. sec_UFV  : + sector dummies (reference: XLP)
  5. mom_UFV  : + idiosyncratic momentum (cumulative sec residuals [t-252, t-21])

Size weights in OLS: log(dynamic market cap) — share count from valuation_consolidated,
price updated daily from Pxs_df.

Residuals saved to DB: factor_residuals_mkt, factor_residuals_size,
                       factor_residuals_sec, factor_residuals_mom
Lambda tables saved  : factor_lambdas_mkt, factor_lambdas_size,
                       factor_lambdas_sec, factor_lambdas_mom

Usage:
    from factor_model_step1 import run
    results = run(Pxs_df, sectors_s)
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

ENGINE          = create_engine("postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db")
BETA_WINDOW     = 252
BETA_HL         = 126
MOM_LONG        = 252
MOM_SKIP        = 21
MOM_LONG_BUFFER = MOM_LONG
MIN_STOCKS      = 150
SECTOR_REF      = 'XLP'


# ==============================================================================
# HELPERS
# ==============================================================================

def clean_ticker(t: str) -> str:
    return t.strip().split(' ')[0].upper()


def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


# ==============================================================================
# UNIVERSE
# ==============================================================================

def get_universe(Pxs_df: pd.DataFrame, sectors_s: pd.Series,
                 extended_st_dt: pd.Timestamp) -> list:
    with ENGINE.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ticker FROM income_data
        """)).fetchall()
    db_tickers  = {r[0].upper() for r in rows}
    etf_tickers = set(sectors_s.values)
    pre_dates   = Pxs_df.index[Pxs_df.index < extended_st_dt]

    universe = []
    for col in Pxs_df.columns:
        if col in ('SPX',) or col in etf_tickers:
            continue
        if col.upper() not in db_tickers:
            continue
        if col not in sectors_s.index:
            continue
        if len(pre_dates) >= BETA_WINDOW:
            col_data = Pxs_df.loc[pre_dates[-BETA_WINDOW:], col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            if int(col_data.notna().sum()) < BETA_WINDOW // 2:
                continue
        universe.append(col)

    print(f"  Universe: {len(universe)} stocks "
          f"(in DB + sector mapped + sufficient history)")
    return universe


# ==============================================================================
# DYNAMIC SIZE
# ==============================================================================

def load_size_df(universe: list, Pxs_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each stock and each date in Pxs_df:
      1. Find last available date in valuation_consolidated before calc_date
      2. Extract Size_db and Price_db (from Pxs_df on that same date)
      3. Compute shares = Size_db / Price_db
      4. Dynamic size = shares * Price on calc_date
      Fallback: if Price_db is NaN, use Size_db as-is.

    Returns DataFrame (date x ticker) of dynamic market caps.
    """
    print("  Loading size data from valuation_consolidated...")
    us_tickers = [t + ' US' for t in universe]

    with ENGINE.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, ticker, "Size"
            FROM valuation_consolidated
            WHERE "Size" IS NOT NULL
              AND ticker = ANY(:tickers)
            ORDER BY ticker, date
        """), {"tickers": us_tickers}).fetchall()

    size_raw           = pd.DataFrame(rows, columns=['date', 'ticker', 'Size'])
    size_raw['date']   = pd.to_datetime(size_raw['date'])
    size_raw['ticker'] = size_raw['ticker'].str.replace(' US', '', regex=False)

    # Pivot: date x ticker — raw DB sizes
    size_pivot = size_raw.pivot_table(
        index='date', columns='ticker', values='Size', aggfunc='last'
    )

    all_dates  = Pxs_df.index
    # Forward fill raw sizes to all Pxs_df dates — gives us last known Size_db per date
    size_ff    = size_pivot.reindex(all_dates).ffill()

    # Also track which DB date each forward-filled value came from
    # We do this by forward-filling a date indicator
    date_grid  = pd.DataFrame(
        index=size_pivot.index,
        columns=size_pivot.columns,
        data=np.tile(size_pivot.index.values.reshape(-1, 1),
                     (1, len(size_pivot.columns)))
    )
    date_grid  = date_grid.reindex(all_dates).ffill()

    print(f"  Raw size pivot: {size_pivot.shape} | "
          f"Forward-filled to {size_ff.shape}")

    # Build dynamic size: shares * current price
    print("  Computing dynamic size (shares * daily price)...")
    dynamic_size = pd.DataFrame(index=all_dates, columns=universe, dtype=float)

    for ticker in universe:
        if ticker not in size_ff.columns:
            continue

        size_db_series  = size_ff[ticker]           # forward-filled Size_db per date
        db_date_series  = date_grid[ticker]          # corresponding DB snapshot date

        px_series       = Pxs_df[ticker]             # daily prices

        # For each calc_date, get Price_db = price on DB snapshot date
        # Vectorized: map db_date -> price
        db_dates_arr    = pd.to_datetime(db_date_series.values)
        # Build price lookup: date -> price for this ticker
        px_lookup       = px_series.to_dict()

        price_db = pd.Series(
            [px_lookup.get(d, np.nan) for d in db_dates_arr],
            index=all_dates
        )

        # shares = Size_db / Price_db (fallback to Size_db if Price_db is NaN)
        shares = size_db_series / price_db.replace(0, np.nan)
        # Where Price_db is NaN, fallback: implied shares = Size_db / 1 (use raw)
        shares_fallback = size_db_series.copy()
        shares          = shares.where(shares.notna(), shares_fallback)

        # Dynamic size = shares * current price
        dyn = shares * px_series
        # Where current price is NaN, fallback to Size_db
        dyn = dyn.where(dyn.notna(), size_db_series)

        dynamic_size[ticker] = dyn

    dynamic_size = dynamic_size.astype(float)
    print(f"  Dynamic size computed: {dynamic_size.shape}")
    return dynamic_size


def get_log_size(dynamic_size: pd.DataFrame,
                 calc_date: pd.Timestamp,
                 valid_idx: pd.Index) -> pd.Series:
    """
    Returns log(dynamic_size) for valid_idx on calc_date.
    Used as OLS weights (not z-scored).
    Falls back to 1.0 where missing.
    """
    if calc_date not in dynamic_size.index:
        return pd.Series(1.0, index=valid_idx)
    s = dynamic_size.loc[calc_date, valid_idx].reindex(valid_idx)
    s = np.log(s.clip(lower=1).fillna(1))
    return s


# ==============================================================================
# SECTOR DUMMIES
# ==============================================================================

def build_sector_dummies(universe: list, sectors_s: pd.Series) -> pd.DataFrame:
    sectors_dedup = sectors_s[~sectors_s.index.duplicated(keep='first')]
    etfs          = sorted(set(
        sectors_dedup.loc[sectors_dedup.index.isin(universe)].dropna().values
    ))
    etfs_use      = [e for e in etfs if e != SECTOR_REF]

    dummies = pd.DataFrame(0, index=universe, columns=etfs_use)
    for stk in universe:
        etf = sectors_dedup.get(stk)
        if etf is not None and etf in etfs_use:
            dummies.loc[stk, etf] = 1

    print(f"  Sector dummies: {len(etfs_use)} sectors "
          f"(reference: {SECTOR_REF} dropped)")
    return dummies


# ==============================================================================
# ROLLING CHARACTERISTICS
# ==============================================================================

def calc_rolling_betas(Pxs_df: pd.DataFrame, universe: list,
                        calc_dates: pd.DatetimeIndex) -> pd.DataFrame:
    print("  Calculating rolling EWMA betas...")
    spx_rets  = Pxs_df['SPX'].pct_change()
    stk_rets  = Pxs_df[universe].pct_change()
    all_dates = Pxs_df.index
    betas     = {}

    for dt in calc_dates:
        window   = all_dates[all_dates < dt][-BETA_WINDOW:]
        if len(window) < BETA_WINDOW // 2:
            continue

        spx_w    = spx_rets.loc[window].values
        stk_w_df = stk_rets.loc[window].fillna(0)
        cols     = stk_w_df.columns.tolist()
        stk_w    = stk_w_df.values

        n        = len(window)
        alpha    = 1 - np.exp(-np.log(2) / BETA_HL)
        weights  = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()

        spx_mean = np.dot(weights, spx_w)
        stk_mean = stk_w.T @ weights
        spx_dev  = spx_w - spx_mean
        stk_dev  = stk_w - stk_mean[np.newaxis, :]

        cov      = (stk_dev * spx_dev[:, np.newaxis] *
                    weights[:, np.newaxis]).sum(axis=0)
        var_spx  = np.dot(weights, spx_dev ** 2)

        beta_t    = cov / var_spx if var_spx > 0 \
                    else np.full(len(cols), np.nan)
        betas[dt] = pd.Series(beta_t, index=cols)

    beta_df = pd.DataFrame(betas).T.reindex(columns=universe)
    beta_df.index.name = 'date'
    print(f"  Betas computed: {len(beta_df)} dates")
    return beta_df


def calc_idio_momentum(resid_sec_df: pd.DataFrame,
                        calc_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Idiosyncratic momentum: cumulative sum of step-4 (sec) residuals
    over [t-252, t-21] in residual-date space, z-scored cross-sectionally.
    """
    print("  Calculating idiosyncratic momentum from sec residuals...")
    all_resid_dates = resid_sec_df.index
    mom_dict        = {}

    for dt in calc_dates:
        past = all_resid_dates[all_resid_dates < dt]
        if len(past) < MOM_LONG + 1:
            continue

        window = past[-MOM_LONG:-MOM_SKIP]
        if len(window) < MOM_LONG - MOM_SKIP - 10:
            continue

        cum_resid = resid_sec_df.loc[window].sum(axis=0)
        valid     = cum_resid.dropna()

        if len(valid) < MIN_STOCKS:
            continue

        mom_dict[dt] = zscore(valid)

    mom_df = pd.DataFrame(mom_dict).T.reindex(columns=resid_sec_df.columns)
    mom_df.index.name = 'date'
    print(f"  Idiosyncratic momentum computed: {len(mom_df)} dates")
    return mom_df


# ==============================================================================
# CROSS-SECTIONAL WLS
# ==============================================================================

def wls_cross_section(y: pd.Series, X: pd.DataFrame,
                       w: pd.Series) -> tuple:
    idx  = y.index.intersection(X.index).intersection(w.index)
    if len(idx) < 10:
        return None, None, None

    y_   = y.loc[idx].values
    X_   = np.column_stack([np.ones(len(idx)), X.loc[idx].values])
    w_   = w.loc[idx].values
    w_   = np.where(np.isnan(w_) | (w_ <= 0), 1.0, w_)
    w_   = w_ / w_.sum()

    W    = np.diag(w_)
    try:
        XtW  = X_.T @ W
        lam  = np.linalg.solve(XtW @ X_, XtW @ y_)
    except np.linalg.LinAlgError:
        return None, None, None

    fitted  = X_ @ lam
    resid   = y_ - fitted
    ss_res  = np.dot(w_, resid ** 2)
    ss_tot  = np.dot(w_, (y_ - np.dot(w_, y_)) ** 2)
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return lam, pd.Series(resid, index=idx), r2


# ==============================================================================
# STORAGE
# ==============================================================================

def save_lambdas(lambda_df: pd.DataFrame, table_name: str):
    with ENGINE.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
    lambda_df.to_sql(table_name, ENGINE, if_exists='replace',
                     index=True, index_label='date')
    print(f"  Lambdas saved to '{table_name}' ({len(lambda_df)} rows)")


def save_residuals(resid_df: pd.DataFrame, table_name: str):
    print(f"  Saving residuals to '{table_name}'...")
    long           = resid_df.stack().reset_index()
    long.columns   = ['date', 'ticker', 'resid']
    long['date']   = pd.to_datetime(long['date'])

    with ENGINE.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        conn.execute(text(f"""
            CREATE TABLE {table_name} (
                date   DATE,
                ticker VARCHAR(20),
                resid  NUMERIC,
                PRIMARY KEY (date, ticker)
            )
        """))
    long.to_sql(table_name, ENGINE, if_exists='append', index=False)
    print(f"  Saved {len(long):,} rows "
          f"({resid_df.shape[0]} dates x {resid_df.shape[1]} stocks)")


# ==============================================================================
# VARIANCE / R2 STATS
# ==============================================================================

def variance_stats(resid_df: pd.DataFrame, label: str,
                    reference_var: float = None) -> float:
    vals = resid_df.values.flatten()
    vals = vals[~np.isnan(vals)]
    var  = float(np.var(vals))
    std  = float(np.std(vals))

    print(f"\n  [{label}]")
    print(f"    Pooled variance : {var:.8f}")
    print(f"    Pooled std dev  : {std:.6f}")
    print(f"    N observations  : {len(vals):,}")
    if reference_var is not None:
        print(f"    % of reference  : {var / reference_var * 100:.2f}%")
    return var


def r2_stats(r2_series: pd.Series, label: str):
    r2 = r2_series.dropna()
    print(f"\n  [{label}] Daily cross-sectional R²:")
    print(f"    Mean   : {r2.mean():.4f}")
    print(f"    Median : {r2.median():.4f}")
    print(f"    10th   : {r2.quantile(0.10):.4f}")
    print(f"    90th   : {r2.quantile(0.90):.4f}")


def lambda_stats(series: pd.Series, label: str) -> float:
    s       = series.dropna()
    mean    = s.mean()
    std     = s.std()
    t_stat  = mean / (std / np.sqrt(len(s)))
    pct_pos = (s > 0).mean() * 100

    print(f"\n  {label}")
    print(f"    N          : {len(s):,}")
    print(f"    Mean       : {mean:+.6f}")
    print(f"    Std        : {std:.6f}")
    print(f"    t-stat     : {t_stat:+.2f}")
    print(f"    % positive : {pct_pos:.1f}%")
    print(f"    Min        : {s.min():+.6f}")
    print(f"    5th pct    : {s.quantile(0.05):+.6f}")
    print(f"    Median     : {s.median():+.6f}")
    print(f"    95th pct   : {s.quantile(0.95):+.6f}")
    print(f"    Max        : {s.max():+.6f}")
    return t_stat


# ==============================================================================
# GENERIC FACTOR STEP RUNNER
# ==============================================================================

def run_factor_step(factor_cols: list,
                     char_by_date: dict,
                     all_rets: pd.DataFrame,
                     dynamic_size: pd.DataFrame,
                     calc_dates: pd.DatetimeIndex,
                     universe: list) -> tuple:
    resid_dict  = {}
    lambda_dict = {}
    r2_dict     = {}

    for dt in calc_dates:
        y = all_rets.loc[dt, universe].dropna()
        if len(y) < MIN_STOCKS:
            continue

        valid_idx = y.index
        X_parts   = []

        for col, char_df in char_by_date.items():
            if dt not in char_df.index:
                valid_idx = pd.Index([])
                break
            s         = char_df.loc[dt].reindex(valid_idx).dropna()
            valid_idx = s.index
            X_parts.append(s.rename(col))

        if len(valid_idx) < MIN_STOCKS or not X_parts:
            continue

        X  = pd.concat(X_parts, axis=1).loc[valid_idx]
        y_ = y.loc[valid_idx]
        w_ = get_log_size(dynamic_size, dt, valid_idx)

        lam, resid, r2 = wls_cross_section(y_, X, w_)
        if resid is None:
            continue

        resid_dict[dt]  = resid
        r2_dict[dt]     = r2
        cols            = ['intercept'] + factor_cols
        lambda_dict[dt] = {**dict(zip(cols, lam)), 'r2': r2}

    resid_df  = pd.DataFrame(resid_dict).T
    lambda_df = pd.DataFrame(lambda_dict).T
    lambda_df.index.name = 'date'
    r2_s      = pd.Series(r2_dict)

    return resid_df, lambda_df, r2_s


# ==============================================================================
# LAMBDA SUMMARY
# ==============================================================================

def print_lambda_summary(lambda_df: pd.DataFrame,
                          factor_cols: list,
                          step_label: str,
                          st_dt: pd.Timestamp,
                          annual_col: str = None):
    lm = lambda_df[lambda_df.index >= st_dt].copy()

    print(f"\n{'='*70}")
    print(f"  LAMBDA DISTRIBUTIONS — {step_label} (from st_dt onwards)")
    print(f"{'='*70}")

    for col in factor_cols:
        if col not in lm.columns:
            continue
        lambda_stats(lm[col], f"lambda_{col}")

        # Annual breakdown for slow-moving signals
        if col == annual_col:
            clean = lm[col].dropna()
            print(f"\n  Annual breakdown ({col}):")
            print(f"  {'Year':<6} {'Mean':>12} {'t-stat':>10} {'%pos':>8}")
            print(f"  {'-'*40}")
            for yr, grp in clean.groupby(clean.index.year):
                mean  = grp.mean()
                t     = mean / (grp.std() / np.sqrt(len(grp)))
                pct_p = (grp > 0).mean() * 100
                print(f"  {yr:<6} {mean:>+12.6f} {t:>+10.2f} {pct_p:>7.1f}%")

            cum = clean.cumsum()
            print(f"\n  Cumulative {col} lambda: {cum.iloc[-1]:+.4f}")

    print("\n--- Intercept ---")
    lambda_stats(lm['intercept'], "lambda_0 (intercept)")


def print_sector_lambdas(lambda_df: pd.DataFrame,
                          sec_cols: list,
                          st_dt: pd.Timestamp):
    lm = lambda_df[lambda_df.index >= st_dt]
    print(f"\n  Sector lambdas ({len(sec_cols)} sectors):")
    print(f"  {'Sector':<10} {'Mean':>10} {'Std':>10} {'t-stat':>10} {'%pos':>8}")
    print(f"  {'-'*52}")
    for col in sorted(sec_cols):
        if col not in lm.columns:
            continue
        s       = lm[col].dropna()
        mean    = s.mean()
        std     = s.std()
        t       = mean / (std / np.sqrt(len(s)))
        pct_pos = (s > 0).mean() * 100
        print(f"  {col:<10} {mean:>+10.6f} {std:>10.6f} {t:>+10.2f} {pct_pos:>7.1f}%")


# ==============================================================================
# MAIN
# ==============================================================================

def run(Pxs_df: pd.DataFrame, sectors_s: pd.Series) -> dict:
    print("=" * 70)
    print("  FACTOR MODEL - STEP 1")
    print("=" * 70)

    # Deduplicate upfront
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]
    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]

    # Start date
    st_input = input("\n  Start date (YYYY-MM-DD, or Enter for 2019-01-01): ").strip()
    st_dt    = pd.Timestamp(st_input) if st_input else pd.Timestamp('2019-01-01')
    print(f"  Start date       : {st_dt.date()}")

    # Extended start: MOM_LONG_BUFFER trading days before st_dt
    all_dates      = Pxs_df.index
    st_dt_loc      = all_dates.searchsorted(st_dt)
    ext_loc        = max(0, st_dt_loc - MOM_LONG_BUFFER)
    extended_st_dt = all_dates[ext_loc]
    print(f"  Extended start   : {extended_st_dt.date()} "
          f"({MOM_LONG_BUFFER} trading days before st_dt)")

    # Setup
    universe     = get_universe(Pxs_df, sectors_s, extended_st_dt)
    dynamic_size = load_size_df(universe, Pxs_df)
    sector_dum   = build_sector_dummies(universe, sectors_s)
    sec_cols     = sector_dum.columns.tolist()

    all_rets   = Pxs_df[universe].pct_change()
    ext_dates  = all_dates[all_dates >= extended_st_dt]
    valid_ext  = ext_dates[
        all_rets.loc[ext_dates, universe].notna().sum(axis=1) >= MIN_STOCKS
    ]
    valid_days = valid_ext[valid_ext >= st_dt]

    print(f"  Extended dates   : {len(valid_ext)} (from {extended_st_dt.date()})")
    print(f"  Valid dates      : {len(valid_days)} "
          f"(from {st_dt.date()}, for variance stats)")

    # --------------------------------------------------------------------------
    # UFV
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  BASELINE: UFV (from st_dt)")
    print("=" * 70)
    raw_mat = all_rets.loc[valid_days, universe]
    UFV     = variance_stats(raw_mat, "UFV - Raw Returns")
    print(f"\n  UFV = {UFV:.8f}")

    # --------------------------------------------------------------------------
    # Rolling betas (from extended_st_dt)
    # --------------------------------------------------------------------------
    beta_df = calc_rolling_betas(Pxs_df, universe, valid_ext)

    # --------------------------------------------------------------------------
    # Size characteristic: z-scored log(dynamic_size), cross-sectional per day
    # --------------------------------------------------------------------------
    print("\n  Building size characteristic (z-scored log market cap)...")
    size_char_dict = {}
    for dt in valid_ext:
        if dt not in dynamic_size.index:
            continue
        s = dynamic_size.loc[dt, universe].dropna()
        s = np.log(s.clip(lower=1))
        if len(s) < MIN_STOCKS:
            continue
        size_char_dict[dt] = zscore(s)
    size_char_df            = pd.DataFrame(size_char_dict).T.reindex(columns=universe)
    size_char_df.index.name = 'date'
    print(f"  Size characteristic built: {len(size_char_df)} dates")

    # --------------------------------------------------------------------------
    # STEP 1: Market Beta (run from extended_st_dt)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 1: Market Beta")
    print("=" * 70)

    dates_ext              = valid_ext.intersection(beta_df.index)
    resid_mkt_full, lambda_mkt, r2_mkt = run_factor_step(
        factor_cols  = ['beta'],
        char_by_date = {'beta': beta_df},
        all_rets     = all_rets,
        dynamic_size = dynamic_size,
        calc_dates   = dates_ext,
        universe     = universe,
    )
    resid_mkt = resid_mkt_full[resid_mkt_full.index >= st_dt]
    mkt_UFV   = variance_stats(resid_mkt, "mkt_UFV - Beta Residuals", UFV)
    r2_stats(r2_mkt[r2_mkt.index >= st_dt], "Market Beta")
    save_lambdas(lambda_mkt[lambda_mkt.index >= st_dt], 'factor_lambdas_mkt')
    save_residuals(resid_mkt, 'factor_residuals_mkt')
    print_lambda_summary(lambda_mkt, ['beta'], "Market Beta", st_dt)

    # --------------------------------------------------------------------------
    # STEP 2: Market Beta + Size
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 2: Market Beta + Size")
    print("=" * 70)

    dates_size              = dates_ext.intersection(size_char_df.index)
    resid_size_full, lambda_size, r2_size = run_factor_step(
        factor_cols  = ['beta', 'size'],
        char_by_date = {'beta': beta_df, 'size': size_char_df},
        all_rets     = all_rets,
        dynamic_size = dynamic_size,
        calc_dates   = dates_size,
        universe     = universe,
    )
    resid_size = resid_size_full[resid_size_full.index >= st_dt]
    size_UFV   = variance_stats(resid_size, "size_UFV - Beta+Size Residuals", mkt_UFV)
    _          = variance_stats(resid_size, "size_UFV vs UFV", UFV)
    r2_stats(r2_size[r2_size.index >= st_dt], "Market Beta + Size")
    save_lambdas(lambda_size[lambda_size.index >= st_dt], 'factor_lambdas_size')
    save_residuals(resid_size, 'factor_residuals_size')
    print_lambda_summary(lambda_size, ['beta', 'size'], "Market Beta + Size",
                         st_dt, annual_col='size')

    # --------------------------------------------------------------------------
    # STEP 3: Market Beta + Size + Sector Dummies
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 3: Market Beta + Size + Sector Dummies")
    print("=" * 70)

    sec_char = {'beta': beta_df, 'size': size_char_df}
    for col in sec_cols:
        sec_char[col] = pd.DataFrame(
            {dt: sector_dum[col] for dt in dates_size}
        ).T

    resid_sec_full, lambda_sec, r2_sec = run_factor_step(
        factor_cols  = ['beta', 'size'] + sec_cols,
        char_by_date = sec_char,
        all_rets     = all_rets,
        dynamic_size = dynamic_size,
        calc_dates   = dates_size,
        universe     = universe,
    )
    resid_sec = resid_sec_full[resid_sec_full.index >= st_dt]
    sec_UFV   = variance_stats(resid_sec, "sec_UFV - Beta+Size+Sector Residuals", size_UFV)
    _         = variance_stats(resid_sec, "sec_UFV vs UFV", UFV)
    r2_stats(r2_sec[r2_sec.index >= st_dt], "Market Beta + Size + Sectors")
    save_lambdas(lambda_sec[lambda_sec.index >= st_dt], 'factor_lambdas_sec')
    save_residuals(resid_sec, 'factor_residuals_sec')
    print_lambda_summary(lambda_sec, ['beta', 'size'],
                         "Market Beta + Size + Sectors", st_dt)
    print_sector_lambdas(lambda_sec, sec_cols, st_dt)

    # --------------------------------------------------------------------------
    # STEP 4: + Idiosyncratic Momentum (from sec residuals)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 4: Market Beta + Size + Sectors + Idiosyncratic Momentum")
    print("=" * 70)

    # Compute idio momentum from full sec residual history (includes buffer)
    mom_df    = calc_idio_momentum(resid_sec_full, valid_days)
    dates_mom = dates_size.intersection(valid_days).intersection(mom_df.index)

    mom_char  = {'beta': beta_df, 'size': size_char_df,
                 **{col: sec_char[col] for col in sec_cols},
                 'idio_mom': mom_df}

    resid_mom, lambda_mom, r2_mom = run_factor_step(
        factor_cols  = ['beta', 'size'] + sec_cols + ['idio_mom'],
        char_by_date = mom_char,
        all_rets     = all_rets,
        dynamic_size = dynamic_size,
        calc_dates   = dates_mom,
        universe     = universe,
    )
    mom_UFV = variance_stats(resid_mom, "mom_UFV - Full Residuals", sec_UFV)
    _       = variance_stats(resid_mom, "mom_UFV vs UFV", UFV)
    r2_stats(r2_mom, "Market Beta + Size + Sectors + Idio Momentum")
    save_lambdas(lambda_mom, 'factor_lambdas_mom')
    save_residuals(resid_mom, 'factor_residuals_mom')
    print_lambda_summary(lambda_mom, ['beta', 'size', 'idio_mom'],
                         "Full Model", st_dt, annual_col='idio_mom')
    print_sector_lambdas(lambda_mom, sec_cols, st_dt)

    # --------------------------------------------------------------------------
    # Variance Reduction Summary
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  VARIANCE REDUCTION SUMMARY")
    print("=" * 70)
    rows = [
        ("UFV (raw returns)",              UFV,     UFV,     None),
        ("mkt_UFV  (+ beta)",              mkt_UFV, UFV,     UFV),
        ("size_UFV (+ size)",              size_UFV,UFV,     mkt_UFV),
        ("sec_UFV  (+ sectors)",           sec_UFV, UFV,     size_UFV),
        ("mom_UFV  (+ idio momentum)",     mom_UFV, UFV,     sec_UFV),
    ]
    print(f"  {'':35} {'Variance':>12} {'% UFV':>8} {'% prev':>8}")
    print(f"  {'-'*68}")
    for label, var, base, prev in rows:
        pct_ufv  = f"{var/base*100:.2f}%"
        pct_prev = f"{var/prev*100:.2f}%" if prev else "---"
        print(f"  {label:<35} {var:>12.8f} {pct_ufv:>8} {pct_prev:>8}")

    return {
        'UFV':              UFV,
        'mkt_UFV':          mkt_UFV,
        'size_UFV':         size_UFV,
        'sec_UFV':          sec_UFV,
        'mom_UFV':          mom_UFV,
        'resid_mkt':        resid_mkt,
        'resid_size':       resid_size,
        'resid_sec':        resid_sec,
        'resid_sec_full':   resid_sec_full,
        'resid_mom':        resid_mom,
        'lambda_mkt':       lambda_mkt,
        'lambda_size':      lambda_size,
        'lambda_sec':       lambda_sec,
        'lambda_mom':       lambda_mom,
        'beta_df':          beta_df,
        'size_char_df':     size_char_df,
        'dynamic_size':     dynamic_size,
        'mom_df':           mom_df,
        'universe':         universe,
        'sec_cols':         sec_cols,
        'st_dt':            st_dt,
        'extended_st_dt':   extended_st_dt,
    }


if __name__ == "__main__":
    print("Usage: from factor_model_step1 import run")
    print("       results = run(Pxs_df, sectors_s)")
