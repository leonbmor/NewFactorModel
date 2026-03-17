#!/usr/bin/env python
# coding: utf-8

"""
Factor Model - Step 1
======================
Sequential Fama-MacBeth factor residualization.
Factors added in order of relevance, validated by variance reduction.

Steps:
  1. UFV       : full variance of raw returns (baseline, from st_dt)
  2. mkt_UFV   : + market beta (EWMA, hl=126, window=252)
  3. size_UFV  : + size (z-scored log market cap, cross-sectional)
  4. sec_UFV   : + sector dummies (reference: XLP)
  5. joint_UFV : + idio momentum + 21d reversal + SI composite (joint multivariate OLS)

Key design:
  - Dynamic size: shares from valuation_consolidated × daily price from Pxs_df
    Cached in DB table 'dynamic_size_df', only new dates computed on each run
  - Common sample: all variance stats computed on the intersection of dates/stocks
    where ALL characteristics are available → clean apples-to-apples comparisons
  - OLS weights: log(dynamic_size), normalized inside WLS
  - All residuals and lambda tables saved to DB

Usage:
    from factor_model_step1 import run
    results = run(Pxs_df, sectors_s)
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

ENGINE           = create_engine("postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db")
DYNAMIC_SIZE_TBL = 'dynamic_size_df'
BETA_WINDOW      = 252
BETA_HL          = 126
VOL_WINDOW       = 84    # shorter than beta to avoid redundancy
VOL_HL           = 42    # EWMA half-life for vol factor

# Macro factor settings — same EWMA structure as market beta
MACRO_COLS = [
    'USGG2YR',              # 2Y nominal rate changes (bps)
    'US10Y2Y_SPREAD_CHG',   # 2Y/10Y spread changes (bps)
    'T10YIE',               # 10Y inflation breakeven changes
    'T5YIFR',               # 5y5y implied inflation changes
    'Crude',                # WTI crude changes
    'XAUUSD',               # Gold changes
    'M2MP',                 # Money velocity level/detrended
    'VIX Mom',              # VIX momentum level
]
MOM_LONG         = 252
MOM_SKIP         = 21
MOM_LONG_BUFFER  = MOM_LONG
MIN_STOCKS       = 150
SECTOR_REF       = 'XLP'
SI_COMPOSITE_TBL  = 'si_composite_df'
SI_HORIZON        = 21        # forward return horizon for SI signal (trading days)
QUALITY_ANCHOR_TBL = 'valuation_metrics_anchors'
OU_REVERSION_TBL  = 'ou_reversion_df'
OU_MEANREV_W      = 60        # O-U fitting window (trading days)
OU_MIN_OBS        = 30        # minimum observations for valid O-U fit
OU_ST_REV_W       = 21        # ST reversal fallback window
OU_VOLUME_W       = 10        # volume normalization rolling window
OU_VOL_CLIP_LO    = 0.5       # volume scalar lower clip
OU_VOL_CLIP_HI    = 3.0       # volume scalar upper clip
OU_WEIGHT_REF     = 30.0      # reference half-life for ou_weight = 1.0
OU_WEIGHT_CAP     = 10.0      # maximum ou_weight


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
# DYNAMIC SIZE — DB CACHED
# ==============================================================================

def _compute_dynamic_size_for_dates(dates_to_calc: pd.DatetimeIndex,
                                     universe: list,
                                     Pxs_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each date in dates_to_calc and each stock in universe:
      shares   = Size_db / Price_db  (last available Size in valuation_consolidated
                                      before calc_date, Price from Pxs_df on same date)
      dyn_size = shares * Price on calc_date
      Fallback: if Price_db is NaN -> use Size_db as-is
    """
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

    size_pivot = size_raw.pivot_table(
        index='date', columns='ticker', values='Size', aggfunc='last'
    )

    # Forward fill to all Pxs_df dates to get last known Size_db per date
    all_px_dates = Pxs_df.index
    size_ff      = size_pivot.reindex(all_px_dates).ffill().bfill()

    # Track which DB date each forward-filled value came from
    date_indicator = pd.DataFrame(
        index=size_pivot.index,
        columns=size_pivot.columns,
        data=np.tile(
            size_pivot.index.values.reshape(-1, 1),
            (1, len(size_pivot.columns))
        )
    )
    date_indicator = date_indicator.reindex(all_px_dates).ffill().bfill()

    # Build price lookup dict per ticker for fast access
    results = {}
    for dt in dates_to_calc:
        if dt not in Pxs_df.index:
            continue
        row = {}
        for ticker in universe:
            if ticker not in size_ff.columns:
                continue
            size_db = size_ff.loc[dt, ticker]
            if pd.isna(size_db):
                continue

            # Price on DB snapshot date
            db_date   = pd.Timestamp(date_indicator.loc[dt, ticker])
            price_db  = Pxs_df.loc[db_date, ticker] \
                        if db_date in Pxs_df.index else np.nan

            # Current price
            price_t   = Pxs_df.loc[dt, ticker]

            if pd.isna(price_db) or price_db == 0 or pd.isna(price_t):
                row[ticker] = size_db          # fallback
            else:
                shares      = size_db / price_db
                row[ticker] = shares * price_t

        results[dt] = row

    df             = pd.DataFrame(results).T
    df.index.name  = 'date'
    df             = df.reindex(columns=universe)
    return df


def load_dynamic_size(universe: list,
                       Pxs_df: pd.DataFrame,
                       all_calc_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load dynamic size from DB cache, computing only missing dates.
    Returns DataFrame: date x ticker (all dates in all_calc_dates).
    """
    # Check which dates already exist in DB
    already_done = set()
    try:
        with ENGINE.connect() as conn:
            # Check table exists
            exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = :t
                )
            """), {"t": DYNAMIC_SIZE_TBL}).scalar()

        if exists:
            with ENGINE.connect() as conn:
                rows = conn.execute(text(f"""
                    SELECT DISTINCT date FROM {DYNAMIC_SIZE_TBL}
                """)).fetchall()
            already_done = {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        pass

    dates_to_calc = [d for d in all_calc_dates if d not in already_done]

    if dates_to_calc:
        print(f"  Computing dynamic size for {len(dates_to_calc)} new dates "
              f"({len(already_done)} already in DB)...")
        new_df = _compute_dynamic_size_for_dates(
            pd.DatetimeIndex(dates_to_calc), universe, Pxs_df
        )
        # Save new dates to DB
        long           = new_df.stack(dropna=False).reset_index()
        long.columns   = ['date', 'ticker', 'size']
        long           = long.dropna(subset=['size'])
        long['date']   = pd.to_datetime(long['date'])

        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {DYNAMIC_SIZE_TBL} (
                    date   DATE,
                    ticker VARCHAR(20),
                    size   NUMERIC,
                    PRIMARY KEY (date, ticker)
                )
            """))
        long.to_sql(DYNAMIC_SIZE_TBL, ENGINE, if_exists='append', index=False)
        print(f"  Saved {len(long):,} new rows to '{DYNAMIC_SIZE_TBL}'")
    else:
        print(f"  Dynamic size: all {len(all_calc_dates)} dates already in DB")

    # Load full table for requested dates
    date_list = [d.date() for d in all_calc_dates]
    print(f"  Loading dynamic size from DB...")
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT date, ticker, size FROM {DYNAMIC_SIZE_TBL}
            WHERE date = ANY(:dates)
        """), {"dates": date_list}).fetchall()

    df             = pd.DataFrame(rows, columns=['date', 'ticker', 'size'])
    df['date']     = pd.to_datetime(df['date'])
    df['size']     = df['size'].astype(float)
    pivot          = df.pivot_table(index='date', columns='ticker',
                                    values='size', aggfunc='last')
    pivot          = pivot.reindex(columns=universe)
    print(f"  Dynamic size loaded: {pivot.shape}")
    return pivot


def get_log_size(dynamic_size: pd.DataFrame,
                 calc_date: pd.Timestamp,
                 valid_idx: pd.Index) -> pd.Series:
    """
    Returns log(dynamic_size) for valid_idx on calc_date.
    Used as OLS weights (not z-scored — normalization inside WLS).
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


def calc_macro_betas(Pxs_df: pd.DataFrame,
                     universe: list,
                     calc_dates: pd.DatetimeIndex) -> dict:
    """
    Compute EWMA rolling betas of each stock vs each macro factor change.
    Same EWMA structure as market beta: BETA_WINDOW=252, BETA_HL=126.

    Inputs (from Pxs_df columns, pre-computed daily changes):
      USGG2YR           2Y nominal rate changes
      USGG10YR_SPREAD   2Y/10Y spread changes (computed as USGG10YR - USGG2YR)
      T10YIE            10Y inflation breakeven changes
      T5YIFR            5y5y implied inflation changes
      Crude Oil USD/Bbl WTI crude changes
      XAUUSD            Gold changes
      M2MP Velocity     Money velocity changes (forward-filled weekly)
      VIX Mom           VIX momentum (already transformed, used as-is)

    For each date t, stock i, macro factor m:
        beta_im = Cov_EWMA(r_i, d_m) / Var_EWMA(d_m)

    Each macro beta series z-scored cross-sectionally per date.

    Returns dict: {macro_col: DataFrame(dates x tickers)} for run_factor_step.
    """
    print("  Calculating macro factor betas...")

    # Build macro change series from Pxs_df
    macro_raw = {}
    for col in MACRO_COLS:
        if col in Pxs_df.columns:
            macro_raw[col] = Pxs_df[col]
        else:
            print(f"  WARNING: '{col}' not found in Pxs_df — skipping")

    if not macro_raw:
        print("  WARNING: no macro factors found — skipping macro step")
        return {}

    avail_cols = list(macro_raw.keys())
    print(f"  Macro factors available: {avail_cols}")

    stk_rets  = Pxs_df[universe].pct_change()
    all_dates = Pxs_df.index
    alpha     = 1 - np.exp(-np.log(2) / BETA_HL)

    # Results: {macro_col: {date: Series(ticker -> beta)}}
    betas_by_macro = {col: {} for col in avail_cols}

    for dt in calc_dates:
        window = all_dates[all_dates < dt][-BETA_WINDOW:]
        if len(window) < BETA_WINDOW // 2:
            continue

        # EWMA weights
        n       = len(window)
        weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()

        # Stock returns over window
        stk_w = stk_rets.loc[window].fillna(0).values   # (n x tickers)
        stk_mean = stk_w.T @ weights                      # (tickers,)
        stk_dev  = stk_w - stk_mean[np.newaxis, :]        # (n x tickers)

        for col in avail_cols:
            macro_s = macro_raw[col].reindex(window).fillna(0).values  # (n,)

            macro_mean = np.dot(weights, macro_s)
            macro_dev  = macro_s - macro_mean
            var_macro  = np.dot(weights, macro_dev ** 2)

            if var_macro <= 0:
                continue

            cov    = (stk_dev * macro_dev[:, np.newaxis] * weights[:, np.newaxis]).sum(axis=0)
            beta_t = cov / var_macro

            betas_by_macro[col][dt] = pd.Series(beta_t, index=universe)

    # Build DataFrames and z-score cross-sectionally
    result = {}
    for col in avail_cols:
        if not betas_by_macro[col]:
            continue
        df = pd.DataFrame(betas_by_macro[col]).T.reindex(columns=universe)
        df.index.name = 'date'
        # Z-score cross-sectionally each date
        df = df.apply(zscore, axis=1)
        result[col] = df
        print(f"  {col}: {len(df)} dates computed")

    return result



def calc_idio_momentum(resid_sec_df: pd.DataFrame,
                        calc_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Idiosyncratic momentum: cumulative sum of sector residuals
    over [t-MOM_LONG, t-MOM_SKIP], z-scored cross-sectionally.
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


def calc_idio_momentum_volscaled(resid_sec_df: pd.DataFrame,
                                  volumeTrd_df: pd.DataFrame,
                                  calc_dates: pd.DatetimeIndex,
                                  vol_lower: float = 0.5,
                                  vol_upper: float = 3.0) -> pd.DataFrame:
    """
    Volume-scaled idiosyncratic momentum.
    volumeTrd_df is assumed to contain precomputed volume scalars
    (e.g. volume(t) / mean(volume[t-10, t-1])), clipped to [vol_lower, vol_upper].
    Cumulative volume-weighted idio return over [t-MOM_LONG, t-MOM_SKIP],
    z-scored cross-sectionally per date.
    """
    print(f"  Calculating volume-scaled idio momentum "
          f"(clip=[{vol_lower}, {vol_upper}])...")

    # Clip scalars to bounds (in case not pre-clipped)
    vol_scalars = volumeTrd_df.clip(lower=vol_lower, upper=vol_upper)

    all_resid_dates = resid_sec_df.index
    mom_dict        = {}

    for dt in calc_dates:
        past = all_resid_dates[all_resid_dates < dt]
        if len(past) < MOM_LONG + 1:
            continue

        window = past[-MOM_LONG:-MOM_SKIP]
        if len(window) < MOM_LONG - MOM_SKIP - 10:
            continue

        # Align window to dates available in both resid and vol scalars
        common_days  = window.intersection(vol_scalars.index)
        if len(common_days) < MOM_LONG - MOM_SKIP - 10:
            continue

        weighted_sum = (resid_sec_df.loc[common_days] *
                        vol_scalars.loc[common_days]).sum(axis=0)
        valid        = weighted_sum.dropna()

        if len(valid) < MIN_STOCKS:
            continue

        mom_dict[dt] = zscore(valid)

    mom_df            = pd.DataFrame(mom_dict).T.reindex(columns=resid_sec_df.columns)
    mom_df.index.name = 'date'
    print(f"  Volume-scaled idio momentum computed: {len(mom_df)} dates")
    return mom_df


def calc_reversal_21d(Pxs_df: pd.DataFrame,
                       universe: list,
                       calc_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Short-term reversal: log(P[t-1] / P[t-22])
    Captures last 21 trading days of raw price performance.
    Expected sign: negative (recent winners mean-revert).
    Z-scored cross-sectionally per date.
    """
    print("  Calculating 21-day short-term reversal from prices...")
    all_px_dates = Pxs_df.index
    rev_dict     = {}

    for dt in calc_dates:
        past = all_px_dates[all_px_dates < dt]
        if len(past) < 22:
            continue

        p_recent = Pxs_df.loc[past[-1],  universe]   # yesterday
        p_old    = Pxs_df.loc[past[-22], universe]   # 21 trading days ago

        valid_mask = (p_recent > 0) & (p_old > 0)
        rev        = np.log(p_recent / p_old).where(valid_mask).dropna()

        if len(rev) < MIN_STOCKS:
            continue

        rev_dict[dt] = zscore(rev)

    rev_df            = pd.DataFrame(rev_dict).T.reindex(columns=universe)
    rev_df.index.name = 'date'
    print(f"  21d reversal computed: {len(rev_df)} dates")
    return rev_df


def load_ohlc_tables(universe: list) -> tuple:
    """
    Load open, high, low prices from DB for universe tickers.
    Tables: daily_open, daily_high, daily_low.
    Columns have ' US' extension in DB; stripped to bare tickers on return.
    Returns: (open_df, high_df, low_df) — each a DataFrame (dates x bare tickers).
    """
    def _load_table(tbl: str) -> pd.DataFrame:
        try:
            with ENGINE.connect() as conn:
                df = pd.read_sql(text(f"SELECT * FROM {tbl}"), conn)
        except Exception as e:
            print(f"  ERROR loading '{tbl}': {e}")
            return pd.DataFrame()
        # Prefer columns with 'date' in name, fall back to 'index'
        date_col = [c for c in df.columns if 'date' in c.lower()]
        if not date_col:
            date_col = [c for c in df.columns if c.lower() == 'index']
        if not date_col:
            print(f"  ERROR loading '{tbl}': no date column found "
                  f"(columns: {list(df.columns[:5])})")
            return pd.DataFrame()
        dc = date_col[0]
        df[dc] = pd.to_datetime(df[dc])
        df = df.set_index(dc).sort_index()
        df.columns = [clean_ticker(c) for c in df.columns]
        keep = [t for t in universe if t in df.columns]
        if not keep:
            print(f"  ERROR: no universe tickers found in '{tbl}' after normalization "
                  f"(sample cols: {list(df.columns[:5])})")
            return pd.DataFrame()
        return df[keep].astype(float)

    print("  Loading OHLC tables from DB...")
    open_df  = _load_table('daily_open')
    high_df  = _load_table('daily_high')
    low_df   = _load_table('daily_low')

    print(f"  daily_open : {open_df.shape if not open_df.empty else 'EMPTY'}")
    print(f"  daily_high : {high_df.shape if not high_df.empty else 'EMPTY'}")
    print(f"  daily_low  : {low_df.shape  if not low_df.empty  else 'EMPTY'}")

    # Check universe coverage
    for name, df in [('daily_open', open_df), ('daily_high', high_df), ('daily_low', low_df)]:
        if not df.empty:
            missing = [t for t in universe if t not in df.columns]
            if missing:
                print(f"  {name}: {len(missing)} universe tickers missing from columns "
                      f"(e.g. {missing[:5]})")
            else:
                print(f"  {name}: all {len(universe)} universe tickers present")

    if open_df.empty or high_df.empty or low_df.empty:
        print("  WARNING: OHLC tables incomplete — will fall back to close-to-close vol")
        return None, None, None

    print(f"  OHLC loaded: {open_df.shape[0]} dates x {len(open_df.columns)} tickers")
    return open_df, high_df, low_df


def calc_vol_factor(Pxs_df: pd.DataFrame,
                    universe: list,
                    calc_dates: pd.DatetimeIndex,
                    open_df: pd.DataFrame = None,
                    high_df: pd.DataFrame = None,
                    low_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Short-window EWMA realized volatility factor.
    Window: VOL_WINDOW (84d), half-life: VOL_HL (42d).
    Shorter than beta (252d/hl=126) to capture distinct variation.

    If OHLC DataFrames provided: uses Garman-Klass estimator —
      σ²_GK = 0.5·(ln(H/L))² - (2·ln2-1)·(ln(C/O))²
    ~8x more efficient than close-to-close. Falls back gracefully if unavailable.

    Z-scored cross-sectionally per date.
    Expected sign: negative (high vol stocks underperform risk-adjusted).
    """
    use_gk = (open_df is not None and high_df is not None and low_df is not None)
    method = "Garman-Klass" if use_gk else "close-to-close"
    print(f"  Calculating vol factor ({method}, window={VOL_WINDOW}d, hl={VOL_HL}d)...")

    alpha        = 1 - np.exp(-np.log(2) / VOL_HL)
    all_px_dates = Pxs_df.index
    vol_dict     = {}

    for dt in calc_dates:
        past = all_px_dates[all_px_dates < dt]
        if len(past) < VOL_WINDOW // 2:
            continue

        window = past[-VOL_WINDOW:]

        if use_gk:
            # Garman-Klass: σ²_GK = 0.5·(ln(H/L))² - (2·ln2-1)·(ln(C/O))²
            C = Pxs_df.loc[window, universe]
            O = open_df.reindex(index=window, columns=universe)
            H = high_df.reindex(index=window, columns=universe)
            L = low_df.reindex(index=window,  columns=universe)

            valid_mask = (C > 0) & (O > 0) & (H > 0) & (L > 0) & (H >= L)
            log_hl     = np.log(H / L).where(valid_mask)
            log_co     = np.log(C / O).where(valid_mask)
            gk_var     = (0.5 * log_hl ** 2
                          - (2 * np.log(2) - 1) * log_co ** 2).clip(lower=0)

            n        = len(window)
            weights  = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
            weights /= weights.sum()

            ewma_var = gk_var.mul(weights, axis=0).sum(axis=0)
            ewma_vol = np.sqrt(ewma_var * 252)

        else:
            px_win = Pxs_df.loc[window, universe]
            rets   = px_win.pct_change().dropna(how='all')
            if len(rets) < VOL_WINDOW // 2:
                continue

            n        = len(rets)
            weights  = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
            weights /= weights.sum()

            ewma_var = (rets ** 2).mul(weights, axis=0).sum(axis=0)
            ewma_vol = np.sqrt(ewma_var * 252)

        valid = ewma_vol.replace(0, np.nan).dropna()
        if len(valid) < MIN_STOCKS:
            continue

        vol_dict[dt] = zscore(valid)

    vol_df            = pd.DataFrame(vol_dict).T.reindex(columns=universe)
    vol_df.index.name = 'date'
    print(f"  Vol factor computed: {len(vol_df)} dates ({method})")
    return vol_df


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


def wls_ridge_cross_section(y: pd.Series, X: pd.DataFrame,
                             w: pd.Series, ridge_lambda: float = 0.0) -> tuple:
    """
    Weighted least squares with optional L2 (ridge) regularization.
    Intercept is NOT penalized — ridge penalty applied to slope coefficients only.
    ridge_lambda=0.0 reduces to standard WLS (same as wls_cross_section).
    """
    idx  = y.index.intersection(X.index).intersection(w.index)
    if len(idx) < 10:
        return None, None, None

    y_   = y.loc[idx].values
    X_   = np.column_stack([np.ones(len(idx)), X.loc[idx].values])
    w_   = w.loc[idx].values
    w_   = np.where(np.isnan(w_) | (w_ <= 0), 1.0, w_)
    w_   = w_ / w_.sum()

    W    = np.diag(w_)
    XtW  = X_.T @ W
    XtWX = XtW @ X_

    # Ridge penalty — intercept (col 0) unpenalized, slopes penalized
    n_slopes = X_.shape[1] - 1
    pen      = np.zeros(X_.shape[1])
    pen[1:]  = ridge_lambda                         # skip intercept
    XtWX_reg = XtWX + np.diag(pen)

    try:
        lam = np.linalg.solve(XtWX_reg, XtW @ y_)
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
# VARIANCE / R2 / LAMBDA STATS
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


def print_lambda_summary(lambda_df: pd.DataFrame,
                          factor_cols: list,
                          step_label: str,
                          common_dates: pd.DatetimeIndex,
                          annual_col: str = None):
    """Stats computed on common_dates only for clean comparability."""
    lm = lambda_df[lambda_df.index.isin(common_dates)].copy()

    print(f"\n{'='*70}")
    print(f"  LAMBDA DISTRIBUTIONS — {step_label} (common sample)")
    print(f"{'='*70}")

    for col in factor_cols:
        if col not in lm.columns:
            continue
        lambda_stats(lm[col], f"lambda_{col}")

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
                          common_dates: pd.DatetimeIndex):
    lm = lambda_df[lambda_df.index.isin(common_dates)]
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
# GENERIC FACTOR STEP RUNNER
# ==============================================================================    return summary


# ==============================================================================
# GENERIC FACTOR STEP RUNNER
# ==============================================================================

def run_factor_step(factor_cols: list,
                     char_by_date: dict,
                     all_rets: pd.DataFrame,
                     dynamic_size: pd.DataFrame,
                     calc_dates: pd.DatetimeIndex,
                     universe: list,
                     ridge_lambda: float = 0.0) -> tuple:
    resid_dict  = {}
    lambda_dict = {}
    r2_dict     = {}

    # Restrict universe to tickers present in all_rets (may be residuals with fewer tickers)
    valid_universe = [t for t in universe if t in all_rets.columns]

    for dt in calc_dates:
        y = all_rets.loc[dt, valid_universe].dropna()
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

        lam, resid, r2 = wls_ridge_cross_section(y_, X, w_, ridge_lambda=ridge_lambda)
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
# SHORT INTEREST COMPOSITE — DB CACHED
# ==============================================================================

def _compute_si_composite_for_dates(dates_to_calc: pd.DatetimeIndex,
                                     universe: list) -> pd.DataFrame:
    """
    For each date in dates_to_calc:
      1. Load SI % Free Float and Utilization from short_interest_data
      2. Forward-fill to calc dates (SI is lower frequency than daily)
      3. Cross-sectional z-score each metric
      4. Equal-weight composite = (z_si_float + z_utilization) / 2
    Returns DataFrame: date x ticker
    """
    us_tickers = [t + ' US' for t in universe]

    with ENGINE.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, ticker,
                   short_interest_pct_free_float,
                   short_interest_shares_k,
                   short_availability_shares_k
            FROM short_interest_data
            WHERE ticker = ANY(:tickers)
            ORDER BY ticker, date
        """), {"tickers": us_tickers}).fetchall()

    if not rows:
        print("  WARNING: No SI data found for universe tickers")
        return pd.DataFrame(index=dates_to_calc, columns=universe, dtype=float)

    df           = pd.DataFrame(rows, columns=[
        'date', 'ticker', 'si_float', 'si_shares_k', 'avail_shares_k'
    ])
    df['date']   = pd.to_datetime(df['date'])
    df['ticker'] = df['ticker'].str.replace(' US', '', regex=False).str.strip()
    for col in ['si_float', 'si_shares_k', 'avail_shares_k']:
        df[col]  = df[col].astype(float)

    df['utilization'] = (
        df['si_shares_k'] / df['avail_shares_k'].replace(0, np.nan)
    ).clip(0, 1)

    # Pivot each metric: date x ticker
    piv_float = df.pivot_table(
        index='date', columns='ticker', values='si_float', aggfunc='last'
    )
    piv_util  = df.pivot_table(
        index='date', columns='ticker', values='utilization', aggfunc='last'
    )

    # Reindex to calc dates and forward-fill (SI updates less than daily)
    piv_float = piv_float.reindex(dates_to_calc).ffill()
    piv_util  = piv_util.reindex(dates_to_calc).ffill()

    # Fill remaining NaNs with cross-sectional median per date
    # Stocks missing SI data get a neutral score (~0 after z-scoring)
    # rather than being dropped from the universe entirely
    piv_float = piv_float.apply(lambda row: row.fillna(row.median()), axis=1)
    piv_util  = piv_util.apply(lambda row: row.fillna(row.median()),  axis=1)

    # Cross-sectional z-score per date, equal-weight composite
    results = {}
    for dt in dates_to_calc:
        z_float = zscore(piv_float.loc[dt].dropna()) \
                  if dt in piv_float.index else pd.Series(dtype=float)
        z_util  = zscore(piv_util.loc[dt].dropna()) \
                  if dt in piv_util.index else pd.Series(dtype=float)

        common  = z_float.index.intersection(z_util.index)
        if len(common) < MIN_STOCKS:
            continue

        composite       = (z_float.loc[common] + z_util.loc[common]) / 2
        results[dt]     = composite

    df_out            = pd.DataFrame(results).T.reindex(columns=universe)
    df_out.index.name = 'date'
    return df_out


def load_si_composite(universe: list,
                       all_calc_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load SI composite from DB cache, computing only missing dates.
    Returns DataFrame: date x ticker.
    """
    already_done = set()
    try:
        with ENGINE.connect() as conn:
            exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = :t
                )
            """), {"t": SI_COMPOSITE_TBL}).scalar()

        if exists:
            with ENGINE.connect() as conn:
                rows = conn.execute(text(
                    f"SELECT DISTINCT date FROM {SI_COMPOSITE_TBL}"
                )).fetchall()
            already_done = {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        pass

    dates_to_calc = [d for d in all_calc_dates if d not in already_done]

    if dates_to_calc:
        print(f"  Computing SI composite for {len(dates_to_calc)} new dates "
              f"({len(already_done)} already in DB)...")
        new_df = _compute_si_composite_for_dates(
            pd.DatetimeIndex(dates_to_calc), universe
        )
        long           = new_df.stack(dropna=False).reset_index()
        long.columns   = ['date', 'ticker', 'si_composite']
        long           = long.dropna(subset=['si_composite'])
        long['date']   = pd.to_datetime(long['date'])

        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {SI_COMPOSITE_TBL} (
                    date         DATE,
                    ticker       VARCHAR(20),
                    si_composite NUMERIC,
                    PRIMARY KEY  (date, ticker)
                )
            """))
        long.to_sql(SI_COMPOSITE_TBL, ENGINE, if_exists='append', index=False)
        print(f"  Saved {len(long):,} new rows to '{SI_COMPOSITE_TBL}'")
    else:
        print(f"  SI composite: all {len(all_calc_dates)} dates already in DB")

    # Load full table for requested dates
    date_list = [d.date() for d in all_calc_dates]
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT date, ticker, si_composite FROM {SI_COMPOSITE_TBL}
            WHERE date = ANY(:dates)
        """), {"dates": date_list}).fetchall()

    df             = pd.DataFrame(rows, columns=['date', 'ticker', 'si_composite'])
    df['date']     = pd.to_datetime(df['date'])
    df['si_composite'] = df['si_composite'].astype(float)
    pivot          = df.pivot_table(
        index='date', columns='ticker', values='si_composite', aggfunc='last'
    )
    pivot          = pivot.reindex(columns=universe)
    # Forward-fill missing dates (SI fetched less frequently than prices)
    pivot          = pivot.reindex(all_calc_dates).ffill()
    print(f"  SI composite loaded: {pivot.shape}")
    return pivot


# ==============================================================================
# O-U MEAN REVERSION — DB cached, computed on common sample dates
# ==============================================================================

def _fit_ou_single(resid_series: pd.Series,
                   px_series: pd.Series) -> tuple:
    """
    Fit AR(1)/O-U to compounded residual price index for one stock on one date.
    Returns (neg_dist_st, halflife) or (nan, nan) on failure.
    """
    from sklearn.linear_model import LinearRegression

    resid_clean = resid_series.replace({np.inf: np.nan, -np.inf: np.nan}).dropna()
    if len(resid_clean) < OU_MIN_OBS:
        return np.nan, np.nan

    anchor_dates = px_series.index[px_series.index >= resid_clean.index[0]]
    if anchor_dates.empty:
        return np.nan, np.nan
    anchor_price = float(px_series.loc[anchor_dates[0]])
    if np.isnan(anchor_price) or anchor_price <= 0:
        return np.nan, np.nan

    px_idx = (1 + resid_clean).cumprod() * anchor_price
    sX1    = px_idx.iloc[:-1].values.reshape(-1, 1)
    sX2    = px_idx.iloc[1:].values

    try:
        mod = LinearRegression()
        mod.fit(sX1, sX2)
        a = float(mod.intercept_)
        b = float(mod.coef_[0])
    except Exception:
        return np.nan, np.nan

    if not (0 < b < 1):
        return np.nan, np.nan

    m = a / (1 - b)
    k = -np.log(b)

    residuals  = sX2 - mod.predict(sX1).flatten()
    resid_std  = float(np.std(residuals))
    if resid_std == 0 or k == 0:
        return np.nan, np.nan

    # Scale to actual price space
    last_px = float(px_series.dropna().iloc[-1])
    if np.isnan(last_px) or last_px <= 0:
        return np.nan, np.nan
    idx_last   = float(px_idx.iloc[-1])
    scale      = last_px / idx_last if idx_last != 0 else 1.0
    m_scaled   = m * scale
    std_scaled = resid_std * scale

    if m_scaled <= 0:
        return np.nan, np.nan

    dist_st  = (last_px - m_scaled) / (std_scaled / np.sqrt(2 * k))
    halflife = np.log(2) / k

    return -dist_st, halflife   # neg_dist_st, halflife


def _compute_ou_for_dates(calc_dates: pd.DatetimeIndex,
                           universe: list,
                           resid_pivot: pd.DataFrame,
                           Pxs_df: pd.DataFrame,
                           volumeTrd_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute O-U mean reversion z-scores for given dates.
    Returns DataFrame (dates x tickers), z-scored final_score.
    """
    # Volume-scale residuals if provided
    resid = resid_pivot.copy()
    if volumeTrd_df is not None:
        common = resid.columns.intersection(volumeTrd_df.columns)
        vol_norm  = (volumeTrd_df[common]
                     .rolling(OU_VOLUME_W).mean()
                     .reindex(resid.index).ffill())
        vol_ratio = (volumeTrd_df[common]
                     .reindex(resid.index).ffill()
                     / vol_norm).clip(OU_VOL_CLIP_LO, OU_VOL_CLIP_HI)
        resid[common] = resid[common] / vol_ratio

    result = {}
    n = len(calc_dates)

    # Pre-compute compounded residual price index for each stock
    # Used for both O-U fitting and ST reversal fallback
    # Anchor each stock to 1.0 at its first available residual date
    cum_resid = (1 + resid.fillna(0)).cumprod()

    for idx, dt in enumerate(calc_dates):
        if (idx + 1) % 50 == 0:
            print(f"  O-U: [{idx+1}/{n}] {dt.date()}", end='\r')

        # Get residual history up to this date
        past_resid = resid[resid.index < dt]
        past_cum   = cum_resid[cum_resid.index < dt]

        # ST reversal from cumulative residual index (fallback)
        # log(cum_resid[t-1] / cum_resid[t-22]) — idiosyncratic, net of all model factors
        st_rev = pd.Series(np.nan, index=universe)
        if len(past_cum) >= OU_ST_REV_W + 1:
            cum_recent = past_cum.iloc[-1]
            cum_old    = past_cum.iloc[-OU_ST_REV_W - 1]
            valid      = (cum_recent > 0) & (cum_old > 0)
            st_rev     = np.log(cum_recent / cum_old).where(valid).reindex(universe)

        neg_dists  = pd.Series(np.nan, index=universe)
        halflives  = pd.Series(np.nan, index=universe)

        for ticker in universe:
            if ticker not in past_resid.columns:
                continue
            stock_resid = past_resid[ticker].dropna().iloc[-OU_MEANREV_W:]
            if ticker not in Pxs_df.columns:
                continue
            neg_dist, hl = _fit_ou_single(stock_resid, Pxs_df[ticker].dropna())
            neg_dists[ticker] = neg_dist
            halflives[ticker] = hl

        # Cross-sectional ranks
        valid_neg = neg_dists.dropna()
        valid_rev = st_rev.dropna()

        ou_rank  = pd.Series(np.nan, index=universe)
        rev_rank = pd.Series(np.nan, index=universe)

        if len(valid_neg) > 1:
            r = valid_neg.rank(method='average', ascending=True)
            ou_rank[valid_neg.index] = (r - 1) / (len(r) - 1)

        if len(valid_rev) > 1:
            r = valid_rev.rank(method='average', ascending=False)
            rev_rank[valid_rev.index] = (r - 1) / (len(r) - 1)

        # ou_weight = min(OU_WEIGHT_REF / halflife, OU_WEIGHT_CAP), 0 if NaN
        ou_weight = (OU_WEIGHT_REF / halflives).clip(upper=OU_WEIGHT_CAP).fillna(0)

        # Weighted blend
        final = (ou_weight * ou_rank.fillna(0) + rev_rank) / (ou_weight + 1)

        # Z-score cross-sectionally
        result[dt] = zscore(final.dropna())

    print(f"\n  O-U computation complete: {n} dates")
    return pd.DataFrame(result).T.reindex(columns=universe)


def load_ou_reversion(universe: list,
                       calc_dates: pd.DatetimeIndex,
                       resid_pivot: pd.DataFrame,
                       Pxs_df: pd.DataFrame,
                       volumeTrd_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Load O-U mean reversion scores from DB cache, computing only missing dates.
    Returns DataFrame: dates x tickers, z-scored final_score.
    """
    already_done = set()
    try:
        with ENGINE.connect() as conn:
            exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = :t
                )
            """), {"t": OU_REVERSION_TBL}).scalar()
        if exists:
            with ENGINE.connect() as conn:
                rows = conn.execute(text(
                    f"SELECT DISTINCT date FROM {OU_REVERSION_TBL}"
                )).fetchall()
            already_done = {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        pass

    dates_to_calc = pd.DatetimeIndex([d for d in calc_dates if d not in already_done])

    if len(dates_to_calc) > 0:
        print(f"  Computing O-U reversion for {len(dates_to_calc)} new dates "
              f"({len(already_done)} already in DB)...")
        new_df       = _compute_ou_for_dates(dates_to_calc, universe,
                                              resid_pivot, Pxs_df, volumeTrd_df)
        long         = new_df.stack(dropna=False).reset_index()
        long.columns = ['date', 'ticker', 'ou_score']
        long         = long.dropna(subset=['ou_score'])
        long['date'] = pd.to_datetime(long['date'])

        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {OU_REVERSION_TBL} (
                    date      DATE,
                    ticker    VARCHAR(20),
                    ou_score  NUMERIC,
                    PRIMARY KEY (date, ticker)
                )
            """))
        long.to_sql(OU_REVERSION_TBL, ENGINE, if_exists='append', index=False)
        print(f"  Saved {len(long):,} new rows to '{OU_REVERSION_TBL}'")
    else:
        print(f"  O-U reversion: all {len(calc_dates)} dates already in DB")

    # Load from DB
    date_list = [d.date() for d in calc_dates]
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT date, ticker, ou_score FROM {OU_REVERSION_TBL}
            WHERE date = ANY(:dates)
        """), {"dates": date_list}).fetchall()

    df           = pd.DataFrame(rows, columns=['date', 'ticker', 'ou_score'])
    df['date']   = pd.to_datetime(df['date'])
    df['ou_score'] = df['ou_score'].astype(float)
    pivot        = df.pivot_table(index='date', columns='ticker',
                                   values='ou_score', aggfunc='last')
    pivot        = pivot.reindex(columns=universe).reindex(calc_dates)
    print(f"  O-U reversion loaded: {pivot.shape}")
    return pivot


# ==============================================================================
# QUALITY FACTOR — load from anchor snapshots, forward-fill to daily
# ==============================================================================

def load_quality_scores(universe: list, calc_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load pre-computed quality composite scores from valuation_metrics_anchors.
    Uses GQF/CQF weights derived from the full anchor history via quality_factor.run(),
    but here we load the raw anchor data and use a simplified composite:
    just average all non-excluded metrics after sector-ranking, then z-score cross-sectionally.

    Returns a DataFrame (dates x tickers) forward-filled to calc_dates.
    """
    import sys, os
    # Try to import quality_factor for proper composite computation
    qf_dir = os.path.dirname(os.path.abspath(__file__))
    if qf_dir not in sys.path:
        sys.path.insert(0, qf_dir)

    try:
        from quality_factor import (load_anchor_dates, load_all_snapshots,
                                     build_derived_metrics, derive_weights,
                                     compute_composite_scores, compute_rate_signal,
                                     QF_MAV_WINDOW, QF_THRESHOLD)
        import importlib, quality_factor as _qf

        print("  Loading quality factor scores from anchor snapshots...")
        # We need Pxs_df for rate signal — store as module-level during run()
        # Pass via closure: caller sets _QF_PXS_DF before calling this
        Pxs_df_ref = load_quality_scores._pxs_df
        sectors_ref = load_quality_scores._sectors_s

        anchor_dates = load_anchor_dates()
        snapshots    = load_all_snapshots(anchor_dates)
        rate_signal  = compute_rate_signal(Pxs_df_ref, QF_MAV_WINDOW, QF_THRESHOLD)

        gqf_w = derive_weights(snapshots, sectors_ref, Pxs_df_ref, regime='growth')
        cqf_w = derive_weights(snapshots, sectors_ref, Pxs_df_ref, regime='conservative')

        anchor_sorted = sorted(snapshots.keys())
        rows = {}
        for anchor in anchor_sorted:
            snap = snapshots[anchor]
            if snap is None or snap.empty:
                continue
            snap = build_derived_metrics(snap)
            rate_dates = rate_signal.index[rate_signal.index <= anchor]
            q = float(rate_signal.loc[rate_dates[-1]]) if not rate_dates.empty else 0.5
            scores = compute_composite_scores(snap, sectors_ref, gqf_w, cqf_w, q)
            rows[anchor] = scores

        if not rows:
            print("  WARNING: No quality scores derived — skipping quality factor")
            return pd.DataFrame(index=calc_dates, columns=universe, dtype=float)

        # Build anchor-date pivot, then forward-fill to calc_dates
        anchor_df = pd.DataFrame(rows).T  # dates x tickers
        anchor_df = anchor_df.reindex(columns=universe)

        # Reindex to calc_dates with forward-fill
        all_dates  = calc_dates.union(anchor_df.index).sort_values()
        quality_ff = anchor_df.reindex(all_dates).ffill()
        quality_ff = quality_ff.reindex(calc_dates)

        # Z-score cross-sectionally each date
        quality_z = quality_ff.apply(zscore, axis=1)

        print(f"  Quality scores: {quality_z.notna().any(axis=1).sum()} dates with data "
              f"| {quality_z.notna().any(axis=0).sum()} tickers")
        return quality_z

    except Exception as e:
        print(f"  WARNING: Could not load quality scores ({e}) — skipping quality factor")
        return pd.DataFrame(index=calc_dates, columns=universe, dtype=float)

load_quality_scores._pxs_df   = None  # set by run() before calling
load_quality_scores._sectors_s = None


# ==============================================================================
# VALUE FACTOR — load from valuation_consolidated, sector-rank with reflexivity
# ==============================================================================

VALUE_METRICS  = ['P/S', 'P/Ee', 'P/Eo', 'sP/S', 'sP/E', 'sP/GP', 'P/GP']
VALUE_TABLE    = 'valuation_consolidated'

# IC-derived t-stat weights (absolute, from factor_residuals_joint study, avg 21d+63d)
# Higher absolute t-stat → more weight. Negatives clipped to 0 (all are positive here).
_VALUE_TSTAT = {
    'P/S':   (3.021 + 5.454) / 2,   # 4.238
    'P/Ee':  (3.444 + 4.477) / 2,   # 3.961
    'P/Eo':  (3.076 + 3.963) / 2,   # 3.520
    'sP/S':  (3.077 + 4.901) / 2,   # 3.989
    'sP/E':  (2.630 + 4.234) / 2,   # 3.432
    'sP/GP': (3.140 + 4.190) / 2,   # 3.665
    'P/GP':  (2.577 + 4.194) / 2,   # 3.386
}
_total = sum(_VALUE_TSTAT.values())
VALUE_WEIGHTS = {m: w / _total for m, w in _VALUE_TSTAT.items()}


def load_value_scores(universe: list, calc_dates: pd.DatetimeIndex,
                      sectors_s: pd.Series) -> pd.DataFrame:
    """
    Build daily value composite score from valuation_consolidated.

    For each valuation date:
      1. Load VALUE_METRICS for all tickers
      2. Within each sector, apply reflexivity treatment for negatives:
         adjusted = metric if > 0 else sector_max_positive + abs(metric)
      3. Rank adjusted values ascending within sector (low = cheap = rank 0→1)
         → invert rank so cheap stocks score HIGH: value_rank = 1 - ascending_rank
      4. Weighted average across metrics using IC-derived t-stat weights
      5. Z-score cross-sectionally

    Forward-fill valuation dates to daily calc_dates.
    Returns DataFrame (dates x tickers), z-scored.
    """
    print("  Loading value factor scores from valuation_consolidated...")

    # Load all valuation dates
    with ENGINE.connect() as conn:
        date_rows = conn.execute(text(f"""
            SELECT DISTINCT date FROM {VALUE_TABLE} ORDER BY date
        """)).fetchall()
    val_dates = [pd.Timestamp(r[0]) for r in date_rows]

    if not val_dates:
        print("  WARNING: No valuation dates found — skipping value factor")
        return pd.DataFrame(index=calc_dates, columns=universe, dtype=float)

    # Load all data in one query for efficiency
    with ENGINE.connect() as conn:
        df_all = pd.read_sql(text(f"""
            SELECT date, ticker, {', '.join(f'"{m}"' for m in VALUE_METRICS)}
            FROM {VALUE_TABLE}
            WHERE ticker IS NOT NULL
            ORDER BY date
        """), conn)

    df_all['date']   = pd.to_datetime(df_all['date'])
    df_all['ticker'] = df_all['ticker'].apply(clean_ticker)

    rows = {}
    for val_date, grp in df_all.groupby('date'):
        snap = grp.drop(columns='date').drop_duplicates('ticker').set_index('ticker')
        for m in VALUE_METRICS:
            if m in snap.columns:
                snap[m] = pd.to_numeric(snap[m], errors='coerce')

        # Attach sector
        snap['_sector'] = snap.index.map(sectors_s)
        snap = snap[snap['_sector'].notna()]
        if snap.empty:
            continue

        metric_scores = {}
        for m in VALUE_METRICS:
            if m not in snap.columns:
                continue
            scores = pd.Series(np.nan, index=snap.index)

            for sec, sec_grp in snap.groupby('_sector'):
                m_vals = sec_grp[m].copy()
                valid  = m_vals.dropna()
                if len(valid) < 3:
                    continue

                # Reflexivity: negatives → sector_max_positive + abs(value)
                pos_mask = valid > 0
                if pos_mask.any():
                    sec_max = valid[pos_mask].max()
                    adjusted = valid.copy()
                    adjusted[~pos_mask] = sec_max + valid[~pos_mask].abs()
                else:
                    # All negative — invert abs so most negative = most expensive
                    adjusted = valid.abs().max() - valid

                # Ascending rank (low P/x = cheap) then invert → cheap scores HIGH
                asc_rank = adjusted.rank(method='average', ascending=True)
                val_rank = 1.0 - (asc_rank - 1) / (len(asc_rank) - 1) if len(asc_rank) > 1 \
                           else pd.Series(0.5, index=asc_rank.index)
                scores.loc[val_rank.index] = val_rank

            metric_scores[m] = scores

        if not metric_scores:
            continue

        # Weighted composite
        composite = pd.Series(0.0, index=snap.index)
        total_w   = 0.0
        for m, w in VALUE_WEIGHTS.items():
            if m in metric_scores:
                s = metric_scores[m].reindex(snap.index)
                valid_mask = s.notna()
                composite[valid_mask] += w * s[valid_mask]
                total_w += w
        if total_w > 0:
            composite /= total_w

        rows[pd.Timestamp(val_date)] = composite.reindex(universe)

    if not rows:
        print("  WARNING: No value scores derived — skipping value factor")
        return pd.DataFrame(index=calc_dates, columns=universe, dtype=float)

    # Build valuation-date pivot, forward-fill to daily calc_dates
    val_df     = pd.DataFrame(rows).T.sort_index()
    all_dates  = calc_dates.union(val_df.index).sort_values()
    value_ff   = val_df.reindex(all_dates).ffill().reindex(calc_dates)

    # Z-score cross-sectionally each date
    value_z = value_ff.apply(zscore, axis=1)

    print(f"  Value scores: {value_z.notna().any(axis=1).sum()} dates with data "
          f"| {value_z.notna().any(axis=0).sum()} tickers")
    return value_z


# ==============================================================================
# MAIN
# ==============================================================================

def run(Pxs_df: pd.DataFrame, sectors_s: pd.Series,
        volumeTrd_df: pd.DataFrame = None) -> dict:
    print("=" * 70)
    print("  FACTOR MODEL - STEP 1")
    print("=" * 70)

    # Deduplicate upfront
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]
    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]

    # Start date
    st_input     = input("\n  Start date (YYYY-MM-DD, or Enter for 2019-01-01): ").strip()
    st_dt        = pd.Timestamp(st_input) if st_input else pd.Timestamp('2019-01-01')
    print(f"  Start date       : {st_dt.date()}")
    ridge_input  = input("  Ridge penalty λ  (or Enter for default=0.1): ").strip()
    RIDGE_LAMBDA = float(ridge_input) if ridge_input else 0.1
    print(f"  Ridge penalty    : {RIDGE_LAMBDA}")
    MACRO_RIDGE  = min(2 * RIDGE_LAMBDA, max(RIDGE_LAMBDA, 0.5))
    print(f"  Macro ridge λ    : {MACRO_RIDGE}")
    vol_input    = input("  Volume-scaled momentum? (y/n) [default=n]: ").strip().lower()
    use_vol_scale = vol_input == 'y'
    if use_vol_scale:
        vol_lo_in  = input("    Vol scalar lower bound [default=0.5]: ").strip()
        vol_hi_in  = input("    Vol scalar upper bound [default=3.0]: ").strip()
        VOL_LOWER  = float(vol_lo_in) if vol_lo_in else 0.5
        VOL_UPPER  = float(vol_hi_in) if vol_hi_in else 3.0
        print(f"  Vol scalar clip  : [{VOL_LOWER}, {VOL_UPPER}]")
    else:
        VOL_LOWER, VOL_UPPER = 0.5, 3.0

    # Extended start: MOM_LONG_BUFFER trading days before st_dt
    all_dates      = Pxs_df.index
    st_dt_loc      = all_dates.searchsorted(st_dt)
    ext_loc        = max(0, st_dt_loc - MOM_LONG_BUFFER)
    extended_st_dt = all_dates[ext_loc]
    print(f"  Extended start   : {extended_st_dt.date()} "
          f"({MOM_LONG_BUFFER} trading days before st_dt)")

    # Setup
    universe   = get_universe(Pxs_df, sectors_s, extended_st_dt)
    sector_dum = build_sector_dummies(universe, sectors_s)
    sec_cols   = sector_dum.columns.tolist()

    all_rets   = Pxs_df[universe].pct_change()

    # Winsorize extreme returns — catches any residual data errors after price cleaning
    RETURN_CLIP = 0.50
    n_clipped   = int((all_rets.abs() > RETURN_CLIP).sum().sum())
    if n_clipped > 0:
        print(f"  Winsorizing {n_clipped} extreme return observations (|ret| > {RETURN_CLIP:.0%})")
    all_rets = all_rets.clip(lower=-RETURN_CLIP, upper=RETURN_CLIP)
    ext_dates  = all_dates[all_dates >= extended_st_dt]
    valid_ext  = ext_dates[
        all_rets.loc[ext_dates, universe].notna().sum(axis=1) >= MIN_STOCKS
    ]
    valid_days = valid_ext[valid_ext >= st_dt]

    print(f"  Extended dates   : {len(valid_ext)} (from {extended_st_dt.date()})")
    print(f"  Valid dates      : {len(valid_days)} "
          f"(from {st_dt.date()}, for variance stats)")

    # Dynamic size and SI composite — load from DB cache, compute only new dates
    dynamic_size = load_dynamic_size(universe, Pxs_df, valid_ext)
    si_composite = load_si_composite(universe, valid_ext)

    # --------------------------------------------------------------------------
    # Build all characteristics upfront
    # --------------------------------------------------------------------------

    # Rolling betas (market)
    beta_df = calc_rolling_betas(Pxs_df, universe, valid_ext)

    # Macro factor betas — same EWMA structure as market beta
    macro_betas = calc_macro_betas(Pxs_df, universe, valid_ext)
    macro_cols  = list(macro_betas.keys())
    print(f"  Macro factors computed: {len(macro_cols)}")

    # Size characteristic: z-scored log(dynamic_size) cross-sectionally per day
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

    # Sector dummies expanded to date index
    dates_ext_common = valid_ext.intersection(beta_df.index).intersection(
        size_char_df.index
    )
    sec_char = {'beta': beta_df, 'size': size_char_df}
    for col in sec_cols:
        sec_char[col] = pd.DataFrame(
            {dt: sector_dum[col] for dt in dates_ext_common}
        ).T

    # --------------------------------------------------------------------------
    # Determine COMMON SAMPLE:
    # Dates where ALL characteristics are available (beta + size + sectors)
    # Then further intersect with mom dates after mom_12m1 is computed
    # --------------------------------------------------------------------------

    # Steps 1-3 common dates: beta + size available
    dates_123 = valid_days.intersection(beta_df.index).intersection(
        size_char_df.index
    )

    # --------------------------------------------------------------------------
    # UFV on common sample
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  BASELINE: UFV (common sample, st_dt onwards)")
    print("=" * 70)
    raw_mat_123 = all_rets.loc[dates_123, universe]
    UFV_123     = variance_stats(raw_mat_123, "UFV - Raw Returns (dates_123)")
    print(f"\n  UFV (dates_123) = {UFV_123:.8f}")

    # --------------------------------------------------------------------------
    # STEP 2: Market Beta — target: raw returns, feature: beta only
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 2: Market Beta")
    print("=" * 70)

    resid_mkt_full, lambda_mkt, r2_mkt = run_factor_step(
        factor_cols  = ['beta'],
        char_by_date = {'beta': beta_df},
        all_rets     = all_rets,
        dynamic_size = dynamic_size,
        calc_dates   = dates_ext_common,
        universe     = universe,
    )

    # --------------------------------------------------------------------------
    # STEP 3: Size — target: mkt residuals, feature: size only
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 3: Size")
    print("=" * 70)

    resid_size_full, lambda_size, r2_size = run_factor_step(
        factor_cols  = ['size'],
        char_by_date = {'size': size_char_df},
        all_rets     = resid_mkt_full,
        dynamic_size = dynamic_size,
        calc_dates   = dates_ext_common,
        universe     = universe,
    )

    # --------------------------------------------------------------------------
    # STEP 4: Macro Factors — target: size residuals, features: macro betas (joint, ridge)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 4: Macro Factors (joint, ridge regularization)")
    print("=" * 70)

    if macro_cols:
        # Full extended dates → macro residuals
        full_dates_macro = dates_ext_common
        for col in macro_cols:
            full_dates_macro = full_dates_macro.intersection(macro_betas[col].index)

        resid_macro_full, lambda_macro, r2_macro = run_factor_step(
            factor_cols  = macro_cols,
            char_by_date = macro_betas,
            all_rets     = resid_size_full,
            dynamic_size = dynamic_size,
            calc_dates   = full_dates_macro,
            universe     = universe,
            ridge_lambda = MACRO_RIDGE,
        )
    else:
        print("  No macro factors available — skipping step 4, using size residuals")
        resid_macro_full = resid_size_full
        lambda_macro     = pd.DataFrame()
        r2_macro         = pd.Series(dtype=float)

    # --------------------------------------------------------------------------
    # STEP 5: Sector Dummies — target: macro residuals, features: sector dummies only
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 5: Sector Dummies")
    print("=" * 70)

    resid_sec_full, lambda_sec, r2_sec = run_factor_step(
        factor_cols  = sec_cols,
        char_by_date = {col: sec_char[col] for col in sec_cols},
        all_rets     = resid_macro_full,
        dynamic_size = dynamic_size,
        calc_dates   = dates_ext_common,
        universe     = universe,
    )

    # --------------------------------------------------------------------------
    # STEP 5: Quality Factor — target: sector residuals, feature: quality only
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 5: + Quality Factor (standalone)")
    print("=" * 70)

    # Quality factor — load from anchor snapshots, forward-fill to valid_ext
    load_quality_scores._pxs_df    = Pxs_df
    load_quality_scores._sectors_s = sectors_s
    quality_df = load_quality_scores(universe, valid_ext)

    # Value factor — load from valuation_consolidated, forward-fill to valid_ext
    value_df = load_value_scores(universe, valid_ext, sectors_s)

    # Vol factor (Garman-Klass)
    open_df, high_df, low_df = load_ohlc_tables(universe)
    vol_df = calc_vol_factor(Pxs_df, universe, valid_ext,
                             open_df=open_df, high_df=high_df, low_df=low_df)

    # --------------------------------------------------------------------------
    # FINAL COMMON SAMPLE
    # --------------------------------------------------------------------------
    common_dates = (dates_123
                    .intersection(vol_df.index)
                    .intersection(si_composite.index)
                    .intersection(quality_df.index[quality_df.notna().any(axis=1)])
                    .intersection(value_df.index[value_df.notna().any(axis=1)]))
    for col in macro_cols:
        common_dates = common_dates.intersection(macro_betas[col].index)
    print(f"\n  Common sample: {len(common_dates)} dates "
          f"(beta + size + macro + sectors + quality + vol + SI + value, from st_dt)")

    if len(common_dates) == 0:
        print("  ERROR: common_dates is empty — check that all factor inputs overlap. Aborting.")
        return None

    # Recompute UFV on final common sample
    print("\n" + "=" * 70)
    print("  BASELINE: UFV (final common sample)")
    print("=" * 70)
    raw_mat = all_rets.loc[common_dates, universe]
    UFV     = variance_stats(raw_mat, "UFV - Raw Returns (common sample)")
    print(f"\n  UFV = {UFV:.8f}")

    # Full extended quality residuals (for idio momentum computation)
    full_dates_quality = valid_ext.intersection(quality_df.index[quality_df.notna().any(axis=1)])
    resid_full_quality, _, _ = run_factor_step(
        factor_cols  = ['quality'],
        char_by_date = {'quality': quality_df},
        all_rets     = resid_sec_full,
        dynamic_size = dynamic_size,
        calc_dates   = full_dates_quality,
        universe     = universe,
        ridge_lambda = 0.0,
    )

    # Common dates quality regression (for variance stats)
    resid_quality, lambda_quality, r2_quality = run_factor_step(
        factor_cols  = ['quality'],
        char_by_date = {'quality': quality_df},
        all_rets     = resid_sec_full,
        dynamic_size = dynamic_size,
        calc_dates   = common_dates,
        universe     = universe,
        ridge_lambda = 0.0,
    )

    # --------------------------------------------------------------------------
    # STEP 6: GK Vol + SI + IdioMom (joint)
    # target: quality residuals
    # idio momentum computed from quality residuals (net of market+size+sectors+quality)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 6: + GK Vol + SI + IdioMom (joint, on quality residuals)")
    print("=" * 70)

    # Idio momentum from quality residuals
    if use_vol_scale:
        if volumeTrd_df is None:
            print("  WARNING: use_vol_scale=True but volumeTrd_df not provided — "
                  "falling back to standard idio momentum")
            mom_df = calc_idio_momentum(resid_full_quality, valid_ext)
        else:
            mom_df = calc_idio_momentum_volscaled(
                resid_full_quality, volumeTrd_df, valid_ext,
                vol_lower=VOL_LOWER, vol_upper=VOL_UPPER
            )
    else:
        mom_df = calc_idio_momentum(resid_full_quality, valid_ext)

    # Update common_dates to include mom — this is the FINAL common_dates
    common_dates = common_dates.intersection(mom_df.index)
    if len(common_dates) == 0:
        print("  ERROR: common_dates empty after intersecting with mom_df. Aborting.")
        return None
    print(f"  Common sample after idio_mom: {len(common_dates)} dates")

    # Check O-U cache override — must happen after common_dates is fully finalized
    ou_override = input("\n  Override existing O-U cached dates? (y/n) [default=n]: ").strip().lower() == 'y'
    ou_already_done = set()
    try:
        with ENGINE.connect() as conn:
            exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = :t
                )
            """), {"t": OU_REVERSION_TBL}).scalar()
        if exists:
            if ou_override:
                print(f"  Dropping '{OU_REVERSION_TBL}' for full recompute...")
                with ENGINE.begin() as conn:
                    conn.execute(text(f"DROP TABLE IF EXISTS {OU_REVERSION_TBL}"))
            else:
                with ENGINE.connect() as conn:
                    rows = conn.execute(text(
                        f"SELECT DISTINCT date FROM {OU_REVERSION_TBL}"
                    )).fetchall()
                ou_already_done = {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        pass
    ou_dates_to_calc = pd.DatetimeIndex([d for d in common_dates if d not in ou_already_done])
    print(f"  O-U cached: {len(ou_already_done)} dates | to compute: {len(ou_dates_to_calc)} dates")

    # Full extended dates → step 6 residuals for DB
    full_dates_mom = (full_dates_quality
                      .intersection(mom_df.index)
                      .intersection(vol_df.index)
                      .intersection(si_composite.index))
    resid_full_mom, _, _ = run_factor_step(
        factor_cols  = ['vol', 'si_composite', 'idio_mom'],
        char_by_date = {'vol': vol_df, 'si_composite': si_composite, 'idio_mom': mom_df},
        all_rets     = resid_full_quality,
        dynamic_size = dynamic_size,
        calc_dates   = full_dates_mom,
        universe     = universe,
        ridge_lambda = RIDGE_LAMBDA,  # L2 regularization — only applied here (3 correlated features)
    )

    # Common dates step 6
    resid_mom, lambda_mom, r2_mom = run_factor_step(
        factor_cols  = ['vol', 'si_composite', 'idio_mom'],
        char_by_date = {'vol': vol_df, 'si_composite': si_composite, 'idio_mom': mom_df},
        all_rets     = resid_full_quality,
        dynamic_size = dynamic_size,
        calc_dates   = common_dates,
        universe     = universe,
        ridge_lambda = RIDGE_LAMBDA,  # L2 regularization — only applied here (3 correlated features)
    )

    # --------------------------------------------------------------------------
    # STEP 7: Value — target: step 6 residuals
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 7: + Value (net of momentum + vol + SI + quality)")
    print("=" * 70)

    # Full extended dates → value residuals (used for O-U fitting)
    full_dates_value = (full_dates_mom
                        .intersection(value_df.index[value_df.notna().any(axis=1)]))
    resid_full_value, _, _ = run_factor_step(
        factor_cols  = ['value'],
        char_by_date = {'value': value_df},
        all_rets     = resid_full_mom,
        dynamic_size = dynamic_size,
        calc_dates   = full_dates_value,
        universe     = universe,
        ridge_lambda = 0.0,
    )

    # Common dates step 7
    resid_joint, lambda_joint, r2_joint = run_factor_step(
        factor_cols  = ['value'],
        char_by_date = {'value': value_df},
        all_rets     = resid_mom,
        dynamic_size = dynamic_size,
        calc_dates   = common_dates,
        universe     = universe,
        ridge_lambda = 0.0,
    )

    # --------------------------------------------------------------------------
    # STEP 8: O-U Mean Reversion — target: value residuals
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 8: + O-U Mean Reversion (final step)")
    print("=" * 70)

    if len(ou_dates_to_calc) > 0:
        print(f"  Computing O-U for {len(ou_dates_to_calc)} new dates...")
        # Volume scaling for O-U: use volumeTrd_df if provided (consistent with idio mom)
        new_ou       = _compute_ou_for_dates(
            ou_dates_to_calc, universe,
            resid_full_value, Pxs_df,
            volumeTrd_df if use_vol_scale else None,
        )
        long         = new_ou.stack(dropna=False).reset_index()
        long.columns = ['date', 'ticker', 'ou_score']
        long         = long.dropna(subset=['ou_score'])
        long['date'] = pd.to_datetime(long['date'])
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {OU_REVERSION_TBL} (
                    date      DATE,
                    ticker    VARCHAR(20),
                    ou_score  NUMERIC,
                    PRIMARY KEY (date, ticker)
                )
            """))
        long.to_sql(OU_REVERSION_TBL, ENGINE, if_exists='append', index=False)
        print(f"  Saved {len(long):,} new rows to '{OU_REVERSION_TBL}'")

    date_list = [d.date() for d in common_dates]
    try:
        with ENGINE.connect() as conn:
            ou_rows = conn.execute(text(f"""
                SELECT date, ticker, ou_score FROM {OU_REVERSION_TBL}
                WHERE date = ANY(:dates)
            """), {"dates": date_list}).fetchall()
    except Exception:
        ou_rows = []

    if ou_rows:
        ou_df             = pd.DataFrame(ou_rows, columns=['date', 'ticker', 'ou_score'])
        ou_df['date']     = pd.to_datetime(ou_df['date'])
        ou_df['ou_score'] = ou_df['ou_score'].astype(float)
        ou_pivot = ou_df.pivot_table(index='date', columns='ticker',
                                      values='ou_score', aggfunc='last')
        ou_pivot = ou_pivot.reindex(columns=universe).reindex(common_dates)
    else:
        ou_pivot = pd.DataFrame(index=common_dates, columns=universe, dtype=float)
    print(f"  O-U scores loaded: {ou_pivot.shape}")

    ou_common = common_dates.intersection(
        ou_pivot.index[ou_pivot.notna().any(axis=1)]
    )

    # Full extended dates → full model residuals for DB
    full_dates_ou = full_dates_value.intersection(
        ou_pivot.index[ou_pivot.notna().any(axis=1)]
    )
    resid_full, _, _ = run_factor_step(
        factor_cols  = ['ou_reversion'],
        char_by_date = {'ou_reversion': ou_pivot},
        all_rets     = resid_full_value,
        dynamic_size = dynamic_size,
        calc_dates   = full_dates_ou,
        universe     = universe,
        ridge_lambda = 0.0,
    )

    resid_ou, lambda_ou, r2_ou = run_factor_step(
        factor_cols  = ['ou_reversion'],
        char_by_date = {'ou_reversion': ou_pivot},
        all_rets     = resid_joint,
        dynamic_size = dynamic_size,
        calc_dates   = ou_common,
        universe     = universe,
        ridge_lambda = 0.0,
    )

    # Restrict early residuals to common sample for variance stats
    resid_mkt   = resid_mkt_full[resid_mkt_full.index.isin(common_dates)]
    resid_size  = resid_size_full[resid_size_full.index.isin(common_dates)]
    resid_macro = resid_macro_full[resid_macro_full.index.isin(common_dates)] if macro_cols else resid_size
    resid_sec   = resid_sec_full[resid_sec_full.index.isin(common_dates)]

    # Step 8 (value) residuals restricted to ou_common for step 9 comparison
    raw_ou              = all_rets.loc[ou_common, universe]
    UFV_ou              = variance_stats(raw_ou, "UFV (ou_common subsample)", UFV)
    resid_joint_ou      = resid_joint[resid_joint.index.isin(ou_common)]
    joint_UFV_ou_scalar = variance_stats(resid_joint_ou, "joint_UFV (ou_common subsample)", UFV_ou)

    # --------------------------------------------------------------------------
    # Variance stats — all on common sample
    # --------------------------------------------------------------------------
    mkt_UFV     = variance_stats(resid_mkt,     "mkt_UFV     - Beta Residuals",           UFV)
    size_UFV    = variance_stats(resid_size,    "size_UFV    - Size Residuals",            mkt_UFV)
    _           = variance_stats(resid_size,    "size_UFV vs UFV",                         UFV)
    if macro_cols:
        macro_UFV = variance_stats(resid_macro, "macro_UFV   - Macro Factor Residuals",    size_UFV)
        _         = variance_stats(resid_macro, "macro_UFV vs UFV",                        UFV)
    else:
        macro_UFV = size_UFV
    sec_UFV     = variance_stats(resid_sec,     "sec_UFV     - Sector Residuals",          macro_UFV)
    _           = variance_stats(resid_sec,     "sec_UFV vs UFV",                          UFV)
    quality_UFV = variance_stats(resid_quality, "quality_UFV - Quality Residuals",         sec_UFV)
    _           = variance_stats(resid_quality, "quality_UFV vs UFV",                      UFV)
    mom_UFV     = variance_stats(resid_mom,     "mom_UFV     - +Vol+SI+IdioMom",           quality_UFV)
    _           = variance_stats(resid_mom,     "mom_UFV vs UFV",                          UFV)
    joint_UFV   = variance_stats(resid_joint,   "joint_UFV   - +Value",                    mom_UFV)
    _           = variance_stats(resid_joint,   "joint_UFV vs UFV",                        UFV)
    ou_UFV      = variance_stats(resid_ou,      "ou_UFV      - +O-U Mean Reversion",       joint_UFV_ou_scalar)
    _           = variance_stats(resid_ou,      "ou_UFV vs UFV",                           UFV_ou)

    # R2 stats
    r2_stats(r2_mkt[r2_mkt.index.isin(common_dates)],          "Step 2: Market Beta")
    r2_stats(r2_size[r2_size.index.isin(common_dates)],         "Step 3: Size")
    if macro_cols:
        r2_stats(r2_macro[r2_macro.index.isin(common_dates)],  "Step 4: Macro Factors")
    r2_stats(r2_sec[r2_sec.index.isin(common_dates)],           "Step 5: Sectors")
    r2_stats(r2_quality[r2_quality.index.isin(common_dates)],   "Step 6: Quality")
    r2_stats(r2_mom,                                             "Step 7: Vol + SI + IdioMom")
    r2_stats(r2_joint,                                           "Step 8: Value")
    r2_stats(r2_ou,                                              "Step 9: O-U Mean Reversion")

    # --------------------------------------------------------------------------
    # Save to DB
    # --------------------------------------------------------------------------
    save_lambdas(lambda_mkt[lambda_mkt.index.isin(common_dates)],          'factor_lambdas_mkt')
    save_lambdas(lambda_size[lambda_size.index.isin(common_dates)],         'factor_lambdas_size')
    if macro_cols:
        save_lambdas(lambda_macro[lambda_macro.index.isin(common_dates)],   'factor_lambdas_macro')
    save_lambdas(lambda_sec[lambda_sec.index.isin(common_dates)],           'factor_lambdas_sec')
    save_lambdas(lambda_quality[lambda_quality.index.isin(common_dates)],   'factor_lambdas_quality')
    save_lambdas(lambda_mom[lambda_mom.index.isin(common_dates)],           'factor_lambdas_mom')
    save_lambdas(lambda_joint[lambda_joint.index.isin(common_dates)],       'factor_lambdas_joint')
    save_lambdas(lambda_ou[lambda_ou.index.isin(ou_common)],                'factor_lambdas_ou')

    save_residuals(resid_mkt_full,    'factor_residuals_mkt')
    save_residuals(resid_size_full,   'factor_residuals_size')
    if macro_cols:
        save_residuals(resid_macro_full, 'factor_residuals_macro')
    save_residuals(resid_sec_full,    'factor_residuals_sec')
    save_residuals(resid_full_quality,'factor_residuals_quality')
    save_residuals(resid_full_mom,    'factor_residuals_mom')
    save_residuals(resid_full_value,  'factor_residuals_joint')   # after value — used by OU diag
    save_residuals(resid_full,        'factor_residuals_ou')      # full model residuals

    # --------------------------------------------------------------------------
    # Lambda distributions
    # --------------------------------------------------------------------------
    print_lambda_summary(lambda_mkt,     ['beta'],
                         "Step 2: Market Beta", common_dates)
    print_lambda_summary(lambda_size,    ['size'],
                         "Step 3: Size", common_dates, annual_col='size')
    if macro_cols:
        print_lambda_summary(lambda_macro, macro_cols,
                             "Step 4: Macro Factors", common_dates, annual_col=macro_cols[0])
        for mc in macro_cols:
            print_lambda_summary(lambda_macro, [mc],
                                 f"Annual — Macro: {mc}", common_dates, annual_col=mc)
    print_lambda_summary(lambda_sec,     sec_cols,
                         "Step 5: Sectors", common_dates)
    print_sector_lambdas(lambda_sec, sec_cols, common_dates)
    print_lambda_summary(lambda_quality, ['quality'],
                         "Step 6: Quality Factor", common_dates, annual_col='quality')
    print_lambda_summary(lambda_mom,     ['vol', 'si_composite', 'idio_mom'],
                         "Step 7: Vol + SI + IdioMom", common_dates, annual_col='si_composite')
    print_lambda_summary(lambda_joint,   ['value'],
                         "Step 8: Value", common_dates, annual_col='value')
    print_lambda_summary(lambda_ou,      ['ou_reversion'],
                         "Step 9: O-U Mean Reversion", ou_common, annual_col='ou_reversion')
    print_sector_lambdas(lambda_ou, sec_cols, ou_common)
    # Annual breakdowns
    print_lambda_summary(lambda_quality, ['quality'],      "Annual — Quality",       common_dates, annual_col='quality')
    print_lambda_summary(lambda_mom,     ['vol'],          "Annual — Vol Factor",    common_dates, annual_col='vol')
    print_lambda_summary(lambda_mom,     ['si_composite'], "Annual — SI",            common_dates, annual_col='si_composite')
    print_lambda_summary(lambda_mom,     ['idio_mom'],     "Annual — Idio Momentum", common_dates, annual_col='idio_mom')
    print_lambda_summary(lambda_joint,   ['value'],        "Annual — Value",         common_dates, annual_col='value')
    print_lambda_summary(lambda_ou,      ['ou_reversion'], "Annual — O-U",           ou_common,    annual_col='ou_reversion')

    # --------------------------------------------------------------------------
    # Variance Reduction Summary
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  VARIANCE REDUCTION SUMMARY (common sample)")
    print("=" * 70)
    summary_rows = [
        ("UFV (raw returns)",                    UFV,         UFV,         None),
        ("mkt_UFV     (+ beta)",                 mkt_UFV,     UFV,         UFV),
        ("size_UFV    (+ size)",                 size_UFV,    UFV,         mkt_UFV),
    ]
    if macro_cols:
        summary_rows.append(("macro_UFV   (+ macro factors)",   macro_UFV,   UFV,  size_UFV))
        summary_rows.append(("sec_UFV     (+ sectors)",         sec_UFV,     UFV,  macro_UFV))
    else:
        summary_rows.append(("sec_UFV     (+ sectors)",         sec_UFV,     UFV,  size_UFV))
    summary_rows += [
        ("quality_UFV (+ quality)",              quality_UFV, UFV,         sec_UFV),
        ("mom_UFV     (+ Vol+SI+IdioMom)",       mom_UFV,     UFV,         quality_UFV),
        ("joint_UFV   (+ value)",                joint_UFV,   UFV,         mom_UFV),
        ("ou_UFV      (+ O-U)*",                 ou_UFV,      UFV_ou,      joint_UFV_ou_scalar),
    ]
    print(f"  {'':<44} {'Variance':>12} {'% UFV':>8} {'% prev':>8}")
    print(f"  {'-'*76}")
    for lbl, var, base, prev in summary_rows:
        pct_ufv  = f"{var/base*100:.2f}%"
        pct_prev = f"{var/prev*100:.2f}%" if prev else "---"
        print(f"  {lbl:<44} {var:>12.8f} {pct_ufv:>8} {pct_prev:>8}")
    print("  * O-U on ou_common subsample; % prev vs step 8 on same subsample")

    # --------------------------------------------------------------------------
    # CONSOLIDATED SUMMARY
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONSOLIDATED MODEL SUMMARY")
    print("=" * 70)

    # Consolidated R² = 1 - ou_UFV / UFV_ou (both on ou_common subsample)
    r2_consolidated = 1.0 - (ou_UFV / UFV_ou) if UFV_ou > 0 else np.nan
    print(f"\n  Consolidated R²  : {r2_consolidated:.4f}  "
          f"({r2_consolidated*100:.2f}% of variance explained)")

    # Consolidated intercept = sum of all step intercepts per date
    # Collect all intercept series on ou_common dates
    intercept_frames = []
    for ldf, label in [
        (lambda_mkt,     'mkt'),
        (lambda_size,    'size'),
        (lambda_sec,     'sec'),
        (lambda_quality, 'quality'),
        (lambda_mom,     'mom'),
        (lambda_joint,   'joint'),
        (lambda_ou,      'ou'),
    ]:
        if ldf is not None and len(ldf) > 0 and 'intercept' in ldf.columns:
            intercept_frames.append(ldf['intercept'].rename(label))
    if macro_cols and len(lambda_macro) > 0 and 'intercept' in lambda_macro.columns:
        intercept_frames.append(lambda_macro['intercept'].rename('macro'))

    if intercept_frames:
        intercept_df           = pd.concat(intercept_frames, axis=1).reindex(ou_common)
        consolidated_intercept = intercept_df.sum(axis=1)
        ci_mean = consolidated_intercept.mean()
        ci_std  = consolidated_intercept.std()
        ci_t    = ci_mean / (ci_std / np.sqrt(len(consolidated_intercept)))
        print(f"\n  Consolidated intercept (sum of all step intercepts):")
        print(f"    Mean   : {ci_mean*100:+.4f}%")
        print(f"    Std    : {ci_std*100:.4f}%")
        print(f"    t-stat : {ci_t:+.2f}")
    else:
        consolidated_intercept = pd.Series(np.nan, index=ou_common)

    # --------------------------------------------------------------------------
    # LAST 5 DAYS SNAPSHOT — factor lambdas, intercepts, R² per step
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  LAST 5 DAYS SNAPSHOT")
    print("=" * 70)

    last5 = ou_common[-5:]

    # Build per-step per-date records
    step_frames = []

    def _add_step(ldf, r2s, step_label, cols):
        if ldf is None or len(ldf) == 0:
            return
        sub = ldf.reindex(last5)
        r2  = r2s.reindex(last5) if r2s is not None and len(r2s) > 0               else pd.Series(np.nan, index=last5)
        for dt in last5:
            row = {'date': dt.date(), 'step': step_label}
            for c in cols:
                v = sub.loc[dt, c] if dt in sub.index and c in sub.columns else np.nan
                row[c] = round(v * 100, 2) if not np.isnan(v) else np.nan
            iv = sub.loc[dt, 'intercept'] if dt in sub.index and 'intercept' in sub.columns else np.nan
            rv = r2.loc[dt] if dt in r2.index else np.nan
            row['intercept'] = round(iv * 100, 2) if not np.isnan(iv) else np.nan
            row['r2']        = round(rv * 100, 2) if not np.isnan(rv) else np.nan
            step_frames.append(row)

    _add_step(lambda_mkt,     r2_mkt,     'Step2_Beta',    ['beta'])
    _add_step(lambda_size,    r2_size,    'Step3_Size',    ['size'])
    if macro_cols and len(lambda_macro) > 0:
        _add_step(lambda_macro, r2_macro, 'Step4_Macro',   macro_cols)
    _add_step(lambda_sec,     r2_sec,     'Step5_Sectors', sec_cols)
    _add_step(lambda_quality, r2_quality, 'Step6_Quality', ['quality'])
    _add_step(lambda_mom,     r2_mom,     'Step7_Mom',     ['vol', 'si_composite', 'idio_mom'])
    _add_step(lambda_joint,   r2_joint,   'Step8_Value',   ['value'])
    _add_step(lambda_ou,      r2_ou,      'Step9_OU',      ['ou_reversion'])

    last5_snapshot = pd.DataFrame(step_frames).set_index(['date', 'step'])                      if step_frames else pd.DataFrame()

    if not last5_snapshot.empty:
        for dt in last5:
            c_int = consolidated_intercept.loc[dt] if dt in consolidated_intercept.index else np.nan
            c_r2  = r2_consolidated
            print(f"\n  {'='*70}")
            print(f"  {dt.date()}   |   Consolidated intercept: {c_int*100:+.2f}%   |   R²: {c_r2*100:.2f}%")
            print(f"  {'='*70}")
            print(f"  {'Step':<20} {'Factor':<24} {'Lambda%':>9} {'Intcpt%':>9} {'R²%':>7}")
            print(f"  {'-'*71}")
            dt_date = dt.date()
            if dt_date not in last5_snapshot.index.get_level_values('date'):
                continue
            for step_label in last5_snapshot.loc[dt_date].index:
                row              = last5_snapshot.loc[(dt_date, step_label)]
                factor_cols_here = [c for c in row.index
                                    if c not in ('intercept', 'r2') and not pd.isna(row[c])]
                intercept_val    = row.get('intercept', np.nan)
                r2_val           = row.get('r2', np.nan)
                for i, fc in enumerate(factor_cols_here):
                    lam_str  = f"{row[fc]:>+9.2f}"
                    if i == 0:
                        int_str = f"{intercept_val:>+9.2f}" if not np.isnan(intercept_val) else f"{'nan':>9}"
                        r2_str  = f"{r2_val:>7.2f}"         if not np.isnan(r2_val)        else f"{'nan':>7}"
                        print(f"  {step_label:<20} {fc:<24} {lam_str} {int_str} {r2_str}")
                    else:
                        print(f"  {'':20} {fc:<24} {lam_str}")

    return {
        'UFV':               UFV,
        'mkt_UFV':           mkt_UFV,
        'size_UFV':          size_UFV,
        'macro_UFV':         macro_UFV,
        'sec_UFV':           sec_UFV,
        'quality_UFV':       quality_UFV,
        'mom_UFV':           mom_UFV,
        'joint_UFV':         joint_UFV,
        'ou_UFV':            ou_UFV,
        'resid_mkt':         resid_mkt,
        'resid_size':        resid_size,
        'resid_sec':         resid_sec,
        'resid_sec_full':    resid_sec_full,
        'resid_quality':     resid_quality,
        'resid_full_quality':resid_full_quality,
        'resid_mom':         resid_mom,
        'resid_full_mom':    resid_full_mom,
        'resid_joint':       resid_joint,
        'resid_full_value':  resid_full_value,
        'resid_ou':          resid_ou,
        'resid_full':        resid_full,
        'lambda_mkt':        lambda_mkt,
        'lambda_size':       lambda_size,
        'lambda_macro':      lambda_macro if macro_cols else pd.DataFrame(),
        'lambda_sec':        lambda_sec,
        'lambda_quality':    lambda_quality,
        'lambda_mom':        lambda_mom,
        'lambda_joint':      lambda_joint,
        'lambda_ou':         lambda_ou,
        'macro_betas':       macro_betas,
        'macro_cols':        macro_cols,
        'resid_macro':       resid_macro,
        'resid_macro_full':  resid_macro_full if macro_cols else resid_size_full,
        'r2_consolidated':   r2_consolidated,
        'consolidated_intercept': consolidated_intercept,
        'last5_snapshot':    last5_snapshot,
        'beta_df':           beta_df,
        'size_char_df':      size_char_df,
        'dynamic_size':      dynamic_size,
        'si_composite':      si_composite,
        'mom_df':            mom_df,
        'vol_df':            vol_df,
        'quality_df':        quality_df,
        'value_df':          value_df,
        'ou_pivot':          ou_pivot,
        'universe':          universe,
        'sec_cols':          sec_cols,
        'common_dates':      common_dates,
        'ou_common':         ou_common,
        'st_dt':             st_dt,
        'extended_st_dt':    extended_st_dt,
    }

if __name__ == "__main__":
    print("Usage: from factor_model_step1 import run")
    print("       results = run(Pxs_df, sectors_s)")
    print("       results = run(Pxs_df, sectors_s, volumeTrd_df=volumeTrd_df)  # vol-scaled momentum")
