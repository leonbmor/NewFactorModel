#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

"""
mvo_backtest.py
===============
Runs three portfolios in parallel and compares performance:
  1. Baseline      -- quality factor only (from primary_factor_backtest.py)
  2. Pure Alpha    -- composite alpha signal, equal-weight + concentration
  3. MVO           -- composite alpha signal, mean-variance optimized weights

MVO uses an ensemble of four covariance matrices (Empirical EWMA,
Ledoit-Wolf, Factor-driven XFX', PCA) with Grinold-Kahn alpha scaling,
eligibility filtering (min matrix count), and floor/cap weight constraints.

User prompts mirror composite_backtest.py exactly.
MVO-specific parameters are function arguments (not prompts).

Assumes all factor_model_step1, quality_factor, primary_factor_backtest,
factor_ic_study, portfolio_risk_decomp, mvo_diagnostics functions are
loaded in the Jupyter kernel.

Entry point
-----------
    results = run_mvo_backtest(
        Pxs_df, sectors_s, weights_by_year, regime_s,
        volumeTrd_df=None,
        # MVO parameters:
        ic=0.04,
        max_weight=0.10,
        min_weight=0.025,
        zscore_cap=2.5,
        min_matrix_count=2,
        pca_var_threshold=0.65,
        universe_mult=5,
        risk_aversion=1.0,
        force_rebuild_cache=True,
    )

Returns
-------
    dict with nav_baseline, nav_alpha, nav_mvo,
         port_baseline, port_alpha, port_mvo,
         and intermediate data
"""

import warnings
import sys
import os
import json
import hashlib
import time
import traceback
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import spearmanr
import matplotlib
try:
    matplotlib.use('Agg')
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import create_engine, text

# Wrapper for momentum exclusions (populated by momentum_exclusions.py pipeline step)
def _load_momentum_exclusions(dt, top_n=None):
    try:
        return load_exclusions(dt, top_n=top_n)
    except Exception:
        return {}


def _load_mr_scores_by_date(start_dt=None, end_dt=None):
    """
    Load all MR scores from momentum_exclusions table grouped by date.
    Returns {date: pd.Series({ticker: score})} for dates with score > 0.
    Used to build the per-date penalty map for MR_K soft penalty mode.
    """
    try:
        where = "WHERE score > 0"
        params = {}
        if start_dt is not None:
            where += " AND date >= :sd"
            params['sd'] = pd.Timestamp(start_dt).date()
        if end_dt is not None:
            where += " AND date <= :ed"
            params['ed'] = pd.Timestamp(end_dt).date()
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT date, ticker, score FROM momentum_exclusions
                {where} ORDER BY date, score DESC
            """), params).fetchall()
        result = {}
        for r in rows:
            dt_ = pd.Timestamp(r[0])
            result.setdefault(dt_, {})[r[1]] = float(r[2])
        return {dt_: pd.Series(d) for dt_, d in result.items()}
    except Exception:
        return {}

warnings.filterwarnings('ignore')

# ── Database connection ────────────────────────────────────────────────────────
_DB_URL = 'postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db'
try:
    ENGINE = create_engine(_DB_URL, pool_pre_ping=True)
    with ENGINE.connect() as _test_conn:
        pass
except Exception as _e:
    print(f"  WARNING: DB connection failed: {_e}")
    ENGINE = None

# ===============================================================================
# PARAMETERS  (all tuneable constants in one place)
# ===============================================================================

# ── Backtest universe / rebalancing ───────────────────────────────────────────
MB_START_DATE  = pd.Timestamp('2019-01-01')
MB_TOP_N       = 20       # default number of stocks
MB_REBAL_FREQ  = 30       # default rebalance frequency (calendar days)
MB_MODEL_VER   = 'v2'

# ── Portfolio / position limits ───────────────────────────────────────────────
AUM              = 5_000_000   # default AUM ($5M)
TRADING_COST_BPS = 10          # one-way cost (bps)
VOLUME_WINDOW    = 10          # rolling window for volume de-trending
ADV_WINDOW       = 20          # days for median ADVP calculation
VOL_LOOKBACK     = 63          # rolling window for vol filter

# ── MVO covariance estimation ─────────────────────────────────────────────────
MVO_LOOKBACK         = 252    # return history for cov estimation (days)
MVO_EWMA_HL          = 126    # EWMA half-life for covariance
MVO_PCA_VAR_THRESH   = 0.65   # PCA variance explained threshold
MVO_DEFAULT_IC       = 0.04   # default IC for Grinold-Kahn alpha scaling
MVO_OVERLAP_TARGET   = 0.65   # target portfolio overlap across cov matrices
MVO_MAX_WEIGHT       = 0.10   # hard weight cap (single name)
MVO_ZSCORE_CAP       = 2.50   # composite z-score winsorisation cap
MVO_MIN_MATRIX_COUNT = 2      # stock must appear in >= N cov matrices
MVO_MIN_WEIGHT       = 0.025  # floor on non-zero single-name weight

# ── Smart hybrid drawdown thresholds ─────────────────────────────────────────
SH_DD_ALPHA       = 0.075  # enter hybrid below this drawdown
SH_DD_HYBRID      = 0.175  # enter MVO below this drawdown
SH_DD_EXIT_ALPHA  = 0.050  # recovery needed to exit hybrid → alpha
SH_DD_EXIT_HYBRID = 0.150  # recovery needed to exit MVO → hybrid
SH_PERSIST_DAYS   = 3      # days signal must persist before regime switch

# ── Dynamic rebalancing triggers ──────────────────────────────────────────────
DYN_TO_THRESHOLD_ALPHA  = 0.25   # turnover trigger in alpha regime
DYN_TO_THRESHOLD_HYBRID = 0.30   # turnover trigger in hybrid regime
DYN_TO_THRESHOLD_MVO    = 0.35   # turnover trigger in MVO regime
DYN_VOLDIFF_CAP         = 0.175  # max vol increase alongside TO trigger
DYN_VOLDIFF_DERISK      = -0.750 # vol de-risk trigger (effectively disabled)
DYN_MIN_HOLD_DAYS       = 10     # minimum days between rebalances

# ── Drawdown policy (strategy 8: Dyn + Hedge + DD) ───────────────────────────
# Each tuple: (dd_threshold, fraction_of_remaining_to_cut)
DD_LEVELS = [
    (0.175, 2/5),  # -17.5%: cut 40% of remaining → 60% exposed
    (0.300, 3/7),  # -30.0%: cut 43% of remaining → 34% exposed
    (0.350, 1/2),  # -35.0%: cut 50% of remaining → 17% exposed
    (0.400, 2/3),  # -40.0%: cut 67% of remaining →  6% exposed
    (0.450, 1/1),  # -45.0%: cut 100%              →  0% exposed
]
# Regime forced by each DD level (index matches DD_LEVELS)
# All DD levels fire at or below SH_DD_HYBRID=-17.5% which is already MVO territory
DD_LEVEL_REGIME = ['mvo', 'mvo', 'mvo', 'mvo', 'mvo']
DD_REENTRY_PCT      = 0.075
DD_REENTRY_CONFIRM  = 5
DD_ANNUAL_RESET_PCT = 0.30

# ── Strategy 9 exclusion filter ───────────────────────────────────────────────
# MR_CAP=0.0  → no exclusions → strategy 9 = strategy 8 (convergence check)
# MR_CAP=0.025 → top 2.5% overextended names excluded (default)
MR_CAP = 0.0

# ── MR momentum penalty (soft penalty mode) ───────────────────────────────────
# Divides Idio_Mom and Mom_12M1 z-scores by exp(MR_K * MR_score) per stock.
# MR_K=0.0  → no penalty (standard composite)
# MR_K=1.0  → full exp(score) penalty
# MR_K=0.5  → gentler curve (default starting point)
MR_K = 0.5   # set > 0 to activate soft penalty mode

# ── Quality weight floor ───────────────────────────────────────────────────────
# Sets a minimum share for Quality within the {Idio_Mom + Mom_12M1 + Quality}
# sub-sum, leaving Value untouched. When floor binds, momentum factors are
# reduced proportionally (preserving their ratio to each other).
# QUALITY_FLOOR=0.0 → inactive, weights unchanged
# QUALITY_FLOOR=0.5 → Quality gets at least 50% of the mom+quality sub-sum
QUALITY_FLOOR = 0.0   # set to e.g. 0.5 to activate

# ── Hedge engine ──────────────────────────────────────────────────────────────
BETA_WINDOW    = 63     # rolling window for beta calculation
CORR_WINDOW    = 63     # rolling window for correlation ranking
EFF_MAV_WINDOW = 20     # MAV window for smoothing effectiveness
EFF_FLOOR      = 0.75   # minimum effectiveness score to qualify
CORR_FLOOR     = 0.50   # minimum correlation to portfolio to qualify
HEDGE_RATIO    = 0.25   # hedge size per instrument (fraction of NAV)
MAX_HEDGE      = 0.50   # maximum total hedge (fraction of NAV)
TRIGGER_ASSETS = ['QQQ', 'SPY']  # assets that trigger hedge on/off

# ── Cache / DB table names ────────────────────────────────────────────────────
MB_COV_CACHE_TBL  = 'mvo_cov_cache'
MB_X_CACHE_TBL    = 'mvo_x_snapshots'
DAILY_PORT_TBL    = 'mvo_daily_portfolios'
DAILY_TRIGGER_TBL = 'mvo_daily_triggers'
MB_DAILY_PORT_TBL = 'mvo_daily_portfolios'   # alias kept for compatibility
MB_MIN_COV_MATRICES = 2

# ── ICS / composite score constants ──────────────────────────────────────────
ICS_MIN_STOCKS = 50
ICS_WEIGHT_MIN = 0.10
ICS_WEIGHT_MAX = 0.50
ICS_MOM_LONG   = 252   # 12M1 momentum lookback (trading days)
ICS_MOM_SKIP   = 21    # 12M1 momentum skip period
ICS_MOM_RESID  = {'v1': 'factor_residuals_vol', 'v2': 'v2_factor_residuals_quality'}
ICS_OU_TBL     = {'v1': 'ou_reversion_df',      'v2': 'v2_ou_reversion_df'}
ICS_VALUE_TBL  = {'v1': 'value_scores_df',       'v2': 'v2_value_scores_df'}

# ── X-matrix / factor model internals ────────────────────────────────────────
_MB_MOM_RESID   = {'v1': 'factor_residuals_vol',     'v2': 'v2_factor_residuals_quality'}
_MB_OU_CACHE    = {'v1': 'ou_reversion_df',           'v2': 'v2_ou_reversion_df'}
_MB_SCALAR_TBLS = {
    'v1': {'Beta':'factor_lambdas_mkt','Size':'factor_lambdas_size',
           'Quality':'factor_lambdas_quality','SI':'factor_lambdas_si',
           'GK_Vol':'factor_lambdas_vol','Idio_Mom':'factor_lambdas_mom',
           'Value':'factor_lambdas_joint','OU':'factor_lambdas_ou'},
    'v2': {'Beta':'v2_factor_lambdas_mkt','Size':'v2_factor_lambdas_size',
           'Quality':'v2_factor_lambdas_quality','SI':'v2_factor_lambdas_si',
           'GK_Vol':'v2_factor_lambdas_vol','Idio_Mom':'v2_factor_lambdas_mom',
           'Value':'v2_factor_lambdas_value','OU':'v2_factor_lambdas_ou'},
}
_MB_SEC_TBL     = {'v1': 'factor_lambdas_sec',  'v2': 'v2_factor_lambdas_sec'}
_MB_LAMBDA_META = {'intercept', 'r2', 'ridge_lambda', 'date'}
_MB_F_LOOKBACK  = 252
_MB_F_EWMA_HL   = 42


# ── Inline dependency: select_with_sector_cap ─────────────────────────────────
def select_with_sector_cap(ranked_df, sector_cap, top_n):
    """Select top_n stocks with max sector_cap per sector (relaxes cap if needed)."""
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


# ── Inline dependency: apply_vol_filter ───────────────────────────────────────
def apply_vol_filter(tickers, dt, Pxs_df, lookback=63, vol_cap_mult=3.0):
    """
    Filter out stocks with annualised volatility exceeding vol_cap_mult × median
    cross-sectional vol over the last `lookback` trading days.
    Returns filtered list of tickers.
    """
    past = [d for d in Pxs_df.index if d <= dt][-lookback:]
    if len(past) < 10:
        return list(tickers)
    rets = Pxs_df.loc[past, [t for t in tickers if t in Pxs_df.columns]].pct_change()
    vols = rets.std() * np.sqrt(252)
    vols = vols.dropna()
    if vols.empty:
        return list(tickers)
    threshold = vols.median() * vol_cap_mult
    surviving = vols[vols <= threshold].index.tolist()
    no_data   = [t for t in tickers if t not in vols.index]
    return surviving + no_data


def get_universe(Pxs_df, sectors_s, extended_st_dt):
    """
    Build stock universe: sector-mapped stocks with sufficient price history.
    Uses sectors_s as the canonical universe — NOT filtered by Pxs_df.columns —
    so the universe is invariant to the end date of Pxs_df (no lookahead bias).
    Stocks not yet in Pxs_df at a given calc_date will produce NaN scores and
    be naturally excluded from that date's cross-section.
    """
    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(
                "SELECT DISTINCT ticker FROM income_data"
            )).fetchall()
        db_tickers = {r[0].upper() for r in rows}
    except Exception as e:
        print(f"  WARNING: DB universe query failed ({e}) — using sectors_s")
        db_tickers = set(sectors_s.index)

    etf_tickers = set(sectors_s.values)
    pre_dates   = Pxs_df.index[Pxs_df.index < extended_st_dt]

    universe = []
    for col in sectors_s.index:
        if col in ('SPX',) or col in etf_tickers:
            continue
        if col.upper() not in db_tickers:
            continue
        # Pre-start history check — skip if stock not yet in Pxs_df or insufficient data
        # Per-calc-date filtering is handled by _get_active_universe
        if col in Pxs_df.columns and len(pre_dates) >= BETA_WINDOW:
            col_data = Pxs_df.loc[pre_dates[-BETA_WINDOW:], col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            if int(col_data.notna().sum()) < BETA_WINDOW // 2:
                continue
        universe.append(col)

    print(f"  Universe: {len(universe)} stocks "
          f"(sector mapped + DB + sufficient pre-start history)")
    return universe


def generate_calc_dates(Pxs_df, step_days=30):
    """Generate rebalancing dates from MB_START_DATE at step_days intervals."""
    end_date = Pxs_df.index.max()
    dates    = []
    current  = MB_START_DATE
    while current <= end_date:
        available = Pxs_df.index[Pxs_df.index >= current]
        if available.empty:
            break
        dates.append(available[0])
        current += pd.Timedelta(days=step_days)
    return sorted(set(dates))




def _ics_zscore(s):
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return s * 0
    return (s - mu) / sd

def _ics_bounded_normalize(raw_series, w_min=ICS_WEIGHT_MIN, w_max=ICS_WEIGHT_MAX):
    s = raw_series.clip(lower=0)
    total = s.sum()
    if total == 0:
        return s * 0 + 1.0 / len(s)
    s = s / total
    # Iterative clipping
    for _ in range(50):
        over  = s > w_max
        under = s < w_min
        if not over.any() and not under.any():
            break
        s[over]  = w_max
        s[under] = w_min
        remainder = 1.0 - s[over].sum() - s[under].sum()
        mid = ~over & ~under
        if mid.any() and s[mid].sum() > 0:
            s[mid] = s[mid] / s[mid].sum() * remainder
    return s / s.sum() if s.sum() > 0 else s

def run_backtest(factor_by_date, calc_dates, Pxs_df,
                 use_vol_filter=False, use_mom_12m1=False, use_mom_idio=False,
                 resid_pivot=None, vol_pivot=None, mktcap_floor=None,
                 sector_cap=None, top_n=20, mom_weight=1.0,
                 prefilt_pct=1.0, conc_factor=1.0):
    """Baseline NAV backtest — pure quality factor, equal-weight."""
    nav          = 1.0
    nav_series   = {}
    portfolio    = []
    port_records = {}
    pxs_columns  = set(Pxs_df.columns)

    for i, rebal_date in enumerate(calc_dates):
        next_date = calc_dates[i + 1] if i + 1 < len(calc_dates) else Pxs_df.index.max()

        if rebal_date in factor_by_date:
            fdf = factor_by_date[rebal_date].copy()
            if mktcap_floor is not None and 'mkt_cap' in fdf.columns:
                fdf = fdf[fdf['mkt_cap'].fillna(0) >= mktcap_floor]
            fdf = fdf.loc[[t for t in fdf.index if t in pxs_columns]]
            if use_vol_filter and len(fdf) > top_n:
                surviving = apply_vol_filter(fdf.index, rebal_date, Pxs_df)
                fdf       = fdf.loc[fdf.index.intersection(surviving)]
            if prefilt_pct < 1.0 and len(fdf) > top_n:
                n_keep = max(top_n, int(np.ceil(len(fdf) * prefilt_pct)))
                fdf    = fdf.nlargest(n_keep, 'factor')
            if len(fdf) < top_n:
                portfolio = []
            else:
                rank_col = 'factor'
                ranked   = fdf.sort_values(rank_col, ascending=False)
                if sector_cap is not None:
                    top = select_with_sector_cap(ranked, sector_cap, top_n)
                else:
                    top = ranked.head(top_n)
                portfolio = [t for t in top.index if t in pxs_columns]
                n_port    = len(portfolio)
                n_top_h   = int(np.ceil(n_port / 2))
                n_bot_h   = n_port - n_top_h
                if conc_factor == 1.0 or n_bot_h == 0:
                    weights = {t: 1.0 / n_port for t in portfolio}
                else:
                    top_alloc = conc_factor / (conc_factor + 1.0)
                    bot_alloc = 1.0 / (conc_factor + 1.0)
                    weights   = {}
                    for j, t in enumerate(portfolio):
                        weights[t] = (top_alloc / n_top_h if j < n_top_h
                                      else bot_alloc / n_bot_h)
                port_records[rebal_date] = (
                    list(top.index) + [None] * (top_n - len(top.index))
                )[:top_n]

        if not portfolio:
            period_dates = Pxs_df.index[
                (Pxs_df.index >= rebal_date) & (Pxs_df.index <= next_date)]
            for d in period_dates:
                nav_series[d] = nav
            continue

        period_dates = Pxs_df.index[
            (Pxs_df.index >= rebal_date) & (Pxs_df.index <= next_date)]
        if len(period_dates) < 2:
            nav_series[rebal_date] = nav
            continue

        px_start   = Pxs_df.loc[period_dates[0],  portfolio]
        px_end     = Pxs_df.loc[period_dates[-1], portfolio]
        stk_rets   = (px_end / px_start - 1).fillna(0)
        w_series   = pd.Series(weights).reindex(portfolio).fillna(0)
        period_ret = (stk_rets * w_series).sum()
        nav       *= (1 + period_ret)

        px_period  = Pxs_df.loc[period_dates, portfolio]
        stk_daily  = px_period.div(px_start, axis=1) - 1
        port_cum   = stk_daily.mul(w_series, axis=1).sum(axis=1)
        period_nav = nav / (1 + period_ret) * (1 + port_cum)
        for d, v in period_nav.items():
            nav_series[d] = v

    nav_s = pd.Series(nav_series).sort_index()
    if MB_START_DATE not in nav_s.index and not nav_s.empty:
        nav_s[MB_START_DATE] = 1.0
        nav_s = nav_s.sort_index()

    port_df = pd.DataFrame.from_dict(
        port_records, orient='index',
        columns=[f'Stock{i+1}' for i in range(top_n)]
    )
    return nav_s, port_df

def _ics_compute_mom_12m1(universe, calc_dates, Pxs_df):
    """Compute 12M-1M momentum: z-scored return from t-252 to t-21."""
    print("  Computing 12M1 momentum scores...")
    all_px_dates   = Pxs_df.index
    valid_universe = [t for t in universe if t in Pxs_df.columns]
    results = {}
    for dt in calc_dates:
        past = all_px_dates[all_px_dates < dt]
        if len(past) < ICS_MOM_LONG + 1:
            continue
        date_start = past[-(ICS_MOM_LONG + 1)]
        date_end   = past[-(ICS_MOM_SKIP + 1)]
        px_start = Pxs_df.loc[date_start, valid_universe]
        px_end   = Pxs_df.loc[date_end,   valid_universe]
        mom = ((px_end - px_start) / px_start.replace(0, np.nan)).dropna()
        if len(mom) < ICS_MIN_STOCKS:
            continue
        results[dt] = _ics_zscore(mom)
    out = pd.DataFrame(results).T
    out.index.name = 'date'
    print(f"  12M1 momentum scores: {len(out)} dates")
    return out.reindex(columns=valid_universe)

def _mb_compute_idio_mom_scores(resid_df, calc_dates):
    """Fallback: compute idio momentum from residuals if kernel version unavailable."""
    if resid_df.empty:
        return pd.DataFrame(index=calc_dates)
    return resid_df.rolling(ICS_MOM_LONG, min_periods=ICS_MOM_LONG//2).sum().reindex(calc_dates).ffill()

def _mb_load_ou(universe, model_version):
    """Load O-U scores directly from DB — fallback if kernel version unavailable."""
    tbl = ICS_OU_TBL.get(model_version, 'v2_ou_reversion_df')
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(f"SELECT date, ticker, ou_score FROM {tbl} ORDER BY date", conn)
        df['date'] = pd.to_datetime(df['date'])
        ou = df.pivot_table(index='date', columns='ticker', values='ou_score', aggfunc='last')
        return ou.reindex(columns=universe).apply(_ics_zscore, axis=1)
    except Exception as e:
        print(f"  WARNING: O-U load failed ({e}) — skipping")
        return pd.DataFrame()


_MIN_HISTORY_DAYS  = 50    # minimum trading days of real price history for inclusion
_STALE_RUN_LIMIT   = 3     # consecutive unchanged closes = stale/backfilled price

def _get_active_universe(universe, Pxs_df, dt,
                          min_days=_MIN_HISTORY_DAYS,
                          stale_run=_STALE_RUN_LIMIT):
    """
    Return subset of universe with genuine price history up to dt.
    Filters out:
      1. Stocks not yet in Pxs_df at dt
      2. Stocks with < min_days of non-stale trading days up to dt
    Stale detection: any run of >stale_run consecutive identical closes
    is treated as backfilled/pre-IPO data and replaced with NaN.
    """
    px_to_dt = Pxs_df.loc[:dt, [t for t in universe if t in Pxs_df.columns]]
    if px_to_dt.empty:
        return []

    active = []
    for tkr in px_to_dt.columns:
        s = px_to_dt[tkr].dropna()
        if s.empty:
            continue
        # Replace stale runs with NaN
        # A run of >stale_run identical values = backfilled
        runs = (s != s.shift()).cumsum()
        run_lengths = runs.map(runs.value_counts())
        s_clean = s.where(run_lengths <= stale_run)
        if s_clean.notna().sum() >= min_days:
            active.append(tkr)
    return active


def _cb_build_composite_scores(universe, calc_dates, Pxs_df, sectors_s,
                                weights_by_year, regime_s, volumeTrd_df,
                                model_version, exclude_factors=None,
                                weights_by_date=None,
                                mr_scores_by_date=None):
    _load_quality   = _ics_load_quality
    _load_idio_mom  = _ics_load_idio_mom
    _calc_idio_mom  = _ics_compute_idio_mom_scores
    _load_value     = _ics_load_value
    _load_ou        = _mb_load_ou
    _load_mom12m1   = _ics_compute_mom_12m1
    exclude_factors = exclude_factors or ['OU']
    first_w = next(iter(weights_by_year.values()))
    active  = [f for f in first_w.columns if f not in exclude_factors]

    # Pre-sort weights_by_date keys for fast lookup
    _wbd_dates = sorted(weights_by_date.keys()) if weights_by_date else []

    print(f"  Active factors: {active}")
    if weights_by_date:
        print(f"  Using point-in-time weights_by_date ({len(_wbd_dates)} dates)")
    print(f"  Point-in-time weights (sample -- year 2022)"
          + (f"  [QUALITY_FLOOR={QUALITY_FLOOR}]:" if QUALITY_FLOOR > 0 else ":"))
    if 2022 in weights_by_year:
        w_sample = weights_by_year[2022][active].copy()
        for r in w_sample.index:
            w_sample.loc[r] = _ics_bounded_normalize(
                w_sample.loc[r], w_min=ICS_WEIGHT_MIN, w_max=ICS_WEIGHT_MAX)
        # Apply quality floor to display
        if QUALITY_FLOOR > 0:
            _mom_factors = [f for f in ('Idio_Mom', 'Mom_12M1') if f in w_sample.columns]
            if 'Quality' in w_sample.columns and _mom_factors:
                for r in w_sample.index:
                    _wm  = sum(w_sample.loc[r, f] for f in _mom_factors)
                    _wq  = w_sample.loc[r, 'Quality']
                    _mqs = _wm + _wq
                    if _mqs > 0 and (_wq / _mqs) < QUALITY_FLOOR:
                        w_sample.loc[r, 'Quality'] = _mqs * QUALITY_FLOOR
                        _rem = _mqs * (1.0 - QUALITY_FLOOR)
                        for f in _mom_factors:
                            w_sample.loc[r, f] = (_rem * (w_sample.loc[r, f] / _wm)
                                                   if _wm > 0 else _rem / len(_mom_factors))
        print(w_sample.round(4))
    print()

    score_dfs = {}
    if 'Quality' in active:
        print("  Loading quality scores...", end=' ')
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter('always')
            score_dfs['Quality'] = _load_quality(universe, calc_dates, Pxs_df, sectors_s)
        _n = score_dfs['Quality'].notna().any(axis=1).sum() if not score_dfs['Quality'].empty else 0
        print(f"{_n} dates" if _n else "WARNING: no data returned")
    if 'Idio_Mom' in active:
        print("  Loading idio momentum...", end=' ')
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter('always')
            resid_df = _load_idio_mom(universe, model_version)
            score_dfs['Idio_Mom'] = _calc_idio_mom(resid_df, calc_dates)
        _n = score_dfs['Idio_Mom'].notna().any(axis=1).sum() if not score_dfs['Idio_Mom'].empty else 0
        print(f"{_n} dates" if _n else "WARNING: no data returned")
    if 'Value' in active:
        print("  Loading value scores...", end=' ')
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter('always')
            score_dfs['Value'] = _load_value(universe, calc_dates, sectors_s, model_version)
        _n = score_dfs['Value'].notna().any(axis=1).sum() if not score_dfs['Value'].empty else 0
        print(f"{_n} dates" if _n else "WARNING: no data returned")
    if 'Mom_12M1' in active:
        print("  Computing 12M1 momentum...", end=' ')
        score_dfs['Mom_12M1'] = _load_mom12m1(universe, calc_dates, Pxs_df)
        _n = len(score_dfs['Mom_12M1'])
        print(f"{_n} dates")
    if 'OU' in active:
        print("  Loading O-U scores...", end=' ')
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter('always')
            ou = _load_ou(universe, model_version)
        if not ou.empty:
            all_d = calc_dates.union(ou.index).sort_values()
            score_dfs['OU'] = ou.reindex(all_d).ffill().reindex(calc_dates)
            print(f"{ou.shape[0]} dates")
        else:
            print("WARNING: no data returned")

    composite_by_date = {}
    n = len(calc_dates)
    _diag_dt = calc_dates[0] if len(calc_dates) > 0 else None
    for i, dt in enumerate(calc_dates):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Building composite [{i+1}/{n}] {dt.date()}", end='\r')

        # ── Per-date universe: only stocks with genuine price history up to dt ─
        active_u = _get_active_universe(universe, Pxs_df, dt)

        if dt == _diag_dt:
            print(f"\n  [DIAG {dt.date()}] active_u={len(active_u)}, today_ts={Pxs_df.index[-1].date()}")
            for fname in active:
                if fname in score_dfs and score_dfs[fname] is not None:
                    sdf = score_dfs[fname]
                    if dt in sdf.index:
                        vals = sdf.loc[dt].reindex(active_u).dropna()
                        print(f"    {fname}: n={len(vals)}, mean={vals.mean():.4f}, std={vals.std():.4f}", flush=True)

        # ── Weight lookup: weights_by_date (preferred) or weights_by_year ────
        if _wbd_dates:
            # Find latest weights_by_date entry strictly before or on dt
            past = [d for d in _wbd_dates if d <= dt]
            if past:
                w_t = weights_by_date[past[-1]]
                # w_t is already a Series(factor -> weight) for current regime
                w_t = w_t.reindex(active).fillna(1.0 / len(active))
                w_t = w_t / w_t.sum()
                if dt == _diag_dt:
                    print(f"    w_t from weights_by_date[{past[-1].date()}]: {w_t.round(4).to_dict()}", flush=True)
            else:
                # Before any weights_by_date entry — fall back to equal weight
                w_t = pd.Series(1.0 / len(active), index=active)
                if dt == _diag_dt:
                    print(f"    w_t: equal weight fallback", flush=True)
        else:
            # Fall back to annual snapshot
            yr = dt.year
            if yr in weights_by_year:
                w_df = weights_by_year[yr]
            else:
                avail   = sorted(weights_by_year.keys())
                nearest = min(avail, key=lambda y: abs(y - yr))
                w_df    = weights_by_year[nearest]

            w_df_active = w_df[active].copy()
            rs = w_df_active.sum(axis=1)
            for ri in w_df_active.index:
                w_df_active.loc[ri] = (w_df_active.loc[ri] / rs[ri]
                                       if rs[ri] > 0 else 1.0 / len(active))

            reg_cands = regime_s[regime_s.index <= dt]
            if reg_cands.empty:
                continue
            r = float(reg_cands.iloc[-1])
            if r not in w_df_active.index:
                r = min(w_df_active.index, key=lambda x: abs(x - r))
            w_t = w_df_active.loc[r]

        parts = []
        # ── Quality floor: redistribute within {Idio_Mom, Mom_12M1, Quality} ──
        if QUALITY_FLOOR > 0:
            _mom_factors = [f for f in ('Idio_Mom', 'Mom_12M1') if f in w_t.index]
            _qual_factor = 'Quality'
            if _qual_factor in w_t.index and _mom_factors:
                _w_mom_sum  = sum(w_t[f] for f in _mom_factors)
                _w_qual     = w_t[_qual_factor]
                _w_mqsum    = _w_mom_sum + _w_qual   # Value untouched
                _qual_share = _w_qual / _w_mqsum if _w_mqsum > 0 else 0.0
                if _qual_share < QUALITY_FLOOR:
                    # Floor binds — redistribute
                    w_t = w_t.copy()
                    _qual_new  = _w_mqsum * QUALITY_FLOOR
                    _mom_rem   = _w_mqsum * (1.0 - QUALITY_FLOOR)
                    w_t[_qual_factor] = _qual_new
                    # Split remainder in original Idio_Mom : Mom_12M1 ratio
                    if _w_mom_sum > 0:
                        for f in _mom_factors:
                            w_t[f] = _mom_rem * (w_t[f] / _w_mom_sum)
                    else:
                        for f in _mom_factors:
                            w_t[f] = _mom_rem / len(_mom_factors)
        # MR penalty: load scores for this date if MR_K > 0
        _mr_pen = None
        if MR_K > 0 and mr_scores_by_date is not None:
            _mr_dt = max((d for d in mr_scores_by_date if d <= dt),
                         default=None)
            if _mr_dt is not None:
                _mr_pen = mr_scores_by_date[_mr_dt]  # Series {ticker: score}

        for fname in active:
            if fname not in score_dfs or score_dfs[fname] is None:
                continue
            sdf = score_dfs[fname]
            if dt not in sdf.index:
                continue
            col = sdf.loc[dt].reindex(active_u)   # per-date universe only
            if col.isna().all():
                continue
            # Re-z-score using active_u cross-section — ensures all factor scores
            # are on the same scale regardless of how they were originally computed
            col = _ics_zscore(col.replace(0, np.nan).dropna())
            if col.empty:
                continue
            col = col.reindex(active_u).fillna(0)
            # Apply MR penalty to momentum factors only
            if MR_K > 0 and _mr_pen is not None and fname in ('Idio_Mom', 'Mom_12M1'):
                pen = _mr_pen.reindex(col.index).fillna(0.0)
                divisor = np.exp(MR_K * pen)
                col = col / divisor
            parts.append(col * w_t[fname])

        if not parts:
            continue

        composite = pd.concat(parts, axis=1).sum(axis=1)
        composite = _ics_zscore(composite.replace(0, np.nan).dropna())
        if len(composite) >= ICS_MIN_STOCKS:
            composite_by_date[dt] = composite
            if dt == _diag_dt:
                print(f"\n  [COMPOSITE DIAG {dt.date()}] top-5, today_ts={Pxs_df.index[-1].date()}")
                print(composite.nlargest(5).round(4).to_string(), flush=True)

    print(f"\n  Composite scores built: {len(composite_by_date)} dates")
    return composite_by_date, score_dfs

# Hedge engine functions embedded below (see HEDGE ENGINE section)

class _SuppressOutput:
    """
    Context manager to suppress stdout in both scripts and Jupyter notebooks.
    Tries fd-level redirect first; falls back to sys.stdout swap if fileno()
    is unavailable (Jupyter kernel streams don't expose a real fd).
    """
    def __enter__(self):
        self._devnull = open(os.devnull, 'w')
        try:
            self._stdout_fd = sys.stdout.fileno()
            self._old_fd    = os.dup(self._stdout_fd)
            os.dup2(self._devnull.fileno(), self._stdout_fd)
            self._use_fd = True
        except Exception:
            # Jupyter: no real fd -- fall back to replacing sys.stdout
            self._orig_stdout = sys.stdout
            sys.stdout = self._devnull
            self._use_fd = False
        return self

    def __exit__(self, *args):
        if self._use_fd:
            os.dup2(self._old_fd, self._stdout_fd)
            os.close(self._old_fd)
        else:
            sys.stdout = self._orig_stdout
        self._devnull.close()



# ── Covariance helpers (inlined from mvo_diagnostics.py) ─────────────────────

def _mvo_ewma_cov(ret_df, hl):
    """EWMA covariance matrix (T x N returns → N x N). Returns np.ndarray."""
    T     = len(ret_df)
    decay = np.log(2) / hl
    w     = np.exp(-decay * np.arange(T - 1, -1, -1))
    w    /= w.sum()
    v     = ret_df.values
    mu    = (w[:, None] * v).sum(0)
    d     = v - mu
    return (d * w[:, None]).T @ d


def _mvo_ewma_vol(ret_df, hl):
    """EWMA volatility per stock. Returns pd.Series."""
    T     = len(ret_df)
    decay = np.log(2) / hl
    w     = np.exp(-decay * np.arange(T - 1, -1, -1))
    w    /= w.sum()
    v     = ret_df.values
    mu    = (w[:, None] * v).sum(0)
    var   = (w[:, None] * (v - mu) ** 2).sum(0)
    return pd.Series(np.sqrt(var), index=ret_df.columns)


def _mvo_ledoit_wolf(ret_matrix):
    """Ledoit-Wolf shrinkage. Returns (Sigma_lw, rho_bar, shrinkage_coef)."""
    from sklearn.covariance import LedoitWolf
    T, N        = ret_matrix.shape
    lw          = LedoitWolf(assume_centered=False)
    lw.fit(ret_matrix)
    Sigma_lw    = lw.covariance_
    shrink_coef = float(lw.shrinkage_)
    std         = np.sqrt(np.diag(Sigma_lw))
    std_mat     = np.outer(std, std)
    with np.errstate(invalid='ignore', divide='ignore'):
        Corr = np.where(std_mat > 0, Sigma_lw / std_mat, 0.0)
    n_off   = N * (N - 1)
    rho_bar = (Corr.sum() - np.trace(Corr)) / n_off if n_off > 0 else 0.0
    return Sigma_lw, rho_bar, shrink_coef


def _mvo_pca_cov(ret_matrix, var_threshold=MVO_PCA_VAR_THRESH):
    """PCA covariance. Returns (Sigma_pca, n_components, var_explained, eigvals)."""
    Sigma            = _mvo_ewma_cov(pd.DataFrame(ret_matrix), MVO_EWMA_HL)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    idx              = np.argsort(eigvals)[::-1]
    eigvals          = np.maximum(eigvals[idx], 0.0)
    eigvecs          = eigvecs[:, idx]
    total_var        = eigvals.sum()
    cum_var          = np.cumsum(eigvals) / total_var if total_var > 0 else np.zeros_like(eigvals)
    n_comp           = max(1, int(np.searchsorted(cum_var, var_threshold) + 1))
    var_expl         = float(cum_var[n_comp - 1])
    Vk               = eigvecs[:, :n_comp]
    Lk               = np.diag(eigvals[:n_comp])
    Sigma_pca        = Vk @ Lk @ Vk.T
    resid_var        = np.diag(Sigma) - np.diag(Sigma_pca)
    Sigma_pca       += np.diag(np.maximum(resid_var, 0.0))
    return Sigma_pca, n_comp, var_expl, eigvals


# ===============================================================================
# SELF-CONTAINED FLOOR/CAP (does not depend on mvo_diagnostics.py version)
# ===============================================================================

def _mb_floor_then_cap(w, min_weight, max_weight, max_iter=20):
    """
    Post-solve weight adjustment:

    Step 1 -- Floor:
        Raise non-zero weights to min_weight, renormalize.
    Step 2 -- Cap (iterative):
        Reduce weights above max_weight to max_weight,
        redistribute excess proportionally to remaining stocks.
    """
    w = w.copy().clip(lower=0)
    if w.sum() == 0:
        return w

    # Step 1: raise sub-floor weights to min_weight then renorm
    w = w.clip(lower=min_weight)
    if w.sum() > 0:
        w = w / w.sum()

    # Step 2: iterative cap
    for _ in range(max_iter):
        over = w > max_weight + 1e-9
        if not over.any():
            break
        excess   = (w[over] - max_weight).sum()
        w[over]  = max_weight
        under    = ~over
        if w[under].sum() > 1e-12:
            w[under] += excess * (w[under] / w[under].sum())
        else:
            w += excess / len(w)

    if w.sum() > 0:
        w = w / w.sum()
    return w



# ===============================================================================
# COVARIANCE CACHE
# ===============================================================================

def _mb_get_cached_dates(force_rebuild):
    """Return set of dates already cached, or empty set if force_rebuild."""
    if force_rebuild:
        return set()
    try:
        with ENGINE.connect() as conn:
            exists = conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = :t
                )
            """), {"t": MB_COV_CACHE_TBL}).scalar()
        if not exists:
            return set()
        with ENGINE.connect() as conn:
            rows = conn.execute(text(
                f"SELECT DISTINCT date FROM {MB_COV_CACHE_TBL}"
            )).fetchall()
        return {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        return set()


def _mb_build_cov_matrices(dt, candidates, Pxs_df, sectors_s,
                             volumeTrd_df, model_version,
                             pca_var_threshold, X_df_cached=None):
    """
    Build all four covariance matrices for the candidate universe on date dt.
    Returns (Sigma_emp, Sigma_lw, Sigma_factor, Sigma_pca, Sigma_ens).
    """
    # Return matrix — strictly up to dt (no lookahead)
    ret_df = (Pxs_df.loc[:dt, candidates].pct_change()
              .dropna(how='all')
              .iloc[-MVO_LOOKBACK:]
              .dropna(axis=1, how='any'))
    valid = ret_df.columns.tolist()

    # Empirical
    Sigma_emp = _mvo_ewma_cov(ret_df, MVO_EWMA_HL)

    # Ledoit-Wolf
    Sigma_lw, _, _ = _mvo_ledoit_wolf(ret_df.values)

    # Factor-driven (self-contained -- no _mvo_factor_cov dependency)
    try:
        F_mat, factor_names_f, sec_cols_f = _mb_build_F(model_version)
        factor_names_f2, sec_cols_f2 = _mb_get_factor_names(model_version)
        X_df = pd.DataFrame(0.0, index=valid, columns=factor_names_f)
        # Use most recent X snapshot if available, else build on the fly
        if X_df_cached is not None:
            X_df = X_df_cached.reindex(index=valid, columns=factor_names_f).fillna(0.0)
        else:
            pxs_slice = Pxs_df.loc[:dt]
            X_built   = _mb_build_X(dt, valid, factor_names_f,
                                     sec_cols_f, pxs_slice, sectors_s,
                                     volumeTrd_df, model_version)
            if X_built is not None:
                X_df = X_built.reindex(index=valid, columns=factor_names_f).fillna(0.0)
        X          = X_df.values
        Sigma_factor_raw = X @ F_mat @ X.T
        emp_mean_var     = np.diag(Sigma_emp).mean()
        factor_mean_var  = np.diag(Sigma_factor_raw).mean()
        if factor_mean_var > 1e-12:
            Sigma_factor = Sigma_factor_raw * (emp_mean_var / factor_mean_var)
        else:
            Sigma_factor = Sigma_emp.copy()
    except Exception as e:
        warnings.warn(f"  Factor cov failed ({e}) -- using empirical")
        Sigma_factor = Sigma_emp.copy()

    # PCA
    Sigma_pca, _, _, _ = _mvo_pca_cov(ret_df.values,
                                        var_threshold=pca_var_threshold)

    # Ensemble
    Sigma_ens = (Sigma_emp + Sigma_lw + Sigma_factor + Sigma_pca) / 4.0

    return valid, Sigma_emp, Sigma_lw, Sigma_factor, Sigma_pca, Sigma_ens


# ===============================================================================
# SELF-CONTAINED X-MATRIX BUILDERS (no portfolio_risk_decomp dependency)
# ===============================================================================




def _mb_zscore(s):
    mu, sd = s.mean(), s.std()
    return s * 0.0 if (sd == 0 or np.isnan(sd)) else (s - mu) / sd


def _mb_get_factor_names(model_version):
    scalar_names = list(_MB_SCALAR_TBLS[model_version].keys())
    try:
        with ENGINE.connect() as conn:
            sdf = pd.read_sql(
                f"SELECT * FROM {_MB_SEC_TBL[model_version]} "
                f"ORDER BY date DESC LIMIT 1", conn
            )
        sec_cols = sorted([c for c in sdf.columns
                           if c not in _MB_LAMBDA_META
                           and pd.api.types.is_float_dtype(sdf[c])])
    except Exception as e:
        warnings.warn(f"Could not load sector columns: {e}")
        sec_cols = []
    head = ['Beta', 'Size']
    tail = ['Quality', 'SI', 'GK_Vol', 'Idio_Mom', 'Value', 'OU']
    factor_names = (
        [f for f in head if f in scalar_names]
        + list(MACRO_COLS) + sec_cols
        + [f for f in tail if f in scalar_names]
    )
    return factor_names, sec_cols


def _mb_ewma_cov_f(df, hl):
    """EWMA covariance for building F matrix."""
    decay = np.log(2) / hl
    T     = len(df)
    w     = np.exp(-decay * np.arange(T - 1, -1, -1))
    w    /= w.sum()
    v     = df.values
    mu    = (w[:, None] * v).sum(0)
    d     = v - mu
    return (d * w[:, None]).T @ d


def _mb_build_F(model_version):
    """Self-contained F matrix builder."""
    tbls   = _MB_SCALAR_TBLS[model_version]
    frames = []
    for fname, tbl in tbls.items():
        try:
            with ENGINE.connect() as conn:
                df = pd.read_sql(
                    f"SELECT * FROM {tbl} ORDER BY date DESC LIMIT {_MB_F_LOOKBACK}",
                    conn
                )
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            num = df.drop(columns=[c for c in _MB_LAMBDA_META if c in df.columns],
                          errors='ignore').select_dtypes(include=np.number)
            col = num.iloc[:, 0].rename(fname)
            frames.append(col)
        except Exception as e:
            warnings.warn(f"Lambda table {tbl} failed: {e}")

    macro_tbl_map = {'v1': 'factor_lambdas_macro', 'v2': 'v2_factor_lambdas_macro'}
    try:
        macro_tbl = macro_tbl_map[model_version]
        with ENGINE.connect() as conn:
            mdf = pd.read_sql(
                f"SELECT * FROM {macro_tbl} ORDER BY date DESC LIMIT {_MB_F_LOOKBACK}",
                conn
            )
        mdf['date'] = pd.to_datetime(mdf['date'])
        mdf = mdf.set_index('date').sort_index()
        mnum = mdf.drop(columns=[c for c in _MB_LAMBDA_META if c in mdf.columns],
                        errors='ignore').select_dtypes(include=np.number)
        for mc in MACRO_COLS:
            if mc in mnum.columns:
                frames.append(mnum[mc].rename(mc))
    except Exception as e:
        warnings.warn(f"Macro lambda failed: {e}")

    try:
        with ENGINE.connect() as conn:
            sdf = pd.read_sql(
                f"SELECT * FROM {_MB_SEC_TBL[model_version]} "
                f"ORDER BY date DESC LIMIT {_MB_F_LOOKBACK}", conn
            )
        sdf['date'] = pd.to_datetime(sdf['date'])
        sdf = sdf.set_index('date').sort_index()
        sec_cols = sorted([c for c in sdf.columns
                           if c not in _MB_LAMBDA_META
                           and pd.api.types.is_float_dtype(sdf[c])])
        for sc in sec_cols:
            frames.append(sdf[sc].rename(sc))
    except Exception as e:
        warnings.warn(f"Sector lambda failed: {e}")
        sec_cols = []

    combined = pd.concat(frames, axis=1).dropna()
    factor_names, sec_cols = _mb_get_factor_names(model_version)
    ordered  = [c for c in factor_names if c in combined.columns]
    combined = combined[ordered]
    F        = _mb_ewma_cov_f(combined, _MB_F_EWMA_HL)
    return F, ordered, sec_cols


def _mb_sector_dummies(universe, sectors_s, sec_cols):
    sect = sectors_s.reindex(universe).fillna('Unknown')
    data = pd.DataFrame(0.0, index=universe, columns=sec_cols)
    for ticker in universe:
        s = sect[ticker]
        if s in sec_cols:
            data.loc[ticker, s] = 1.0
    return data


def _mb_build_X(dt, universe, factor_names, sec_cols,
                 Pxs_df, sectors_s, volumeTrd_df, model_version):
    """Self-contained X matrix builder -- uses kernel factor model functions."""
    pxs_to_dt  = Pxs_df.loc[:dt]
    if len(pxs_to_dt) < BETA_WINDOW // 2:
        return None
    calc_dates = pd.DatetimeIndex([dt])

    # Resolve kernel functions
    # These functions come from factor_model_step1 (run before mvo_backtest)
    try:
        beta_df = calc_rolling_betas(pxs_to_dt, universe, calc_dates)
        beta_s  = _mb_zscore(beta_df.iloc[-1].reindex(universe)).rename('Beta')

        size_df = load_dynamic_size(universe, pxs_to_dt, calc_dates)
        size_s  = _mb_zscore(np.log(size_df.iloc[-1].reindex(universe).clip(lower=1e-6))).rename('Size')

        macro_betas  = calc_macro_betas(pxs_to_dt, universe, calc_dates)
        macro_series = {}
        for mc in MACRO_COLS:
            if mc in macro_betas and not macro_betas[mc].empty:
                macro_series[mc] = _mb_zscore(macro_betas[mc].iloc[-1].reindex(universe)).rename(mc)
            else:
                macro_series[mc] = pd.Series(0.0, index=universe, name=mc)

        dummies = _mb_sector_dummies(universe, sectors_s, sec_cols)

        quality_df = load_quality_scores(universe, calc_dates, pxs_to_dt, sectors_s)
        quality_s  = _mb_zscore(quality_df.reindex(calc_dates).iloc[-1].reindex(universe)).rename('Quality')

        recent_dates = pxs_to_dt.index[-60:]
        si_full = load_si_composite(universe, recent_dates)
        si_s    = _mb_zscore(si_full.reindex(recent_dates).ffill().iloc[-1].reindex(universe)).rename('SI')

        open_df, high_df, low_df = load_ohlc_tables(universe)
        vol_df = calc_vol_factor(pxs_to_dt, universe, calc_dates,
                                 open_df=open_df, high_df=high_df, low_df=low_df)
        vol_s  = _mb_zscore(vol_df.iloc[-1].reindex(universe)).rename('GK_Vol')

        try:
            with ENGINE.connect() as conn:
                res_mom = pd.read_sql(
                    f"SELECT * FROM {_MB_MOM_RESID[model_version]} ORDER BY date", conn)
            res_mom['date'] = pd.to_datetime(res_mom['date'])
            if 'ticker' in res_mom.columns and 'resid' in res_mom.columns:
                res_mom = res_mom.pivot_table(
                    index='date', columns='ticker', values='resid', aggfunc='last'
                ).reindex(columns=universe)
            else:
                res_mom = res_mom.set_index('date').reindex(columns=universe)
            res_mom = res_mom[res_mom.index <= dt]
            if volumeTrd_df is not None:
                mom_df = calc_idio_momentum_volscaled(res_mom, volumeTrd_df, calc_dates)
            else:
                mom_df = calc_idio_momentum(res_mom, calc_dates)
            mom_s = _mb_zscore(mom_df.iloc[-1].reindex(universe)).rename('Idio_Mom')
        except Exception as e:
            warnings.warn(f"  Idio momentum failed ({e}) -- filling 0")
            mom_s = pd.Series(0.0, index=universe, name='Idio_Mom')

        value_df = load_value_scores(universe, calc_dates, sectors_s)
        value_s  = _mb_zscore(value_df.reindex(calc_dates).iloc[-1].reindex(universe)).rename('Value')

        try:
            with ENGINE.connect() as conn:
                ou_raw = pd.read_sql(
                    f"SELECT date, ticker, ou_score FROM {_MB_OU_CACHE[model_version]} "
                    f"WHERE date <= '{dt.date()}' ORDER BY date DESC "
                    f"LIMIT {len(universe) * 5}", conn
                )
            ou_raw['date'] = pd.to_datetime(ou_raw['date'])
            ou_pivot = ou_raw.pivot_table(
                index='date', columns='ticker',
                values='ou_score', aggfunc='last'
            ).reindex(columns=universe)
            ou_s = _mb_zscore(ou_pivot.iloc[-1].reindex(universe)).rename('OU')
        except Exception as e:
            warnings.warn(f"  O-U load failed ({e}) -- filling 0")
            ou_s = pd.Series(0.0, index=universe, name='OU')

        scalar_map = {
            'Beta': beta_s, 'Size': size_s, 'Quality': quality_s,
            'SI': si_s, 'GK_Vol': vol_s, 'Idio_Mom': mom_s,
            'Value': value_s, 'OU': ou_s,
        }
        cols = []
        for fname in factor_names:
            if fname in scalar_map:
                cols.append(scalar_map[fname])
            elif fname in MACRO_COLS:
                cols.append(macro_series.get(
                    fname, pd.Series(0.0, index=universe, name=fname)
                ))
            elif fname in sec_cols:
                cols.append(dummies[fname].rename(fname))
            else:
                cols.append(pd.Series(0.0, index=universe, name=fname))

        return pd.concat(cols, axis=1).reindex(universe).fillna(0.0)

    except Exception as e:
        warnings.warn(f"  X build failed for {dt.date()}: {e}")
        return None


def _mb_month_end_dates(date_index, latest_date):
    idx   = pd.DatetimeIndex(date_index)
    s     = pd.Series(idx, index=idx)
    mends = s.groupby([s.index.year, s.index.month]).last().values
    return pd.DatetimeIndex(mends).union([latest_date]).sort_values()


# ===============================================================================
# X SNAPSHOT CACHE (PostgreSQL-backed)
# ===============================================================================



def _mb_save_x_snapshot(dt, model_version, xdf):
    """Persist a single X snapshot to the DB cache table (upsert)."""
    rec = {
        'date'         : dt.strftime('%Y-%m-%d'),
        'model_version': model_version,
        'tickers'      : json.dumps(xdf.index.tolist()),
        'factors'      : json.dumps(xdf.columns.tolist()),
        'values'       : json.dumps(xdf.values.tolist()),
    }
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {MB_X_CACHE_TBL} (
                date          DATE         NOT NULL,
                model_version VARCHAR(10)  NOT NULL,
                tickers       TEXT         NOT NULL,
                factors       TEXT         NOT NULL,
                values        TEXT         NOT NULL,
                PRIMARY KEY (date, model_version)
            )
        """))
        conn.execute(text(f"""
            INSERT INTO {MB_X_CACHE_TBL} (date, model_version, tickers, factors, values)
            VALUES (:date, :model_version, :tickers, :factors, :values)
            ON CONFLICT (date, model_version) DO UPDATE
                SET tickers = EXCLUDED.tickers,
                    factors = EXCLUDED.factors,
                    values  = EXCLUDED.values
        """), rec)


def _mb_load_x_cache(model_version):
    """Load all cached X snapshots for a given model version. Returns dict {date: df}."""
    try:
        with ENGINE.connect() as conn:
            exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = :t
                )
            """), {"t": MB_X_CACHE_TBL}).scalar()
        if not exists:
            return {}
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT date, tickers, factors, values
                FROM {MB_X_CACHE_TBL}
                WHERE model_version = :mv
                ORDER BY date
            """), {"mv": model_version}).fetchall()
        cached = {}
        for row in rows:
            dt      = pd.Timestamp(row[0])
            tickers = json.loads(row[1])
            factors = json.loads(row[2])
            vals    = json.loads(row[3])
            cached[dt] = pd.DataFrame(vals, index=tickers, columns=factors)
        return cached
    except Exception as e:
        warnings.warn(f"  X cache load failed: {e}")
        return {}


def _mb_clear_x_cache(model_version):
    """Delete all cached X snapshots for a given model version."""
    try:
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                DELETE FROM {MB_X_CACHE_TBL}
                WHERE model_version = :mv
            """), {"mv": model_version})
        print(f"  X cache cleared for model_version={model_version}")
    except Exception:
        pass


# ===============================================================================
# X SNAPSHOT BUILDER
# ===============================================================================

def _mb_build_x_snapshots(rebal_dates, Pxs_df, sectors_s,
                            volumeTrd_df, model_version,
                            force_rebuild):
    """
    Build X exposure matrix snapshots with DB caching.

    Behavior:
    - force_rebuild=False (default): load cache, compute only missing dates,
      save new ones. On typical daily runs this means only today is computed.
    - force_rebuild=True: clear cache, rebuild all month-end dates from scratch.

    Returns dict {date: DataFrame(universe x factors)}.
    """
    extended_st_dt = Pxs_df.index[0]
    universe       = get_universe(Pxs_df, sectors_s, extended_st_dt)
    factor_names, sec_cols = _mb_get_factor_names(model_version)

    # All dates we need: month-ends from backtest start + latest date
    # Bound to MB_START_DATE -- no point building X before backtest begins
    pxs_idx_bounded = Pxs_df.index[Pxs_df.index >= MB_START_DATE - pd.Timedelta(days=90)]
    x_snap_dates = _mb_month_end_dates(pxs_idx_bounded, rebal_dates[-1])

    if force_rebuild:
        _mb_clear_x_cache(model_version)
        cached = {}
        print(f"  force_rebuild=True: rebuilding all {len(x_snap_dates)} snapshots...")
    else:
        cached = _mb_load_x_cache(model_version)
        print(f"  X cache loaded: {len(cached)} dates already cached")

    # Only compute dates not already in cache
    to_build = [d for d in x_snap_dates if d not in cached]
    print(f"  Dates to build: {len(to_build)}"
          f"  ({to_build[0].date() if to_build else 'none'}"
          f" -> {to_build[-1].date() if to_build else 'none'})")

    X_snapshots = dict(cached)
    n_built = 0
    for i, xdt in enumerate(to_build):
        print(f"  Building snapshot {i+1}/{len(to_build)}: {xdt.date()}...", end='\r')
        with _SuppressOutput():
            xdf = _mb_build_X(xdt, universe, factor_names, sec_cols,
                              Pxs_df, sectors_s, volumeTrd_df, model_version)
        if xdf is not None:
            xdf = xdf.fillna(0.0)
            X_snapshots[xdt] = xdf
            try:
                _mb_save_x_snapshot(xdt, model_version, xdf)
            except Exception as e:
                warnings.warn(f"  Failed to save X snapshot {xdt.date()}: {e}")
            n_built += 1

    if to_build:
        print(f"\n  Built and cached: {n_built}/{len(to_build)} new snapshots")
    print(f"  Total X snapshots available: {len(X_snapshots)}")
    return X_snapshots, universe, factor_names, sec_cols


# ===============================================================================
# DAILY PORTFOLIO CACHE (PostgreSQL-backed)
# ===============================================================================

def _make_make_params_hash(ic, max_weight, min_weight, zscore_cap, pca_var_threshold,
                      universe_mult, risk_aversion, top_n, conc_factor,
                      prefilt_pct, min_cov_matrices, model_version):
    """Stable 12-char hash of all portfolio construction parameters."""
    params = dict(
        ic=ic, maxw=max_weight, minw=min_weight, zc=zscore_cap,
        pca=pca_var_threshold, um=universe_mult, ra=risk_aversion,
        n=top_n, conc=conc_factor, pf=round(prefilt_pct, 4),
        mcm=min_cov_matrices, mv=model_version,
    )
    return hashlib.md5(
        json.dumps(params, sort_keys=True).encode()
    ).hexdigest()[:12]


def _ensure_daily_cache_table():
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {MB_DAILY_PORT_TBL} (
                date          DATE        NOT NULL,
                model_version VARCHAR(10) NOT NULL,
                params_hash   VARCHAR(16) NOT NULL,
                strategy      VARCHAR(16) NOT NULL,
                weights_json  TEXT        NOT NULL,
                PRIMARY KEY (date, model_version, params_hash, strategy)
            )
        """))


def _get_cached_portfolio_dates(params_hash, model_version):
    try:
        _ensure_daily_cache_table()
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT DISTINCT date FROM {MB_DAILY_PORT_TBL}
                WHERE params_hash = :ph AND model_version = :mv
                  AND strategy = 'alpha'
            """), {'ph': params_hash, 'mv': model_version}).fetchall()
        return {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        return set()


def _save_daily_portfolios(dt, model_version, params_hash,
                            w_alpha, w_mvo, w_hybrid, w_smart=None):
    for strategy, w in [('alpha', w_alpha), ('mvo', w_mvo),
                        ('hybrid', w_hybrid), ('smart', w_smart)]:
        if w is None or w.empty:
            continue
        wj = json.dumps({t: float(v) for t, v in w.items()})
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                INSERT INTO {MB_DAILY_PORT_TBL}
                    (date, model_version, params_hash, strategy, weights_json)
                VALUES (:dt, :mv, :ph, :st, :wj)
                ON CONFLICT (date, model_version, params_hash, strategy)
                DO UPDATE SET weights_json = EXCLUDED.weights_json
            """), {'dt': dt.strftime('%Y-%m-%d'), 'mv': model_version,
                   'ph': params_hash, 'st': strategy, 'wj': wj})


def load_daily_portfolios(params_hash, model_version='v2'):
    """
    Load all cached daily portfolios for given params_hash.
    Returns: {date: {'alpha': pd.Series, 'mvo': pd.Series, 'hybrid': pd.Series}}
    """
    _ensure_daily_cache_table()
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT date, strategy, weights_json
            FROM {MB_DAILY_PORT_TBL}
            WHERE params_hash = :ph AND model_version = :mv
            ORDER BY date
        """), {'ph': params_hash, 'mv': model_version}).fetchall()
    result = {}
    for row in rows:
        dt = pd.Timestamp(row[0])
        st = row[1]
        w  = pd.Series(json.loads(row[2]))
        result.setdefault(dt, {})[st] = w
    return result


# ===============================================================================
# MVO WEIGHT SOLVER -- exact replica of mvo_diagnostics logic
# ===============================================================================

def _mb_max_alpha_portfolio(Sigma, alpha, tickers, risk_aversion,
                              max_weight, min_weight):
    """
    Unconstrained MVO (w>=0, sum=1), then post-solve floor+cap.
    Identical to mvo_diagnostics._mvo_max_alpha_portfolio.
    """
    import cvxpy as cp
    try:
        N = len(tickers)
        w = cp.Variable(N)
        a = alpha.reindex(tickers).fillna(0.0).values
        prob = cp.Problem(
            cp.Maximize(a @ w - (risk_aversion / 2) * cp.quad_form(w, Sigma)),
            [cp.sum(w) == 1, w >= 0]
        )
        prob.solve(solver=cp.OSQP, verbose=False)
        if prob.status not in ['optimal', 'optimal_inaccurate']:
            prob.solve(solver=cp.SCS, verbose=False)
        if prob.status in ['optimal', 'optimal_inaccurate'] and w.value is not None:
            w_sol = pd.Series(np.maximum(w.value, 0.0), index=tickers)
            if w_sol.sum() > 0:
                w_sol = w_sol / w_sol.sum()
            return _mb_floor_then_cap(w_sol, min_weight, max_weight)
    except Exception as e:
        warnings.warn(f"  MVO solver failed: {e}")
    # Fallback: equal weight
    w_eq = pd.Series(1.0 / len(tickers), index=tickers)
    return _mb_floor_then_cap(w_eq, min_weight, max_weight)


def _advp_filter_and_replace(weights: pd.Series,
                              candidates: pd.Series,
                              dt: pd.Timestamp,
                              Pxs_df: pd.DataFrame,
                              volumeRaw_df: pd.DataFrame,
                              current_aum: float,
                              advp_cap: float,
                              min_weight: float,
                              max_weight: float,
                              top_n: int,
                              conc_factor: float = 1.0,
                              target_n: int = None,
                              full_universe: pd.Series = None) -> pd.Series:
    """
    Apply ADVP liquidity filter to a portfolio, replacing illiquid stocks
    with the next-best liquid candidates from the ranked candidates Series.

    target_n: target number of stocks after filtering. Defaults to top_n.
              For hybrid portfolios, pass len(weights) to preserve the
              larger union without truncating to top_n.
    """
    if target_n is None:
        target_n = top_n
    """
    Apply ADVP liquidity filter to a portfolio, replacing illiquid stocks
    with the next-best liquid candidates from the ranked candidates Series.

    Parameters
    ----------
    weights    : current portfolio weights (top_n stocks)
    candidates : composite scores for full candidate universe (ranked desc)
    current_aum: AUM at this date (AUM × running_nav)
    advp_cap   : fraction of ADV allowed per stock
    min_weight : minimum portfolio weight — stocks below this capacity are excluded
    top_n      : target number of stocks
    conc_factor: used to rebuild weights after replacement

    Returns
    -------
    Filtered and replaced weights Series, always attempting top_n stocks.
    """
    if volumeRaw_df is None or current_aum <= 0 or weights.empty:
        return weights, set()

    past_dates = [d for d in Pxs_df.index if d <= dt][-ADV_WINDOW:]
    pxs_cols   = set(Pxs_df.columns)

    def _is_liquid(tkr):
        if tkr not in pxs_cols or tkr not in volumeRaw_df.columns:
            return False  # no volume data — treat as illiquid (conservative)
        px_s  = Pxs_df.loc[past_dates, tkr].dropna()
        vol_s = volumeRaw_df.loc[past_dates, tkr].dropna()
        com   = px_s.index.intersection(vol_s.index)
        if len(com) < 3: return False
        dv = (px_s.reindex(com) * vol_s.reindex(com).replace({0: np.nan})).median()
        return (dv * advp_cap) / current_aum >= min_weight

    # Split current portfolio into liquid / illiquid
    liquid   = [t for t in weights.index if _is_liquid(t)]
    illiquid = [t for t in weights.index if not _is_liquid(t)]

    if not illiquid:
        return weights, set()   # nothing to do — (weights, affected_tickers)

    # Replace illiquid with next-best liquid candidates
    # First search pre-filtered candidates, then full_universe if still short
    replacements = []
    _already = set(liquid) | set(illiquid)

    def _fill_from(pool):
        for tkr in pool.sort_values(ascending=False).index:
            if len(liquid) + len(replacements) >= target_n:
                break
            if tkr in _already or tkr not in pxs_cols:
                continue
            if _is_liquid(tkr):
                replacements.append(tkr)
                _already.add(tkr)

    _fill_from(candidates)

    # If still short, search full universe beyond pre-filtered candidates
    if len(liquid) + len(replacements) < target_n and full_universe is not None:
        _extra = full_universe.drop(
            index=[t for t in full_universe.index if t in _already],
            errors='ignore')
        _fill_from(_extra)

    final = liquid + replacements
    n     = len(final)
    if n == 0:
        return weights, set(illiquid)   # fallback — nothing survived

    # Track affected tickers: excluded (illiquid, not replaced) + capped
    _excluded = set(illiquid) - set(replacements)

    # Compute ADVP caps for all stocks
    _advp_caps = {}
    for tkr in final:
        if tkr not in pxs_cols or tkr not in volumeRaw_df.columns:
            continue
        px_s  = Pxs_df.loc[past_dates, tkr].dropna()
        vol_s = volumeRaw_df.loc[past_dates, tkr].dropna()
        com   = px_s.index.intersection(vol_s.index)
        if len(com) < 3: continue
        dv    = (px_s.reindex(com) * vol_s.reindex(com).replace({0: np.nan})).median()
        cap_w = (dv * advp_cap) / current_aum
        if cap_w < max_weight:
            _advp_caps[tkr] = max(cap_w, min_weight)

    # Assign stocks to tiers (top half gets concentrated allocation)
    n_top  = int(np.ceil(n / 2))
    n_bot  = n - n_top
    top_tks = final[:n_top]
    bot_tks = final[n_top:]

    # Build weights tier by tier, respecting ADVP caps and min/max bounds
    # Each tier gets a budget: top gets top_a, bot gets bot_a
    if conc_factor == 1.0 or n_bot == 0:
        top_a = 1.0
        bot_a = 0.0
    else:
        top_a = conc_factor / (conc_factor + 1.0)
        bot_a = 1.0 / (conc_factor + 1.0)

    def _alloc_tier(tickers, budget):
        """Allocate budget across tickers respecting ADVP caps and max_weight."""
        if not tickers or budget <= 0:
            return pd.Series(dtype=float), 0.0
        w = pd.Series(budget / len(tickers), index=tickers)
        # Iteratively cap and redistribute within tier
        for _ in range(20):
            changed = False
            for tkr in list(w.index):
                cap = min(_advp_caps.get(tkr, max_weight), max_weight)
                if w[tkr] > cap + 1e-9:
                    w[tkr] = cap
                    changed = True
            # Renorm to budget
            if w.sum() > 1e-12:
                w = w / w.sum() * budget
            if not changed:
                break
        return w, budget - w.sum()

    w_top, excess_top = _alloc_tier(top_tks, top_a)
    w_bot, excess_bot = _alloc_tier(bot_tks, bot_a)

    # If top tier has excess (e.g. all stocks capped), try to absorb in bottom
    # and vice versa
    if excess_top > 1e-9 and not w_bot.empty:
        w_bot2, _ = _alloc_tier(list(w_bot.index), bot_a + excess_top)
        w_bot = w_bot2
    if excess_bot > 1e-9 and not w_top.empty:
        w_top2, _ = _alloc_tier(list(w_top.index), top_a + excess_bot)
        w_top = w_top2

    w_new = pd.concat([w_top, w_bot])
    if w_new.empty:
        return weights, _excluded

    # Final renorm to sum to 1.0
    if w_new.sum() > 0:
        w_new = w_new / w_new.sum()

    _capped = {t for t in final if t in _advp_caps and
               w_new.get(t, 0) <= _advp_caps[t] + 1e-9 and
               t not in _excluded}
    # Only flag as capped if the ADVP cap was actually binding
    _capped = {t for t in _advp_caps if t in w_new and
               _advp_caps[t] < (top_a / n_top if t in top_tks else
                                 bot_a / n_bot if n_bot > 0 else top_a / n_top)}

    _affected = _excluded | _capped
    w_final = w_new / w_new.sum() if w_new.sum() > 0 else w_new
    return w_final, _affected


def _apply_advp_cap(weights: pd.Series,
                    dt:      pd.Timestamp,
                    Pxs_df:  pd.DataFrame,
                    volumeRaw_df: pd.DataFrame,
                    current_aum: float,
                    advp_cap:    float,
                    min_weight:  float,
                    max_weight:  float) -> tuple:
    """
    Apply ADVP cap to portfolio weights.

    For each stock, compute median dollar ADV over last ADV_WINDOW days,
    then cap the dollar allocation at advp_cap * ADV.

    If the cap falls below min_weight, the stock is excluded entirely
    (including it at min_weight would violate the cap).

    Excess weight is redistributed iteratively to remaining uncapped stocks.

    Returns (capped_weights, capped_set) where capped_set is the set of
    tickers where the cap was binding.
    """
    if volumeRaw_df is None or current_aum <= 0:
        return weights, set()

    tickers = weights.index.tolist()

    # Compute median dollar ADV and derived max weight for each stock
    past_dates = [d for d in Pxs_df.index if d <= dt][-ADV_WINDOW:]
    adv_cap_w  = {}
    excluded   = set()
    for tkr in tickers:
        if tkr not in Pxs_df.columns or tkr not in volumeRaw_df.columns:
            adv_cap_w[tkr] = max_weight
            continue
        px_s   = Pxs_df.loc[past_dates, tkr].dropna()
        vol_s  = volumeRaw_df.loc[past_dates, tkr].dropna()
        common = px_s.index.intersection(vol_s.index)
        if len(common) < 3:
            adv_cap_w[tkr] = max_weight
            continue
        dollar_vol = (px_s.reindex(common) * vol_s.reindex(common).replace({0: np.nan})).median()
        max_dollar = dollar_vol * advp_cap
        cap_w      = max_dollar / current_aum if current_aum > 0 else max_weight
        if cap_w < min_weight:
            # Cap is below floor — exclude stock entirely
            excluded.add(tkr)
            adv_cap_w[tkr] = 0.0
        else:
            adv_cap_w[tkr] = min(cap_w, max_weight)

    # Remove excluded stocks and redistribute their weight
    w = weights.copy()
    if excluded:
        w = w.drop(index=excluded, errors='ignore')
        if w.sum() > 0:
            w = w / w.sum()   # renormalise before capping

    # Iterative water-filling for remaining stocks
    # Set of tickers where ADVP cap (not max_weight) is the binding constraint
    advp_binding = {tkr for tkr, cap in adv_cap_w.items()
                    if cap < max_weight - 1e-6 and tkr not in excluded}

    # Track raw ADVP caps (before min with max_weight) to distinguish binding constraint
    advp_only_cap = {}   # cap derived purely from ADVP (may be > max_weight)
    for tkr in tickers:
        if tkr in excluded:
            advp_only_cap[tkr] = 0.0
        elif tkr not in Pxs_df.columns or tkr not in volumeRaw_df.columns:
            advp_only_cap[tkr] = max_weight * 10   # effectively unconstrained by ADVP
        else:
            advp_only_cap[tkr] = adv_cap_w.get(tkr, max_weight * 10)
            # If adv_cap_w was set as min(cap_w, max_weight), recover raw cap_w
            # by checking if it equals max_weight (means ADVP wasn't binding)

    water_capped = set()   # only stocks where ADVP cap (not max_weight) was binding
    for _ in range(20):
        excess = 0.0
        newly_capped = set()
        for tkr in w.index:
            cap = adv_cap_w.get(tkr, max_weight)
            if w.get(tkr, 0) > cap + 1e-8:
                excess += w[tkr] - cap
                w[tkr]  = cap
                # Only flag if ADVP cap is strictly below max_weight
                if adv_cap_w.get(tkr, max_weight) < max_weight - 1e-6:
                    newly_capped.add(tkr)
        water_capped |= newly_capped
        if excess < 1e-6:
            break
        uncapped = [t for t in w.index if t not in water_capped]
        if not uncapped:
            break
        total_uncapped = w.reindex(uncapped).sum()
        if total_uncapped <= 0:
            break
        for tkr in uncapped:
            w[tkr] += excess * (w[tkr] / total_uncapped)

    # Re-normalise and apply max_weight cap only
    # (floor is NOT applied here — caller's _mb_floor_then_cap handles that
    #  for alpha/hybrid; for MVO cached weights we just cap and renorm)
    w = w[w > 1e-8]
    if w.sum() > 0:
        w = w / w.sum()
    # Apply max_weight cap iteratively
    for _ in range(20):
        over = w > max_weight + 1e-9
        if not over.any():
            break
        excess  = (w[over] - max_weight).sum()
        w[over] = max_weight
        under   = ~over
        if w[under].sum() > 1e-12:
            w[under] += excess * (w[under] / w[under].sum())
        else:
            w += excess / len(w)
    # Return only water-filled stocks as capped (*** flag) — excluded handled upstream
    return w, water_capped


def _mb_solve_mvo(dt, candidates, composite_scores, Pxs_df, sectors_s,
                   volumeTrd_df, model_version, pca_var_threshold,
                   ic, max_weight, min_weight, zscore_cap,
                   risk_aversion,
                   X_snapshots=None, snapshot_dates=None,
                   top_n=20, min_cov_matrices=2):
    """
    Exact replica of mvo_diagnostics.run_mvo_diagnostics portfolio logic:
      1. Build 4 cov matrices + ensemble
      2. Run _mb_max_alpha_portfolio on each of the 5 matrices
      3. Count matrix appearances per stock (excluding ensemble)
      4. Eligible = stocks appearing in >= min_cov_matrices of the 4 matrices
         (if min_cov_matrices=0, use all candidates)
      5. Final solve: _mb_max_alpha_portfolio on ensemble, eligible universe
    Returns (weights pd.Series, diagnostics dict).
    """
    # -- Get cached X snapshot -------------------------------------------------
    X_df_cached = None
    if X_snapshots is not None and snapshot_dates is not None:
        valid_snaps = [d for d in snapshot_dates if d <= dt]
        if valid_snaps:
            X_df_cached = X_snapshots[valid_snaps[-1]]

    # -- Return history — strictly up to dt (no lookahead) --------------------
    ret_df = (Pxs_df.loc[:dt, candidates].pct_change()
              .dropna(how='all')
              .iloc[-MVO_LOOKBACK:]
              .dropna(axis=1, how='any'))
    valid = [t for t in candidates if t in ret_df.columns]
    if len(valid) < top_n:
        warnings.warn(f"  {dt.date()} SKIP: only {len(valid)} valid candidates")
        return pd.Series(dtype=float), {}

    # -- Build covariance matrices ---------------------------------------------
    try:
        valid, Sigma_emp, Sigma_lw, Sigma_factor, Sigma_pca, Sigma_ens =             _mb_build_cov_matrices(dt, valid, Pxs_df, sectors_s,
                                    volumeTrd_df, model_version, pca_var_threshold,
                                    X_df_cached=X_df_cached)
    except Exception as e:
        warnings.warn(f"  {dt.date()} COV MATRIX FAILED: {e}")
        return pd.Series(dtype=float), {}

    # -- Alpha signal ----------------------------------------------------------
    vol_daily = _mvo_ewma_vol(ret_df[valid], MVO_EWMA_HL)
    vol_ann   = vol_daily * np.sqrt(252)
    z_s       = composite_scores.reindex(valid).fillna(0.0)
    z_capped  = z_s.clip(-zscore_cap, zscore_cap)
    alpha     = (ic * vol_daily * z_capped).fillna(0.0)

    # -- Per-matrix MVO (unconstrained, post-solve floor+cap) ------------------
    matrices = {
        'E': Sigma_emp,
        'L': Sigma_lw,
        'F': Sigma_factor,
        'P': Sigma_pca,
        'Ens': Sigma_ens,
    }
    top_n_by = {}
    for mname, Sigma in matrices.items():
        w_m = _mb_max_alpha_portfolio(Sigma, alpha, valid,
                                       risk_aversion, max_weight, min_weight)
        top_n_by[mname] = w_m.nlargest(top_n).index.tolist()

    # -- Eligibility: count appearances in 4 individual matrices (not ensemble) -
    count_matrices = ['E', 'L', 'F', 'P']
    selection_count = {t: sum(1 for m in count_matrices
                              if t in top_n_by[m])
                       for t in valid}

    if min_cov_matrices > 0:
        eligible = [t for t in valid
                    if selection_count.get(t, 0) >= min_cov_matrices]
        # Graceful fallback: lower threshold until we have enough stocks
        thresh = min_cov_matrices
        while len(eligible) < top_n and thresh > 0:
            thresh -= 1
            eligible = [t for t in valid
                        if selection_count.get(t, 0) >= thresh]
    else:
        eligible = valid  # alpha-only mode

    # -- Final MVO on ensemble cov, eligible universe --------------------------
    elig_idx  = [valid.index(t) for t in eligible]
    S_ens_el  = Sigma_ens[np.ix_(elig_idx, elig_idx)]
    alpha_el  = alpha.reindex(eligible).fillna(0.0)

    w_out = _mb_max_alpha_portfolio(S_ens_el, alpha_el, eligible,
                                     risk_aversion, max_weight, min_weight)
    # Trim to exactly top_n by keeping highest-weight stocks, renorm
    if len(w_out) > top_n:
        w_out = w_out.nlargest(top_n)
        w_out = w_out / w_out.sum()


    # -- Diagnostics dict for display ------------------------------------------
    diag = {
        'n_valid':    len(valid),
        'n_eligible': len(eligible),
        'vol_ann'        : vol_ann,
        'z_s'            : z_s,
        'alpha_ann'      : alpha * 252,
        'top_n_by'       : {k: v for k, v in top_n_by.items() if k != 'Ens'},
        'selection_count': selection_count,
        'eligible'       : eligible,
        'ic_used'        : ic,
    }
    return w_out, diag



# ===============================================================================
# BACKTEST ENGINE
# ===============================================================================

def _mb_run_nav(weights_by_date, calc_dates, Pxs_df, cost_by_date=None):
    """
    Compute NAV series from a dict of {rebal_date: pd.Series(weights)}.
    Mirrors run_backtest NAV logic from primary_factor_backtest.py.
    cost_by_date: optional dict {date: cost_fraction} applied at rebalance.
    """
    nav        = 1.0
    nav_series = {}
    portfolio  = []
    weights    = {}

    for i, rebal_date in enumerate(calc_dates):
        next_date = (calc_dates[i+1] if i+1 < len(calc_dates)
                     else Pxs_df.index.max())

        if rebal_date in weights_by_date:
            w = weights_by_date[rebal_date]
            w = w[w > 1e-6]
            if len(w) > 0 and w.sum() > 0:
                # Apply trading cost at rebalance date
                if cost_by_date and rebal_date in cost_by_date:
                    nav *= (1 - cost_by_date[rebal_date])
                portfolio = w.index.tolist()
                weights   = (w / w.sum()).to_dict()

        period_dates = Pxs_df.index[
            (Pxs_df.index >= rebal_date) & (Pxs_df.index <= next_date)
        ]

        if not portfolio or len(period_dates) < 2:
            for d in period_dates:
                nav_series[d] = nav
            continue

        px_start   = Pxs_df.loc[period_dates[0],  portfolio]
        px_end     = Pxs_df.loc[period_dates[-1], portfolio]
        stk_rets   = (px_end / px_start - 1).fillna(0)
        w_s        = pd.Series(weights).reindex(portfolio).fillna(0)
        period_ret = (stk_rets * w_s).sum()
        nav       *= (1 + period_ret)

        px_period  = Pxs_df.loc[period_dates, portfolio]
        stk_daily  = px_period.div(px_start, axis=1) - 1
        port_cum   = stk_daily.mul(w_s, axis=1).sum(axis=1)
        period_nav = nav / (1 + period_ret) * (1 + port_cum)
        for d, v in period_nav.items():
            nav_series[d] = v

    nav_s = pd.Series(nav_series).sort_index()
    if MB_START_DATE not in nav_s.index:
        nav_s[MB_START_DATE] = 1.0
        nav_s = nav_s.sort_index()
    return nav_s


# ===============================================================================
# COMPARISON PLOT
# ===============================================================================

def _mb_plot(nav_baseline, nav_alpha, nav_mvo, regime_s, nav_hybrid=None,
             nav_smart=None, nav_dynamic=None,
             nav_dyn_hedged=None, nav_dd=None):
    """Four panels: NAV, relative vs baseline, drawdown, regime."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 16),
                             gridspec_kw={'height_ratios': [3, 2, 2, 1]})
    fig.patch.set_facecolor('#FAFAF9')
    for ax in axes:
        ax.set_facecolor('#FAFAF9')

    GRAY  = '#888780'; TEAL = '#1D9E75'; BLUE = '#378ADD'; CORAL = '#D85A30'

    common = nav_baseline.index               .intersection(nav_alpha.index)               .intersection(nav_mvo.index)
    if nav_hybrid is not None:
        common = common.intersection(nav_hybrid.index)
    if nav_smart is not None:
        common = common.intersection(nav_smart.index)
    if nav_dynamic is not None and not nav_dynamic.empty:
        common = common.intersection(nav_dynamic.index)
    if nav_dyn_hedged is not None and not nav_dyn_hedged.empty:
        common = common.intersection(nav_dyn_hedged.index)
    if nav_dd is not None and not nav_dd.empty:
        common = common.intersection(nav_dd.index)

    nb  = nav_baseline.reindex(common)
    na  = nav_alpha.reindex(common)
    nm  = nav_mvo.reindex(common)
    nh  = nav_hybrid.reindex(common)     if nav_hybrid     is not None else None
    ns  = nav_smart.reindex(common)      if nav_smart      is not None else None
    nd  = nav_dynamic.reindex(common)    if (nav_dynamic    is not None and not nav_dynamic.empty)    else None
    ndh = nav_dyn_hedged.reindex(common) if (nav_dyn_hedged is not None and not nav_dyn_hedged.empty) else None
    ndd = nav_dd.reindex(common)         if (nav_dd         is not None and not nav_dd.empty)         else None

    # Regime shading
    reg = regime_s.reindex(common, method='ffill').fillna(0)
    REGIME_BG = {0.0: '#E1F5EE', 0.5: '#F1EFE8', 1.0: '#FAECE7'}
    for ax in axes[:3]:
        prev_r, prev_d = float(reg.iloc[0]), common[0]
        for d, r in reg.items():
            r = float(r)
            if r != prev_r or d == common[-1]:
                ax.axvspan(prev_d, d,
                           color=REGIME_BG.get(prev_r, '#F1EFE8'),
                           alpha=0.3, linewidth=0)
                prev_r, prev_d = r, d

    # Panel 1: NAV
    ax = axes[0]
    ps = [(nb,'Baseline',GRAY,1.2),(na,'Pure alpha',TEAL,1.5),(nm,'MVO',BLUE,1.8)]
    if nh  is not None: ps.append((nh, 'Hybrid',         '#7F77DD', 1.5))
    if ns  is not None: ps.append((ns, 'Smart Hybrid',   '#E8A838', 1.8))
    if nd  is not None: ps.append((nd, 'Dynamic',        '#9B59B6', 1.8))
    if ndh is not None: ps.append((ndh,'Dyn+Hedge',      '#2ECC71', 2.0))
    if ndd is not None: ps.append((ndd,'Dyn+Hedge+DD',   '#E74C3C', 2.0))
    for nav_s, lbl, color, lw in ps:
        ax.plot(nav_s.index.to_numpy(), nav_s.values,
                label=lbl, color=color, linewidth=lw)
    ax.set_ylabel("NAV", fontsize=10, color='#5F5E5A')
    ax.set_title("Strategy NAV Comparison",
                 fontsize=12, fontweight='500', color='#2C2C2A')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.85)
    ax.grid(color='#D3D1C7', linewidth=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel 2: Relative to baseline
    ax2 = axes[1]
    rs = [(na,'Pure alpha vs baseline',TEAL),(nm,'MVO vs baseline',BLUE)]
    if nh  is not None: rs.append((nh, 'Hybrid vs baseline',       '#7F77DD'))
    if ns  is not None: rs.append((ns, 'Smart Hybrid vs baseline', '#E8A838'))
    if nd  is not None: rs.append((nd, 'Dynamic vs baseline',      '#9B59B6'))
    if ndh is not None: rs.append((ndh,'Dyn+Hedge vs baseline',    '#2ECC71'))
    if ndd is not None: rs.append((ndd,'Dyn+Hedge+DD vs baseline', '#E74C3C'))
    for nav_s, lbl, color in rs:
        rel = (nav_s / nb - 1) * 100
        ax2.plot(rel.index.to_numpy(), rel.values,
                 label=lbl, color=color, linewidth=1.2)
        ax2.fill_between(rel.index.to_numpy(), rel.values, 0,
                         where=(rel.values >= 0), color=color, alpha=0.08)
        ax2.fill_between(rel.index.to_numpy(), rel.values, 0,
                         where=(rel.values < 0), color=CORAL, alpha=0.08)
    ax2.axhline(0, color=GRAY, linewidth=0.8, linestyle='--')
    ax2.set_ylabel("Relative to baseline (%)", fontsize=10, color='#5F5E5A')
    ax2.legend(fontsize=8, loc='upper left', framealpha=0.85)
    ax2.grid(color='#D3D1C7', linewidth=0.5)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    # Panel 3: Drawdown
    ax3 = axes[2]
    ds = [(nb,'Baseline',GRAY),(na,'Pure alpha',TEAL),(nm,'MVO',BLUE)]
    if nh  is not None: ds.append((nh, 'Hybrid',       '#7F77DD'))
    if ns  is not None: ds.append((ns, 'Smart Hybrid', '#E8A838'))
    if nd  is not None: ds.append((nd, 'Dynamic',      '#9B59B6'))
    if ndh is not None: ds.append((ndh,'Dyn+Hedge',    '#2ECC71'))
    if ndd is not None: ds.append((ndd,'Dyn+Hedge+DD', '#E74C3C'))
    for nav_s, lbl, color in ds:
        dd = (nav_s / nav_s.cummax() - 1) * 100
        ax3.plot(dd.index.to_numpy(), dd.values,
                 label=lbl, color=color, linewidth=1.0)
    ax3.axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    ax3.set_ylabel("Drawdown (%)", fontsize=10, color='#5F5E5A')
    ax3.legend(fontsize=8, loc='lower left', framealpha=0.85)
    ax3.grid(color='#D3D1C7', linewidth=0.5)
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

    # Panel 4: Regime
    ax4 = axes[3]
    ax4.fill_between(reg.index.to_numpy(), reg.values, 0,
                     color=BLUE, alpha=0.25)
    ax4.plot(reg.index.to_numpy(), reg.values, color=BLUE, linewidth=1.0)
    ax4.set_yticks([0.0, 0.5, 1.0])
    ax4.set_yticklabels(['Easy', 'Neutral', 'Tight'], fontsize=8)
    ax4.set_ylabel("Regime", fontsize=10, color='#5F5E5A')
    ax4.grid(color='#D3D1C7', linewidth=0.5)
    ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig




# ===============================================================================
# DAILY PORTFOLIO CACHE BUILDER
# ===============================================================================

# -- Suppress output -----------------------------------------------------------
# _Suppress = _SuppressOutput (already defined above)


# -- Parameter hash ------------------------------------------------------------

def _ensure_tables():
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {DAILY_PORT_TBL} (
                date          DATE        NOT NULL,
                model_version VARCHAR(10) NOT NULL,
                params_hash   VARCHAR(16) NOT NULL,
                strategy      VARCHAR(16) NOT NULL,
                weights_json  TEXT        NOT NULL,
                PRIMARY KEY (date, model_version, params_hash, strategy)
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {DAILY_TRIGGER_TBL} (
                date               DATE        NOT NULL,
                model_version      VARCHAR(10) NOT NULL,
                params_hash        VARCHAR(16) NOT NULL,
                implied_turnover   NUMERIC(8,6),
                vol_diff           NUMERIC(8,6),
                PRIMARY KEY (date, model_version, params_hash)
            )
        """))


def _get_cached_dates(ph, mv):
    try:
        _ensure_tables()
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT DISTINCT date FROM {DAILY_PORT_TBL}
                WHERE params_hash=:ph AND model_version=:mv AND strategy='alpha'
            """), {'ph': ph, 'mv': mv}).fetchall()
        return {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        return set()


def _save_portfolios(dt, mv, ph, portfolios):
    for strategy, w in portfolios.items():
        if w is None or w.empty:
            continue
        wj = json.dumps({t: float(v) for t, v in w.items()})
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                INSERT INTO {DAILY_PORT_TBL}
                    (date, model_version, params_hash, strategy, weights_json)
                VALUES (:dt, :mv, :ph, :st, :wj)
                ON CONFLICT (date, model_version, params_hash, strategy)
                DO UPDATE SET weights_json = EXCLUDED.weights_json
            """), {'dt': dt.strftime('%Y-%m-%d'), 'mv': mv,
                   'ph': ph, 'st': strategy, 'wj': wj})


def _save_triggers(dt, mv, ph, implied_to, vol_diff):
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {DAILY_TRIGGER_TBL}
                (date, model_version, params_hash,
                 implied_turnover, vol_diff)
            VALUES (:dt, :mv, :ph, :ito, :vd)
            ON CONFLICT (date, model_version, params_hash)
            DO UPDATE SET
                implied_turnover = EXCLUDED.implied_turnover,
                vol_diff         = EXCLUDED.vol_diff
        """), {'dt': dt.strftime('%Y-%m-%d'), 'mv': mv, 'ph': ph,
               'ito': float(implied_to), 'vd': float(vol_diff)})


# -- Vol helper ----------------------------------------------------------------
def _portfolio_vol(w, Pxs_df, dt, lookback=VOL_LOOKBACK):
    """Annualized realized vol of portfolio w over last `lookback` trading days."""
    if w is None or w.empty:
        return 0.0
    tickers = [t for t in w.index if t in Pxs_df.columns]
    if not tickers:
        return 0.0
    px = Pxs_df.loc[:dt, tickers].iloc[-lookback:]
    if len(px) < 10:
        return 0.0
    rets   = px.pct_change().dropna()
    w_s    = w.reindex(tickers).fillna(0)
    if w_s.sum() > 0:
        w_s = w_s / w_s.sum()
    port_r = (rets * w_s).sum(axis=1)
    return float(port_r.std() * np.sqrt(252))


# -- Main builder --------------------------------------------------------------

# ================================================================================
# HEDGE ENGINE (embedded from hedge_engine.py)
# ================================================================================




# ================================================================================
# HELPERS
# ================================================================================

def _portfolio_returns(weights_by_date: dict,
                       Pxs_df:          pd.DataFrame,
                       all_dates:        pd.DatetimeIndex) -> pd.Series:
    """
    Build daily portfolio return series using fixed rebalance weights.
    Between rebalances, apply the last rebalance weights to daily stock returns.
    """
    rebal_dates = sorted(weights_by_date.keys())
    port_rets   = {}

    for i, dt in enumerate(all_dates):
        # Find last rebalance date <= dt
        past = [d for d in rebal_dates if d <= dt]
        if not past:
            continue
        last_rebal = past[-1]
        w = weights_by_date[last_rebal]
        tickers = [t for t in w.index if t in Pxs_df.columns]
        if not tickers:
            continue
        prev_dt_idx = all_dates.get_loc(dt)
        if prev_dt_idx == 0:
            continue
        prev_dt = all_dates[prev_dt_idx - 1]
        if prev_dt not in Pxs_df.index or dt not in Pxs_df.index:
            continue
        px_prev = Pxs_df.loc[prev_dt, tickers]
        px_cur  = Pxs_df.loc[dt,      tickers]
        ret     = (px_cur / px_prev - 1).fillna(0)
        port_rets[dt] = (w.reindex(tickers).fillna(0) * ret).sum()

    return pd.Series(port_rets, name='portfolio_ret')


def _compute_beta(port_ret_s: pd.Series,
                  inst_ret_s: pd.Series,
                  window:     int) -> float:
    """OLS beta of portfolio returns to instrument returns over last `window` days."""
    common = port_ret_s.dropna().index.intersection(inst_ret_s.dropna().index)
    common = common[-window:] if len(common) >= window else common
    if len(common) < window // 2:
        return np.nan
    p = port_ret_s.reindex(common)
    h = inst_ret_s.reindex(common)
    var_h = h.var()
    return p.cov(h) / var_h if var_h > 0 else np.nan


def _compute_corr(port_ret_s: pd.Series,
                  inst_ret_s: pd.Series,
                  window:     int) -> float:
    """Rolling correlation of portfolio to instrument over last `window` days."""
    common = port_ret_s.dropna().index.intersection(inst_ret_s.dropna().index)
    common = common[-window:] if len(common) >= window else common
    if len(common) < window // 2:
        return np.nan
    return port_ret_s.reindex(common).corr(inst_ret_s.reindex(common))


def _get_effectiveness(signal_df: pd.DataFrame,
                       dt:        pd.Timestamp,
                       mav_win:   int) -> float:
    """Smoothed effectiveness at date dt (MAV of last mav_win values).
    Returns 1.0 (neutral) if no effectiveness history available yet."""
    if 'effectiveness' not in signal_df.columns:
        return 1.0
    past = signal_df['effectiveness'].loc[:dt].dropna()
    if past.empty:
        return 1.0
    return past.iloc[-mav_win:].mean()


# ================================================================================
# MAIN ENTRY POINT
# ================================================================================

def run_hedge_backtest(Pxs_df:            pd.DataFrame,
                       multi:             dict,
                       portfolio_weights: dict,
                       rebal_dates:       list,
                       hedges_l:          list,
                       beta_window:       int   = BETA_WINDOW,
                       corr_window:       int   = CORR_WINDOW,
                       eff_mav_window:    int   = EFF_MAV_WINDOW,
                       eff_floor:         float = EFF_FLOOR,
                       corr_floor:        float = CORR_FLOOR,
                       hedge_ratio:       float = HEDGE_RATIO,
                       max_hedge:         float = MAX_HEDGE,
                       trigger_assets:    list  = TRIGGER_ASSETS) -> dict:
    """
    Run hedge backtest in parallel with main portfolio.

    Parameters
    ----------
    Pxs_df            : price DataFrame
    multi             : output of run_macro_hedge_cached
    portfolio_weights : {date: pd.Series} rebalance weights
    rebal_dates       : sorted list of portfolio rebalancing dates
    hedges_l          : list of hedge instrument tickers
    (all other params: see module defaults)

    Returns
    -------
    dict with keys:
      'hedge_account_by_date' : {date: float}  hedge account balance each day
      'hedge_sweep_by_date'   : {date: float}  amount swept to portfolio at rebal
      'hedge_log'             : list of event dicts
      'summary_by_year'       : pd.DataFrame
      'summary_by_instrument' : pd.DataFrame
      'port_ret_s'            : pd.Series  daily portfolio returns
    """
    results      = multi['results']
    all_dates    = Pxs_df.index
    rebal_set    = set(rebal_dates)

    # Validate trigger assets
    for ta in trigger_assets:
        if ta not in results:
            print(f"  WARNING: trigger asset {ta} not in multi results — "
                  f"will be treated as always OFF")

    # Validate hedges
    hedges_l = [h for h in hedges_l if h in results and h in Pxs_df.columns]
    missing  = [h for h in hedges_l if h not in results or h not in Pxs_df.columns]
    if missing:
        print(f"  WARNING: skipping {missing} — not in multi results or Pxs_df")

    print('=' * 72)
    print('  HEDGE BACKTEST ENGINE')
    print('=' * 72)
    print(f"\n  Trigger assets : {trigger_assets}")
    print(f"  Hedge universe : {hedges_l}")
    print(f"  Parameters     : beta_win={beta_window}d  corr_win={corr_window}d  "
          f"eff_mav={eff_mav_window}d")
    print(f"                   eff_floor={eff_floor}  corr_floor={corr_floor:.0%}  "
          f"hedge_ratio={hedge_ratio:.0%}  max_hedge={max_hedge:.0%}")

    # Build portfolio return series
    print("\n  Building portfolio return series...")
    port_ret_s = _portfolio_returns(portfolio_weights, Pxs_df, all_dates)

    # Pre-compute instrument return series
    inst_ret = {inst: Pxs_df[inst].pct_change().fillna(0)
                for inst in hedges_l if inst in Pxs_df.columns}

    # ── Main loop ─────────────────────────────────────────────────────────────
    hedge_account      = 0.0    # running hedge account balance (as % of NAV)
    hedge_sweep        = {}     # {rebal_date: amount swept to portfolio}
    hedge_account_hist = {}     # {date: balance}
    hedge_log          = []     # all hedge events

    active_hedges      = {}     # {inst: {'entry_px': float, 'weight': float}}
    hedge_on           = False  # current hedge state
    prev_trigger       = 0      # previous day's trigger signal

    print("\n  Running hedge simulation...\n")

    for dt in all_dates:
        if dt not in port_ret_s.index:
            hedge_account_hist[dt] = hedge_account
            continue

        # ── Sweep hedge account at portfolio rebalancing dates ────────────────
        # Only sweep when no active hedges — deferred if hedges still open
        if dt in rebal_set and not hedge_on:
            if hedge_account != 0.0:
                sweep_amt        = hedge_account
                hedge_sweep[dt]  = hedge_sweep.get(dt, 0) + sweep_amt
                print(f"  {'='*68}")
                print(f"  PORTFOLIO REBALANCE {dt.date()}  |  "
                      f"Hedge account sweep: {sweep_amt:+.4%} of NAV")
                print(f"  {'='*68}")
                hedge_account = 0.0
        elif dt in rebal_set and hedge_on:
            print(f"  PORTFOLIO REBALANCE {dt.date()}  |  "
                  f"Hedges active — sweep deferred")

        # ── Compute trigger signal (1-day lag) ────────────────────────────────
        trigger_on = False
        for ta in trigger_assets:
            if ta in results:
                sig_df = results[ta]['signal_df']
                if dt in sig_df.index:
                    # 1-day lag: use previous day's signal
                    prev_dates = sig_df.index[sig_df.index < dt]
                    if len(prev_dates) > 0:
                        prev_sig = sig_df.loc[prev_dates[-1], 'signal']
                        if prev_sig == 1:
                            trigger_on = True
                            break

        # ── Hedge activation ──────────────────────────────────────────────────
        if trigger_on and not hedge_on:
            # New hedge event — select instruments
            selected = _select_hedge_instruments(
                dt, hedges_l, results, port_ret_s, inst_ret,
                beta_window, corr_window, eff_mav_window,
                eff_floor, corr_floor, trigger_assets
            )

            if selected:
                hedge_on      = True
                active_hedges = {}
                n_inst        = len(selected)
                total_hedge   = min(n_inst * hedge_ratio, max_hedge)

                # Allocate by beta * effectiveness, normalized to total_hedge
                raw_scores = {inst: d['beta'] * d['effectiveness']
                              for inst, d in selected.items()
                              if not np.isnan(d['beta']) and
                                 not np.isnan(d['effectiveness'])}
                total_score = sum(raw_scores.values())

                print(f"\n  {'─'*68}")
                print(f"  HEDGE OPEN  {dt.date()}  |  "
                      f"{n_inst} instruments  total_hedge={total_hedge:.1%}")
                print(f"  {'─'*68}")
                print(f"  {'Instrument':<10}  {'Corr':>7}  {'Beta':>7}  "
                      f"{'Eff (MAV)':>10}  {'Score':>8}  {'Weight':>8}")
                print(f"  {'-'*60}")

                for inst, d in selected.items():
                    raw_sc = raw_scores.get(inst, 0)
                    w      = (raw_sc / total_score * total_hedge
                              if total_score > 0 else total_hedge / n_inst)
                    active_hedges[inst] = {
                        'weight':   w,
                        'entry_px': Pxs_df.loc[dt, inst]
                                    if dt in Pxs_df.index else np.nan,
                        'entry_dt': dt,
                        'beta':     d['beta'],
                        'corr':     d['corr'],
                        'eff':      d['effectiveness'],
                        'score':    raw_sc,
                    }
                    print(f"  {inst:<10}  {d['corr']:>7.3f}  {d['beta']:>7.3f}  "
                          f"{d['effectiveness']:>10.3f}  {raw_sc:>8.3f}  {w:>7.2%}")

                print(f"  {'─'*60}")
                print(f"  Total hedge: {total_hedge:.1%}  "
                      f"(allocated {sum(v['weight'] for v in active_hedges.values()):.1%})")

                hedge_log.append({
                    'date': dt, 'event': 'OPEN',
                    'instruments': list(active_hedges.keys()),
                    'total_hedge': total_hedge,
                    'details': {inst: dict(v) for inst, v in active_hedges.items()},
                })
            else:
                print(f"\n  {dt.date()}  Trigger ON but no qualifying instruments — "
                      f"no hedge activated")

        # ── Hedge unwind ──────────────────────────────────────────────────────
        elif not trigger_on and hedge_on:
            hedge_on = False
            print(f"\n  {'─'*68}")
            print(f"  HEDGE CLOSE {dt.date()}")
            print(f"  {'─'*68}")
            print(f"  {'Instrument':<10}  {'Entry':>12}  {'Exit':>12}  "
                  f"{'Weight':>8}  {'P&L':>10}")
            print(f"  {'-'*57}")

            total_pnl = 0.0
            for inst, h in active_hedges.items():
                if dt in Pxs_df.index and h['entry_px'] and not np.isnan(h['entry_px']):
                    exit_px      = Pxs_df.loc[dt, inst]
                    inst_ret_pct = (exit_px / h['entry_px'] - 1)
                    # Short position: profit when instrument falls
                    # Deduct 10bps entry + 10bps exit = 20bps round-trip per instrument
                    trade_cost = 2 * TRADING_COST_BPS / 10000 * h['weight']
                    pnl        = -inst_ret_pct * h['weight'] - trade_cost
                    total_pnl += pnl
                    print(f"  {inst:<10}  {h['entry_px']:>12.4f}  {exit_px:>12.4f}  "
                          f"{h['weight']:>8.2%}  {pnl:>+9.4%}")
                else:
                    pnl = 0.0
                    print(f"  {inst:<10}  {'n/a':>12}  {'n/a':>12}  "
                          f"{h['weight']:>8.2%}  {'n/a':>10}")

            hedge_account += total_pnl
            total_costs = sum(2 * TRADING_COST_BPS / 10000 * h['weight']
                              for h in active_hedges.values())
            print(f"  {'─'*57}")
            print(f"  Episode P&L : {total_pnl:+.4%}  (incl. {total_costs*100:.2f}bps round-trip costs)")
            print(f"  Hedge acct  : {hedge_account:+.4%} (pending sweep at next rebal)")

            days_held = (dt - active_hedges[list(active_hedges.keys())[0]]['entry_dt']).days
            hedge_log.append({
                'date': dt, 'event': 'CLOSE',
                'instruments': list(active_hedges.keys()),
                'total_pnl': total_pnl,
                'days_held': days_held,
                'hedge_account': hedge_account,
                'details': {inst: dict(v) for inst, v in active_hedges.items()},
            })
            active_hedges = {}

        # ── Mark-to-market active hedges (display only, not booked to account) ─
        elif hedge_on and active_hedges:
            daily_pnl = 0.0
            prev_dates_all = all_dates[all_dates < dt]
            if len(prev_dates_all) > 0:
                prev_dt = prev_dates_all[-1]
                for inst, h in active_hedges.items():
                    if (inst in Pxs_df.columns and
                            prev_dt in Pxs_df.index and dt in Pxs_df.index):
                        r = Pxs_df.loc[dt, inst] / Pxs_df.loc[prev_dt, inst] - 1
                        daily_pnl += -r * h['weight']
            # MTM tracked separately — not added to hedge_account
            # hedge_account only updates at close

        hedge_account_hist[dt] = hedge_account

    # ── Final summary ─────────────────────────────────────────────────────────
    _print_hedge_summary(hedge_log, Pxs_df, results)

    return {
        'hedge_account_by_date': hedge_account_hist,
        'hedge_sweep_by_date':   hedge_sweep,
        'hedge_log':             hedge_log,
        'port_ret_s':            port_ret_s,
        'active_hedges':         active_hedges,
    }


# ================================================================================
# INSTRUMENT SELECTION
# ================================================================================

def _select_hedge_instruments(dt:            pd.Timestamp,
                               hedges_l:     list,
                               results:      dict,
                               port_ret_s:   pd.Series,
                               inst_ret:     dict,
                               beta_window:  int,
                               corr_window:  int,
                               eff_mav_win:  int,
                               eff_floor:    float,
                               corr_floor:   float,
                               trigger_assets: list) -> dict:
    """
    Select qualifying hedge instruments at activation date dt.
    Returns dict {inst: {'beta', 'corr', 'effectiveness'}} for selected instruments.
    """
    # Compute corr and effectiveness for all instruments
    scores = {}
    for inst in hedges_l:
        if inst not in inst_ret or inst not in results:
            continue
        corr = _compute_corr(port_ret_s, inst_ret[inst], corr_window)
        eff  = _get_effectiveness(results[inst]['signal_df'], dt, eff_mav_win)
        beta = _compute_beta(port_ret_s, inst_ret[inst], beta_window)
        scores[inst] = {'corr': corr, 'eff': eff, 'beta': beta,
                        'effectiveness': eff}

    # Sort by correlation descending
    ranked = sorted(scores.items(),
                    key=lambda x: x[1]['corr'] if not np.isnan(x[1]['corr']) else -999,
                    reverse=True)

    _DEFAULT_EFF  = 0.75   # default when effectiveness data not yet available
    _DEFAULT_CORR = 0.75   # default when correlation history insufficient

    selected = {}

    # Always check trigger assets first (QQQ, SPY)
    for ta in trigger_assets:
        if ta in scores:
            d    = scores[ta]
            corr = d['corr'] if not np.isnan(d.get('corr', np.nan)) else _DEFAULT_CORR
            eff  = d['eff']  if not np.isnan(d.get('eff',  np.nan)) else _DEFAULT_EFF
            if corr >= corr_floor and eff >= eff_floor:
                selected[ta] = d

    # Add top 3 from correlation ranking (excluding already selected)
    n_from_ranking = 0
    for inst, d in ranked:
        if inst in selected:
            continue
        if n_from_ranking >= 3:
            break
        corr = d['corr'] if not np.isnan(d.get('corr', np.nan)) else _DEFAULT_CORR
        eff  = d['eff']  if not np.isnan(d.get('eff',  np.nan)) else _DEFAULT_EFF
        if corr < corr_floor:
            break   # ranked by corr, so all remaining will also fail
        if eff >= eff_floor:
            selected[inst] = d
            n_from_ranking += 1

    return selected


# ================================================================================
# SUMMARY
# ================================================================================

def _print_hedge_summary(hedge_log:  list,
                         Pxs_df:    pd.DataFrame,
                         results:   dict) -> None:
    """Print per-year and per-instrument hedge summary."""
    if not hedge_log:
        print("\n  No hedge events recorded.")
        return

    close_events = [e for e in hedge_log if e['event'] == 'CLOSE']
    if not close_events:
        print("\n  No completed hedge episodes.")
        return

    print(f"\n{'='*72}")
    print(f"  HEDGE SUMMARY")
    print(f"{'='*72}")

    # Overall stats
    all_pnl    = [e['total_pnl'] for e in close_events]
    n_pos      = sum(1 for p in all_pnl if p > 0)
    print(f"\n  Total episodes  : {len(close_events)}")
    print(f"  Hit rate        : {n_pos/len(close_events)*100:.1f}%  "
          f"({n_pos} positive / {len(close_events)-n_pos} negative)")
    print(f"  Total P&L       : {sum(all_pnl):+.4%}")
    print(f"  Avg P&L/episode : {np.mean(all_pnl):+.4%}")
    print(f"  Avg days held   : {np.mean([e.get('days_held',0) for e in close_events]):.1f}")

    # Per-year breakdown
    print(f"\n  {'─'*68}")
    print(f"  PER-YEAR BREAKDOWN")
    print(f"  {'─'*68}")
    print(f"  {'Year':<6}  {'Episodes':>9}  {'Hit Rate':>9}  "
          f"{'Total P&L':>10}  {'Avg P&L':>10}  {'Avg Days':>9}")
    print(f"  {'-'*60}")

    years = sorted(set(e['date'].year for e in close_events))
    for yr in years:
        yr_ev  = [e for e in close_events if e['date'].year == yr]
        yr_pnl = [e['total_pnl'] for e in yr_ev]
        yr_pos = sum(1 for p in yr_pnl if p > 0)
        yr_days= np.mean([e.get('days_held', 0) for e in yr_ev])
        print(f"  {yr:<6}  {len(yr_ev):>9}  "
              f"{yr_pos/len(yr_ev)*100:>8.1f}%  "
              f"{sum(yr_pnl):>+9.4%}  "
              f"{np.mean(yr_pnl):>+9.4%}  "
              f"{yr_days:>9.1f}")

    # Per-instrument breakdown (only instruments that were used)
    inst_events = {}
    for e in close_events:
        for inst in e.get('instruments', []):
            if inst not in inst_events:
                inst_events[inst] = []
            w   = e['details'].get(inst, {}).get('weight', 0)
            px_entry = e['details'].get(inst, {}).get('entry_px', np.nan)
            entry_dt = e['details'].get(inst, {}).get('entry_dt', e['date'])
            # Recompute instrument-level P&L from entry to close
            close_dt = e['date']
            if (px_entry and not np.isnan(px_entry) and
                    inst in Pxs_df.columns and close_dt in Pxs_df.index):
                exit_px  = Pxs_df.loc[close_dt, inst]
                inst_pnl = -(exit_px / px_entry - 1) * w
            else:
                inst_pnl = np.nan
            inst_events[inst].append({
                'pnl': inst_pnl, 'weight': w,
                'eff': e['details'].get(inst, {}).get('eff', np.nan),
                'beta': e['details'].get(inst, {}).get('beta', np.nan),
            })

    if inst_events:
        print(f"\n  {'─'*68}")
        print(f"  PER-INSTRUMENT BREAKDOWN")
        print(f"  {'─'*68}")
        print(f"  {'Instrument':<12}  {'Uses':>5}  {'Hit Rate':>9}  "
              f"{'Total P&L':>10}  {'Avg Weight':>11}  {'Avg Beta':>9}  {'Avg Eff':>8}")
        print(f"  {'-'*68}")
        for inst in sorted(inst_events.keys()):
            evs     = inst_events[inst]
            pnls    = [e['pnl'] for e in evs if not np.isnan(e['pnl'])]
            pos     = sum(1 for p in pnls if p > 0)
            avg_w   = np.mean([e['weight'] for e in evs])
            avg_b   = np.nanmean([e['beta'] for e in evs])
            avg_e   = np.nanmean([e['eff']  for e in evs])
            hit_str = f"{pos/len(pnls)*100:.1f}%" if pnls else "n/a"
            pnl_str = f"{sum(pnls):+.4%}" if pnls else "n/a"
            print(f"  {inst:<12}  {len(evs):>5}  {hit_str:>9}  "
                  f"{pnl_str:>10}  {avg_w:>10.2%}  {avg_b:>9.3f}  {avg_e:>8.3f}")


def run_daily_cache_build(
    Pxs_df, sectors_s, weights_by_year, regime_s,
    volumeTrd_df=None,
    weights_by_date=None,
    volumeRaw_df=None,       # raw share volume for ADVP filter
    aum_by_date=None,        # optional {date: aum} from prior backtest for correct ADVP
    ic=0.03, max_weight=0.075, min_weight=0.025,
    zscore_cap=2.5, pca_var_threshold=0.65,
    universe_mult=5, risk_aversion=10,
    top_n=25, conc_factor=2.0, prefilt_pct=0.5,
    min_cov_matrices=2, model_version='v2',
    mode='incremental',   # 'incremental' | 'rebuild'
    # Legacy params kept for backward compatibility — ignored
    force_rebuild=None, excl_only=None,
    pct_df=None, ou_scores_df=None,
):
    """
    mode='incremental' (default — safe daily use)
        Adds today's date only. Uses all cached portfolios as-is.
        Standard portfolios (strategies 1-8) and excl portfolios
        (strategy 9) are both updated incrementally.
        Use this every day after running the upstream pipeline.

    mode='rebuild'
        Wipes ALL portfolios (standard + excl) and recomputes from
        scratch using the current composite scores, MR_K penalty and
        exclusion lists as cached by momentum_exclusions.
        Use this when QUALITY_FLOOR, MR_K or any portfolio construction
        parameter changes.

    Note: force_rebuild and excl_only are deprecated and ignored.
    """
    if force_rebuild is not None or excl_only is not None:
        print("  WARNING: force_rebuild/excl_only are deprecated. Use mode='incremental' or mode='rebuild'.")

    ph = _make_make_params_hash(ic, max_weight, min_weight, zscore_cap, pca_var_threshold,
                      universe_mult, risk_aversion, top_n, conc_factor,
                      prefilt_pct, min_cov_matrices, model_version)

    print("=" * 72)
    print("  DAILY PORTFOLIO CACHE BUILDER")
    print("=" * 72)
    print(f"\n  params_hash={ph}  model={model_version}  mode={mode}")
    print(f"  IC={ic}  ra={risk_aversion}  N={top_n}  "
          f"max_w={max_weight}  min_w={min_weight}")
    print(f"  universe_mult={universe_mult}  conc={conc_factor}  "
          f"prefilt={prefilt_pct}  min_cov={min_cov_matrices}")
    print(f"  MR_K={MR_K}  MR_CAP={MR_CAP}  QUALITY_FLOOR={QUALITY_FLOOR}")
    print(f"  SH thresholds: alpha<{SH_DD_ALPHA:.0%}  "
          f"hybrid<{SH_DD_HYBRID:.0%}  else MVO")

    _ensure_tables()

    if mode == 'rebuild':
        # Wipe everything — standard + excl + triggers
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                DELETE FROM {DAILY_PORT_TBL}
                WHERE params_hash=:ph AND model_version=:mv
            """), {'ph': ph, 'mv': model_version})
            conn.execute(text(f"""
                DELETE FROM {DAILY_TRIGGER_TBL}
                WHERE params_hash=:ph AND model_version=:mv
            """), {'ph': ph, 'mv': model_version})
        print("  Cache wiped (mode=rebuild)")
        cached = set()
    else:
        # incremental — find what's already cached
        cached = _get_cached_dates(ph, model_version)
        print(f"  Already cached: {len(cached)} dates")

    # All trading days since MB_START_DATE
    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]
    pxs_cols  = set(Pxs_df.columns)

    all_dates   = Pxs_df.index
    st_loc      = all_dates.searchsorted(MB_START_DATE)
    ext_loc     = max(0, st_loc - MOM_LONG_BUFFER)
    ext_st      = all_dates[ext_loc]
    daily_dates = all_dates[all_dates >= MB_START_DATE]
    last_date   = daily_dates[-1]   # always recompute today
    to_compute  = sorted([d for d in daily_dates if d not in cached or d == last_date])

    print(f"\n  Total trading days : {len(daily_dates)}")
    print(f"  Dates to compute   : {len(to_compute)}")

    if not to_compute:
        print("\n  OK Cache fully up to date")
        return ph

    # -- [1/3] Composite scores ------------------------------------------------
    print("\n  [1/3] Building composite scores...")
    dates_idx = pd.DatetimeIndex(to_compute)
    # Standard composite — always penalty-free (feeds strategies 1-8)
    with _SuppressOutput():
        composite_by_date, _ = _cb_build_composite_scores(
            universe          = get_universe(Pxs_df, sectors_s, ext_st),
            calc_dates        = dates_idx,
            Pxs_df            = Pxs_df,
            sectors_s         = sectors_s,
            weights_by_year   = weights_by_year,
            regime_s          = regime_s,
            volumeTrd_df      = volumeTrd_df,
            model_version     = model_version,
            exclude_factors   = ['OU'],
            weights_by_date   = weights_by_date,
            mr_scores_by_date = None,   # never penalise standard portfolios
        )
    print(f"  Composite scores: {len(composite_by_date)} dates")
    # Penalised composite — only built when MR_K > 0, feeds excl portfolios only
    _mr_s_cache = None
    composite_penalised = {}
    if MR_K > 0:
        print(f"  Building penalised composite (MR_K={MR_K}) for excl portfolios...")
        _mr_s_cache = _load_mr_scores_by_date()
        with _SuppressOutput():
            composite_penalised, _ = _cb_build_composite_scores(
                universe          = get_universe(Pxs_df, sectors_s, ext_st),
                calc_dates        = dates_idx,
                Pxs_df            = Pxs_df,
                sectors_s         = sectors_s,
                weights_by_year   = weights_by_year,
                regime_s          = regime_s,
                volumeTrd_df      = volumeTrd_df,
                model_version     = model_version,
                exclude_factors   = ['OU'],
                weights_by_date   = weights_by_date,
                mr_scores_by_date = _mr_s_cache,
            )
        print(f"  Penalised composite: {len(composite_penalised)} dates")

    # -- [2/3] X snapshots ----------------------------------------------------
    print("  [2/3] Loading X snapshots from cache...")
    with _SuppressOutput():
        X_snapshots, _, _, _ = _mb_build_x_snapshots(
            to_compute, Pxs_df, sectors_s,
            volumeTrd_df, model_version, False
        )
    snapshot_dates = sorted(X_snapshots.keys())
    print(f"  X snapshots: {len(snapshot_dates)}")

    # -- [3/3] Per-day computation ---------------------------------------------
    print(f"\n  [3/3] Computing daily portfolios...")
    print(f"  {'Date':<14} {'TO':>6} {'DVol':>7}  {'Elapsed':>7}  {'ETA':>7}")
    print(f"  {'-'*50}")

    n_cands    = top_n * universe_mult
    n_saved    = 0
    n_errors   = 0
    t_start    = time.time()

    # State trackers
    w_live     = pd.Series(dtype=float)   # drifted live portfolio for TO/vol computation
    w_hyb_prev = pd.Series(dtype=float)   # previous hybrid weights (for hybrid computation)
    sh_regime  = 'alpha'

    for idx, dt in enumerate(to_compute):
        if dt not in composite_by_date:
            continue

        comp_scores = composite_by_date[dt]          # clean — feeds strategies 1-8
        cands       = comp_scores.dropna()
        cands       = cands.loc[[t for t in cands.index if t in pxs_cols]]
        n_pf        = max(n_cands, int(np.ceil(len(cands) * prefilt_pct)))

        # Penalised composite for excl portfolios (strategy 9)
        comp_scores_pen = composite_penalised.get(dt, comp_scores) if composite_penalised else comp_scores
        cands       = cands.nlargest(min(n_pf, len(cands)))

        if len(cands) < top_n:
            continue

        valid_snap = [d for d in snapshot_dates if d <= dt]
        if not valid_snap:
            continue

        # -- Alpha -------------------------------------------------------------
        ranked     = cands.sort_values(ascending=False)
        alpha_port = [t for t in ranked.head(top_n).index if t in pxs_cols]
        n_p        = len(alpha_port)
        n_top_h    = int(np.ceil(n_p / 2))
        n_bot_h    = n_p - n_top_h
        if conc_factor == 1.0 or n_bot_h == 0:
            w_alpha = pd.Series(1.0 / n_p, index=alpha_port)
        else:
            top_a = conc_factor / (conc_factor + 1.0)
            bot_a = 1.0 / (conc_factor + 1.0)
            w_alpha = pd.Series({
                t: (top_a / n_top_h if j < n_top_h else bot_a / n_bot_h)
                for j, t in enumerate(alpha_port)
            })

        # -- MVO ---------------------------------------------------------------
        try:
            with _SuppressOutput():
                w_mvo, _ = _mb_solve_mvo(
                    dt=dt, candidates=cands.index.tolist()[:n_cands],
                    composite_scores=comp_scores, Pxs_df=Pxs_df,
                    sectors_s=sectors_s, volumeTrd_df=volumeTrd_df,
                    model_version=model_version,
                    pca_var_threshold=pca_var_threshold,
                    ic=ic, max_weight=max_weight, min_weight=min_weight,
                    zscore_cap=zscore_cap, risk_aversion=risk_aversion,
                    X_snapshots=X_snapshots, snapshot_dates=snapshot_dates,
                    top_n=top_n, min_cov_matrices=min_cov_matrices,
                )
        except Exception:
            n_errors += 1
            continue

        if w_mvo.empty or w_mvo.sum() == 0:
            continue
        w_mvo_nz = w_mvo[w_mvo > 1e-6]

        # -- Hybrid ------------------------------------------------------------
        all_t    = list(set(w_alpha.index) | set(w_mvo_nz.index))
        w_hybrid = (w_alpha.reindex(all_t).fillna(0.0) +
                    w_mvo_nz.reindex(all_t).fillna(0.0)) / 2.0
        w_hybrid = w_hybrid[w_hybrid > 0]
        if w_hybrid.sum() > 0:
            w_hybrid = w_hybrid / w_hybrid.sum()

        w_hyb_prev = w_hybrid.copy()

        # -- Trigger variables -------------------------------------------------
        # Drift w_live by today's price moves for accurate TO computation.
        # w_live resets to w_alpha on rebalance (proxy — regime selected in backtest).
        if not w_live.empty and idx > 0:
            prev_dt_live = to_compute[idx - 1]
            tks_live = [t for t in w_live.index if t in pxs_cols]
            if tks_live and prev_dt_live in Pxs_df.index and dt in Pxs_df.index:
                px_prev_live = Pxs_df.loc[prev_dt_live, tks_live]
                px_cur_live  = Pxs_df.loc[dt,           tks_live]
                w_drifted = w_live.reindex(tks_live) * (px_cur_live / px_prev_live).fillna(1)
                if w_drifted.sum() > 0:
                    w_live = w_drifted / w_drifted.sum()

        # Implied turnover: use alpha as the reference portfolio
        # (regime selection happens in run_mvo_backtest, not here)
        if w_live.empty:
            implied_to = 1.0
        else:
            all_t_to   = list(set(w_alpha.index) | set(w_live.index))
            implied_to = (w_alpha.reindex(all_t_to).fillna(0) -
                          w_live.reindex(all_t_to).fillna(0)).abs().sum() / 2

        # Vol difference: alpha vs live
        vol_new  = _portfolio_vol(w_alpha, Pxs_df, dt)
        vol_live = _portfolio_vol(w_live,  Pxs_df, dt) if not w_live.empty else vol_new
        vol_diff = vol_new - vol_live

        # Update live portfolio — reset to alpha on TO trigger, drift otherwise
        _to_thr  = DYN_TO_THRESHOLD_ALPHA
        if implied_to > _to_thr or w_live.empty:
            w_live = w_alpha.copy()

        # -- ADVP filter with replacement --------------------------------------
        # Apply liquidity filter using correct AUM at this date.
        # Stocks below min_weight capacity are dropped and replaced by the
        # next-best ranked liquid candidate from the composite score universe.
        def _advp_filter_replace(w: pd.Series, label: str) -> pd.Series:
            if volumeRaw_df is None or w.empty:
                return w
            _aum = float(aum_by_date[aum_by_date.index <= dt].iloc[-1])                    if (aum_by_date is not None and not aum_by_date.empty
                       and len(aum_by_date[aum_by_date.index <= dt]) > 0) \
                   else AUM
            past_d = [d for d in Pxs_df.index if d <= dt][-ADV_WINDOW:]

            def _liquid(tkr):
                if tkr not in Pxs_df.columns or tkr not in volumeRaw_df.columns:
                    return True
                px_s  = Pxs_df.loc[past_d, tkr].dropna()
                vol_s = volumeRaw_df.loc[past_d, tkr].dropna()
                com   = px_s.index.intersection(vol_s.index)
                if len(com) < 3: return True
                dv = (px_s.reindex(com) * vol_s.reindex(com).replace({0: np.nan})).median()
                return (dv * advp_cap) / _aum >= min_weight

            liquid  = [t for t in w.index if _liquid(t)]
            removed = [t for t in w.index if not _liquid(t)]
            if not removed:
                return w

            # Replace removed stocks with next-best liquid candidates from composite
            all_cands_ranked = cands.sort_values(ascending=False)
            replacements = []
            for tkr in all_cands_ranked.index:
                if len(liquid) + len(replacements) >= top_n:
                    break
                if tkr in liquid or tkr in removed or tkr not in pxs_cols:
                    continue
                if _liquid(tkr):
                    replacements.append(tkr)

            final = liquid + replacements
            if not final:
                return w  # no change if nothing survives

            # Equal weight replacements, preserve relative weights of survivors
            n_final   = len(final)
            n_liquid  = len(liquid)
            n_rep     = len(replacements)
            if n_rep == 0:
                # Just renormalise survivors
                w_new = w.reindex(liquid)
                return w_new / w_new.sum() if w_new.sum() > 0 else w_new
            # Survivors keep their relative weights; replacements get equal share
            survivor_share = n_liquid / n_final
            replace_share  = n_rep    / n_final
            w_surv = w.reindex(liquid)
            if w_surv.sum() > 0:
                w_surv = w_surv / w_surv.sum() * survivor_share
            w_rep = pd.Series(replace_share / n_rep, index=replacements)
            w_out = pd.concat([w_surv, w_rep])
            return w_out / w_out.sum() if w_out.sum() > 0 else w_out

        w_alpha  = _advp_filter_replace(w_alpha,  'alpha')
        w_mvo_nz = _advp_filter_replace(w_mvo_nz, 'mvo')
        # Recompute hybrid after filtering alpha and mvo
        all_t    = list(set(w_alpha.index) | set(w_mvo_nz.index))
        w_hybrid = (w_alpha.reindex(all_t).fillna(0.0) +
                    w_mvo_nz.reindex(all_t).fillna(0.0)) / 2.0
        w_hybrid = w_hybrid[w_hybrid > 0]
        if w_hybrid.sum() > 0:
            w_hybrid = w_hybrid / w_hybrid.sum()

        # -- Save --------------------------------------------------------------
        try:
            # Save candidates list (full ranked universe for ADVP filtering at runtime)
            _cands_to_save = cands.sort_values(ascending=False)
            _save_portfolios(dt, model_version, ph, {
                'alpha'     : w_alpha,
                'mvo'       : w_mvo_nz,
                'hybrid'    : w_hybrid,
                'candidates': _cands_to_save,  # composite scores, not weights
            })
            _save_triggers(dt, model_version, ph, implied_to, vol_diff)
            n_saved += 1
        except Exception as e:
            warnings.warn(f"  Save failed {dt.date()}: {e}")
            n_errors += 1
            continue

        # Excl portfolios — isolated block, never affects standard save
        try:
            _mr_n = int(MR_CAP * len(cands)) if MR_CAP > 0 else 0
            excl_d = _load_momentum_exclusions(dt, top_n=_mr_n) if _mr_n > 0 else {}
            # Use penalised composite for excl candidate ranking
            cands_pen = comp_scores_pen.dropna()
            cands_pen = cands_pen.loc[[t for t in cands_pen.index if t in pxs_cols]]
            cands_pen = cands_pen.nlargest(min(max(n_cands, int(np.ceil(len(cands_pen) * prefilt_pct))), len(cands_pen)))
            if excl_d:
                ce = cands_pen[~cands_pen.index.isin(excl_d.keys())]
            else:
                ce = cands_pen
            if len(ce) >= top_n and (excl_d or MR_K > 0):
                re_ = ce.sort_values(ascending=False)
                ap_ = [t for t in re_.head(top_n).index if t in pxs_cols]
                n_  = len(ap_)
                if conc_factor == 1.0 or n_ < 2:
                    w_ae = pd.Series(1.0/n_, index=ap_)
                else:
                    nt_ = int(np.ceil(n_/2)); nb_ = n_-nt_
                    ta_ = conc_factor/(conc_factor+1.0); ba_ = 1.0/(conc_factor+1.0)
                    w_ae = pd.Series({t:(ta_/nt_ if j<nt_ else ba_/nb_)
                                      for j,t in enumerate(ap_)})
                try:
                    with _SuppressOutput():
                        w_me,_ = _mb_solve_mvo(
                            dt=dt, candidates=ce.index.tolist()[:n_cands],
                            composite_scores=comp_scores_pen, Pxs_df=Pxs_df,
                            sectors_s=sectors_s, volumeTrd_df=volumeTrd_df,
                            model_version=model_version,
                            pca_var_threshold=pca_var_threshold,
                            ic=ic, max_weight=max_weight, min_weight=min_weight,
                            zscore_cap=zscore_cap, risk_aversion=risk_aversion,
                            X_snapshots=X_snapshots, snapshot_dates=snapshot_dates,
                            top_n=top_n, min_cov_matrices=min_cov_matrices,
                        )
                    w_me_nz = w_me[w_me > 1e-6]
                    at_ = list(set(w_ae.index)|set(w_me_nz.index))
                    w_he = (w_ae.reindex(at_).fillna(0)+w_me_nz.reindex(at_).fillna(0))/2.0
                    w_he = w_he[w_he > 0]
                    if w_he.sum() > 0: w_he = w_he/w_he.sum()
                except Exception:
                    w_me_nz = w_mvo_nz.copy(); w_he = w_hybrid.copy(); w_ae = w_alpha.copy()
            else:
                w_ae = w_alpha.copy(); w_me_nz = w_mvo_nz.copy(); w_he = w_hybrid.copy()
            _save_portfolios(dt, model_version, ph,
                             {'alpha_excl':w_ae,'mvo_excl':w_me_nz,'hybrid_excl':w_he})
        except Exception as e:
            warnings.warn(f"  Excl save failed {dt.date()}: {e}")

        # Progress every 50 dates
        if idx % 50 == 0 or idx == len(to_compute) - 1:
            elapsed   = time.time() - t_start
            rate      = elapsed / max(idx + 1, 1)
            remaining = rate * (len(to_compute) - idx - 1)
            print(f"  {str(dt.date()):<14} "
                  f"{implied_to*100:>5.1f}%  {vol_diff*100:>+6.1f}%  "
                  f"{elapsed:>5.0f}s  {remaining:>5.0f}s")

    elapsed_total = time.time() - t_start
    print(f"\n  OK Done: {n_saved} dates saved, {n_errors} errors  "
          f"({elapsed_total/60:.1f} min)")
    print(f"  params_hash = '{ph}'")
    print(f"\n  Load trigger data with:")
    print(f"    triggers = pd.read_sql(")
    print(f"        \"SELECT * FROM {DAILY_TRIGGER_TBL} WHERE params_hash='{ph}'\",")
    print(f"        ENGINE, index_col='date', parse_dates=['date'])")
    return ph


def _run_excl_only_cache_build(
    ph, model_version, Pxs_df, sectors_s,
    weights_by_year, regime_s, volumeTrd_df,
    ic, max_weight, min_weight, zscore_cap,
    pca_var_threshold, universe_mult, risk_aversion,
    top_n, conc_factor, prefilt_pct, min_cov_matrices,
    weights_by_date=None,
):
    """Rebuild only excl portfolios. Standard portfolios + triggers untouched."""
    import time
    _ensure_tables()
    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]
    pxs_cols  = set(Pxs_df.columns)

    existing  = load_daily_portfolios(ph, model_version)
    std_dates = sorted([dt for dt,v in existing.items() if 'alpha' in v])
    excl_done = {dt for dt,v in existing.items() if 'alpha_excl' in v}
    last_date = std_dates[-1] if std_dates else None
    # Always recompute last date; rebuild any missing excl dates
    to_build  = [dt for dt in std_dates if dt not in excl_done or dt == last_date]
    print(f"  Standard portfolios: {len(std_dates)}  Excl cached: {len(excl_done)}  To build: {len(to_build)}")
    if not to_build:
        print("  OK — excl portfolios fully up to date"); return ph

    all_dates = Pxs_df.index
    ext_st    = all_dates[max(0, all_dates.searchsorted(MB_START_DATE) - MOM_LONG_BUFFER)]
    # Excl-only always needs the penalised composite (that's the whole point)
    _mr_s_eo = _load_mr_scores_by_date() if MR_K > 0 else None
    with _SuppressOutput():
        composite_penalised_eo, _ = _cb_build_composite_scores(
            universe=get_universe(Pxs_df, sectors_s, ext_st),
            calc_dates=pd.DatetimeIndex(to_build), Pxs_df=Pxs_df,
            sectors_s=sectors_s, weights_by_year=weights_by_year,
            regime_s=regime_s, volumeTrd_df=volumeTrd_df,
            model_version=model_version, exclude_factors=['OU'],
            weights_by_date=None, mr_scores_by_date=_mr_s_eo,
        )
    with _SuppressOutput():
        X_snapshots,_,_,_ = _mb_build_x_snapshots(
            to_build, Pxs_df, sectors_s, volumeTrd_df, model_version, False)
    snapshot_dates = sorted(X_snapshots.keys())
    n_cands = top_n * universe_mult
    t0 = time.time(); n_saved = n_errors = 0

    for idx, dt in enumerate(to_build):
        std = existing.get(dt, {})
        w_a = std.get('alpha', pd.Series(dtype=float))
        w_m = std.get('mvo',   pd.Series(dtype=float))
        w_h = std.get('hybrid',pd.Series(dtype=float))
        if w_a.empty: continue
        try:
            cs = composite_penalised_eo.get(dt, pd.Series(dtype=float))
            if cs.empty: raise ValueError("no scores")
            cands = cs.reindex([t for t in cs.index if t in pxs_cols]).dropna().sort_values(ascending=False)
            if len(cands) < top_n: raise ValueError("insufficient candidates")
        except Exception:
            _save_portfolios(dt, model_version, ph,
                             {'alpha_excl':w_a.copy(),'mvo_excl':w_m.copy(),'hybrid_excl':w_h.copy()})
            n_saved += 1; continue
        try:
            _mr_n = int(MR_CAP * len(cands)) if MR_CAP > 0 else 0
            excl_d = _load_momentum_exclusions(dt, top_n=_mr_n) if _mr_n > 0 else {}
            ce = cands[~cands.index.isin(excl_d.keys())] if excl_d else cands
            # Rebuild if there are hard exclusions OR if MR_K > 0 (penalised composite differs)
            if len(ce) >= top_n and (excl_d or MR_K > 0):
                re_ = ce.sort_values(ascending=False)
                ap_ = [t for t in re_.head(top_n).index if t in pxs_cols]
                n_  = len(ap_)
                if conc_factor == 1.0 or n_ < 2:
                    w_ae = pd.Series(1.0/n_, index=ap_)
                else:
                    nt_ = int(np.ceil(n_/2)); nb_ = n_-nt_
                    ta_ = conc_factor/(conc_factor+1.0); ba_ = 1.0/(conc_factor+1.0)
                    w_ae = pd.Series({t:(ta_/nt_ if j<nt_ else ba_/nb_) for j,t in enumerate(ap_)})
                try:
                    with _SuppressOutput():
                        w_me,_ = _mb_solve_mvo(
                            dt=dt, candidates=ce.index.tolist()[:n_cands],
                            composite_scores=cs, Pxs_df=Pxs_df,
                            sectors_s=sectors_s, volumeTrd_df=volumeTrd_df,
                            model_version=model_version, pca_var_threshold=pca_var_threshold,
                            ic=ic, max_weight=max_weight, min_weight=min_weight,
                            zscore_cap=zscore_cap, risk_aversion=risk_aversion,
                            X_snapshots=X_snapshots, snapshot_dates=snapshot_dates,
                            top_n=top_n, min_cov_matrices=min_cov_matrices,
                        )
                    w_me_nz = w_me[w_me>1e-6]
                    at_ = list(set(w_ae.index)|set(w_me_nz.index))
                    w_he = (w_ae.reindex(at_).fillna(0)+w_me_nz.reindex(at_).fillna(0))/2.0
                    w_he = w_he[w_he>0]
                    if w_he.sum() > 0: w_he = w_he/w_he.sum()
                except Exception:
                    w_me_nz=w_m.copy(); w_he=w_h.copy(); w_ae=w_a.copy()
            else:
                w_ae=w_a.copy(); w_me_nz=w_m.copy(); w_he=w_h.copy()
            _save_portfolios(dt, model_version, ph,
                             {'alpha_excl':w_ae,'mvo_excl':w_me_nz,'hybrid_excl':w_he})
            n_saved += 1
        except Exception as e:
            warnings.warn(f"  Excl failed {dt.date()}: {e}"); n_errors += 1
        if idx % 100 == 0 or idx == len(to_build)-1:
            el = time.time()-t0; eta = el/max(idx+1,1)*(len(to_build)-idx-1)
            print(f"  [{idx+1}/{len(to_build)}] {dt.date()}  {el:.0f}s  eta={eta:.0f}s", end='\r')

    print(f"\n  OK: {n_saved} excl dates saved, {n_errors} errors ({(time.time()-t0)/60:.1f} min)")
    print(f"  Standard portfolios: UNTOUCHED")
    return ph


# ===============================================================================
# ENTRY POINT
# ===============================================================================

def run_mvo_backtest(Pxs_df, sectors_s, weights_by_year, regime_s,
                     volumeTrd_df=None,
                     volumeRaw_df=None,          # raw share volume for ADVP filter
                     weights_by_date=None,       # point-in-time factor weights (preferred)
                     # MVO parameters (not user prompts)
                     ic=0.04,
                     max_weight=0.10,
                     min_weight=0.025,
                     zscore_cap=2.5,
                     pca_var_threshold=0.65,
                     universe_mult=5,
                     risk_aversion=1.0,
                     # Deprecated — use run_daily_cache_build(mode='rebuild') instead
                     force_rebuild_cache=False,
                     portfolio_cache_override=False,
                     # Hedge layer (optional)
                     hedge_multi=None,      # output of run_macro_hedge_cached
                     hedges_l=None,         # list of hedge instrument tickers
                     hedge_trigger_assets=None):  # default ['QQQ','SPX']
    """
    Run the full MVO backtest (strategies 1-9).
    Always reads portfolios from the cache built by run_daily_cache_build().

    Parameters
    ----------
    Pxs_df, sectors_s, weights_by_year, regime_s, volumeTrd_df
        -- same as composite_backtest.py
    weights_by_date : point-in-time factor weights from compute_rolling_regime_weights()
    ic              : Grinold-Kahn IC scaling (default 0.04)
    max_weight      : single-name cap (default 0.10)
    min_weight      : single-name floor (default 0.025)
    zscore_cap      : alpha z-score winsorization (default 2.5)
    pca_var_threshold : PCA variance explained threshold (default 0.65)
    universe_mult   : candidate pool = port_n x universe_mult (default 5)
    risk_aversion   : MVO risk aversion lambda (default 1.0)

    Note: force_rebuild_cache and portfolio_cache_override are deprecated.
          Use run_daily_cache_build(mode='rebuild') to rebuild the cache.
    """
    if force_rebuild_cache or portfolio_cache_override:
        print("  WARNING: force_rebuild_cache and portfolio_cache_override are deprecated.")
        print("           Use run_daily_cache_build(mode='rebuild') instead.")

    print("=" * 72)
    print("  MVO BACKTEST -- Baseline vs Pure Alpha vs MVO")
    print("=" * 72)
    print(f"\n  MVO params: IC={ic}, max_w={max_weight}, min_w={min_weight}, "
          f"zscore_cap={zscore_cap}")
    print(f"  PCA thresh={pca_var_threshold}, universe_mult={universe_mult}, "
          f"risk_aversion={risk_aversion}")
    print(f"  MR_K={MR_K}  MR_CAP={MR_CAP}  QUALITY_FLOOR={QUALITY_FLOOR}\n")

    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]

    # De-trend raw daily volume: divide by rolling mean to get relative volume
    # volumeTrd_df input is expected to be RAW daily traded volume per stock
    if volumeTrd_df is not None:
        volumeRaw_df = volumeTrd_df.copy()   # preserve raw volume for ADV cap
        vol_roll     = (volumeTrd_df
                        .replace({0: np.nan})
                        .fillna(method='ffill')
                        .fillna(0)
                        .rolling(VOLUME_WINDOW).mean())
        volumeTrd_df = volumeTrd_df / vol_roll.replace({0: np.nan})
    else:
        volumeRaw_df = None

    # -- User prompts (identical to composite_backtest.py) ---------------------
    print("  PORTFOLIO CONSTRUCTION OPTIONS:")
    topn_input    = input(f"  Number of stocks [default={MB_TOP_N}]: ").strip()
    rebal_input   = input(f"  Rebalancing frequency in days [default={MB_REBAL_FREQ}]: ").strip()
    sector_input  = input("  Max stocks per sector (or Enter to skip): ").strip()
    mktcap_input  = input("  Min market cap floor ($M, or Enter to skip): ").strip()
    advp_input    = input(f"  ADVP cap — max % of median daily $ volume per stock [default=4%]: ").strip()
    vol_input     = input("  Apply vol filter? (y/n) [default=n]: ").strip().lower()
    prefilt_input = input("  Pre-filter fraction by composite score 0<x<=1 (or Enter for none): ").strip()
    conc_input    = input("  Concentration factor for Pure Alpha >=1.0 (or Enter for equal weight): ").strip()

    top_n        = int(topn_input)           if topn_input    else MB_TOP_N
    rebal_freq   = int(rebal_input)          if rebal_input   else MB_REBAL_FREQ
    sector_cap   = int(sector_input)         if sector_input  else None
    mktcap_floor = float(mktcap_input) * 1e6 if mktcap_input  else None
    advp_cap     = float(advp_input) / 100   if advp_input    else 0.04
    use_vol      = vol_input == 'y'
    prefilt_pct  = float(prefilt_input)      if prefilt_input else 1.0
    if prefilt_pct <= 0 or prefilt_pct > 1:
        prefilt_pct = 1.0
    conc_factor  = float(conc_input) if conc_input else 1.0
    if conc_factor < 1.0:
        conc_factor = 1.0

    min_cov_inp      = input(f"  Min cov matrices for stock selection "
                             f"(0=alpha-only, default={MB_MIN_COV_MATRICES}): ").strip()
    min_cov_matrices = int(min_cov_inp) if min_cov_inp else MB_MIN_COV_MATRICES
    min_cov_matrices = max(0, min(min_cov_matrices, 4))

    enforce_hybrid_floor = False

    n_cands = top_n * universe_mult
    print(f"\n  Settings: N={top_n}, rebal={rebal_freq}d, "
          f"sector_cap={sector_cap}, mktcap_floor={mktcap_input or 'none'}, "
          f"advp_cap={advp_cap:.1%} (AUM=${AUM:,.0f}), "
          f"vol_filter={use_vol}, prefilt={prefilt_pct:.0%}, "
          f"conc={conc_factor:.1f}x (pure alpha only)")
    print(f"  MVO candidate pool: {n_cands} stocks ({universe_mult}x{top_n})  |  "
          f"min_cov_matrices={min_cov_matrices}\n")

    # -- Daily portfolio cache setup -------------------------------------------
    params_hash = _make_make_params_hash(
        ic, max_weight, min_weight, zscore_cap, pca_var_threshold,
        universe_mult, risk_aversion, top_n, conc_factor,
        prefilt_pct, min_cov_matrices, MB_MODEL_VER,
    )
    print(f"  Portfolio cache: params_hash={params_hash}")
    _ensure_daily_cache_table()
    cached_port_dates = _get_cached_portfolio_dates(params_hash, MB_MODEL_VER)
    print(f"  Portfolio cache: {len(cached_port_dates)} dates already cached\n")

    # -- Universe and calc dates -----------------------------------------------
    all_dates      = Pxs_df.index
    st_dt_loc      = all_dates.searchsorted(MB_START_DATE)
    ext_loc        = max(0, st_dt_loc - MOM_LONG_BUFFER)
    extended_st_dt = all_dates[ext_loc]
    universe       = get_universe(Pxs_df, sectors_s, extended_st_dt)
    calc_dates     = generate_calc_dates(Pxs_df, step_days=rebal_freq)
    calc_dates_idx = pd.DatetimeIndex(calc_dates)
    pxs_cols       = set(Pxs_df.columns)

    print(f"  Universe: {len(universe)} stocks  |  "
          f"Rebalance dates: {len(calc_dates)}")

    # -- [1/4] Composite alpha scores -----------------------------------------
    print("\n[1/4] Building composite alpha scores...")
    # Standard composite — always penalty-free (feeds strategies 1-8)
    _mr_scores_by_date = None
    composite_by_date, _ = _cb_build_composite_scores(
        universe          = universe,
        calc_dates        = calc_dates_idx,
        Pxs_df            = Pxs_df,
        sectors_s         = sectors_s,
        weights_by_year   = weights_by_year,
        regime_s          = regime_s,
        volumeTrd_df      = volumeTrd_df,
        model_version     = MB_MODEL_VER,
        exclude_factors   = ['OU'],
        weights_by_date   = weights_by_date,
        mr_scores_by_date = None,   # never penalise standard portfolios
    )
    # Penalised composite for strategy 9 only
    _composite_penalised = {}
    if MR_K > 0:
        print(f"  MR_K={MR_K} — building penalised composite for strategy 9...")
        _mr_scores_by_date = _load_mr_scores_by_date()
        print(f"  MR penalty scores: {len(_mr_scores_by_date)} dates with score > 0")
        _composite_penalised, _ = _cb_build_composite_scores(
            universe          = universe,
            calc_dates        = calc_dates_idx,
            Pxs_df            = Pxs_df,
            sectors_s         = sectors_s,
            weights_by_year   = weights_by_year,
            regime_s          = regime_s,
            volumeTrd_df      = volumeTrd_df,
            model_version     = MB_MODEL_VER,
            exclude_factors   = ['OU'],
            weights_by_date   = weights_by_date,
            mr_scores_by_date = _mr_scores_by_date,
        )

    # -- [2/4] X snapshots (cached) --------------------------------------------
    print("\n[2/4] Building X snapshots (monthly, cached)...")
    X_snapshots, _, _, _ = _mb_build_x_snapshots(
        calc_dates, Pxs_df, sectors_s,
        volumeTrd_df, MB_MODEL_VER, force_rebuild_cache
    )
    snapshot_dates = sorted(X_snapshots.keys())
    if snapshot_dates:
        print(f"  X snapshots available: {len(snapshot_dates)}  "
              f"({snapshot_dates[0].date()} -> {snapshot_dates[-1].date()})")

    # -- [3/4] Compute portfolio weights per rebalance date --------------------
    print("\n[3/4] Computing portfolio weights...")

    alpha_weights_by_date  = {}   # pure alpha (equal/concentration)
    mvo_weights_by_date    = {}   # MVO
    hybrid_weights_by_date = {}   # hybrid (built incrementally for turnover display)
    quality_factor_by_date = {}   # baseline
    all_stock_returns      = []   # pool of individual stock returns across all periods
    _last_mvo_diag         = {}   # last known MVO diagnostics for dynamic display
    _diag_by_date          = {}   # {date: mvo_diag} for each rebal date
    # Per-strategy turnover tracking for cost calculation
    _prev_w = {"baseline": pd.Series(dtype=float),
               "alpha":    pd.Series(dtype=float),
               "mvo":      pd.Series(dtype=float),
               "hybrid":   pd.Series(dtype=float),
               "smart":    pd.Series(dtype=float)}
    _cost_by_date = {"baseline": {}, "alpha": {}, "mvo": {},
                     "hybrid": {}, "smart": {}}
    all_turnover_ratios    = []   # turnover % at each rebalance
    all_portfolio_returns        = []   # hybrid portfolio return per holding period
    smart_hybrid_weights_by_date = {}   # drawdown-regime-aware hybrid
    _sh_nav_running = 1.0              # running NAV for smart hybrid regime detection
    _sh_hwm         = 1.0              # high-water mark for regime detection
    _sh_regime_counts = {"alpha": 0, "hybrid": 0, "mvo": 0}  # regime prevalence
    _sh_regime_by_date = {}   # {date: regime_str} for yearly breakdown

    # Running NAV per strategy for correct per-strategy ADVP AUM
    _running_nav        = 1.0   # alpha (strategies 1-2)
    _running_nav_mvo    = 1.0   # MVO (strategy 3)
    _running_nav_hybrid = 1.0   # hybrid (strategy 4)
    _running_nav_smart  = 1.0   # smart hybrid (strategy 5)
    _running_nav_dyn    = 1.0   # dynamic (strategies 6-9, uses last known dyn weights)
    _prev_rebal_dt      = None
    _advp_capped        = {}   # {dt: set of tickers where ADVP cap was binding}

    # Load quality scores for baseline
    print("  Loading quality scores for baseline...")
    all_tickers  = list(sectors_s.index)
    with _SuppressOutput():
        quality_wide = get_quality_scores(
            calc_dates         = calc_dates_idx,
            universe           = all_tickers,
            Pxs_df             = Pxs_df,
            sectors_s          = sectors_s,
            use_cached_weights = True,
            force_recompute    = False,
        )
    print(f"  Quality scores loaded: {len(quality_wide)} dates")

    n = len(calc_dates)
    for i, dt in enumerate(calc_dates):
        print(f"  Processing [{i+1}/{n}] {dt.date()}...", end='\r')

        # -- Update running NAVs per strategy for correct ADVP AUM -------------
        if _prev_rebal_dt is not None:
            _days_between = [d for d in Pxs_df.index
                             if _prev_rebal_dt <= d <= dt]
            for _di in range(1, len(_days_between)):
                _d0 = _days_between[_di - 1]
                _d1 = _days_between[_di]
                # Alpha
                _w_nav = alpha_weights_by_date.get(_prev_rebal_dt, pd.Series(dtype=float))
                _tks   = [t for t in _w_nav.index if t in Pxs_df.columns]
                if _tks:
                    _r = (_w_nav.reindex(_tks).fillna(0) *
                          (Pxs_df.loc[_d1, _tks] / Pxs_df.loc[_d0, _tks] - 1).fillna(0)).sum()
                    _running_nav *= (1 + _r)
                # MVO
                _w_nav = mvo_weights_by_date.get(_prev_rebal_dt, pd.Series(dtype=float))
                _tks   = [t for t in _w_nav.index if t in Pxs_df.columns]
                if _tks:
                    _r = (_w_nav.reindex(_tks).fillna(0) *
                          (Pxs_df.loc[_d1, _tks] / Pxs_df.loc[_d0, _tks] - 1).fillna(0)).sum()
                    _running_nav_mvo *= (1 + _r)
                # Hybrid
                _w_nav = hybrid_weights_by_date.get(_prev_rebal_dt, pd.Series(dtype=float))
                _tks   = [t for t in _w_nav.index if t in Pxs_df.columns]
                if _tks:
                    _r = (_w_nav.reindex(_tks).fillna(0) *
                          (Pxs_df.loc[_d1, _tks] / Pxs_df.loc[_d0, _tks] - 1).fillna(0)).sum()
                    _running_nav_hybrid *= (1 + _r)
                # Smart hybrid (uses same weights as alpha until smart hybrid is built)
                _w_nav = (smart_hybrid_weights_by_date.get(_prev_rebal_dt) or
                          alpha_weights_by_date.get(_prev_rebal_dt, pd.Series(dtype=float)))
                _tks   = [t for t in _w_nav.index if t in Pxs_df.columns]
                if _tks:
                    _r = (_w_nav.reindex(_tks).fillna(0) *
                          (Pxs_df.loc[_d1, _tks] / Pxs_df.loc[_d0, _tks] - 1).fillna(0)).sum()
                    _running_nav_smart *= (1 + _r)
                # Dynamic — uses last known dynamic weights (or alpha as fallback)
                try:
                    _dyn_dates = sorted([d for d in dyn_weights_by_date if d <= _d0])
                    _w_nav = (dyn_weights_by_date[_dyn_dates[-1]] if _dyn_dates
                              else alpha_weights_by_date.get(_prev_rebal_dt, pd.Series(dtype=float)))
                except NameError:
                    _w_nav = alpha_weights_by_date.get(_prev_rebal_dt, pd.Series(dtype=float))
                _tks   = [t for t in _w_nav.index if t in Pxs_df.columns]
                if _tks:
                    _r = (_w_nav.reindex(_tks).fillna(0) *
                          (Pxs_df.loc[_d1, _tks] / Pxs_df.loc[_d0, _tks] - 1).fillna(0)).sum()
                    _running_nav_dyn *= (1 + _r)
        current_aum         = AUM * _running_nav
        current_aum_mvo     = AUM * _running_nav_mvo
        current_aum_hybrid  = AUM * _running_nav_hybrid
        current_aum_smart   = AUM * _running_nav_smart
        current_aum_dyn     = AUM * _running_nav_dyn
        _prev_rebal_dt = dt
        current_aum = AUM * _running_nav

        # -- Baseline quality factor -------------------------------------------
        if dt in quality_wide.index:
            scores = quality_wide.loc[dt].dropna()
            if not scores.empty:
                fdf = scores.rename('factor').to_frame()
                fdf['Sector'] = fdf.index.map(sectors_s)
                fdf = fdf.dropna(subset=['Sector'])
                fdf = fdf.loc[[t for t in fdf.index if t in pxs_cols]]
                if not fdf.empty:
                    quality_factor_by_date[dt] = fdf

        if dt not in composite_by_date:
            continue

        comp_scores = composite_by_date[dt]

        # -- Candidate universe ------------------------------------------------
        cands = comp_scores.dropna()
        cands = cands.loc[[t for t in cands.index if t in pxs_cols]]

        # Vol filter
        if use_vol and len(cands) > top_n:
            surviving = apply_vol_filter(cands.index, dt, Pxs_df)
            cands = cands.reindex(surviving).dropna()

        # Pre-filter
        n_prefilt = max(n_cands, int(np.ceil(len(cands) * prefilt_pct)))
        cands = cands.nlargest(min(n_prefilt, len(cands)))

        # Sector cap pre-filter (for pure alpha path)
        if sector_cap is not None:
            sec_counts = {}
            cands_filtered = []
            for t in cands.index:
                sec = sectors_s.get(t, 'Unknown')
                if sec_counts.get(sec, 0) < sector_cap * universe_mult:
                    cands_filtered.append(t)
                    sec_counts[sec] = sec_counts.get(sec, 0) + 1
            cands = cands.reindex(cands_filtered).dropna()

        if len(cands) < top_n:
            continue

        candidates = cands.index.tolist()

        # -- ADVP liquidity filter: build liquid universe before alpha ranking --
        # Two thresholds:
        #   alpha_liquid: cap_w >= min_weight (stock must be holdable at floor)
        #   mvo_liquid:   cap_w >= min_weight / top_n (any non-trivial allocation)
        # For MVO: if fewer than n_cands liquid stocks in cands, search deeper
        # into the full alpha-ranked universe until n_cands reached or exhausted.
        current_aum_filt = AUM * _running_nav
        if volumeRaw_df is not None and current_aum_filt > 0:
            past_dates_filt  = [d for d in Pxs_df.index if d <= dt][-ADV_WINDOW:]
            mvo_min_w        = min_weight / top_n

            def _is_alpha_liquid(tkr):
                if tkr not in Pxs_df.columns or tkr not in volumeRaw_df.columns:
                    return True
                px_s   = Pxs_df.loc[past_dates_filt, tkr].dropna()
                vol_s  = volumeRaw_df.loc[past_dates_filt, tkr].dropna()
                common = px_s.index.intersection(vol_s.index)
                if len(common) < 3:
                    return True
                dv = (px_s.reindex(common) * vol_s.reindex(common).replace({0: np.nan})).median()
                return (dv * advp_cap) / current_aum_filt >= min_weight

            def _is_mvo_liquid(tkr):
                if tkr not in Pxs_df.columns or tkr not in volumeRaw_df.columns:
                    return True
                px_s   = Pxs_df.loc[past_dates_filt, tkr].dropna()
                vol_s  = volumeRaw_df.loc[past_dates_filt, tkr].dropna()
                common = px_s.index.intersection(vol_s.index)
                if len(common) < 3:
                    return True
                dv = (px_s.reindex(common) * vol_s.reindex(common).replace({0: np.nan})).median()
                return (dv * advp_cap) / current_aum_filt >= mvo_min_w

            # Alpha liquid: strict threshold, from pre-filtered cands only
            alpha_liquid_set = {t for t in cands.index if _is_alpha_liquid(t)}
            cands_liquid     = cands.reindex([t for t in cands.index
                                              if t in alpha_liquid_set])

            # MVO liquid: permissive threshold, search full universe if needed
            # Start from pre-filtered cands, then go deeper into full ranking
            full_ranked = comp_scores.dropna()
            full_ranked = full_ranked.loc[[t for t in full_ranked.index
                                           if t in pxs_cols]]
            full_ranked = full_ranked.sort_values(ascending=False)

            mvo_liquid_list = [t for t in cands.index if _is_mvo_liquid(t)]
            if len(mvo_liquid_list) < n_cands:
                # Search beyond pre-filtered cands
                already = set(cands.index)
                for tkr in full_ranked.index:
                    if len(mvo_liquid_list) >= n_cands:
                        break
                    if tkr in already:
                        continue
                    if _is_mvo_liquid(tkr):
                        mvo_liquid_list.append(tkr)
                        already.add(tkr)

            cands_liquid_mvo = full_ranked.reindex(
                [t for t in full_ranked.index if t in set(mvo_liquid_list)]
            ).dropna()
        else:
            cands_liquid     = cands
            cands_liquid_mvo = cands

        # -- Pure alpha weights (concentration) --------------------------------
        ranked  = cands_liquid.sort_values(ascending=False)
        if sector_cap is not None:
            top_sel = select_with_sector_cap(
                ranked.rename('factor').to_frame().assign(
                    Sector=ranked.index.map(sectors_s)
                ),
                sector_cap, top_n
            )
            alpha_port = top_sel.index.tolist()
        else:
            alpha_port = ranked.head(top_n).index.tolist()
        alpha_port = [t for t in alpha_port if t in pxs_cols]

        n_port   = len(alpha_port)
        n_top_h  = int(np.ceil(n_port / 2))
        n_bot_h  = n_port - n_top_h
        if conc_factor == 1.0 or n_bot_h == 0:
            alpha_w = {t: 1.0 / n_port for t in alpha_port}
        else:
            top_alloc = conc_factor / (conc_factor + 1.0)
            bot_alloc = 1.0 / (conc_factor + 1.0)
            alpha_w   = {}
            for j, t in enumerate(alpha_port):
                alpha_w[t] = (top_alloc / n_top_h if j < n_top_h
                              else bot_alloc / n_bot_h)
        alpha_weights_by_date[dt] = pd.Series(alpha_w)
        # Apply ADVP cap to alpha weights
        if volumeRaw_df is not None and advp_cap > 0:
            w_alpha_capped, capped_a = _apply_advp_cap(
                pd.Series(alpha_w), dt, Pxs_df, volumeRaw_df,
                current_aum, advp_cap, min_weight, max_weight)
            if capped_a:
                alpha_weights_by_date[dt] = w_alpha_capped
                _advp_capped[dt] = _advp_capped.get(dt, set()) | capped_a

        # Update running NAV from last rebal to now using previous alpha weights
        if _prev_rebal_dt is not None and _prev_rebal_dt in alpha_weights_by_date:
            w_prev_alpha = alpha_weights_by_date[_prev_rebal_dt]
            prev_px_dates = [d for d in Pxs_df.index
                             if _prev_rebal_dt <= d <= dt]
            if len(prev_px_dates) >= 2:
                tks_nav = [t for t in w_prev_alpha.index if t in Pxs_df.columns]
                if tks_nav:
                    px_start = Pxs_df.loc[prev_px_dates[0],  tks_nav]
                    px_end   = Pxs_df.loc[prev_px_dates[-1], tks_nav]
                    period_ret = (w_prev_alpha.reindex(tks_nav).fillna(0) *
                                  (px_end / px_start - 1).fillna(0)).sum()
                    _running_nav *= (1 + period_ret)
        _prev_rebal_dt = dt

        # Apply ADVP cap to alpha weights
        current_aum = AUM * _running_nav
        w_alpha_s   = pd.Series(alpha_w)
        w_alpha_s, capped_alpha = _apply_advp_cap(
            w_alpha_s, dt, Pxs_df, volumeRaw_df,
            current_aum, advp_cap, min_weight, max_weight)
        alpha_weights_by_date[dt] = w_alpha_s
        if capped_alpha:
            _advp_capped[dt] = _advp_capped.get(dt, set()) | capped_alpha
        # Trading cost for alpha
        w_new_a = pd.Series(alpha_w)
        w_old_a = _prev_w["alpha"]
        all_t_a = list(set(w_new_a.index) | set(w_old_a.index))
        to_a    = (w_new_a.reindex(all_t_a).fillna(0) -
                   w_old_a.reindex(all_t_a).fillna(0)).abs().sum() / 2
        _cost_by_date["alpha"][dt] = to_a * TRADING_COST_BPS / 10000
        _prev_w["alpha"] = w_new_a

        # -- MVO weights -------------------------------------------------------
        # Use top n_cands from broader MVO liquid universe
        candidates_liquid = cands_liquid_mvo.index.tolist()
        mvo_cands = candidates_liquid[:n_cands]
        valid_snap = [d for d in snapshot_dates if d <= dt]
        if not valid_snap:
            continue

        # Load from cache if available (skip MVO solve for weights, keep for diagnostics)
        if dt in cached_port_dates:
            import json
            try:
                with ENGINE.connect() as conn:
                    rows = conn.execute(text(f"""
                        SELECT strategy, weights_json FROM {MB_DAILY_PORT_TBL}
                        WHERE date=:dt AND model_version=:mv AND params_hash=:ph
                    """), {'dt': dt.strftime('%Y-%m-%d'), 'mv': MB_MODEL_VER,
                           'ph': params_hash}).fetchall()
                port_cache = {r[0]: pd.Series(json.loads(r[1])) for r in rows}
                if 'alpha' in port_cache and 'mvo' in port_cache:
                    # Apply ADVP filter with replacement using correct per-strategy AUM
                    _candidates_cache = port_cache.get('candidates', pd.Series(dtype=float))
                    if volumeRaw_df is not None and not _candidates_cache.empty:
                        _w_alpha_c = _advp_filter_and_replace(
                            port_cache['alpha'], _candidates_cache, dt,
                            Pxs_df, volumeRaw_df, current_aum,
                            advp_cap, min_weight, max_weight, top_n, conc_factor)
                        _w_mvo_c = _advp_filter_and_replace(
                            port_cache['mvo'], _candidates_cache, dt,
                            Pxs_df, volumeRaw_df, current_aum_mvo,
                            advp_cap, min_weight, max_weight, top_n, conc_factor=1.0)
                        _all_t_hyb = list(set(_w_alpha_c.index) | set(_w_mvo_c.index))
                        _w_hyb_c   = (_w_alpha_c.reindex(_all_t_hyb).fillna(0.0) +
                                      _w_mvo_c.reindex(_all_t_hyb).fillna(0.0)) / 2.0
                        _w_hyb_c   = _w_hyb_c[_w_hyb_c > 0]
                        if _w_hyb_c.sum() > 0:
                            _w_hyb_c = _w_hyb_c / _w_hyb_c.sum()
                        port_cache['alpha']  = _w_alpha_c
                        port_cache['mvo']    = _w_mvo_c
                        port_cache['hybrid'] = _w_hyb_c
                    # Write filtered weights back to all strategy weight dicts
                    alpha_weights_by_date[dt]  = port_cache['alpha']
                    mvo_weights_by_date[dt]    = port_cache['mvo']
                    w_hyb_cached = port_cache.get('hybrid', pd.Series(dtype=float))
                    hybrid_weights_by_date[dt] = w_hyb_cached
                    # Compute MVO/hybrid trading costs for cached dates
                    w_mvo_c  = port_cache['mvo']
                    w_old_mc = _prev_w["mvo"]
                    all_t_mc = list(set(w_mvo_c.index) | set(w_old_mc.index))
                    to_mc    = (w_mvo_c.reindex(all_t_mc).fillna(0) -
                                w_old_mc.reindex(all_t_mc).fillna(0)).abs().sum() / 2
                    _cost_by_date["mvo"][dt] = to_mc * TRADING_COST_BPS / 10000
                    _prev_w["mvo"] = w_mvo_c
                    if not w_hyb_cached.empty:
                        w_old_hc = _prev_w["hybrid"]
                        all_t_hc = list(set(w_hyb_cached.index) | set(w_old_hc.index))
                        to_hc    = (w_hyb_cached.reindex(all_t_hc).fillna(0) -
                                    w_old_hc.reindex(all_t_hc).fillna(0)).abs().sum() / 2
                        _cost_by_date["hybrid"][dt] = to_hc * TRADING_COST_BPS / 10000
                        _prev_w["hybrid"] = w_hyb_cached
                    # Update hybrid weights for turnover tracking
                    hybrid_weights_by_date[dt] = w_hyb_cached
                    # Update smart hybrid regime tracking using daily return
                    all_hyb_dates_c = sorted([d for d in hybrid_weights_by_date if d <= dt])
                    if all_hyb_dates_c:
                        last_hyb_dt_c = all_hyb_dates_c[-1]
                        w_prev_c      = hybrid_weights_by_date[last_hyb_dt_c]
                        tks_c         = [t for t in w_prev_c.index if t in pxs_cols]
                        prev_days_c   = [d for d in Pxs_df.index if d < dt]
                        if tks_c and prev_days_c and dt in Pxs_df.index:
                            prev_day_c = prev_days_c[-1]
                            if prev_day_c in Pxs_df.index:
                                px_sc = Pxs_df.loc[prev_day_c, tks_c]
                                px_ec = Pxs_df.loc[dt,          tks_c]
                                pr_c  = (w_prev_c.reindex(tks_c).fillna(0) *
                                         (px_ec / px_sc - 1).fillna(0)).sum()
                                _sh_nav_running = _sh_nav_running * (1 + pr_c)
                    # Compute dd from alpha portfolio for regime display
                    _alpha_dates_c = sorted([d for d in alpha_weights_by_date if d <= dt])
                    if _alpha_dates_c and len(_alpha_dates_c) > 1:
                        _mini_nav_c = 1.0; _mini_hwm_c = 1.0
                        for _ia in range(1, len(_alpha_dates_c)):
                            _d0c = _alpha_dates_c[_ia - 1]; _d1c = _alpha_dates_c[_ia]
                            _w_c  = alpha_weights_by_date[_d0c]
                            _tk_c = [t for t in _w_c.index if t in pxs_cols]
                            if _tk_c and _d0c in Pxs_df.index and _d1c in Pxs_df.index:
                                _r_c = (_w_c.reindex(_tk_c).fillna(0) *
                                        (Pxs_df.loc[_d1c, _tk_c] /
                                         Pxs_df.loc[_d0c, _tk_c] - 1).fillna(0)).sum()
                                _mini_nav_c *= (1 + _r_c)
                            _mini_hwm_c = max(_mini_hwm_c, _mini_nav_c)
                        _sh_nav_running = _mini_nav_c
                        _sh_hwm         = _mini_hwm_c
                    _sh_hwm = max(_sh_hwm, _sh_nav_running)
                    # Run MVO solve for diagnostics, then fall through to display
                    try:
                        _, _diag_cached = _mb_solve_mvo(
                            dt=dt, candidates=mvo_cands,
                            composite_scores=comp_scores, Pxs_df=Pxs_df,
                            sectors_s=sectors_s, volumeTrd_df=volumeTrd_df,
                            model_version=MB_MODEL_VER,
                            pca_var_threshold=pca_var_threshold,
                            ic=ic, max_weight=max_weight, min_weight=min_weight,
                            zscore_cap=zscore_cap, risk_aversion=risk_aversion,
                            X_snapshots=X_snapshots, snapshot_dates=snapshot_dates,
                            top_n=top_n, min_cov_matrices=min_cov_matrices,
                        )
                        _diag_by_date[dt] = _diag_cached
                        _last_mvo_diag    = _diag_cached
                        mvo_diag          = _diag_cached
                    except Exception:
                        mvo_diag = {}
                    # Use cached weights for display
                    w_mvo    = port_cache['mvo']
                    w_mvo_nz = w_mvo[w_mvo > 1e-6]
            except Exception:
                pass  # fall through to recompute

        if dt not in mvo_weights_by_date:
            try:
                # Debug: show filter funnel when MVO regime is active
                _dd_disp = _sh_nav_running / _sh_hwm - 1
                if _dd_disp < -SH_DD_HYBRID:
                    print(f"\n  [MVO filter funnel @ {dt.date()}]"
                          f"  full_cands={len(cands)}"
                          f"  alpha_liquid={len(cands_liquid)}"
                          f"  mvo_liquid={len(cands_liquid_mvo)}"
                          f"  mvo_cands(top {n_cands})={len(mvo_cands)}")
                w_mvo, mvo_diag = _mb_solve_mvo(
                        dt            = dt,
                        candidates    = mvo_cands,
                        composite_scores = comp_scores,
                        Pxs_df        = Pxs_df,
                        sectors_s     = sectors_s,
                        volumeTrd_df  = volumeTrd_df,
                        model_version = MB_MODEL_VER,
                        pca_var_threshold = pca_var_threshold,
                        ic            = ic,
                        max_weight    = max_weight,
                        min_weight    = min_weight,
                        zscore_cap    = zscore_cap,
                        risk_aversion = risk_aversion,
                        X_snapshots   = X_snapshots,
                        snapshot_dates = snapshot_dates,
                        top_n             = top_n,
                        min_cov_matrices  = min_cov_matrices,
                    )
                if _dd_disp < -SH_DD_HYBRID:
                    print(f"  [MVO solve result @ {dt.date()}]"
                          f"  valid_in_cov={mvo_diag.get('n_valid', '?')}"
                          f"  eligible={mvo_diag.get('n_eligible', '?')}"
                          f"  final_portfolio={len(w_mvo[w_mvo > 1e-6]) if not w_mvo.empty else 0}")
            except Exception as e:
                print(f"  {dt.date()} MVO ERROR: {e}")
                w_mvo, mvo_diag = pd.Series(dtype=float), {}
        else:
            # Already loaded from cache — use cached weights and diag
            w_mvo    = mvo_weights_by_date[dt]
            mvo_diag = _diag_by_date.get(dt, _last_mvo_diag)

            # Self-healing cache: if cached portfolio is too small after ADVP
            # filtering, invalidate and recompute fresh for this date
            if volumeRaw_df is not None and AUM * _running_nav > 0:
                current_aum_chk = current_aum_dyn
                past_d_chk = [d for d in Pxs_df.index if d <= dt][-ADV_WINDOW:]
                n_liquid_chk = 0
                for tkr in w_mvo[w_mvo > 1e-6].index:
                    if tkr not in Pxs_df.columns or tkr not in volumeRaw_df.columns:
                        n_liquid_chk += 1; continue
                    px_s  = Pxs_df.loc[past_d_chk, tkr].dropna()
                    vol_s = volumeRaw_df.loc[past_d_chk, tkr].dropna()
                    com   = px_s.index.intersection(vol_s.index)
                    if len(com) < 3:
                        n_liquid_chk += 1; continue
                    dv = (px_s.reindex(com) * vol_s.reindex(com).replace({0: np.nan})).median()
                    if (dv * advp_cap) / current_aum_chk >= min_weight:
                        n_liquid_chk += 1
                if n_liquid_chk < top_n // 2:
                    # Cache is stale — identify illiquid names and recompute fresh
                    _illiquid = []
                    for tkr in w_mvo[w_mvo > 1e-6].index:
                        if tkr not in Pxs_df.columns or tkr not in volumeRaw_df.columns:
                            continue
                        px_s  = Pxs_df.loc[past_d_chk, tkr].dropna()
                        vol_s = volumeRaw_df.loc[past_d_chk, tkr].dropna()
                        com   = px_s.index.intersection(vol_s.index)
                        if len(com) < 3:
                            _illiquid.append(tkr); continue
                        dv = (px_s.reindex(com) * vol_s.reindex(com).replace({0: np.nan})).median()
                        if (dv * advp_cap) / current_aum_chk < min_weight:
                            _illiquid.append(tkr)
                    del mvo_weights_by_date[dt]
                    try:
                        w_mvo, mvo_diag = _mb_solve_mvo(
                            dt               = dt,
                            candidates       = cands_liquid_mvo.index.tolist()[:n_cands],
                            composite_scores = comp_scores,
                            Pxs_df           = Pxs_df,
                            sectors_s        = sectors_s,
                            volumeTrd_df     = volumeTrd_df,
                            model_version    = MB_MODEL_VER,
                            pca_var_threshold = pca_var_threshold,
                            ic               = ic,
                            max_weight       = max_weight,
                            min_weight       = min_weight,
                            zscore_cap       = zscore_cap,
                            risk_aversion    = risk_aversion,
                            X_snapshots      = X_snapshots,
                            snapshot_dates   = snapshot_dates,
                            top_n            = top_n,
                            min_cov_matrices = min_cov_matrices,
                        )
                        print(f"\n  [CACHE INVALIDATED @ {dt.date()}] "
                              f"only {n_liquid_chk}/{len(w_mvo[w_mvo>1e-6])} liquid stocks "
                              f"at AUM=${current_aum_chk:,.0f} — recomputed fresh"
                              + (f"\n  ADVP excluded: {', '.join(sorted(_illiquid))}"
                                 if _illiquid else ""))
                    except Exception as e:
                        print(f"  {dt.date()} MVO recompute ERROR: {e}")
                        w_mvo, mvo_diag = pd.Series(dtype=float), {}

        if not w_mvo.empty and w_mvo.sum() > 0:
            w_mvo_nz = w_mvo[w_mvo > 1e-6]

            # Apply ADVP cap to MVO weights with adaptive max_weight.
            # When few stocks survive ADVP exclusion, relax max_weight so the
            # portfolio can still be built (1/n_surviving at minimum).
            current_aum_mvo = current_aum_dyn
            if volumeRaw_df is not None and current_aum_mvo > 0:
                past_d_mvo = [d for d in Pxs_df.index if d <= dt][-ADV_WINDOW:]
                n_surviving = 0
                for tkr in w_mvo_nz.index:
                    if tkr not in Pxs_df.columns or tkr not in volumeRaw_df.columns:
                        n_surviving += 1; continue
                    px_s  = Pxs_df.loc[past_d_mvo, tkr].dropna()
                    vol_s = volumeRaw_df.loc[past_d_mvo, tkr].dropna()
                    com   = px_s.index.intersection(vol_s.index)
                    if len(com) < 3:
                        n_surviving += 1; continue
                    dv = (px_s.reindex(com) * vol_s.reindex(com).replace({0: np.nan})).median()
                    if (dv * advp_cap) / current_aum_mvo >= min_weight:
                        n_surviving += 1
                # Relax max_weight if too few liquid stocks
                effective_max_w = max(max_weight,
                                      1.0 / max(n_surviving, 1)) if n_surviving > 0 else max_weight
            else:
                effective_max_w = max_weight

            w_mvo_nz, capped_mvo = _apply_advp_cap(
                w_mvo_nz, dt, Pxs_df, volumeRaw_df,
                current_aum_dyn, advp_cap, min_weight, effective_max_w)
            if capped_mvo:
                _advp_capped[dt] = _advp_capped.get(dt, set()) | capped_mvo

            # Nuclear option: if fewer than top_n stocks survive ADVP exclusion,
            # supplement with next-best liquid stocks from full alpha-ranked universe
            if len(w_mvo_nz) < top_n and volumeRaw_df is not None:
                current_aum_sup = current_aum_dyn
                comp_dt = composite_by_date.get(dt, pd.Series(dtype=float))
                if not comp_dt.empty and current_aum_sup > 0:
                    full_ranked_sup = comp_dt.dropna()
                    full_ranked_sup = full_ranked_sup.loc[
                        [t for t in full_ranked_sup.index if t in pxs_cols]
                    ].sort_values(ascending=False)
                    existing = set(w_mvo_nz.index)
                    past_d_sup = [d for d in Pxs_df.index if d <= dt][-ADV_WINDOW:]
                    added = []
                    for tkr in full_ranked_sup.index:
                        if len(w_mvo_nz) + len(added) >= top_n:
                            break
                        if tkr in existing:
                            continue
                        # Check liquidity at alpha threshold
                        if tkr not in Pxs_df.columns or tkr not in volumeRaw_df.columns:
                            added.append(tkr); continue
                        px_s  = Pxs_df.loc[past_d_sup, tkr].dropna()
                        vol_s = volumeRaw_df.loc[past_d_sup, tkr].dropna()
                        com   = px_s.index.intersection(vol_s.index)
                        if len(com) < 3:
                            added.append(tkr); continue
                        dv = (px_s.reindex(com) * vol_s.reindex(com).replace({0: np.nan})).median()
                        if (dv * advp_cap) / current_aum_sup >= min_weight:
                            added.append(tkr)
                    if added:
                        # Add new stocks with equal share of remaining weight
                        n_new   = len(added)
                        n_exist = len(w_mvo_nz)
                        # Redistribute: existing stocks give up weight proportionally
                        new_w_per  = 1.0 / (n_exist + n_new)
                        w_new_stocks = pd.Series(new_w_per, index=added)
                        w_mvo_nz = pd.concat([w_mvo_nz * (n_exist * new_w_per / w_mvo_nz.sum()
                                               if w_mvo_nz.sum() > 0 else 1),
                                              w_new_stocks])
                        if w_mvo_nz.sum() > 0:
                            w_mvo_nz = w_mvo_nz / w_mvo_nz.sum()
                        # Apply ADVP cap again to new combined portfolio
                        w_mvo_nz, _ = _apply_advp_cap(
                            w_mvo_nz, dt, Pxs_df, volumeRaw_df,
                            current_aum_sup, advp_cap, min_weight, effective_max_w)

            mvo_weights_by_date[dt] = w_mvo_nz
            _last_mvo_diag    = mvo_diag  # keep for dynamic display
            _diag_by_date[dt] = mvo_diag  # store per date for dynamic display
            # Pre-compute hybrid for caching (same logic as display block)
            _alpha_w_cache = alpha_weights_by_date.get(dt, pd.Series(dtype=float))
            if not _alpha_w_cache.empty:
                _all_t_c = list(set(_alpha_w_cache.index) | set(w_mvo_nz.index))
                _w_hyb_c = (_alpha_w_cache.reindex(_all_t_c).fillna(0.0) +
                             w_mvo_nz.reindex(_all_t_c).fillna(0.0)) / 2.0
                _w_hyb_c = _w_hyb_c[_w_hyb_c > 0]
                if _w_hyb_c.sum() > 0:
                    _w_hyb_c = _w_hyb_c / _w_hyb_c.sum()
            else:
                _w_hyb_c = w_mvo_nz.copy()
            try:
                _save_daily_portfolios(dt, MB_MODEL_VER, params_hash,
                                       alpha_weights_by_date.get(dt),
                                       w_mvo_nz, _w_hyb_c)
            except Exception as e:
                warnings.warn(f"  Cache save failed for {dt.date()}: {e}")
            vol_ann   = mvo_diag.get('vol_ann',   pd.Series(dtype=float))
            z_s       = mvo_diag.get('z_s',       pd.Series(dtype=float))
            alpha_ann = mvo_diag.get('alpha_ann', pd.Series(dtype=float))
            top_n_by  = mvo_diag.get('top_n_by', {})  # keys: E, L, F, P

            # Fill diagnostics for alpha-only stocks missing from MVO universe
            comp_dt = composite_by_date.get(dt, pd.Series(dtype=float))
            for tkr in alpha_weights_by_date.get(dt, pd.Series()).index:
                if tkr not in vol_ann.index and tkr in Pxs_df.columns:
                    # Annualised vol from last 63 trading days
                    px_hist = Pxs_df[tkr].dropna()
                    px_hist = px_hist[px_hist.index <= dt].tail(64)
                    if len(px_hist) >= 10:
                        daily_vol = px_hist.pct_change().dropna().std()
                        vol_ann[tkr]   = daily_vol * np.sqrt(252)
                        alpha_ann[tkr] = comp_dt.get(tkr, 0.0)  # raw composite score as proxy
                        # z-score: normalise composite score vs cross-section
                        if not comp_dt.empty and comp_dt.std() > 0:
                            z_s[tkr] = (comp_dt.get(tkr, 0.0) - comp_dt.mean()) / comp_dt.std()
                        else:
                            z_s[tkr] = 0.0

            # -- Build hybrid weights for display ------------------------------
            alpha_w_dt = alpha_weights_by_date.get(dt, pd.Series(dtype=float))
            if not alpha_w_dt.empty:
                all_t  = list(set(alpha_w_dt.index) | set(w_mvo_nz.index))
                w_hyb  = (alpha_w_dt.reindex(all_t).fillna(0.0) +
                          w_mvo_nz.reindex(all_t).fillna(0.0)) / 2.0
                w_hyb  = w_hyb[w_hyb > 0]
                w_hyb  = w_hyb / w_hyb.sum()
                if enforce_hybrid_floor:
                    for _ in range(50):
                        below = w_hyb < min_weight - 1e-9
                        if not below.any():
                            break
                        shortfall = (min_weight - w_hyb[below]).sum()
                        w_hyb[below] = min_weight
                        above = ~below
                        if w_hyb[above].sum() > shortfall + 1e-9:
                            w_hyb[above] -= shortfall * (w_hyb[above] / w_hyb[above].sum())
                        w_hyb = w_hyb.clip(lower=0)
                        w_hyb = w_hyb / w_hyb.sum()
            else:
                w_hyb = w_mvo_nz.copy()
            mvo_only_tickers = set(w_mvo_nz.index) - set(alpha_w_dt.index)
            alpha_only_tickers = set(alpha_w_dt.index) - set(w_mvo_nz.index)
            n_common = len(set(alpha_w_dt.index) & set(w_mvo_nz.index))

            # Trading cost for MVO (skip if already computed in cache block)
            if dt not in _cost_by_date["mvo"]:
                w_old_m  = _prev_w["mvo"]
                all_t_m  = list(set(w_mvo_nz.index) | set(w_old_m.index))
                to_m     = (w_mvo_nz.reindex(all_t_m).fillna(0) -
                            w_old_m.reindex(all_t_m).fillna(0)).abs().sum() / 2
                _cost_by_date["mvo"][dt] = to_m * TRADING_COST_BPS / 10000
                _prev_w["mvo"] = w_mvo_nz

            # Store hybrid weights immediately so turnover can reference prior dates
            # Apply ADVP cap to hybrid weights
            w_hyb, capped_hyb = _apply_advp_cap(
                w_hyb, dt, Pxs_df, volumeRaw_df,
                AUM * _running_nav, advp_cap, min_weight, max_weight)
            if capped_hyb:
                _advp_capped[dt] = _advp_capped.get(dt, set()) | capped_hyb
            hybrid_weights_by_date[dt] = w_hyb
            # Trading cost for hybrid (skip if already computed in cache block)
            if dt not in _cost_by_date["hybrid"]:
                w_old_h  = _prev_w["hybrid"]
                all_t_h  = list(set(w_hyb.index) | set(w_old_h.index))
                to_h     = (w_hyb.reindex(all_t_h).fillna(0) -
                            w_old_h.reindex(all_t_h).fillna(0)).abs().sum() / 2
                _cost_by_date["hybrid"][dt] = to_h * TRADING_COST_BPS / 10000
                _prev_w["hybrid"] = w_hyb

            # -- Determine smart hybrid regime for this date -------------------
            # Use alpha_weights_by_date as NAV proxy — available in this loop.
            # Dynamic weights aren't built yet; the dynamic loop runs later.
            # dd_disp drives regime_lbl display only, not portfolio construction.
            _alpha_dates_so_far = sorted([d for d in alpha_weights_by_date if d <= dt])
            _mini_nav = 1.0
            _mini_hwm = 1.0
            if _alpha_dates_so_far and len(_alpha_dates_so_far) > 1:
                _mini_nav = 1.0
                _mini_hwm = 1.0
                for _idx_a in range(1, len(_alpha_dates_so_far)):
                    _d0 = _alpha_dates_so_far[_idx_a - 1]
                    _d1 = _alpha_dates_so_far[_idx_a]
                    _w_a  = alpha_weights_by_date[_d0]
                    _tks  = [t for t in _w_a.index if t in Pxs_df.columns]
                    if _tks and _d0 in Pxs_df.index and _d1 in Pxs_df.index:
                        _r = (_w_a.reindex(_tks).fillna(0) *
                              (Pxs_df.loc[_d1, _tks] /
                               Pxs_df.loc[_d0, _tks] - 1).fillna(0)).sum()
                        _mini_nav *= (1 + _r)
                    _mini_hwm = max(_mini_hwm, _mini_nav)
                dd_disp = _mini_nav / _mini_hwm - 1
            else:
                dd_disp = 0.0
            _sh_nav_running = _mini_nav if _alpha_dates_so_far else 1.0
            _sh_hwm         = max(_sh_hwm, _sh_nav_running)

            if dd_disp >= -SH_DD_ALPHA:
                w_disp   = alpha_weights_by_date.get(dt, w_hyb)
                regime_lbl = f"alpha (dd={dd_disp*100:+.1f}%)"
                _sh_regime_counts["alpha"] += 1
                _sh_regime_by_date[dt] = "alpha"
            elif dd_disp >= -SH_DD_HYBRID:
                w_disp   = w_hyb
                regime_lbl = f"hybrid (dd={dd_disp*100:+.1f}%)"
                _sh_regime_counts["hybrid"] += 1
                _sh_regime_by_date[dt] = "hybrid"
            else:
                w_disp   = w_mvo_nz
                regime_lbl = f"MVO (dd={dd_disp*100:+.1f}%)"
                _sh_regime_counts["mvo"] += 1
                _sh_regime_by_date[dt] = "mvo"

            if w_disp.empty or w_disp.sum() == 0:
                w_disp = w_hyb
            w_disp = w_disp[w_disp > 1e-6]
            w_disp = w_disp / w_disp.sum()
            # Re-apply max_weight cap after renorm using adaptive max based on n stocks
            _eff_max_disp = max(max_weight, 1.0 / max(len(w_disp), 1))
            for _ in range(20):
                over = w_disp > _eff_max_disp + 1e-9
                if not over.any():
                    break
                excess     = (w_disp[over] - _eff_max_disp).sum()
                w_disp[over] = _eff_max_disp
                under      = ~over
                if w_disp[under].sum() > 1e-12:
                    w_disp[under] += excess * (w_disp[under] / w_disp[under].sum())
            if w_disp.sum() > 0:
                w_disp = w_disp / w_disp.sum()

            # -- Per-date display: smart hybrid portfolio ----------------------
            eff_n  = 1.0 / (w_disp**2).sum() if len(w_disp) > 0 else 0
            n_disp = len(w_disp)

            # -- Turnover vs previous smart hybrid portfolio -------------------
            prev_hyb_dates = sorted([d for d in hybrid_weights_by_date if d < dt])
            if prev_hyb_dates:
                w_prev   = hybrid_weights_by_date[prev_hyb_dates[-1]]
                all_t_to = list(set(w_disp.index) | set(w_prev.index))
                turnover = (w_disp.reindex(all_t_to).fillna(0.0) -
                            w_prev.reindex(all_t_to).fillna(0.0)).abs().sum() * 100
                turn_str = f"  TO={turnover:.0f}%"
                all_turnover_ratios.append(turnover)
            else:
                turn_str = ""

            # -- Individual stock contributions for this rebal period ----------
            next_rebal_dates = [d for d in calc_dates if d > dt]
            period_end = next_rebal_dates[0] if next_rebal_dates else Pxs_df.index[-1]
            contrib_s  = pd.Series(dtype=float)
            if period_end in Pxs_df.index and dt in Pxs_df.index:
                px_start  = Pxs_df.loc[dt,         [t for t in w_disp.index if t in Pxs_df.columns]]
                px_end_p  = Pxs_df.loc[period_end, [t for t in w_disp.index if t in Pxs_df.columns]]
                stk_ret   = (px_end_p / px_start - 1).fillna(0.0)
                port_ret  = (w_disp.reindex(stk_ret.index).fillna(0.0) * stk_ret).sum()
                contrib_s = stk_ret
                c_avg    = contrib_s.mean() * 100
                c_med    = contrib_s.median() * 100
                c_best   = contrib_s.max() * 100
                c_worst  = contrib_s.min() * 100
                c_std    = contrib_s.std() * 100
                contrib_str = (f"  port={port_ret*100:+.1f}%  "
                               f"avg={c_avg:+.1f}%  med={c_med:+.1f}%  "
                               f"best={c_best:+.1f}%  worst={c_worst:+.1f}%  "
                               f"std={c_std:.1f}%")
                all_stock_returns.extend(contrib_s.tolist())
                all_portfolio_returns.append(port_ret * 100)
                period_str = f"  [{dt.date()} -> {period_end.date()}]"
            else:
                contrib_str = ""
                period_str  = ""

            # Source tag: use top_n_by from MVO diag; alpha-only = not in any matrix
            mvo_tickers_set   = set(w_mvo_nz.index)
            alpha_tickers_set = set(alpha_w_dt.index) if not alpha_w_dt.empty else set()

            n_illiquid = len(cands) - len(cands_liquid)
            liq_str    = f"  liq_filt={n_illiquid}" if n_illiquid > 0 else ""
            # Portfolio vol, SPY vol, QQQ vol
            _past_d  = [d for d in Pxs_df.index if d <= dt][-VOL_WINDOW:]
            _pr_d    = pd.Series(0.0, index=_past_d)
            for _dv in _past_d:
                _pv = [d for d in _past_d if d < _dv]
                if _pv:
                    _tv = [t for t in w_disp.index if t in Pxs_df.columns]
                    _rv = (Pxs_df.loc[_dv, _tv] / Pxs_df.loc[_pv[-1], _tv] - 1).fillna(0)
                    _pr_d[_dv] = (w_disp.reindex(_tv).fillna(0) * _rv).sum()
            _pvol_d  = _pr_d.std() * np.sqrt(252) * 100
            _spy_col = 'SPX' if 'SPX' in Pxs_df.columns else None
            _qqq_col = 'QQQ' if 'QQQ' in Pxs_df.columns else None
            _svol_d  = (Pxs_df.loc[_past_d, _spy_col].pct_change().dropna().std() * np.sqrt(252) * 100
                        if _spy_col else 0.0)
            _qvol_d  = (Pxs_df.loc[_past_d, _qqq_col].pct_change().dropna().std() * np.sqrt(252) * 100
                        if _qqq_col else 0.0)
            vol_str  = f"  port_vol={_pvol_d:.1f}%"
            if _svol_d > 0:  vol_str += f"  spx_vol={_svol_d:.1f}%"
            if _qvol_d > 0:  vol_str += f"  qqq_vol={_qvol_d:.1f}%"
            # Excluded stocks at this date
            _mr_n_d = int(MR_CAP * len(cands)) if MR_CAP > 0 else 0
            _excl_d = _load_momentum_exclusions(dt, top_n=_mr_n_d) if _mr_n_d > 0 else {}
            print(f"\n  -- {dt.date()}  [{i+1}/{n}]  regime={regime_lbl}  "
                  f"n={n_disp}  eff_N={eff_n:.1f}  "
                  f"min={w_disp.min():.1%}  max={w_disp.max():.1%}"
                  f"{turn_str}{liq_str}{vol_str} --")
            if _excl_d:
                # Preserve score-descending order (dict already ordered), show score
                _excl_str = ', '.join(f"{t}({s:.2f})" for t, s in _excl_d.items())
                print(f"  *** Excluded ({len(_excl_d)}): {_excl_str}")
            if MR_K > 0:
                # Soft penalty mode — show top-10 MR scores regardless of exclusion
                _mr_top10 = _load_momentum_exclusions(dt, top_n=10)
                if _mr_top10:
                    _mr_str = ', '.join(f"{t}({s:.2f})" for t, s in _mr_top10.items())
                    print(f"  MR top-10 (k={MR_K}): {_mr_str}")
            if contrib_str:
                print(f"  period{period_str}{contrib_str}")
            print(f"  {'Ticker':<8}  {'Weight%':>7}  {'AnnAlpha%':>10}  "
                  f"{'AnnVol%':>8}  {'Z-score':>8}  {'Contrib%':>9}  {'Sector':<28}  Source")
            print(f"  {'-'*105}")
            if _advp_capped.get(dt):
                print(f"  *** ADVP cap active (AUM=${current_aum_dyn:,.0f}, "
                      f"cap={advp_cap:.1%} ADV): "
                      f"{', '.join(sorted(_advp_capped[dt]))}")
            for tkr, wt in w_disp.sort_values(ascending=False).items():
                sec   = sectors_s.get(tkr, '')
                a     = alpha_ann.get(tkr, 0.0) * 100
                v     = vol_ann.get(tkr,   0.0) * 100
                z     = z_s.get(tkr,       0.0)
                c     = contrib_s.get(tkr, np.nan) * 100 if not contrib_s.empty else np.nan
                c_str = f"{c:>+8.2f}%" if not np.isnan(c) else f"{'n/a':>9}"
                # Source: show matrix tags if in MVO universe, else 'alpha'
                if tkr in mvo_tickers_set:
                    in_m = '/'.join(m for m in ['E','L','F','P']
                                    if tkr in top_n_by.get(m, []))
                    src  = (in_m + '/E') if in_m else 'mvo'
                else:
                    src  = 'alpha'
                advp_flag = '***' if tkr in _advp_capped.get(dt, set()) else '   '
                print(f"  {tkr:<8}  {wt*100:>6.2f}%  {a:>+9.1f}%  "
                      f"{v:>7.1f}%  {z:>+8.3f}  {c_str}  {sec:<28}  {src}{advp_flag}")

    # hybrid_weights_by_date is already fully populated from the main loop above

    print(f"\n  Weights computed: "
          f"alpha={len(alpha_weights_by_date)}, "
          f"mvo={len(mvo_weights_by_date)}, "
          f"hybrid={len(hybrid_weights_by_date)}, "
          f"baseline={len(quality_factor_by_date)}")
    if _advp_capped:
        n_capped_dates  = len(_advp_capped)
        all_capped_tkrs = set().union(*_advp_capped.values())
        print(f"  ADVP cap triggered: {n_capped_dates} rebal dates, "
              f"{len(all_capped_tkrs)} unique tickers: {sorted(all_capped_tkrs)}")

    # -- [4/4] NAV series ------------------------------------------------------
    print("\n[4/4] Computing NAV series...")

    nav_alpha    = _mb_run_nav(alpha_weights_by_date,  calc_dates, Pxs_df,
                              cost_by_date=_cost_by_date["alpha"])
    nav_mvo      = _mb_run_nav(mvo_weights_by_date,    calc_dates, Pxs_df,
                              cost_by_date=_cost_by_date["mvo"])
    nav_hybrid   = _mb_run_nav(hybrid_weights_by_date, calc_dates, Pxs_df,
                              cost_by_date=_cost_by_date["hybrid"])

    # -- Smart hybrid: switch regime based on running drawdown of hybrid NAV --
    smart_hybrid_weights_by_date = {}
    nav_h_hwm = 1.0
    for rdt in sorted(hybrid_weights_by_date.keys()):
        if rdt in nav_hybrid.index:
            nav_h_hwm = max(nav_h_hwm, nav_hybrid.loc[rdt])
            dd = nav_hybrid.loc[rdt] / nav_h_hwm - 1
        else:
            dd = 0.0
        if dd >= -SH_DD_ALPHA:
            w_smart = alpha_weights_by_date.get(rdt, pd.Series(dtype=float))
        elif dd >= -SH_DD_HYBRID:
            w_smart = hybrid_weights_by_date.get(rdt, pd.Series(dtype=float))
        else:
            w_smart = mvo_weights_by_date.get(rdt, pd.Series(dtype=float))
        if not w_smart.empty and w_smart.sum() > 0:
            smart_hybrid_weights_by_date[rdt] = w_smart / w_smart.sum()
            # Save smart weights to cache
            try:
                _save_daily_portfolios(rdt, MB_MODEL_VER, params_hash,
                                       None, None, None,
                                       w_smart=smart_hybrid_weights_by_date[rdt])
            except Exception as e:
                warnings.warn(f"  Smart cache save failed for {rdt.date()}: {e}")
    # Trading costs for smart hybrid
    _prev_w_sm = pd.Series(dtype=float)
    for rdt in sorted(smart_hybrid_weights_by_date.keys()):
        w_sm     = smart_hybrid_weights_by_date[rdt]
        all_t_sm = list(set(w_sm.index) | set(_prev_w_sm.index))
        to_sm    = (w_sm.reindex(all_t_sm).fillna(0) -
                    _prev_w_sm.reindex(all_t_sm).fillna(0)).abs().sum() / 2
        _cost_by_date["smart"][rdt] = to_sm * TRADING_COST_BPS / 10000
        _prev_w_sm = w_sm

    nav_smart = _mb_run_nav(smart_hybrid_weights_by_date, calc_dates, Pxs_df,
                             cost_by_date=_cost_by_date["smart"])

    # ── DD forced regimes (populated after NAV is built, consumed by dynamic loop)
    _dd_forced_regimes = {}   # {date: regime_string} — set in pre-pass below

    # ── Dynamic rebalancing strategy ─────────────────────────────────────────
    # Uses daily cached trigger variables to decide when to rebalance.
    # Rebalance if ANY of:
    #   1. Regime switch (always, ignores min hold)
    #   2. TO > DYN_TO_THRESHOLD AND vol_diff < DYN_VOLDIFF_CAP
    #   3. vol_diff < DYN_VOLDIFF_DERISK (de-risk, ignores min hold)
    # All subject to DYN_MIN_HOLD_DAYS except regime switch and de-risk.
    dyn_weights_by_date = {}
    _cost_by_date["dynamic"] = {}
    _dyn_rebal_log = []   # collect rebal info for forward-looking display
    _do_rebal_log  = False
    _s9_wbd        = {}   # strategy 9 weights — populated after strategy 9 block
    try:
        triggers_dyn = load_daily_portfolios(params_hash, MB_MODEL_VER)
        with ENGINE.connect() as _conn:
            _trig_rows = _conn.execute(text(f"""
                SELECT date, implied_turnover, vol_diff
                FROM {DAILY_TRIGGER_TBL}
                WHERE params_hash=:ph AND model_version=:mv
                ORDER BY date
            """), {'ph': params_hash, 'mv': MB_MODEL_VER}).fetchall()
        trig_df = pd.DataFrame(
            _trig_rows,
            columns=['date', 'implied_turnover', 'vol_diff']
        )
        trig_df['date'] = pd.to_datetime(trig_df['date'])
        trig_df = trig_df.set_index('date')

        if not trig_df.empty and triggers_dyn:
            w_dyn_live         = pd.Series(dtype=float)
            w_dyn_live_prev_dt = None
            last_rebal_dt      = None
            prev_regime        = None
            deployed_regime    = None
            dyn_hwm            = None   # HWM for drawdown computation

            for dt in sorted(trig_df.index):
                if dt not in triggers_dyn:
                    continue
                row    = trig_df.loc[dt]
                vd_val = float(row['vol_diff'])

                # ── Compute regime from actual deployed NAV ────────────────
                # Use nav_smart as the reference NAV series (fully computed)
                if dyn_hwm is None:
                    dyn_hwm = float(nav_smart.iloc[0]) if nav_smart is not None and not nav_smart.empty else 1.0
                if nav_smart is not None and dt in nav_smart.index:
                    dyn_nav_val = float(nav_smart.loc[dt])
                    dyn_hwm     = max(dyn_hwm, dyn_nav_val)
                    dd_dyn      = dyn_nav_val / dyn_hwm - 1
                else:
                    dd_dyn = 0.0

                if dd_dyn >= -SH_DD_ALPHA:
                    regime = 'alpha'
                elif dd_dyn >= -SH_DD_HYBRID:
                    regime = 'hybrid'
                else:
                    regime = 'mvo'

                # Select weights for this regime
                w_new = triggers_dyn[dt].get(regime, pd.Series(dtype=float))
                if w_new.empty:
                    w_new = triggers_dyn[dt].get('alpha', pd.Series(dtype=float))
                if w_new.empty:
                    continue

                days_held = (dt - last_rebal_dt).days if last_rebal_dt else 999

                # Drift deployed portfolio to today for accurate TO computation
                if not w_dyn_live.empty and w_dyn_live_prev_dt is not None:
                    tks_drift = [t for t in w_dyn_live.index if t in Pxs_df.columns]
                    if tks_drift and w_dyn_live_prev_dt in Pxs_df.index and dt in Pxs_df.index:
                        px_prev_d = Pxs_df.loc[w_dyn_live_prev_dt, tks_drift]
                        px_cur_d  = Pxs_df.loc[dt,                 tks_drift]
                        w_drift_d = w_dyn_live.reindex(tks_drift) * (px_cur_d / px_prev_d).fillna(1)
                        if w_drift_d.sum() > 0:
                            w_dyn_live = w_drift_d / w_drift_d.sum()
                w_dyn_live_prev_dt = dt

                # Recompute TO against actual drifted deployed portfolio
                if w_dyn_live.empty:
                    to_val = 1.0
                else:
                    all_t_to = list(set(w_new.index) | set(w_dyn_live.index))
                    to_val   = (w_new.reindex(all_t_to).fillna(0) -
                                w_dyn_live.reindex(all_t_to).fillna(0)).abs().sum() / 2

                # Determine if we should rebalance — all triggers respect min hold
                # Regime switch: only if signal regime differs from DEPLOYED regime
                regime_switch = (deployed_regime is not None and
                                 regime != deployed_regime and
                                 days_held >= DYN_MIN_HOLD_DAYS)

                # DD-forced regime: override deployed_regime immediately, no min hold
                dd_forced = _dd_forced_regimes.get(dt)
                dd_regime_override = (dd_forced is not None and
                                      dd_forced != deployed_regime)
                if dd_regime_override:
                    # Force regime to DD-mandated level (hybrid or mvo)
                    # Use the most conservative of signal regime and DD-forced regime
                    _regime_order = {'alpha': 0, 'hybrid': 1, 'mvo': 2}
                    regime = dd_forced if (_regime_order.get(dd_forced, 0) >
                                           _regime_order.get(regime, 0)) else regime
                    w_new = triggers_dyn[dt].get(regime, w_new)
                    if w_new.empty:
                        w_new = triggers_dyn[dt].get('smart', pd.Series(dtype=float))
                # Regime-specific TO threshold
                _to_thresh = (DYN_TO_THRESHOLD_MVO    if deployed_regime == 'mvo' else
                              DYN_TO_THRESHOLD_HYBRID if deployed_regime == 'hybrid' else
                              DYN_TO_THRESHOLD_ALPHA)
                to_trigger    = (to_val > _to_thresh and
                                 vd_val < DYN_VOLDIFF_CAP and
                                 days_held >= DYN_MIN_HOLD_DAYS)
                derisk        = (vd_val < DYN_VOLDIFF_DERISK and
                                 days_held >= DYN_MIN_HOLD_DAYS)

                should_rebal  = regime_switch or to_trigger or derisk or dd_regime_override

                if should_rebal or w_dyn_live.empty:
                    # Trading cost — measured against drifted deployed portfolio
                    if not w_dyn_live.empty:
                        all_t_dyn = list(set(w_new.index) | set(w_dyn_live.index))
                        to_dyn    = (w_new.reindex(all_t_dyn).fillna(0) -
                                     w_dyn_live.reindex(all_t_dyn).fillna(0)).abs().sum() / 2
                        _cost_by_date["dynamic"][dt] = to_dyn * TRADING_COST_BPS / 10000
                    else:
                        to_dyn = 1.0  # first rebalance

                    # Trigger label and type
                    if w_dyn_live.empty:
                        trigger_lbl  = 'init'
                        trigger_type = 'init'
                    elif dd_regime_override:
                        trigger_lbl  = f'DD-regime->{regime}'
                        trigger_type = 'dd_regime'
                    elif regime_switch:
                        trigger_lbl  = f'regime->{regime}'
                        trigger_type = 'regime'
                    elif derisk:
                        trigger_lbl  = f'derisk(vd={vd_val*100:+.1f}%)'
                        trigger_type = 'derisk'
                    else:
                        trigger_lbl  = f'TO={to_val*100:.1f}%,vd={vd_val*100:+.1f}% (thr={_to_thresh*100:.0f}%)'
                        trigger_type = 'turnover'

                    # Store rebalance info for forward-looking display (second pass)
                    _dyn_rebal_log.append({
                        'dt': dt, 'trigger': trigger_lbl, 'trigger_type': trigger_type,
                        'regime': regime, 'w': w_new.copy(), 'to_dyn': to_dyn,
                        'days_held': days_held, 'eff_n': 1.0/(w_new**2).sum() if len(w_new)>0 else 0,
                        'diag_dt': (_past_d[-1] if (_past_d := [d for d in sorted(mvo_weights_by_date.keys()) if d <= dt]) else
                                    ([d for d in sorted(mvo_weights_by_date.keys()) if d > dt] or [None])[0]),
                    })

                    w_dyn_live      = w_new.copy()
                    w_dyn_live_prev_dt = dt   # reset drift baseline to rebalance date
                    last_rebal_dt   = dt
                    deployed_regime = regime   # update deployed regime on actual rebalance

                dyn_weights_by_date[dt] = w_dyn_live
                prev_regime = regime   # always track signal regime

            n_rebal_total = len(_cost_by_date['dynamic'])
            print(f"\n  Dynamic strategy: {len(dyn_weights_by_date)} daily weights, "
                  f"{n_rebal_total} rebalances  "
                  f"(avg {len(dyn_weights_by_date)/max(n_rebal_total,1):.1f} days between rebalances)")

            # -- Second pass: deferred to after strategy 9 is built (needs _s9_wbd) --
            _do_rebal_log = True   # flag; actual display happens after strategy 9
        else:
            print("  Dynamic strategy: no trigger data -- run run_daily_cache_build() first")
    except Exception as e:
        print(f"  Dynamic strategy failed: {e}")
        traceback.print_exc()

    nav_dynamic = (_mb_run_nav(dyn_weights_by_date, sorted(dyn_weights_by_date.keys()),
                               Pxs_df, cost_by_date=_cost_by_date["dynamic"])
                   if dyn_weights_by_date else pd.Series(dtype=float))

    # Note: ADVP filter was already applied correctly in the first pass using
    # AUM * _running_nav at each rebalance date. No second-pass needed.

    # Baseline trading costs -- equal weight assumed, turnover from port changes
    _prev_bl = pd.Series(dtype=float)
    for rdt in sorted(quality_factor_by_date.keys()):
        fdf    = quality_factor_by_date[rdt]
        ranked = fdf.sort_values('factor', ascending=False).head(top_n)
        w_bl   = pd.Series(1.0 / len(ranked), index=ranked.index)
        all_t_bl = list(set(w_bl.index) | set(_prev_bl.index))
        to_bl    = (w_bl.reindex(all_t_bl).fillna(0) -
                    _prev_bl.reindex(all_t_bl).fillna(0)).abs().sum() / 2
        _cost_by_date["baseline"][rdt] = to_bl * TRADING_COST_BPS / 10000
        _prev_bl = w_bl

    nav_baseline_raw, port_baseline = run_backtest(
        factor_by_date = quality_factor_by_date,
        calc_dates     = calc_dates,
        Pxs_df         = Pxs_df,
        use_vol_filter = use_vol,
        mktcap_floor   = mktcap_floor,
        sector_cap     = sector_cap,
        top_n          = top_n,
        prefilt_pct    = prefilt_pct,
        conc_factor    = 1.0,
    )
    # Apply baseline costs post-hoc
    nav_baseline = nav_baseline_raw.copy()
    for rdt in sorted(_cost_by_date["baseline"].keys()):
        if rdt in nav_baseline.index:
            cost_factor = 1 - _cost_by_date["baseline"][rdt]
            nav_baseline.loc[rdt:] *= cost_factor

    # Port DataFrames
    port_alpha = pd.DataFrame.from_dict(
        {dt: (list(w.index) + [None] * max(0, top_n - len(w)))[:top_n]
         for dt, w in alpha_weights_by_date.items()},
        orient='index',
        columns=[f'Stock{i+1}' for i in range(top_n)]
    )
    port_mvo = pd.DataFrame.from_dict(
        {dt: (list(w[w > 1e-6].index) +
              [None] * max(0, top_n - (w > 1e-6).sum()))[:top_n]
         for dt, w in mvo_weights_by_date.items()},
        orient='index',
        columns=[f'Stock{i+1}' for i in range(top_n)]
    )
    max_hybrid_n = max((len(w) for w in hybrid_weights_by_date.values()), default=top_n)
    port_hybrid = pd.DataFrame.from_dict(
        {dt: (list(w[w > 1e-6].index) +
              [None] * max(0, max_hybrid_n - (w > 1e-6).sum()))[:max_hybrid_n]
         for dt, w in hybrid_weights_by_date.items()},
        orient='index',
        columns=[f'Stock{i+1}' for i in range(max_hybrid_n)]
    ) if hybrid_weights_by_date else pd.DataFrame()

    # -- Hedge layer (optional) ------------------------------------------------
    nav_dyn_hedged = None
    hedge_results  = None
    if hedge_multi is not None and hedges_l is not None:
        _trigger_assets = hedge_trigger_assets or ['QQQ']
        print(f"\n  {'='*72}")
        print(f"  RUNNING HEDGE LAYER")
        print(f"  {'='*72}")
        hedge_results = run_hedge_backtest(
            Pxs_df            = Pxs_df,
            multi             = hedge_multi,
            portfolio_weights = dyn_weights_by_date,
            rebal_dates       = [r['dt'] for r in _dyn_rebal_log],
            hedges_l          = hedges_l,
            trigger_assets    = _trigger_assets,
        )
        # Build hedged NAV: apply sweep amounts to dynamic NAV at each rebal date
        if nav_dynamic is not None and not nav_dynamic.empty:
            nav_dyn_hedged = nav_dynamic.copy()
            sweep_by_date  = hedge_results['hedge_sweep_by_date']
            cum_hedge      = 0.0
            for dt in nav_dyn_hedged.index:
                if dt in sweep_by_date:
                    cum_hedge += sweep_by_date[dt]
                # Hedged NAV = portfolio NAV * (1 + cumulative hedge contribution)
                nav_dyn_hedged[dt] = nav_dynamic[dt] * (1 + cum_hedge)
            nav_dyn_hedged.name = 'nav_dyn_hedged'

    # -- DD regime pre-pass -------------------------------------------------------
    # Now that nav_dyn_hedged is available, identify dates where DD events fire
    # and populate _dd_forced_regimes so the dynamic loop (already run) can be
    # consulted for display; more importantly this feeds into future re-runs via
    # the cache rebuild. The dynamic loop already consumed _dd_forced_regimes
    # (empty on first pass) — the real fix is the corrected sh_nav in cache build.
    _nav_for_dd = (nav_dyn_hedged if nav_dyn_hedged is not None and not nav_dyn_hedged.empty
                   else nav_smart  if nav_smart      is not None and not nav_smart.empty
                   else None)
    if _nav_for_dd is not None:
        _pre_hwm    = _nav_for_dd.iloc[0]
        _pre_active = False
        _pre_next   = 0
        for _dt, _raw_nav in _nav_for_dd.items():
            if not _pre_active:
                _pre_hwm = max(_pre_hwm, _raw_nav)
            _pre_dd = _raw_nav / _pre_hwm - 1
            if _pre_next < len(DD_LEVELS):
                _thr, _ = DD_LEVELS[_pre_next]
                if _pre_dd <= -_thr:
                    _dd_forced_regimes[_dt] = DD_LEVEL_REGIME[_pre_next]
                    _pre_active = True
                    _pre_next  += 1
            if _pre_active and _raw_nav / max(_nav_for_dd.iloc[0], 1e-9) > (1 - DD_LEVELS[0][0] + DD_REENTRY_PCT):
                _pre_active = False
                _pre_next   = 0
                _pre_hwm    = _raw_nav
    if _dd_forced_regimes:
        print(f"\n  DD regime pre-pass: {len(_dd_forced_regimes)} forced-regime dates "
              f"({sum(1 for r in _dd_forced_regimes.values() if r=='mvo')} mvo)")

    # -- DD policy (8th strategy: Dynamic + Hedge + DD) -----------------------
    nav_dd = None
    _dd_log = []
    if nav_dyn_hedged is not None and not nav_dyn_hedged.empty:
        _lvl_str = '  '.join([f"-{d:.0%}→cut {c:.0%}" for d, c in DD_LEVELS])
        print(f"\n  {'='*72}")
        print(f"  DRAWDOWN POLICY  ({_lvl_str})")
        print(f"  Re-entry: +{DD_REENTRY_PCT:.0%} from trough (confirm {DD_REENTRY_CONFIRM}d, MVO mode)  "
              f"|  Annual reset if YTD dd < {DD_ANNUAL_RESET_PCT:.0%}")
        print(f"  HWM resets at each full re-entry")
        print(f"  {'='*72}")

        base_ret   = nav_dyn_hedged.pct_change().fillna(0)
        mvo_ret_s  = nav_mvo.pct_change().fillna(0) if nav_mvo is not None else base_ret

        dd_nav      = pd.Series(index=nav_dyn_hedged.index, dtype=float)
        dd_nav.iloc[0] = nav_dyn_hedged.iloc[0]
        exposure_s  = pd.Series(index=nav_dyn_hedged.index, dtype=float)
        exposure_s.iloc[0] = 1.0

        hwm          = nav_dyn_hedged.iloc[0]   # resets on full re-entry
        exposure     = 1.0
        trough       = nav_dyn_hedged.iloc[0]
        theo_nav     = nav_dyn_hedged.iloc[0]
        dd_active    = False
        confirm_cnt  = 0
        # Track which DD levels have fired (by index into DD_LEVELS)
        next_level_idx = 0   # next level to watch for

        for i, dt in enumerate(nav_dyn_hedged.index[1:], 1):
            prev_nav = dd_nav.iloc[i-1]

            # Update theoretical MVO NAV only when DD active
            if dd_active:
                theo_nav = theo_nav * (1 + mvo_ret_s.get(dt, 0))

            # Actual return scaled by exposure
            raw_r    = base_ret.get(dt, 0)
            curr_nav = prev_nav * (1 + raw_r * exposure)
            dd_nav.iloc[i]   = curr_nav
            exposure_s.iloc[i] = exposure

            # Update HWM
            if not dd_active:
                hwm = max(hwm, curr_nav)
            dd_current = curr_nav / hwm - 1

            # ── Check next DD level ────────────────────────────────────────
            if dd_active and next_level_idx < len(DD_LEVELS):
                dd_thresh, cut_frac = DD_LEVELS[next_level_idx]
                if dd_current <= -dd_thresh:
                    old_exp  = exposure
                    exposure = exposure * (1 - cut_frac)
                    trough    = curr_nav
                    theo_nav  = curr_nav
                    confirm_cnt = 0
                    next_level_idx += 1   # arm the next level
                    port_cost = (old_exp - exposure) * TRADING_COST_BPS / 10000
                    curr_nav  = curr_nav * (1 - port_cost)
                    dd_nav.iloc[i]    = curr_nav
                    exposure_s.iloc[i] = exposure
                    _dd_log.append({
                        'date': dt, 'event': f'DE-GROSS lv{next_level_idx}',
                        'dd': dd_current, 'exposure_from': old_exp,
                        'exposure_to': exposure,
                    })
                    print(f"\n  {dt.date()}  [DD POLICY: DE-GROSS lv{next_level_idx}]  "
                          f"dd={dd_current:.1%}  cut {cut_frac:.0%} of remaining  "
                          f"exposure {old_exp:.1%}→{exposure:.1%}  "
                          f"costs≈{port_cost*100:.2f}bps")

            # ── Check initial DD trigger (not yet active) ───────────────────
            elif not dd_active and next_level_idx < len(DD_LEVELS):
                dd_thresh, cut_frac = DD_LEVELS[next_level_idx]
                if dd_current <= -dd_thresh:
                    old_exp  = exposure
                    exposure = exposure * (1 - cut_frac)
                    dd_active = True
                    trough    = curr_nav
                    theo_nav  = curr_nav
                    confirm_cnt = 0
                    next_level_idx += 1
                    port_cost = (old_exp - exposure) * TRADING_COST_BPS / 10000
                    curr_nav  = curr_nav * (1 - port_cost)
                    dd_nav.iloc[i]    = curr_nav
                    exposure_s.iloc[i] = exposure
                    _dd_log.append({
                        'date': dt, 'event': f'DE-GROSS lv{next_level_idx}',
                        'dd': dd_current, 'exposure_from': old_exp,
                        'exposure_to': exposure,
                    })
                    print(f"\n  {dt.date()}  [DD POLICY: DE-GROSS lv{next_level_idx}]  "
                          f"dd={dd_current:.1%}  cut {cut_frac:.0%} of remaining  "
                          f"exposure {old_exp:.1%}→{exposure:.1%}  "
                          f"costs≈{port_cost*100:.2f}bps")

            # ── Re-entry logic (runs independently of DD level checks) ──────
            if dd_active and exposure < 1.0:
                trough = min(trough, curr_nav)
                theo_recovery = theo_nav / trough - 1

                # Condition 1: theoretical recovery from trough
                recovery_ok = theo_recovery >= DD_REENTRY_PCT
                # Condition 2: new calendar year AND YTD DD within limit
                yr_start_nav = dd_nav[dd_nav.index.year == dt.year]
                ytd_dd = (curr_nav / yr_start_nav.iloc[0] - 1
                          if len(yr_start_nav) > 1 else 0)
                new_year_ok = (dt.month == 1 and dt.day <= 10 and
                               ytd_dd >= -DD_ANNUAL_RESET_PCT)

                if recovery_ok or new_year_ok:
                    confirm_cnt += 1
                else:
                    confirm_cnt = 0

                if confirm_cnt >= DD_REENTRY_CONFIRM:
                    old_exp  = exposure
                    exposure = 1.0
                    dd_active = False
                    next_level_idx = 0   # reset all levels
                    hwm = curr_nav       # reset HWM from here
                    port_cost = (1.0 - old_exp) * TRADING_COST_BPS / 10000
                    curr_nav  = curr_nav * (1 - port_cost)
                    dd_nav.iloc[i]    = curr_nav
                    exposure_s.iloc[i] = exposure
                    confirm_cnt = 0
                    trigger = "recovery" if recovery_ok else "annual reset"
                    _dd_log.append({
                        'date': dt, 'event': 'RE-ENTRY 100%',
                        'theo_recovery': theo_recovery,
                        'exposure_from': old_exp, 'exposure_to': 1.0,
                        'trigger': trigger,
                    })
                    print(f"\n  {dt.date()}  [DD POLICY: FULL RE-ENTRY ({trigger})]  "
                          f"theoretical recovery={theo_recovery:.1%}  "
                          f"exposure {old_exp:.1%}→100%  MVO mode  "
                          f"costs≈{port_cost*100:.2f}bps  [HWM RESET]")

        nav_dd = dd_nav
        nav_dd.name = 'nav_dd'

    # ── Strategy 9: Dyn+Hedge+DD+Excl (parasitic on strategy 8) ─────────────
    # Uses strategy 8's rebalance dates/regimes. At MR_CAP=0 uses standard
    # portfolios directly → converges naturally to strategy 8 (consistency check).
    nav_dd_excl   = None
    _dd_excl_log  = []
    _s9_wbd       = {}   # strategy 9 weights by date (filled below)
    _s9_costs     = {}

    try:
        _port_cache = load_daily_portfolios(params_hash, MB_MODEL_VER)
        _has_excl   = any('alpha_excl' in v for v in _port_cache.values()) if _port_cache else False

        if _has_excl and _dyn_rebal_log:
            # -- Build strategy 9 weights following strategy 8's rebal schedule --
            # Convergence condition: both MR_CAP and MR_K are zero → S9 = S8 exactly
            if MR_CAP == 0.0 and MR_K == 0.0:
                _s9_wbd   = dyn_weights_by_date        # same object — no copy needed
                _s9_costs = _cost_by_date.get('dynamic', {})
            else:
                _w9_prev        = pd.Series(dtype=float)
                _s9_running_nav = 1.0
                _s9_nav_log     = []   # diagnostic
                for _rec9 in _dyn_rebal_log:
                    _dt9 = _rec9['dt']; _reg9 = _rec9['regime']
                    _p9  = _port_cache.get(_dt9, {})
                    # MVO regime: skip filter, use standard MVO portfolio
                    if _reg9 == 'mvo':
                        _w9 = _p9.get('mvo', pd.Series(dtype=float))
                    else:
                        _w9 = _p9.get(f'{_reg9}_excl', _p9.get(_reg9, pd.Series(dtype=float)))
                    if _w9.empty: _w9 = _rec9['w'].copy()

                    # Apply ADVP filter with replacement using correct running AUM
                    _cands9 = _p9.get('candidates', pd.Series(dtype=float))
                    _s9_aum = AUM * _s9_running_nav
                    if volumeRaw_df is not None and not _cands9.empty and not _w9.empty:
                        _w9_before = _w9.copy()
                        _w9 = _advp_filter_and_replace(
                            _w9, _cands9, _dt9, Pxs_df, volumeRaw_df,
                            _s9_aum, advp_cap, min_weight, max_weight,
                            top_n, conc_factor)
                        # Log if filter changed anything
                        _changed = set(_w9_before.index) != set(_w9.index)
                        if _changed and _dt9.year >= 2024:
                            _removed = set(_w9_before.index) - set(_w9.index)
                            _added   = set(_w9.index) - set(_w9_before.index)
                            print(f"  [S9 ADVP] {_dt9.date()}  AUM=${_s9_aum:,.0f}"
                                  f"  removed={sorted(_removed)}  added={sorted(_added)}")

                    if not _w9_prev.empty:
                        _at9 = list(set(_w9.index)|set(_w9_prev.index))
                        _to9 = (_w9.reindex(_at9).fillna(0)-_w9_prev.reindex(_at9).fillna(0)).abs().sum()/2
                        _s9_costs[_dt9] = _to9 * TRADING_COST_BPS / 10000
                    _s9_wbd[_dt9] = _w9.copy(); _w9_prev = _w9.copy()

                    # Update running NAV to next rebalance for ADVP AUM tracking
                    _next9 = [r['dt'] for r in _dyn_rebal_log if r['dt'] > _dt9]
                    _end9  = _next9[0] if _next9 else Pxs_df.index[-1]
                    _nav9d = [d for d in Pxs_df.index if _dt9 <= d <= _end9]
                    if len(_nav9d) >= 2:
                        _tk9 = [t for t in _w9.index if t in Pxs_df.columns]
                        if _tk9:
                            _ret9 = (_w9.reindex(_tk9).fillna(0) *
                                     (Pxs_df.loc[_nav9d[-1], _tk9] /
                                      Pxs_df.loc[_nav9d[0],  _tk9] - 1).fillna(0)).sum()
                            _s9_running_nav *= (1 + _ret9)
                    _s9_nav_log.append((_dt9, _s9_running_nav, _s9_aum))

                # Print NAV log for recent dates
                print(f"\n  [S9 NAV tracking] last 5 rebalances:")
                for _d, _n, _a in _s9_nav_log[-5:]:
                    print(f"    {_d.date()}  nav={_n:.3f}  AUM=${_a:,.0f}")
                # Fill forward (same dates as strategy 8)
                _lw9 = pd.Series(dtype=float)
                for _dt9f in sorted(dyn_weights_by_date.keys()):
                    if _dt9f in _s9_wbd: _lw9 = _s9_wbd[_dt9f]
                    elif not _lw9.empty: _s9_wbd[_dt9f] = _lw9

            # -- Pre-hedge NAV --
            _nav_dyn9 = (_mb_run_nav(_s9_wbd, sorted(_s9_wbd.keys()),
                                     Pxs_df, cost_by_date=_s9_costs)
                         if _s9_wbd else pd.Series(dtype=float))

            # -- Hedge layer --
            if MR_CAP == 0.0 and MR_K == 0.0:
                _nav_hdg9 = nav_dyn_hedged.copy() if nav_dyn_hedged is not None else _nav_dyn9.copy()
            elif hedge_results is not None and not _nav_dyn9.empty:
                try:
                    _hr9 = run_hedge_backtest(
                        Pxs_df=Pxs_df, multi=hedge_multi,
                        portfolio_weights=_s9_wbd,
                        rebal_dates=[r['dt'] for r in _dyn_rebal_log],
                        hedges_l=hedges_l,
                        trigger_assets=hedge_trigger_assets or ['QQQ'],
                    )
                    _nav_hdg9 = _nav_dyn9.copy(); _ch9 = 0.0
                    for _dt9h in _nav_hdg9.index:
                        if _dt9h in _hr9['hedge_sweep_by_date']:
                            _ch9 += _hr9['hedge_sweep_by_date'][_dt9h]
                        _nav_hdg9[_dt9h] = _nav_dyn9[_dt9h] * (1 + _ch9)
                except Exception:
                    _nav_hdg9 = _nav_dyn9.copy()
            else:
                _nav_hdg9 = _nav_dyn9.copy() if not _nav_dyn9.empty else None

            # -- DD layer --
            if MR_CAP == 0.0 and MR_K == 0.0:
                nav_dd_excl = nav_dd.copy() if nav_dd is not None else None
                if nav_dd_excl is not None: nav_dd_excl.name = 'nav_dd_excl'
            elif _nav_hdg9 is not None and not _nav_hdg9.empty:
                _br9 = _nav_hdg9.pct_change().fillna(0)
                _mr9 = nav_mvo.pct_change().fillna(0) if nav_mvo is not None else _br9
                _dd9 = pd.Series(index=_nav_hdg9.index, dtype=float)
                _dd9.iloc[0] = _nav_hdg9.iloc[0]
                _hwm9=_nav_hdg9.iloc[0]; _exp9=1.0; _tr9=_nav_hdg9.iloc[0]
                _th9=_nav_hdg9.iloc[0]; _act9=False; _cf9=0; _nx9=0
                _exp9s = pd.Series(index=_nav_hdg9.index, dtype=float); _exp9s.iloc[0]=1.0
                for _i9,_dt9 in enumerate(_nav_hdg9.index[1:],1):
                    _pv9=_dd9.iloc[_i9-1]
                    if _act9: _th9=_th9*(1+_mr9.get(_dt9,0))
                    _cv9=_pv9*(1+_br9.get(_dt9,0)*_exp9)
                    _dd9.iloc[_i9]=_cv9; _exp9s.iloc[_i9]=_exp9
                    if not _act9: _hwm9=max(_hwm9,_cv9)
                    _cdd9=_cv9/_hwm9-1
                    if _nx9<len(DD_LEVELS):
                        _th9l,_ct9=DD_LEVELS[_nx9]
                        if _cdd9<=-_th9l:
                            _oe9=_exp9; _exp9=_exp9*(1-_ct9)
                            _tr9=_cv9; _th9=_cv9; _cf9=0; _nx9+=1; _act9=True
                            _cv9=_cv9*(1-(_oe9-_exp9)*TRADING_COST_BPS/10000)
                            _dd9.iloc[_i9]=_cv9; _exp9s.iloc[_i9]=_exp9
                            _dd_excl_log.append({'date':_dt9,'event':f'DE-GROSS lv{_nx9}',
                                                 'dd':_cdd9,'exposure_from':_oe9,'exposure_to':_exp9})
                    if _act9 and _exp9<1.0:
                        _tr9=min(_tr9,_cv9); _trec9=_th9/_tr9-1
                        _ys9=pd.Timestamp(f'{_dt9.year}-01-01')
                        _ytd9=((_cv9/_dd9[_dd9.index>=_ys9].iloc[0]-1)
                               if not _dd9[_dd9.index>=_ys9].empty else 0.0)
                        _jan9=(_dt9.month==1 and _dt9.day<=10 and _ytd9>=-DD_ANNUAL_RESET_PCT)
                        _cf9 = _cf9+1 if _trec9>=DD_REENTRY_PCT else 0
                        if _cf9>=DD_REENTRY_CONFIRM or _jan9:
                            _oe9=_exp9; _exp9=1.0; _act9=False; _nx9=0
                            _hwm9=_cv9; _cf9=0
                            _cv9=_cv9*(1-_oe9*TRADING_COST_BPS/10000)
                            _dd9.iloc[_i9]=_cv9; _exp9s.iloc[_i9]=_exp9
                            _dd_excl_log.append({'date':_dt9,'event':'RE-ENTRY 100%',
                                                 'theo_recovery':_trec9,'exposure_from':_oe9,
                                                 'exposure_to':1.0,
                                                 'trigger':'recovery' if not _jan9 else 'annual reset'})
                nav_dd_excl = _dd9; nav_dd_excl.name = 'nav_dd_excl'
            _dg9=[e for e in _dd_excl_log if 'DE-GROSS' in e['event']]
            _re9=[e for e in _dd_excl_log if 'RE-ENTRY' in e['event']]
            print(f"\n  Strategy 9 DD: {len(_dg9)} de-gross, {len(_re9)} re-entries")
        else:
            if not _has_excl:
                print("\n  Strategy 9: no excl portfolios in cache — "
                      "run run_daily_cache_build(excl_only=True) first")
    except Exception as _e9:
        print(f"  Strategy 9 failed: {_e9}"); traceback.print_exc()

    # -- Deferred rebalancing log (needs _s9_wbd which is now populated) ------
    if _do_rebal_log and _dyn_rebal_log:
        try:
            _spx_c = ('SPY' if 'SPY' in Pxs_df.columns else
                      '^GSPC' if '^GSPC' in Pxs_df.columns else None)
            print(f"\n  {'='*95}")
            print(f"  DYNAMIC REBALANCING LOG  (Strategy 9 — Dyn+Hedge+DD+Excl)")
            print(f"  {'='*95}")
            _rdl = [r['dt'] for r in _dyn_rebal_log]
            for ri, rec in enumerate(_dyn_rebal_log):
                dt_r  = rec['dt']
                w_r   = _s9_wbd.get(dt_r, rec['w']) if _s9_wbd else rec['w']
                nxt   = _rdl[ri+1] if ri+1 < len(_rdl) else Pxs_df.index[-1]
                cstr  = ''
                if dt_r in Pxs_df.index and nxt in Pxs_df.index:
                    tk_ = [t for t in w_r.index if t in Pxs_df.columns]
                    sr_ = (Pxs_df.loc[nxt, tk_] / Pxs_df.loc[dt_r, tk_] - 1).fillna(0)
                    pr_ = (w_r.reindex(tk_).fillna(0) * sr_).sum()
                    cstr = (f"  port={pr_*100:+.1f}%  avg={sr_.mean()*100:+.1f}%  "
                            f"med={sr_.median()*100:+.1f}%  best={sr_.max()*100:+.1f}%  "
                            f"worst={sr_.min()*100:+.1f}%  std={sr_.std()*100:.1f}%")
                _tm = {'init':'INIT','regime':'REGIME SWITCH','derisk':'DE-RISK','turnover':'TURNOVER'}
                _tl = _tm.get(rec.get('trigger_type','turnover'),'TURNOVER')
                # Vol
                _pr = [d for d in Pxs_df.index if d <= dt_r][-VOL_WINDOW:]
                _prs = pd.Series(0.0, index=_pr)
                for _dv in _pr:
                    _pv = [d for d in _pr if d < _dv]
                    if _pv:
                        _tv = [t for t in w_r.index if t in Pxs_df.columns]
                        _rv = (Pxs_df.loc[_dv,_tv]/Pxs_df.loc[_pv[-1],_tv]-1).fillna(0)
                        _prs[_dv] = (w_r.reindex(_tv).fillna(0)*_rv).sum()
                _pv_ = _prs.std()*np.sqrt(252)*100
                _sv_ = (Pxs_df.loc[_pr,_spx_c].pct_change().dropna().std()*np.sqrt(252)*100
                        if _spx_c else 0.0)
                # Exclusions
                _ex_ = _load_momentum_exclusions(
                    dt_r, top_n=(int(MR_CAP*len(rec['w'])) if MR_CAP>0 and rec.get('regime')!='mvo' else 0))
                print(f"\n  -- {dt_r.date()}  [rebal #{ri+1}]  [{_tl}]  "
                      f"trigger={rec['trigger']}  regime={rec['regime']}  "
                      f"n={len(w_r)}  eff_N={rec['eff_n']:.1f}  "
                      f"held={rec['days_held']}d  TO={rec['to_dyn']*100:.1f}%  "
                      f"port_vol={_pv_:.1f}%"
                      + (f"  spx_vol={_sv_:.1f}%" if _sv_>0 else "") + "  --")
                if _ex_:
                    print(f"  *** Excluded ({len(_ex_)}): {', '.join(sorted(_ex_.keys()))}")
                if cstr:
                    print(f"  period [{dt_r.date()} -> {nxt.date()}]{cstr}")
                print(f"  {'Ticker':<8}  {'Weight%':>7}  {'AnnAlpha%':>10}  "
                      f"{'AnnVol%':>8}  {'Z-score':>8}  {'Sector':<28}  Source")
                print(f"  {'-'*95}")
                _dd_ = rec['diag_dt']
                _dg_ = (_diag_by_date.get(_dd_) or _last_mvo_diag if _dd_ else _last_mvo_diag)
                _al_ = _dg_.get('alpha_ann', pd.Series(dtype=float))
                _vl_ = _dg_.get('vol_ann',   pd.Series(dtype=float))
                _zl_ = _dg_.get('z_s',       pd.Series(dtype=float))
                _tp_ = _dg_.get('top_n_by',  {})
                _mv_ = set(mvo_weights_by_date.get(_dd_ or dt_r, pd.Series(dtype=float)).index)
                for tkr, wt in w_r.sort_values(ascending=False).items():
                    sec = sectors_s.get(tkr,'')
                    a   = _al_.get(tkr,0.0)*100; v = _vl_.get(tkr,0.0)*100; z = _zl_.get(tkr,0.0)
                    if tkr in _mv_:
                        im_ = '/'.join(m for m in ['E','L','F','P'] if tkr in _tp_.get(m,[]))
                        src = (im_+'/E') if im_ else 'mvo'
                    else:
                        src = 'alpha'
                    af_ = '***' if tkr in _advp_capped.get(dt_r, set()) else '   '
                    print(f"  {tkr:<8}  {wt*100:>6.2f}%  {a:>+9.1f}%  "
                          f"{v:>7.1f}%  {z:>+8.3f}  {sec:<28}  {src}{af_}")
        except Exception as _rle:
            print(f"  Rebalancing log failed: {_rle}"); traceback.print_exc()

        # -- Gross exposure plot -----------------------------------------------
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
            fig.patch.set_facecolor('#FAFAF9')
            for ax in axes:
                ax.set_facecolor('#FAFAF9')
                ax.grid(color='#D3D1C7', linewidth=0.4)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            axes[0].fill_between(exposure_s.index.to_numpy(),
                                 exposure_s.to_numpy(), 0,
                                 where=exposure_s.to_numpy() < 1.0,
                                 color='#D85A30', alpha=0.4, step='post',
                                 label='reduced exposure')
            axes[0].fill_between(exposure_s.index.to_numpy(),
                                 exposure_s.to_numpy(), 0,
                                 where=exposure_s.to_numpy() >= 1.0,
                                 color='#1D9E75', alpha=0.3, step='post',
                                 label='full exposure')
            axes[0].set_ylim(0, 1.2)
            axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            axes[0].set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
            axes[0].set_title('Gross Exposure (DD Policy)',
                              fontsize=10, fontweight='500', color='#2C2C2A')
            axes[0].legend(fontsize=8)

            axes[1].plot(nav_dyn_hedged.index.to_numpy(),
                         nav_dyn_hedged.to_numpy(),
                         color='#378ADD', lw=1.0, label='Dynamic+Hedge')
            axes[1].plot(nav_dd.index.to_numpy(),
                         nav_dd.to_numpy(),
                         color='#1D9E75', lw=1.0, label='Dyn+Hedge+DD')
            axes[1].set_title('NAV: Dynamic+Hedge vs DD Policy',
                              fontsize=10, fontweight='500', color='#2C2C2A')
            axes[1].legend(fontsize=8)
            plt.tight_layout()
            plt.show()
        except Exception as _e:
            print(f"  (exposure plot failed: {_e})")

        # Summary
        degross_events = [e for e in _dd_log if 'DE-GROSS' in e['event']]
        reentry_events = [e for e in _dd_log if 'RE-ENTRY' in e['event']]
        print(f"\n  DD Policy summary: {len(degross_events)} de-gross events, "
              f"{len(reentry_events)} re-entries")
        if degross_events:
            print(f"  {'Date':<12}  {'Event':<18}  {'DD':>8}  {'Exp From':>9}  {'Exp To':>8}")
            print(f"  {'-'*60}")
            for e in _dd_log:
                print(f"  {str(e['date'].date()):<12}  {e['event']:<18}  "
                      f"{e.get('dd', e.get('theo_recovery', 0)):>+7.1%}  "
                      f"{e.get('exposure_from', 0):>8.1%}  "
                      f"{e.get('exposure_to', 0):>8.1%}")
        if not _dd_log:
            print(f"  DD Policy: no events triggered over backtest period")

    # -- Summary ---------------------------------------------------------------
    print(f"\n  {'='*72}")
    print(f"  COMPARISON")
    print(f"  {'='*72}")
    print(f"  All returns net of {TRADING_COST_BPS}bps trading costs")
    print(f"  {'Strategy':<30} {'CAGR':>8} {'Vol':>8} "
          f"{'Sharpe':>8} {'MDD':>8} {'CAGR/DD':>9}")
    print(f"  {'-'*69}")
    for nav_s, lbl in [
        (nav_baseline, 'Baseline (quality)'),
        (nav_alpha,    f'Pure Alpha (conc={conc_factor:.1f}x)'),
        (nav_mvo,      f'MVO (IC={ic}, max={max_weight:.0%})'),
        (nav_hybrid,   'Hybrid (Alpha+MVO avg)'),
        (nav_smart,    f'Smart Hybrid (<{SH_DD_ALPHA:.0%}=alpha,<{SH_DD_HYBRID:.0%}=hyb,else MVO)'),
        (nav_dynamic,      f'Dynamic (TO>a{DYN_TO_THRESHOLD_ALPHA:.0%}/h{DYN_TO_THRESHOLD_HYBRID:.0%}/m{DYN_TO_THRESHOLD_MVO:.0%},hold={DYN_MIN_HOLD_DAYS}d)'),
        (nav_dyn_hedged,   'Dynamic + Hedge'),
        (nav_dd,           f'Dyn+Hedge+DD ({len(DD_LEVELS)} levels)'),
        (nav_dd_excl,      f'Dyn+Hedge+DD+Excl (MR_CAP={MR_CAP:.1%})'),
    ]:
        if nav_s is None or nav_s.empty:
            print(f"  {lbl:<30} {'(no data -- run run_daily_cache_build first)':>40}")
            continue
        n_yrs  = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
        cagr   = (nav_s.iloc[-1] / nav_s.iloc[0]) ** (1/n_yrs) - 1
        vol    = nav_s.pct_change().dropna().std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else np.nan
        mdd    = ((nav_s / nav_s.cummax()) - 1).min()
        cagr_dd = cagr / abs(mdd) if mdd != 0 else np.nan
        print(f"  {lbl:<30} {cagr*100:>7.1f}% {vol*100:>7.1f}% "
              f"{sharpe:>8.2f} {mdd*100:>7.1f}% {cagr_dd:>8.2f}x")

    _aum_nav = (nav_dd         if nav_dd         is not None and not nav_dd.empty         else
                nav_dyn_hedged if nav_dyn_hedged  is not None and not nav_dyn_hedged.empty else
                nav_dynamic    if nav_dynamic     is not None and not nav_dynamic.empty    else
                nav_smart)

    print(f"\n  Yearly returns  (starting AUM: ${AUM/1e6:.1f}M)")
    print(f"  {'Year':<6} {'Baseline':>10} {'Pure Alpha':>12} {'MVO':>9} {'Hybrid':>9} "
          f"{'Smart':>9} {'Dynamic':>9} {'Dyn+Hdg':>9} {'DD Pol':>8} {'Excl':>8} {'AUM($M)':>9}")
    print(f"  {'-'*116}")

    all_years = sorted(set(nav_baseline.index.year))
    for yr in all_years:
        def yr_ret(nav_s):
            yr_nav = nav_s[nav_s.index.year == yr]
            if len(yr_nav) < 2: return np.nan
            return (yr_nav.iloc[-1] / yr_nav.iloc[0] - 1) * 100
        if _aum_nav is not None and not _aum_nav.empty:
            _nav_to_yr = _aum_nav[_aum_nav.index.year <= yr]
            closing_aum = AUM * float(_nav_to_yr.iloc[-1]) if not _nav_to_yr.empty else AUM
        else:
            closing_aum = AUM
        excl_str = (f"{yr_ret(nav_dd_excl):>+7.2f}%"
                    if nav_dd_excl is not None and not nav_dd_excl.empty else f"{'n/a':>8}")
        print(f"  {yr:<6} {yr_ret(nav_baseline):>+9.2f}%  "
              f"{yr_ret(nav_alpha):>+10.2f}%  "
              f"{yr_ret(nav_mvo):>+8.2f}%  "
              f"{yr_ret(nav_hybrid):>+8.2f}%  "
              f"{yr_ret(nav_smart):>+8.2f}%  "
              f"{yr_ret(nav_dynamic):>+8.2f}%  "
              f"{yr_ret(nav_dyn_hedged):>+7.2f}%  "
              f"{yr_ret(nav_dd):>+7.2f}%  "
              f"{excl_str}  ${closing_aum/1e6:>7.1f}M")

    # -- Today snapshot --------------------------------------------------------
    today = Pxs_df.index[-1]
    print(f"\n  Computing today's snapshot ({today.date()})...")

    def _append_today(port_df, scores_dict, label):
        """Append today's top-N to port_df if not already present."""
        if today in port_df.index:
            return port_df
        if today not in scores_dict:
            print(f"  WARNING: no {label} scores for today")
            return port_df
        scores = scores_dict[today]
        if hasattr(scores, 'rename'):
            fdf = scores.rename('factor').to_frame()
        else:
            fdf = pd.Series(scores).rename('factor').to_frame()
        fdf['Sector'] = fdf.index.map(sectors_s)
        fdf = fdf.dropna(subset=['Sector'])
        fdf = fdf.loc[[t for t in fdf.index if t in pxs_cols]]
        if prefilt_pct < 1.0 and len(fdf) > top_n:
            n_keep = max(top_n, int(np.ceil(len(fdf) * prefilt_pct)))
            fdf    = fdf.nlargest(n_keep, 'factor')
        if len(fdf) < top_n:
            print(f"  WARNING: only {len(fdf)} {label} stocks for today")
            return port_df
        ranked = fdf.sort_values('factor', ascending=False)
        if sector_cap is not None:
            top = select_with_sector_cap(ranked, sector_cap, top_n)
        else:
            top = ranked.head(top_n)
        row = {f'Stock{i+1}': t for i, t in enumerate(top.index)}
        for j in range(len(top), top_n):
            row[f'Stock{j+1}'] = None
        today_row = pd.DataFrame([row], index=[today])
        print(f"  {label} today: {top.index.tolist()}")
        return pd.concat([port_df, today_row])

    # Composite scores for today — always clean (strategy display uses unpenalised)
    if today not in composite_by_date:
        today_comp, _ = _cb_build_composite_scores(
            universe=universe, calc_dates=pd.DatetimeIndex([today]),
            Pxs_df=Pxs_df, sectors_s=sectors_s,
            weights_by_year=weights_by_year, regime_s=regime_s,
            volumeTrd_df=volumeTrd_df, model_version=MB_MODEL_VER,
            exclude_factors=['OU'], weights_by_date=weights_by_date,
            mr_scores_by_date=None,
        )
    else:
        today_comp = {today: composite_by_date[today]}

    # Quality scores for today
    today_qual = {}
    if today in quality_wide.index:
        s = quality_wide.loc[today].dropna()
        if not s.empty:
            today_qual[today] = s

    port_alpha    = _append_today(port_alpha,    today_comp, 'Pure Alpha')
    port_mvo      = _append_today(port_mvo,      today_comp, 'MVO')
    port_baseline = _append_today(port_baseline, today_qual, 'Baseline')
    if not port_hybrid.empty:
        port_hybrid = _append_today(port_hybrid, today_comp, 'Smart Hybrid')

    print(f"\n  port_baseline : {len(port_baseline)} dates x {top_n} stocks (last = today)")
    print(f"  port_alpha    : {len(port_alpha)} dates x {top_n} stocks (last = today)")
    print(f"  port_mvo      : {len(port_mvo)} dates x {top_n} stocks (last = today)")
    print(f"  port_hybrid   : {len(port_hybrid)} dates x {len(port_hybrid.columns) if not port_hybrid.empty else 0} stocks (last = today)")


    def _live_pnl_hybrid(weights_by_date, Pxs_df):
        if not weights_by_date:
            print("  No hybrid weights available.")
            return
        today_ts   = Pxs_df.index[-1]
        past_dates = sorted([d for d in weights_by_date if d < today_ts])
        if not past_dates:
            print("  No past rebalance dates found.")
            return
        rebal_dt = past_dates[-1]
        w0       = weights_by_date[rebal_dt]
        w0       = w0[w0 > 1e-6]
        tickers  = [t for t in w0.index if t in Pxs_df.columns]
        if not tickers:
            print("  No valid tickers in portfolio.")
            return
        w0 = w0.reindex(tickers) / w0.reindex(tickers).sum()

        # Price series starting from the day AFTER rebalance to today
        all_px    = Pxs_df.loc[rebal_dt:, tickers].copy()
        if len(all_px) < 2:
            print("  Insufficient price data since rebalance.")
            return
        # Use rebalance date close as cost basis, show P&L from next trading day
        px_base   = all_px.iloc[0]
        px        = all_px.iloc[1:]   # exclude rebalance date itself
        px_norm   = px / px_base
        port_val  = (px_norm * w0.values).sum(axis=1) * 100
        daily_ret = port_val.pct_change()
        daily_ret.iloc[0] = port_val.iloc[0] / 100 - 1  # first day vs rebal close
        days_held = (today_ts - rebal_dt).days

        print(f"\n  Rebalance date : {rebal_dt.date()}  ({days_held} calendar days ago)")
        print(f"  Stocks         : {len(tickers)}")
        print(f"  Cum P&L        : {port_val.iloc[-1] - 100:+.2f}%")
        print(f"\n  {'Date':<12} {'Port':>8} {'Day P&L':>10} {'Cum P&L':>10}  Top contributors")
        print("  " + "-" * 80)

        for dt in daily_ret.index:
            day_ret  = daily_ret.loc[dt]
            cum_ret  = port_val.loc[dt] / 100 - 1
            loc      = px.index.get_loc(dt)
            px_prev  = px_base if loc == 0 else px.iloc[loc - 1]
            w_drift  = w0 * (px_prev / px_base)
            if w_drift.sum() > 0:
                w_drift = w_drift / w_drift.sum()
            stk_ret  = (px.loc[dt] / px_prev - 1).fillna(0)
            contrib  = (stk_ret * w_drift).sort_values()
            top_neg  = contrib.nsmallest(3)
            top_pos  = contrib.nlargest(3)
            parts    = ([f"{t}:{v*100:+.1f}%" for t, v in top_neg.items()] +
                        [f"{t}:{v*100:+.1f}%" for t, v in top_pos.items()])
            print(f"  {str(dt.date()):<12} {port_val.loc[dt]:>7.2f}  "
                  f"{day_ret*100:>+8.2f}%  {cum_ret*100:>+8.2f}%  "
                  + "  ".join(parts))

    # =========================================================================
    # UNIFIED CURRENT PORTFOLIO VIEW
    # =========================================================================
    # Collect drifted weights for each strategy (as % of total AUM)
    # De-grossed strategies: weights sum to gross exposure, not 100%
    today_ts = Pxs_df.index[-1]

    def _drift_weights(wbd, nav_s=None, dd_gross=1.0):
        """Return drifted weights for the most recent rebalance, scaled by gross exposure."""
        past = sorted([d for d in wbd if d <= today_ts])
        if not past: return pd.Series(dtype=float), None
        ldt = past[-1]
        w   = wbd[ldt]; w = w[w > 1e-6]
        tks = [t for t in w.index if t in Pxs_df.columns]
        if tks and ldt in Pxs_df.index and ldt < today_ts:
            px_r = Pxs_df.loc[ldt, tks]; px_t = Pxs_df.loc[today_ts, tks]
            wd   = w.reindex(tks) * (px_t / px_r).fillna(1)
            if wd.sum() > 0: wd = wd / wd.sum()
        else:
            wd = w.reindex(tks) if tks else w
        # Scale by current gross exposure
        wd = wd * dd_gross
        return wd, ldt

    # DD gross exposure for strategy 8/9
    _dd_gross = 1.0
    if 'exposure_s' in dir() or 'exposure_s' in locals():
        try:
            _exp_to_date = exposure_s[exposure_s.index <= today_ts]
            if not _exp_to_date.empty:
                _dd_gross = float(_exp_to_date.iloc[-1])
        except Exception:
            _dd_gross = 1.0

    # Get drifted weights per strategy
    _strat_labels = ['Alpha', 'MVO', 'Hybrid', 'Smart', 'Dynamic', 'Dyn+Hdg', 'DD Pol', 'Excl']

    # Build per-strategy weight dicts from their respective weight trackers
    _alpha_wbd  = {dt: alpha_weights_by_date[dt]
                   for dt in sorted(alpha_weights_by_date) if dt <= today_ts}
    _mvo_wbd    = {dt: mvo_weights_by_date[dt]
                   for dt in sorted(mvo_weights_by_date) if dt <= today_ts}
    _hybrid_wbd = {dt: hybrid_weights_by_date[dt]
                   for dt in sorted(hybrid_weights_by_date) if dt <= today_ts}

    _w_alpha_d,  _ldt_alpha  = _drift_weights(_alpha_wbd)
    _w_mvo_d,    _ldt_mvo    = _drift_weights(_mvo_wbd)
    _w_hybrid_d, _ldt_hybrid = _drift_weights(_hybrid_wbd)
    _w_smart_d,  _ldt_smart  = _drift_weights(smart_hybrid_weights_by_date)
    _w_dyn_d,    _ldt_dyn    = _drift_weights(dyn_weights_by_date)
    _w_hedge_d,  _ldt_hedge  = _drift_weights(dyn_weights_by_date)
    _w_dd_d,     _ldt_dd     = _drift_weights(dyn_weights_by_date, dd_gross=_dd_gross)
    _w_s9_d,     _ldt_s9     = _drift_weights(_s9_wbd if _s9_wbd else {}, dd_gross=_dd_gross)

    # Add hedge positions as negative weights when active
    _hedge_positions = {}
    if hedge_results is not None:
        active_h = hedge_results.get('active_hedges', {})
        for inst, h in active_h.items():
            _hedge_positions[inst] = -abs(h['weight'])

    def _add_hedges(w, hedge_pos):
        if not hedge_pos: return w
        w = w.copy()
        for inst, wh in hedge_pos.items():
            w[inst] = w.get(inst, 0.0) + wh
        return w

    _w_hedge_d = _add_hedges(_w_hedge_d, _hedge_positions)
    _w_dd_d    = _add_hedges(_w_dd_d,    _hedge_positions)
    _w_s9_d    = _add_hedges(_w_s9_d,    _hedge_positions)

    _strat_weights = [_w_alpha_d, _w_mvo_d, _w_hybrid_d, _w_smart_d,
                      _w_dyn_d, _w_hedge_d, _w_dd_d, _w_s9_d]
    _strat_ldts    = [_ldt_alpha, _ldt_mvo, _ldt_hybrid, _ldt_smart,
                      _ldt_dyn, _ldt_hedge, _ldt_dd, _ldt_s9]

    # Union of all stocks
    _all_tks = set()
    for _w in _strat_weights:
        if _w is not None: _all_tks |= set(_w.index)
    _all_tks = sorted(_all_tks)

    # Last-day return per stock
    _prev_ts = Pxs_df.index[-2] if len(Pxs_df.index) >= 2 else today_ts
    _day_ret = {}
    for tkr in _all_tks:
        if tkr in Pxs_df.columns:
            p1 = Pxs_df.loc[_prev_ts, tkr]; p2 = Pxs_df.loc[today_ts, tkr]
            _day_ret[tkr] = (p2 / p1 - 1) * 100 if p1 > 0 else 0.0
        else:
            _day_ret[tkr] = 0.0

    # Sort rows by average weight across strategies descending
    _avg_w = {tkr: np.mean([(_w.get(tkr, 0.0) if _w is not None else 0.0)
                             for _w in _strat_weights]) for tkr in _all_tks}
    _all_tks_sorted = sorted(_all_tks, key=lambda t: -_avg_w[t])

    print("\n  " + "=" * 72)
    print("  CURRENT LIVE PORTFOLIOS — DRIFTED WEIGHTS (% of AUM)")
    print("  " + "=" * 72)

    # Header: rebalance dates per strategy
    _hdr1 = f"  {'':8}  {'Day%':>6}"
    _hdr2 = f"  {'Ticker':8}  {'Day%':>6}"
    for lbl, ldt in zip(_strat_labels, _strat_ldts):
        _hdr1 += f"  {lbl:>8}"
        _hdr2 += f"  {(ldt.strftime('%m/%d') if ldt else 'n/a'):>8}"
    print(_hdr1)
    print("  " + "-" * 8 + "  " + "-" * 6 +
          ("  " + "-" * 8) * len(_strat_labels))
    print(f"  {'Last reb:':8}  {'':6}" + "".join(
        f"  {(ldt.strftime('%m/%d') if ldt else 'n/a'):>8}"
        for ldt in _strat_ldts))
    print("  " + "-" * 8 + "  " + "-" * 6 +
          ("  " + "-" * 8) * len(_strat_labels))

    for tkr in _all_tks_sorted:
        dr   = _day_ret.get(tkr, 0.0)
        row  = f"  {tkr:<8}  {dr:>+5.1f}%"
        any_nonzero = False
        for _w in _strat_weights:
            wt = (_w.get(tkr, 0.0) if _w is not None else 0.0) * 100
            if abs(wt) >= 0.01: any_nonzero = True
            row += f"  {wt:>7.2f}%" if abs(wt) >= 0.01 else f"  {'':>8}"
        if any_nonzero:
            print(row)

    # =========================================================================
    # 10-DAY DAILY P&L TABLE — ALL 9 STRATEGIES
    # =========================================================================
    print("\n  " + "=" * 72)
    print("  DAILY P&L — LAST 10 TRADING DAYS (all strategies)")
    print("  " + "=" * 72)

    _nav_series = [
        ('Pure Alpha', nav_alpha),
        ('MVO',        nav_mvo),
        ('Hybrid',     nav_hybrid),
        ('Smart',      nav_smart),
        ('Dynamic',    nav_dynamic),
        ('Dyn+Hdg',    nav_dyn_hedged),
        ('DD Pol',     nav_dd),
        ('Excl',       nav_dd_excl),
    ]

    # Add hedge NAV adjustments for hedged strategies
    # nav_dyn_hedged and nav_dd already include hedges from backtest computation

    _last10 = [d for d in Pxs_df.index if d <= today_ts][-10:]

    # Build daily returns from NAV series
    _daily_rets = {}
    for lbl, nav_s in _nav_series:
        if nav_s is None or nav_s.empty:
            _daily_rets[lbl] = {}; continue
        nav_sub = nav_s[nav_s.index <= today_ts]
        rets = {}
        for i, dt in enumerate(_last10):
            nav_at = nav_sub[nav_sub.index <= dt]
            if nav_at.empty: continue
            # Previous available NAV
            prev_dates = [d for d in nav_sub.index if d < dt]
            if not prev_dates: continue
            prev_nav = float(nav_sub.loc[prev_dates[-1]])
            curr_nav = float(nav_at.iloc[-1])
            rets[dt] = (curr_nav / prev_nav - 1) * 100
        _daily_rets[lbl] = rets

    # Print header
    _col_lbls = [lbl for lbl, _ in _nav_series]
    print(f"\n  {'Date':<12}" + "".join(f"  {lbl:>9}" for lbl in _col_lbls))
    print("  " + "-" * 12 + ("  " + "-" * 9) * len(_col_lbls))

    _cum = {lbl: 1.0 for lbl in _col_lbls}
    for dt in _last10:
        row = f"  {dt.strftime('%Y-%m-%d'):<12}"
        for lbl in _col_lbls:
            r = _daily_rets[lbl].get(dt, None)
            if r is not None:
                _cum[lbl] *= (1 + r / 100)
                row += f"  {r:>+8.2f}%"
            else:
                row += f"  {'n/a':>9}"
        print(row)

    # Cumulative row
    print("  " + "-" * 12 + ("  " + "-" * 9) * len(_col_lbls))
    cum_row = f"  {'Cumul':12}"
    for lbl in _col_lbls:
        c = (_cum[lbl] - 1) * 100
        cum_row += f"  {c:>+8.2f}%"
    print(cum_row)



    # -- Live hedge status -----------------------------------------------------
    if hedge_results is not None:
        print("\n  " + "=" * 72)
        print("  LIVE HEDGE STATUS")
        print("  " + "=" * 72)
        active = hedge_results.get('active_hedges', {})
        acct   = hedge_results['hedge_account_by_date']
        acct_s = pd.Series(acct)
        curr_bal = acct_s.iloc[-1] if not acct_s.empty else 0.0

        if active:
            # Find open entry from hedge_log
            open_events = [e for e in hedge_results['hedge_log'] if e['event'] == 'OPEN']
            last_open   = open_events[-1] if open_events else None
            entry_dt    = last_open['date'] if last_open else None
            days_open   = (today_ts - entry_dt).days if entry_dt else 0

            # Unrealised P&L
            unreal_pnl = 0.0
            for inst, h in active.items():
                if inst in Pxs_df.columns and today_ts in Pxs_df.index:
                    if h['entry_px'] and not np.isnan(h['entry_px']):
                        curr_px   = Pxs_df.loc[today_ts, inst]
                        unreal_pnl += -(curr_px / h['entry_px'] - 1) * h['weight']

            print(f"\n  Status         : HEDGED  (open since {entry_dt.date() if entry_dt else 'n/a'}, {days_open}d)")
            print(f"  Unrealised P&L : {unreal_pnl:+.4%}")
            print(f"  Hedge acct bal : {curr_bal:+.4%} (closed episodes)")
            print(f"\n  {'Instrument':<10}  {'Weight':>8}  {'Entry Px':>10}  {'Curr Px':>10}  {'Unreal P&L':>12}  {'Beta':>7}  {'Eff':>7}")
            print("  " + "-" * 70)
            for inst, h in active.items():
                if inst in Pxs_df.columns and today_ts in Pxs_df.index:
                    curr_px  = Pxs_df.loc[today_ts, inst]
                    inst_pnl = -(curr_px / h['entry_px'] - 1) * h['weight']                                if h['entry_px'] and not np.isnan(h['entry_px']) else np.nan
                    print(f"  {inst:<10}  {h['weight']:>8.2%}  "
                          f"{h['entry_px']:>10.4f}  {curr_px:>10.4f}  "
                          f"{inst_pnl:>+11.4%}  {h['beta']:>7.3f}  {h['eff']:>7.3f}")
        else:
            print(f"\n  Status         : FLAT (no active hedges)")
            print(f"  Hedge acct bal : {curr_bal:+.4%} (pending sweep at next rebal)"
                  if curr_bal != 0 else
                  f"  Hedge acct bal : 0.00% (clean)")

    # -- Hypothetical Dyn+Hedge+DD+Excl rebalance as of today ----------------
    # Only show if today is NOT already a rebalance day for strategy 9
    _today_is_s9_rebal = any(r['dt'] == today_ts for r in _dyn_rebal_log)
    w_smart_t = None   # will hold today's excl-filtered portfolio
    if _today_is_s9_rebal:
        print("\n  " + "=" * 72)
        print(f"  NOTE: Today ({today_ts.date()}) IS a strategy 9 rebalance day.")
        print(f"  The current live portfolio above already reflects today's rebalance.")
        print("  " + "=" * 72)
        # Use the actual S9 rebalance portfolio as w_smart_t for trade summary
        w_smart_t = _s9_wbd.get(today_ts, pd.Series(dtype=float))
    else:
        print("\n  " + "=" * 72)
        print(f"  HYPOTHETICAL DYN+HEDGE+DD+EXCL REBALANCE AS OF {today_ts.date()}")
        print("  " + "=" * 72)
    if not _today_is_s9_rebal:
      try:
        if today_ts in composite_by_date:
            comp_today = composite_by_date[today_ts]
        elif today_comp:
            comp_today = today_comp[today_ts]
        else:
            comp_today = None

        if comp_today is not None:
            cands_t = comp_today.dropna()
            cands_t = cands_t.loc[[t for t in cands_t.index if t in pxs_cols]]
            n_pf    = max(n_cands, int(np.ceil(len(cands_t) * prefilt_pct)))
            cands_t = cands_t.nlargest(min(n_pf, len(cands_t)))

            # Apply excl filter (skip in MVO regime)
            _dd_t    = _sh_nav_running / _sh_hwm - 1
            _reg_raw = ('alpha' if _dd_t >= -SH_DD_ALPHA else
                        'hybrid' if _dd_t >= -SH_DD_HYBRID else 'mvo')
            _mr_n_t  = int(MR_CAP * len(cands_t)) if (MR_CAP > 0 and _reg_raw != 'mvo') else 0
            _excl_t  = _load_momentum_exclusions(today_ts, top_n=_mr_n_t) if _mr_n_t > 0 else {}
            cands_excl_t = cands_t[~cands_t.index.isin(_excl_t.keys())] if _excl_t else cands_t

            mvo_cands_t  = cands_excl_t.index.tolist()[:n_cands]
            valid_snap_t = [d for d in snapshot_dates if d <= today_ts]

            if valid_snap_t:
                w_mvo_t, mvo_diag_t = _mb_solve_mvo(
                    dt=today_ts, candidates=mvo_cands_t,
                    composite_scores=comp_today, Pxs_df=Pxs_df,
                    sectors_s=sectors_s, volumeTrd_df=volumeTrd_df,
                    model_version=MB_MODEL_VER,
                    pca_var_threshold=pca_var_threshold,
                    ic=ic, max_weight=max_weight, min_weight=min_weight,
                    zscore_cap=zscore_cap, risk_aversion=risk_aversion,
                    X_snapshots=X_snapshots, snapshot_dates=snapshot_dates,
                    top_n=top_n, min_cov_matrices=min_cov_matrices,
                )

                # Alpha weights (excl-filtered)
                ranked_t     = cands_excl_t.sort_values(ascending=False)
                alpha_port_t = [t for t in ranked_t.head(top_n).index if t in pxs_cols]
                n_pt         = len(alpha_port_t)
                n_top_t      = int(np.ceil(n_pt / 2))
                n_bot_t      = n_pt - n_top_t
                if conc_factor == 1.0 or n_bot_t == 0:
                    w_alpha_t = pd.Series(1.0 / n_pt, index=alpha_port_t)
                else:
                    top_a = conc_factor / (conc_factor + 1.0)
                    bot_a = 1.0 / (conc_factor + 1.0)
                    w_alpha_t = pd.Series({
                        t: (top_a / n_top_t if j < n_top_t else bot_a / n_bot_t)
                        for j, t in enumerate(alpha_port_t)
                    })

                if not w_mvo_t.empty and w_mvo_t.sum() > 0:
                    w_mvo_t_nz = w_mvo_t[w_mvo_t > 1e-6]

                    # Blind hybrid for today
                    all_t_hyp = list(set(w_alpha_t.index) | set(w_mvo_t_nz.index))
                    w_hyp_t   = (w_alpha_t.reindex(all_t_hyp).fillna(0.0) +
                                 w_mvo_t_nz.reindex(all_t_hyp).fillna(0.0)) / 2.0
                    w_hyp_t   = w_hyp_t[w_hyp_t > 0]
                    w_hyp_t   = w_hyp_t / w_hyp_t.sum()

                    # Current drawdown from smart hybrid running NAV
                    dd_today = _sh_nav_running / _sh_hwm - 1
                    if dd_today >= -SH_DD_ALPHA:
                        w_smart_t  = w_alpha_t
                        regime_t   = f"alpha (dd={dd_today*100:+.1f}%)"
                    elif dd_today >= -SH_DD_HYBRID:
                        w_smart_t  = w_hyp_t
                        regime_t   = f"hybrid (dd={dd_today*100:+.1f}%)"
                    else:
                        w_smart_t  = w_mvo_t_nz
                        regime_t   = f"MVO (dd={dd_today*100:+.1f}%)"
                    w_smart_t = w_smart_t[w_smart_t > 1e-6]
                    w_smart_t = w_smart_t / w_smart_t.sum()

                    mvo_set_t   = set(w_mvo_t_nz.index)
                    alpha_set_t = set(w_alpha_t.index)
                    top_n_by_t  = mvo_diag_t.get('top_n_by', {})
                    eff_n_t     = 1.0 / (w_smart_t**2).sum()

                    # Portfolio vol and SPX vol
                    _spx_c = ('SPY' if 'SPY' in Pxs_df.columns else
                              '^GSPC' if '^GSPC' in Pxs_df.columns else None)
                    _past_t = [d for d in Pxs_df.index if d <= today_ts][-VOL_WINDOW:]
                    _pr_t = pd.Series(0.0, index=_past_t)
                    for _dv in _past_t:
                        _pv = [d for d in _past_t if d < _dv]
                        if _pv:
                            _tv = [t for t in w_smart_t.index if t in Pxs_df.columns]
                            _rv = (Pxs_df.loc[_dv,_tv]/Pxs_df.loc[_pv[-1],_tv]-1).fillna(0)
                            _pr_t[_dv] = (w_smart_t.reindex(_tv).fillna(0)*_rv).sum()
                    _pvol_t = _pr_t.std() * np.sqrt(252) * 100
                    _svol_t = 0.0
                    if _spx_c:
                        _svol_t = Pxs_df.loc[_past_t,_spx_c].pct_change().dropna().std()*np.sqrt(252)*100

                    print(f"\n  regime={regime_t}  n={len(w_smart_t)}  "
                          f"eff_N={eff_n_t:.1f}  "
                          f"min={w_smart_t.min():.1%}  max={w_smart_t.max():.1%}  "
                          f"port_vol={_pvol_t:.1f}%"
                          + (f"  spx_vol={_svol_t:.1f}%" if _svol_t > 0 else ""))
                    if _excl_t:
                        _excl_t_str = ', '.join(f"{t}({s:.2f})" for t, s in _excl_t.items())
                        print(f"  Excluded ({len(_excl_t)}): {_excl_t_str}")
                    if MR_K > 0:
                        _mr_hyp10 = _load_momentum_exclusions(today_ts, top_n=10)
                        if _mr_hyp10:
                            _mr_h_str = ', '.join(f"{t}({s:.2f})" for t, s in _mr_hyp10.items())
                            print(f"  MR top-10 (k={MR_K}): {_mr_h_str}")
                    vol_ann_t   = mvo_diag_t.get('vol_ann',   pd.Series(dtype=float))
                    z_s_t       = mvo_diag_t.get('z_s',       pd.Series(dtype=float))
                    alpha_ann_t = mvo_diag_t.get('alpha_ann', pd.Series(dtype=float))
                    print(f"  {'Ticker':<8}  {'Weight%':>7}  {'AnnAlpha%':>10}  "
                          f"{'AnnVol%':>8}  {'Z-score':>8}  {'Sector':<28}  Source")
                    print("  " + "-" * 95)
                    for tkr, wt in w_smart_t.sort_values(ascending=False).items():
                        sec  = sectors_s.get(tkr, '')
                        a    = alpha_ann_t.get(tkr, 0.0) * 100
                        v    = vol_ann_t.get(tkr,   0.0) * 100
                        z    = z_s_t.get(tkr,       0.0)
                        if tkr in mvo_set_t:
                            in_m = '/'.join(m for m in ['E','L','F','P']
                                            if tkr in top_n_by_t.get(m, []))
                            src  = (in_m + '/E') if in_m else 'mvo'
                        else:
                            src = 'alpha'
                        # Apply ADVP cap check for today's hypothetical
                        curr_aum_t = AUM * _running_nav
                        advp_flag_t = ''
                        if volumeRaw_df is not None and curr_aum_t > 0:
                            past_d = [d for d in Pxs_df.index if d <= today_ts][-ADV_WINDOW:]
                            if tkr in Pxs_df.columns and tkr in volumeRaw_df.columns:
                                px_s  = Pxs_df.loc[past_d, tkr].dropna()
                                vol_s = volumeRaw_df.loc[past_d, tkr].dropna()
                                com   = px_s.index.intersection(vol_s.index)
                                if len(com) >= 3:
                                    adv_d = (px_s.reindex(com) * vol_s.reindex(com)).median()
                                    cap_w = adv_d * advp_cap / curr_aum_t
                                    if wt > cap_w + 1e-4:
                                        advp_flag_t = '***'
                        print(f"  {tkr:<8}  {wt*100:>6.2f}%  {a:>+9.1f}%  "
                              f"{v:>7.1f}%  {z:>+8.3f}  {sec:<28}  {src}{advp_flag_t}")
        else:
            print("  No composite scores available for today.")
      except Exception as e:
        print(f"  Could not compute hypothetical smart hybrid rebalance: {e}")

    # -- Unified trade summary table -------------------------------------------
    print("\n  " + "=" * 72)
    print(f"  TRADE SUMMARY  —  {today_ts.date()}")
    print("  " + "=" * 72)
    try:
        # Determine if today has any ACTUAL events for each strategy
        dyn_rebal_today   = today_ts in set([r['dt'] for r in _dyn_rebal_log])
        hedge_event_today = hedge_results is not None and any(
            e['date'].date() == today_ts.date()
            for e in hedge_results.get('hedge_log', []))
        dd_event_today    = any(
            e['date'].date() == today_ts.date() for e in _dd_log)
        has_actual = dyn_rebal_today or hedge_event_today or dd_event_today

        # ── Trade delta calculation ──────────────────────────────────────────
        # Simple: Δ = new_portfolio - old_portfolio, NAs filled with zero.
        # "old" = last rebalance weights drifted to today's prices
        # "new" = today's rebalanced portfolio (or same as old if no rebalance)

        # Dynamic old: last rebal weights drifted to today
        past_dyn       = sorted([d for d in dyn_weights_by_date if d <= today_ts])
        w_dyn_deployed = dyn_weights_by_date[past_dyn[-1]] if past_dyn else pd.Series(dtype=float)

        if dyn_rebal_today and len(past_dyn) >= 2:
            w_dyn_new = w_dyn_deployed   # post-rebal (already set by backtest loop)
            prev_recs = [r for r in _dyn_rebal_log if r['dt'] < today_ts]
            prev_dt   = prev_recs[-1]['dt'] if prev_recs else past_dyn[-2]
            w_prev    = prev_recs[-1]['w']  if prev_recs else dyn_weights_by_date[prev_dt]
            tks = [t for t in w_prev.index if t in Pxs_df.columns]
            if tks and prev_dt in Pxs_df.index and today_ts in Pxs_df.index:
                d = w_prev.reindex(tks) * (Pxs_df.loc[today_ts, tks] /
                                            Pxs_df.loc[prev_dt,  tks]).fillna(1)
                w_dyn_old = d / d.sum() if d.sum() > 0 else w_prev
            else:
                w_dyn_old = w_prev
        else:
            # No rebalance today — hypothetical new = same as what backtest holds today
            # (forward-filled from last rebalance, already ADVP-filtered)
            w_dyn_new = w_dyn_deployed
            last_recs = [r for r in _dyn_rebal_log if r['dt'] <= today_ts]
            if last_recs:
                last_dt = last_recs[-1]['dt']
                w_last  = last_recs[-1]['w']
                tks = [t for t in w_last.index if t in Pxs_df.columns]
                if tks and last_dt in Pxs_df.index and today_ts in Pxs_df.index:
                    d = w_last.reindex(tks) * (Pxs_df.loc[today_ts, tks] /
                                                Pxs_df.loc[last_dt,  tks]).fillna(1)
                    w_dyn_old = d / d.sum() if d.sum() > 0 else w_last
                else:
                    w_dyn_old = w_last
            else:
                w_dyn_old = w_dyn_deployed

        # Strategy 9 old: last S9 rebal weights drifted to today
        _s9_rebal_dates_t = [r['dt'] for r in _dyn_rebal_log if r['dt'] <= today_ts]
        if _s9_rebal_dates_t and _s9_rebal_dates_t[-1] == today_ts:
            _s9_ldt = _s9_rebal_dates_t[-2] if len(_s9_rebal_dates_t) >= 2 else None
        else:
            _s9_ldt = _s9_rebal_dates_t[-1] if _s9_rebal_dates_t else None

        # Strategy 9 new: today's S9 rebalanced portfolio (or last if no rebal)
        _s9_new = _s9_wbd.get(today_ts, _s9_wbd.get(_s9_ldt, pd.Series(dtype=float)))

        _w_s9_old = w_dyn_old   # fallback
        if _s9_ldt and _s9_wbd:
            _s9_lw = _s9_wbd.get(_s9_ldt, pd.Series(dtype=float))
            if not _s9_lw.empty:
                _s9_tk = [t for t in _s9_lw.index if t in Pxs_df.columns]
                if _s9_tk and _s9_ldt in Pxs_df.index and today_ts in Pxs_df.index:
                    _s9_d = _s9_lw.reindex(_s9_tk) * (Pxs_df.loc[today_ts, _s9_tk] /
                                                        Pxs_df.loc[_s9_ldt,  _s9_tk]).fillna(1)
                    _w_s9_old = _s9_d / _s9_d.sum() if _s9_d.sum() > 0 else _s9_lw
                else:
                    _w_s9_old = _s9_lw

        # Hedge positions
        active_hedges_now = hedge_results.get('active_hedges', {}) if hedge_results else {}

        # Universe = union of old + new portfolios only (no external sources)
        all_tickers = (set(w_dyn_old.index) | set(w_dyn_new.index) |
                       set(_w_s9_old.index) | set(_s9_new.index))
        if active_hedges_now:
            all_tickers |= set(active_hedges_now.keys())

        curr_exposure = exposure_s.iloc[-1] if (nav_dd is not None and
                        not nav_dd.empty and len(exposure_s) > 0) else 1.0

        rows = []
        for tkr in sorted(all_tickers):
            old_dyn   = w_dyn_old.get(tkr, 0.0)
            new_dyn   = w_dyn_new.get(tkr, 0.0)
            delta_dyn = (new_dyn - old_dyn) * 100

            # Dyn+Hedge column: same stock changes + hedge positions
            if tkr in active_hedges_now:
                # Hedge instrument — show current short weight (no change if already open)
                h = active_hedges_now[tkr]
                if hedge_event_today:
                    # Hedge just opened or closed today
                    close_events_today = [e for e in hedge_results.get('hedge_log',[])
                                         if e['date'].date() == today_ts.date()
                                         and e['event'] == 'CLOSE'
                                         and tkr in e.get('instruments',[])]
                    open_events_today  = [e for e in hedge_results.get('hedge_log',[])
                                         if e['date'].date() == today_ts.date()
                                         and e['event'] == 'OPEN'
                                         and tkr in e.get('instruments',[])]
                    if open_events_today:
                        delta_hdg = -h['weight'] * 100   # new short
                    elif close_events_today:
                        delta_hdg = +h['weight'] * 100   # covering short
                    else:
                        delta_hdg = 0.0
                else:
                    delta_hdg = 0.0  # active but no change today
            else:
                delta_hdg = delta_dyn  # same stock changes as Dynamic

            # DD Policy column: stock changes scaled by exposure
            delta_dd = delta_dyn * curr_exposure
            if dd_event_today:
                dd_ev_today = [e for e in _dd_log
                               if e['date'].date() == today_ts.date()]
                if dd_ev_today and tkr not in active_hedges_now:
                    ev = dd_ev_today[-1]
                    old_exp = ev.get('exposure_from', curr_exposure)
                    new_exp = ev.get('exposure_to', curr_exposure)
                    delta_dd = old_dyn * (new_exp - old_exp) * 100

            # Excl (S9) column: new = today's S9 portfolio; old = previous S9 drifted
            new_s9   = _s9_new.get(tkr, 0.0)
            delta_s9 = (new_s9 - _w_s9_old.get(tkr, 0.0)) * 100

            if abs(delta_dyn) > 0.01 or abs(delta_hdg) > 0.01 or abs(delta_dd) > 0.01 or abs(delta_s9) > 0.01:
                rows.append((tkr, delta_dyn, delta_hdg, delta_dd, delta_s9))

        label = "[ACTUAL TRADES]" if has_actual else "[THEORETICAL — hypothetical rebalance if triggered today]"
        print(f"\n  {label}")
        if rows:
            print(f"\n  {'Ticker':<10}  {'Δ Dynamic':>12}  {'Δ Dyn+Hedge':>13}  {'Δ DD Policy':>13}  {'Δ Excl (S9)':>13}")
            print(f"  {'-'*66}")
            for tkr, dd, dh, dp, ds9 in sorted(rows, key=lambda x: x[4], reverse=True):
                hedge_tag = '  [SHORT]' if tkr in active_hedges_now else ''
                print(f"  {tkr:<10}  {dd:>+11.2f}%  {dh:>+12.2f}%  {dp:>+12.2f}%  {ds9:>+12.2f}%{hedge_tag}")
        else:
            if has_actual:
                print(f"\n  No material allocation changes  (threshold: 0.01%)")
            else:
                print(f"\n  Hypothetical portfolio identical to current drifted weights — "
                      f"no trades would be made if rebalancing today")
    except Exception as e:
        print(f"  Could not compute trade summary: {e}")

    # -- Trading cost summary -------------------------------------------------
    _cost_by_date['dynamic_excl'] = _s9_costs if _s9_costs else _cost_by_date.get('dynamic', {})
    print("\n  " + "=" * 72)
    print(f"  TRADING COSTS SUMMARY  (@ {TRADING_COST_BPS}bps per side)")
    print("  " + "=" * 72)
    strat_labels = {
        "baseline":     "Baseline",
        "alpha":        "Pure Alpha",
        "mvo":          "MVO",
        "hybrid":       "Hybrid",
        "smart":        "Smart Hybrid",
        "dynamic":      "Dynamic",
        "dynamic_excl": f"Excl (S9)",
    }

    # Build hedge costs by year from hedge_log
    hedge_costs_by_year = {}
    if hedge_results is not None:
        for e in hedge_results.get('hedge_log', []):
            if e['event'] == 'CLOSE':
                yr = e['date'].year
                ep_cost = sum(2 * TRADING_COST_BPS / 10000 * h['weight']
                              for h in e.get('details', {}).values())
                hedge_costs_by_year[yr] = hedge_costs_by_year.get(yr, 0) + ep_cost
    show_hedge = bool(hedge_costs_by_year)

    # Build DD policy costs by year from _dd_log
    dd_costs_by_year = {}
    if _dd_log:
        for e in _dd_log:
            yr = e['date'].year
            traded_pct = abs(e.get('exposure_to', 0) - e.get('exposure_from', 0))
            ep_cost    = traded_pct * TRADING_COST_BPS / 10000
            dd_costs_by_year[yr] = dd_costs_by_year.get(yr, 0) + ep_cost
    show_dd = bool(dd_costs_by_year)

    # Header
    print(f"\n  {'Year':<6}", end="")
    for s in strat_labels:
        print(f"  {strat_labels[s]:>13}", end="")
    if show_hedge:
        print(f"  {'Hedge Costs':>13}", end="")
    if show_dd:
        print(f"  {'DD Costs':>13}", end="")
    print()
    sep_width = 78 + (16 if show_hedge else 0) + (16 if show_dd else 0)
    print(f"  {'-'*sep_width}")

    all_years = sorted(set(d.year for costs in _cost_by_date.values()
                           for d in costs.keys()))
    for yr in all_years:
        print(f"  {yr:<6}", end="")
        for s in strat_labels:
            yr_cost = sum(v for d, v in _cost_by_date[s].items() if d.year == yr)
            print(f"  {yr_cost*100:>12.2f}%", end="")
        if show_hedge:
            hc = hedge_costs_by_year.get(yr, 0)
            print(f"  {hc*100:>12.2f}%", end="")
        if show_dd:
            dc = dd_costs_by_year.get(yr, 0)
            print(f"  {dc*100:>12.2f}%", end="")
        print()
    # Total row
    print(f"  {'Total':<6}", end="")
    for s in strat_labels:
        tot = sum(_cost_by_date[s].values())
        print(f"  {tot*100:>12.2f}%", end="")
    if show_hedge:
        print(f"  {sum(hedge_costs_by_year.values())*100:>12.2f}%", end="")
    if show_dd:
        print(f"  {sum(dd_costs_by_year.values())*100:>12.2f}%", end="")
    print()
    # Annualized row
    n_yrs_cost = (max(all_years) - min(all_years) + 1) if all_years else 1
    print(f"  {'Ann.':<6}", end="")
    for s in strat_labels:
        tot = sum(_cost_by_date[s].values())
        print(f"  {tot/n_yrs_cost*100:>12.2f}%", end="")
    if show_hedge:
        print(f"  {sum(hedge_costs_by_year.values())/n_yrs_cost*100:>12.2f}%", end="")
    if show_dd:
        print(f"  {sum(dd_costs_by_year.values())/n_yrs_cost*100:>12.2f}%", end="")
    print()

    # -- Dynamic trigger type summary --
    if _dyn_rebal_log:
        type_counts = Counter(r.get('trigger_type', 'turnover') for r in _dyn_rebal_log)
        total_r     = len(_dyn_rebal_log)
        print(f"\n  Dynamic rebalancing trigger summary ({total_r} total):")
        print(f"  {'Trigger':<20}  {'Count':>6}  {'%':>7}  Distribution")
        print(f"  {'-'*55}")
        for ttype, tlbl in [('init',      'Initialisation'),
                             ('dd_regime', 'DD regime force'),
                             ('regime',    'Regime switch'),
                             ('turnover',  'Turnover > thres'),
                             ('derisk',    'De-risk (vol)')]:
            cnt = type_counts.get(ttype, 0)
            bar = '#' * int(cnt / total_r * 30)
            print(f"  {tlbl:<20}  {cnt:>6}  {cnt/total_r*100:>6.1f}%  {bar}")

    # -- Dynamic rebalancing frequency stats --
    if _cost_by_date.get('dynamic'):
        rebal_dates_dyn = sorted(_cost_by_date['dynamic'].keys())
        if len(rebal_dates_dyn) > 1:
            gaps = [(rebal_dates_dyn[i+1] - rebal_dates_dyn[i]).days
                    for i in range(len(rebal_dates_dyn)-1)]
            print(f"\n  Dynamic rebalancing frequency ({len(rebal_dates_dyn)} rebalances):")
            print(f"    Avg holding period : {np.mean(gaps):.1f} days")
            print(f"    Median             : {np.median(gaps):.1f} days")
            print(f"    Min                : {min(gaps)} days")
            print(f"    Max                : {max(gaps)} days")
            print(f"    First rebalance    : {rebal_dates_dyn[0].date()}")
            print(f"    Last rebalance     : {rebal_dates_dyn[-1].date()}")
            # Yearly breakdown
            print(f"\n    {'Year':<6}  {'Rebalances':>12}  {'Avg hold (days)':>16}")
            print(f"    {'-'*38}")
            all_years_dyn = sorted(set(d.year for d in rebal_dates_dyn))
            for yr in all_years_dyn:
                yr_rebals = [d for d in rebal_dates_dyn if d.year == yr]
                yr_gaps   = [(rebal_dates_dyn[rebal_dates_dyn.index(d)+1] - d).days
                             for d in yr_rebals
                             if rebal_dates_dyn.index(d)+1 < len(rebal_dates_dyn)
                             and rebal_dates_dyn[rebal_dates_dyn.index(d)+1].year == yr]
                avg_hold  = f"{np.mean(yr_gaps):.1f}" if yr_gaps else "n/a"
                print(f"    {yr:<6}  {len(yr_rebals):>12}  {avg_hold:>16}")

    # -- Consolidated portfolio return statistics (per holding period) --------
    # Recompute from strategy 9 (replaces smart hybrid stats)
    if _s9_wbd and _dyn_rebal_log:
        _s9_rdates = sorted([r['dt'] for r in _dyn_rebal_log])
        all_portfolio_returns = []
        all_stock_returns     = []
        all_turnover_ratios   = []
        for _i9s, _dt9s in enumerate(_s9_rdates):
            _w9s = _s9_wbd.get(_dt9s, pd.Series(dtype=float))
            if _w9s.empty: continue
            if _i9s > 0:
                _w9p = _s9_wbd.get(_s9_rdates[_i9s-1], pd.Series(dtype=float))
                if not _w9p.empty:
                    _at9 = list(set(_w9s.index)|set(_w9p.index))
                    all_turnover_ratios.append(
                        (_w9s.reindex(_at9).fillna(0)-_w9p.reindex(_at9).fillna(0)).abs().sum()*100)
            _nxt9 = _s9_rdates[_i9s+1] if _i9s+1 < len(_s9_rdates) else Pxs_df.index[-1]
            if _dt9s in Pxs_df.index and _nxt9 in Pxs_df.index:
                _tk9 = [t for t in _w9s.index if t in Pxs_df.columns]
                _sr9 = (Pxs_df.loc[_nxt9,_tk9]/Pxs_df.loc[_dt9s,_tk9]-1).fillna(0)
                all_portfolio_returns.append((_w9s.reindex(_tk9).fillna(0)*_sr9).sum()*100)
                all_stock_returns.extend(_sr9.tolist())
    import matplotlib.pyplot as plt
    if all_portfolio_returns:
        port_arr = np.array(all_portfolio_returns)
        q_labels = ['Q1 (0-20%)', 'Q2 (20-40%)', 'Q3 (40-60%)', 'Q4 (60-80%)', 'Q5 (80-100%)']
        port_q   = np.percentile(port_arr, [0, 20, 40, 60, 80, 100])

        print("\n  " + "=" * 72)
        print("  PORTFOLIO RETURN STATISTICS — STRATEGY 9 (per holding period)")
        print("  " + "=" * 72)
        print(f"\n  Total periods      : {len(port_arr)}")
        print(f"  Mean               : {port_arr.mean():>+.2f}%")
        print(f"  Median             : {np.median(port_arr):>+.2f}%")
        print(f"  Std Dev            : {port_arr.std():>.2f}%")
        print(f"  Min                : {port_arr.min():>+.2f}%")
        print(f"  Max                : {port_arr.max():>+.2f}%")
        pct_pos = (port_arr > 0).mean() * 100
        print(f"  % positive periods : {pct_pos:.1f}%")
        print(f"\n  Quintile boundaries:")
        for i, lbl in enumerate(q_labels):
            print(f"    {lbl:<18}  {port_q[i]:>+7.2f}%  ->  {port_q[i+1]:>+7.2f}%")
        if all_turnover_ratios:
            print(f"\n  Avg turnover per rebalance : {np.mean(all_turnover_ratios):.1f}%")
            print(f"  Median turnover            : {np.median(all_turnover_ratios):.1f}%")

        total_periods = sum(_sh_regime_counts.values())
        if total_periods > 0:
            print(f"\n  Smart hybrid regime prevalence ({total_periods} rebalances):")
            regime_display = {'alpha': 'alpha', 'hybrid': 'hybrid', 'mvo': 'mvo'}
            regime_bar     = {'alpha': 'a', 'hybrid': 'h', 'mvo': 'M'}
            for regime, count in _sh_regime_counts.items():
                bar = regime_bar[regime] * int(count / total_periods * 40)
                print(f"    {regime_display[regime]:<8}  {count:>4}  ({count/total_periods*100:>5.1f}%)  {bar}")

            # Yearly breakdown
            if _sh_regime_by_date:
                years = sorted(set(d.year for d in _sh_regime_by_date))
                print(f"\n  Regime breakdown by year:")
                print(f"  {'Year':<6}  {'Total':>6}  {'Alpha':>6}  {'Hybrid':>7}  {'MVO':>5}  Distribution")
                print(f"  {'-'*65}")
                for yr in years:
                    yr_dates   = {d: r for d, r in _sh_regime_by_date.items() if d.year == yr}
                    yr_total   = len(yr_dates)
                    yr_alpha   = sum(1 for r in yr_dates.values() if r == "alpha")
                    yr_hybrid  = sum(1 for r in yr_dates.values() if r == "hybrid")
                    yr_mvo     = sum(1 for r in yr_dates.values() if r == "mvo")
                    # Mini bar: alpha=green block, h=yellow, m=red
                    bar = ('a' * yr_alpha + 'h' * yr_hybrid + 'M' * yr_mvo)
                    print(f"  {yr:<6}  {yr_total:>6}  "
                          f"{yr_alpha:>4} ({yr_alpha/yr_total*100:>4.0f}%)  "
                          f"{yr_hybrid:>4} ({yr_hybrid/yr_total*100:>4.0f}%)  "
                          f"{yr_mvo:>3} ({yr_mvo/yr_total*100:>4.0f}%)  {bar}")

        # Portfolio returns histogram
        fig_port, ax_port = plt.subplots(figsize=(12, 4))
        fig_port.patch.set_facecolor('#FAFAF9')
        ax_port.set_facecolor('#FAFAF9')
        n_bins_p = min(40, max(10, len(port_arr) // 5))
        ax_port.hist(port_arr, bins=n_bins_p, color='#1D9E75', alpha=0.75,
                     edgecolor='white', linewidth=0.4)
        ax_port.axvline(port_arr.mean(),     color='#378ADD', linewidth=1.5,
                        linestyle='--', label=f"Mean {port_arr.mean():+.1f}%")
        ax_port.axvline(np.median(port_arr), color='#D85A30', linewidth=1.5,
                        linestyle='--', label=f"Median {np.median(port_arr):+.1f}%")
        ax_port.axvline(0, color='#888780', linewidth=0.8, linestyle=':')
        ax_port.set_xlabel("Hybrid portfolio return per holding period (%)",
                           fontsize=10, color='#5F5E5A')
        ax_port.set_ylabel("Count", fontsize=10, color='#5F5E5A')
        ax_port.set_title("Distribution of Portfolio Returns per Holding Period",
                          fontsize=12, fontweight='500', color='#2C2C2A')
        ax_port.legend(fontsize=9, framealpha=0.85)
        ax_port.grid(color='#D3D1C7', linewidth=0.5)
        ax_port.spines['top'].set_visible(False)
        ax_port.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

    # -- Consolidated individual stock return statistics ----------------------
    if all_stock_returns:
        ret_arr = np.array(all_stock_returns) * 100  # in percent
        q_labels = ['Q1 (0-20%)', 'Q2 (20-40%)', 'Q3 (40-60%)', 'Q4 (60-80%)', 'Q5 (80-100%)']
        quintiles = np.percentile(ret_arr, [0, 20, 40, 60, 80, 100])

        print("\n  " + "=" * 72)
        print("  CONSOLIDATED INDIVIDUAL STOCK RETURN STATISTICS — STRATEGY 9")
        print("  " + "=" * 72)
        print(f"\n  Total observations : {len(ret_arr)}")
        print(f"  Mean               : {ret_arr.mean():>+.2f}%")
        print(f"  Median             : {np.median(ret_arr):>+.2f}%")
        print(f"  Std Dev            : {ret_arr.std():>.2f}%")
        print(f"  Min                : {ret_arr.min():>+.2f}%")
        print(f"  Max                : {ret_arr.max():>+.2f}%")
        print(f"\n  Quintile boundaries:")
        for i, lbl in enumerate(q_labels):
            print(f"    {lbl:<18}  {quintiles[i]:>+7.2f}%  ->  {quintiles[i+1]:>+7.2f}%")
        # Histogram
        fig_hist, ax_hist = plt.subplots(figsize=(12, 5))
        fig_hist.patch.set_facecolor('#FAFAF9')
        ax_hist.set_facecolor('#FAFAF9')
        pct5, pct95 = np.percentile(ret_arr, 5), np.percentile(ret_arr, 95)
        clip_arr = ret_arr[(ret_arr >= pct5) & (ret_arr <= pct95)]
        n_bins   = min(60, max(20, len(ret_arr) // 20))
        ax_hist.hist(ret_arr, bins=n_bins, color='#378ADD', alpha=0.75,
                     edgecolor='white', linewidth=0.4)
        ax_hist.axvline(ret_arr.mean(),    color='#1D9E75', linewidth=1.5,
                        linestyle='--', label=f"Mean {ret_arr.mean():+.1f}%")
        ax_hist.axvline(np.median(ret_arr), color='#D85A30', linewidth=1.5,
                        linestyle='--', label=f"Median {np.median(ret_arr):+.1f}%")
        ax_hist.axvline(0, color='#888780', linewidth=0.8, linestyle=':')
        ax_hist.set_xlabel("Individual stock return per holding period (%)",
                           fontsize=10, color='#5F5E5A')
        ax_hist.set_ylabel("Count", fontsize=10, color='#5F5E5A')
        ax_hist.set_title("Distribution of Individual Stock Returns (all periods)",
                          fontsize=12, fontweight='500', color='#2C2C2A')
        ax_hist.legend(fontsize=9, framealpha=0.85)
        ax_hist.grid(color='#D3D1C7', linewidth=0.5)
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

    _mb_plot(nav_baseline, nav_alpha, nav_mvo, regime_s, nav_hybrid, nav_smart,
             nav_dynamic, nav_dyn_hedged, nav_dd)

    return {
        'nav_baseline'           : nav_baseline,
        'nav_alpha'              : nav_alpha,
        'nav_mvo'                : nav_mvo,
        'nav_hybrid'                  : nav_hybrid,
        'nav_smart'                   : nav_smart,
        'smart_hybrid_weights_by_date': smart_hybrid_weights_by_date,
        'nav_dynamic'                 : nav_dynamic,
        'nav_dyn_hedged'              : nav_dyn_hedged,
        'nav_dd'                      : nav_dd,
        'nav_dd_excl'                 : nav_dd_excl,
        'hedge_results'               : hedge_results,
        'dyn_weights_by_date'         : dyn_weights_by_date,
        'dyn_rebal_dates'             : [r['dt'] for r in _dyn_rebal_log],
        'port_baseline'          : port_baseline,
        'port_alpha'             : port_alpha,
        'port_mvo'               : port_mvo,
        'port_hybrid'            : port_hybrid,
        'alpha_weights_by_date'  : alpha_weights_by_date,
        'mvo_weights_by_date'    : mvo_weights_by_date,
        'hybrid_weights_by_date' : hybrid_weights_by_date,
        'composite_by_date'      : composite_by_date,
        'regime_s'               : regime_s,
        'weights_by_year'        : weights_by_year,
    }

    # -- Load daily trigger variables from cache if available ------------------
    try:
        with ENGINE.connect() as _conn2:
            _trig_rows2 = _conn2.execute(text(f"""
                SELECT date, implied_turnover, vol_diff, drawdown, live_strategy
                FROM {DAILY_TRIGGER_TBL}
                WHERE params_hash=:ph AND model_version=:mv
                ORDER BY date
            """), {'ph': params_hash, 'mv': MB_MODEL_VER}).fetchall()
        triggers_df = pd.DataFrame(
            _trig_rows2,
            columns=['date','implied_turnover','vol_diff','drawdown','live_strategy']
        )
        triggers_df['date'] = pd.to_datetime(triggers_df['date'])
        triggers_df = triggers_df.set_index('date')
        if not triggers_df.empty:
            print(f"\n  Daily trigger variables loaded: {len(triggers_df)} dates")
            print(f"  (implied_turnover, vol_diff, drawdown, live_strategy)")
            results['daily_triggers'] = triggers_df
        else:
            print("\n  No daily trigger data found -- run run_daily_cache_build() first")
            results['daily_triggers'] = pd.DataFrame()
    except Exception as e:
        warnings.warn(f"  Could not load trigger data: {e}")
        results['daily_triggers'] = pd.DataFrame()

    return results


# =============================================================================
# NEW UNIFIED BACKTEST — single-pass, per-strategy NAV tracking
# =============================================================================

def run_backtest(
    Pxs_df,
    sectors_s,
    weights_by_year,
    regime_s,
    volumeTrd_df        = None,
    weights_by_date     = None,
    mode                = 'incremental',   # 'incremental' | 'rebuild'
    rebuild_cov         = False,           # True only when new stocks added to DB
    # Portfolio construction
    top_n               = 25,
    universe_mult       = 5,
    conc_factor         = 2.0,
    prefilt_pct         = 0.5,
    min_weight          = MVO_MIN_WEIGHT,
    max_weight          = MVO_MAX_WEIGHT,
    ic                  = MVO_DEFAULT_IC,
    zscore_cap          = MVO_ZSCORE_CAP,
    pca_var_threshold   = MVO_PCA_VAR_THRESH,
    risk_aversion       = 10,
    min_cov_matrices    = MB_MIN_COV_MATRICES,
    model_version       = MB_MODEL_VER,
    # ADVP
    advp_cap            = 0.04,
    # Dynamic rebalancing
    min_hold_days       = DYN_MIN_HOLD_DAYS,
    to_thresh_alpha     = DYN_TO_THRESHOLD_ALPHA,
    to_thresh_hybrid    = DYN_TO_THRESHOLD_HYBRID,
    to_thresh_mvo       = DYN_TO_THRESHOLD_MVO,
    voldiff_cap         = DYN_VOLDIFF_CAP,
    # Smart hybrid / DD regime thresholds
    sh_dd_alpha         = SH_DD_ALPHA,
    sh_dd_hybrid        = SH_DD_HYBRID,
    sh_dd_exit_alpha    = SH_DD_EXIT_ALPHA,
    sh_dd_exit_hybrid   = SH_DD_EXIT_HYBRID,
    sh_persist_days     = SH_PERSIST_DAYS,
    # Drawdown policy
    dd_levels           = DD_LEVELS,
    dd_level_regime     = DD_LEVEL_REGIME,
    dd_reentry_pct      = DD_REENTRY_PCT,
    dd_reentry_confirm  = DD_REENTRY_CONFIRM,
    # Hedge — omit hedge_multi / hedges_l to disable hedge layer (6 strategies)
    hedge_multi         = None,            # from run_macro_hedge_cached
    hedges_l            = None,            # list of hedge instrument tickers
    eff_floor           = EFF_FLOOR,
    corr_floor          = CORR_FLOOR,
    hedge_ratio         = HEDGE_RATIO,
    max_hedge           = MAX_HEDGE,
    hedge_trigger_assets= TRIGGER_ASSETS,
    # MR exclusion
    mr_k                = MR_K,
    mr_cap              = MR_CAP,
    # Misc
    aum                 = AUM,
    trading_cost_bps    = TRADING_COST_BPS,
    rebal_freq          = 15,
    quality_floor       = QUALITY_FLOOR,
):
    """
    Unified single-pass backtest for all 9 strategies.
    Each strategy tracks its own NAV, drawdown, and rebalancing schedule.
    ADVP filter uses each strategy's exact current AUM at every rebalance.
    """
    import json, warnings
    import numpy as np
    import pandas as pd
    from sqlalchemy import text

    print(f"\n{'='*72}")
    print(f"  UNIFIED BACKTEST  |  mode={mode}  |  top_n={top_n}  |  AUM=${aum/1e6:.1f}M")
    print(f"{'='*72}")

    # ── 0. Setup ──────────────────────────────────────────────────────────────
    # Derive raw volume from volumeTrd_df (same structure, used for ADVP)
    volumeRaw_df = volumeTrd_df.copy() if volumeTrd_df is not None else None

    # Hedge layer only active when hedge_multi and hedges_l are provided
    hedge_enabled = hedge_multi is not None and hedges_l is not None

    # ── User prompts (same as run_mvo_backtest) ───────────────────────────────
    print("  PORTFOLIO CONSTRUCTION OPTIONS:")
    topn_input    = input(f"  Number of stocks [default={top_n}]: ").strip()
    rebal_input   = input(f"  Rebalancing frequency in days [default={rebal_freq}]: ").strip()
    advp_input    = input(f"  ADVP cap — max % of median daily $ volume per stock [default={advp_cap:.0%}]: ").strip()
    prefilt_input = input("  Pre-filter fraction by composite score 0<x<=1 (or Enter for none): ").strip()
    conc_input    = input("  Concentration factor for Pure Alpha >=1.0 (or Enter for equal weight): ").strip()
    min_cov_inp   = input(f"  Min cov matrices for stock selection "
                          f"(0=alpha-only, default={min_cov_matrices}): ").strip()

    top_n            = int(topn_input)           if topn_input    else top_n
    rebal_freq       = int(rebal_input)          if rebal_input   else rebal_freq
    advp_cap         = float(advp_input) / 100   if advp_input    else advp_cap
    prefilt_pct      = float(prefilt_input)      if prefilt_input else prefilt_pct
    if prefilt_pct <= 0 or prefilt_pct > 1: prefilt_pct = 1.0
    conc_factor      = float(conc_input)         if conc_input    else conc_factor
    if conc_factor < 1.0: conc_factor = 1.0
    min_cov_matrices = int(min_cov_inp)          if min_cov_inp   else min_cov_matrices
    min_cov_matrices = max(0, min(min_cov_matrices, 4))

    n_cands = top_n * universe_mult
    print(f"\n  Settings: N={top_n}, rebal={rebal_freq}d, "
          f"advp_cap={advp_cap:.1%}, prefilt={prefilt_pct:.0%}, "
          f"conc={conc_factor:.1f}x, min_cov={min_cov_matrices}")
    print(f"  MVO candidate pool: {n_cands} stocks ({universe_mult}x{top_n})\n")
    ph           = _make_make_params_hash(ic, max_weight, min_weight, zscore_cap,
                                          pca_var_threshold, universe_mult, risk_aversion,
                                          top_n, conc_factor, prefilt_pct,
                                          min_cov_matrices, model_version)
    start_date   = MB_START_DATE
    trading_days = Pxs_df.index[Pxs_df.index >= start_date]
    pxs_cols     = set(Pxs_df.columns)
    n_cands      = top_n * universe_mult
    ext_st       = MB_START_DATE - pd.Timedelta(days=365)
    today_ts     = Pxs_df.index[-1]

    # ── 1. Build composite scores (all dates) ─────────────────────────────────
    print("  Building composite scores...")
    universe    = get_universe(Pxs_df, sectors_s, ext_st)
    calc_dates  = pd.DatetimeIndex(generate_calc_dates(Pxs_df, step_days=rebal_freq))
    calc_dates  = calc_dates[calc_dates >= start_date]
    # Ensure last trading day is always included for hypothetical trade summary
    if today_ts not in calc_dates:
        calc_dates_full = calc_dates.append(pd.DatetimeIndex([today_ts]))
    else:
        calc_dates_full = calc_dates

    composite_by_date, _ = _cb_build_composite_scores(
        universe        = universe,
        calc_dates      = calc_dates_full,
        Pxs_df          = Pxs_df,
        sectors_s       = sectors_s,
        weights_by_year = weights_by_year,
        regime_s        = regime_s,
        volumeTrd_df    = volumeTrd_df,
        model_version   = model_version,
        exclude_factors = ['OU'],
        weights_by_date = weights_by_date,
    )
    print(f"  Composite scores: {len(composite_by_date)} dates")
    # Diagnostic: print top-5 composite scores at first calc_date to detect lookahead
    _diag_dt = sorted(composite_by_date.keys())[0]
    _diag_s  = composite_by_date[_diag_dt].dropna().sort_values(ascending=False)
    print(f"\n  [DIAG] Composite at {_diag_dt.date()} top-5: "
          + "  ".join(f"{t}={v:.4f}" for t, v in _diag_s.head(5).items()))
    if rebuild_cov:
        _mb_clear_x_cache(model_version)
        print("  X snapshot cache cleared (rebuild_cov=True)")
    X_snapshots = dict(_mb_load_x_cache(model_version))
    snapshot_dates = sorted(X_snapshots.keys())
    print(f"  X snapshots: {len(X_snapshots)} already cached  "
          f"(new ones will be built on-demand during main loop)")

    # Helper to get/build X snapshot for a given date on demand
    _x_snap_universe     = get_universe(Pxs_df, sectors_s, Pxs_df.index[0])
    _x_factor_names, _x_sec_cols = _mb_get_factor_names(model_version)
    _x_snap_dates_needed = set(_mb_month_end_dates(
        Pxs_df.index[Pxs_df.index >= start_date - pd.Timedelta(days=90)],
        Pxs_df.index[-1]))

    def _ensure_x_snapshot(dt):
        """Build X snapshot for the month-end prior to dt if not yet cached."""
        # Find the most recent month-end date <= dt that we need
        needed = [d for d in _x_snap_dates_needed if d <= dt and d not in X_snapshots]
        for x_dt in sorted(needed):
            with _SuppressOutput():
                xdf = _mb_build_X(x_dt, _x_snap_universe, _x_factor_names,
                                  _x_sec_cols, Pxs_df, sectors_s,
                                  volumeTrd_df, model_version)
            if xdf is not None:
                xdf = xdf.fillna(0.0)
                X_snapshots[x_dt] = xdf
                if x_dt not in snapshot_dates:
                    snapshot_dates.append(x_dt)
                    snapshot_dates.sort()
                try:
                    _mb_save_x_snapshot(x_dt, model_version, xdf)
                except Exception:
                    pass

    # ── 3. Quality scores for baseline ───────────────────────────────────────
    print("  Loading quality scores...")
    all_tickers = list(sectors_s.index)
    quality_wide = get_quality_scores(
        calc_dates         = calc_dates_full,
        universe           = all_tickers,
        Pxs_df             = Pxs_df,
        sectors_s          = sectors_s,
        use_cached_weights = True,
        force_recompute    = False,
    )

    # ── 4. Cache management ───────────────────────────────────────────────────
    _ensure_tables()
    if mode == 'rebuild':
        with ENGINE.connect() as conn:
            conn.execute(text(f"""
                DELETE FROM {DAILY_PORT_TBL}
                WHERE params_hash=:ph AND model_version=:mv
            """), {'ph': ph, 'mv': model_version})
            conn.commit()
        print("  Cache wiped (rebuild mode)")

    cached_dates = set(_get_cached_dates(ph, model_version))

    # ── 5. Per-strategy state ─────────────────────────────────────────────────
    # Strategies: 0=Baseline, 1=Alpha, 2=MVO, 3=Hybrid, 4=Smart,
    #             5=Dynamic, 6=Dyn+Hedge, 7=DD Policy, 8=Excl
    N_STRAT = 9
    S_BASE, S_ALPHA, S_MVO, S_HYB, S_SMART, S_DYN, S_HEDGE, S_DD, S_EXCL = range(9)

    # Static strategies rebalance on calc_dates; dynamic on triggers
    STATIC  = {S_BASE, S_ALPHA, S_MVO, S_HYB, S_SMART}
    # Dynamic strategies — hedge/DD/excl only active when hedge layer enabled
    DYNAMIC = {S_DYN, S_HEDGE, S_DD, S_EXCL} if hedge_enabled else {S_DYN}
    ACTIVE  = set(range(N_STRAT)) if hedge_enabled else \
              {S_BASE, S_ALPHA, S_MVO, S_HYB, S_SMART, S_DYN}
    print(f"  Strategies active: {len(ACTIVE)}  "
          f"({'hedge+DD+Excl enabled' if hedge_enabled else 'no hedge/DD/Excl'})")

    state = [{
        'nav':          1.0,
        'hwm':          1.0,
        'dd':           0.0,
        'gross':        1.0,          # de-grossing level (DD policy)
        'trough':       1.0,          # nav trough for re-entry measurement
        'theo_nav':     1.0,          # theoretical nav (no de-gross costs)
        'weights':      pd.Series(dtype=float),
        'last_rebal':   None,
        'days_held':    0,
        'regime':       'alpha',      # current signal regime
        'deployed_regime': None,      # regime of last actual rebalance
        'dd_level':     -1,           # current DD level index (-1 = none)
        'dd_active':    False,        # True when de-grossed
        'dd_regime_forced': False,    # True when DD forced a regime change
        'reentry_count':0,
        'hedge_acct':   0.0,
        'active_hedges':{},
        'hedge_log':    [],           # list of completed hedge episodes
        'costs':        0.0,          # cumulative trading costs (as NAV fraction)
        'costs_by_year':{},           # {year: cumulative costs in that year}
        'nav_series':   {},           # {date: nav}
        'weights_by_date': {},        # {date: weights}
        'rebal_log':    [],           # list of rebal events
        'dd_log':       [],           # list of DD events
    } for _ in range(N_STRAT)]

    # NAV series storage
    nav_records   = {s: {} for s in range(N_STRAT)}   # {date: nav_level}
    gross_records = {s: {} for s in range(N_STRAT)}   # {date: gross_exposure}

    # ── Helper: build alpha portfolio from composite scores ───────────────────
    def _build_alpha(dt, cands, excl_set=None):
        """Build concentrated alpha portfolio from ranked candidates."""
        c = cands.dropna()
        c = c.loc[[t for t in c.index if t in pxs_cols]]
        if excl_set:
            c = c[~c.index.isin(excl_set)]
        n_pf = max(n_cands, int(np.ceil(len(c) * prefilt_pct)))
        c    = c.nlargest(min(n_pf, len(c)))
        ap   = [t for t in c.sort_values(ascending=False).head(top_n).index
                if t in pxs_cols]
        n    = len(ap)
        if n == 0: return pd.Series(dtype=float)
        if conc_factor == 1.0 or n < 2:
            return pd.Series(1.0 / n, index=ap)
        nt = int(np.ceil(n / 2)); nb = n - nt
        ta = conc_factor / (conc_factor + 1.0)
        ba = 1.0 / (conc_factor + 1.0)
        return pd.Series({t: (ta/nt if j < nt else ba/nb)
                          for j, t in enumerate(ap)})

    # ── Helper: build MVO portfolio ───────────────────────────────────────────
    def _build_mvo(dt, cands, comp_scores):
        valid_snaps = [d for d in snapshot_dates if d <= dt]
        if not valid_snaps: return pd.Series(dtype=float)
        with _SuppressOutput():
            w, _ = _mb_solve_mvo(
                dt=dt, candidates=cands.head(n_cands).index.tolist(),
                composite_scores=comp_scores, Pxs_df=Pxs_df,
                sectors_s=sectors_s, volumeTrd_df=volumeTrd_df,
                model_version=model_version,
                pca_var_threshold=pca_var_threshold,
                ic=ic, max_weight=max_weight, min_weight=min_weight,
                zscore_cap=zscore_cap, risk_aversion=risk_aversion,
                X_snapshots=X_snapshots, snapshot_dates=snapshot_dates,
                top_n=top_n, min_cov_matrices=min_cov_matrices,
            )
        return w[w > 1e-6] if not w.empty else w

    # ── Helper: apply ADVP filter with correct strategy AUM ──────────────────
    def _apply_advp(w, cands, dt, strat_idx, target_n=None):
        if volumeRaw_df is None or w.empty: return w, set()
        cur_aum = aum * state[strat_idx]['nav'] * state[strat_idx]['gross']
        if cur_aum <= 0: return w, set()
        # Full composite scores as fallback universe for replacements
        _full_u = composite_by_date.get(last_calc_dt)
        if _full_u is not None:
            _full_u = _full_u.dropna().loc[
                [t for t in _full_u.index if t in pxs_cols]]
        result = _advp_filter_and_replace(
            w, cands, dt, Pxs_df, volumeRaw_df,
            cur_aum, advp_cap, min_weight, max_weight, top_n, conc_factor,
            target_n=target_n if target_n is not None else top_n,
            full_universe=_full_u)
        if isinstance(result, tuple):
            return result
        return result, set()

    # ── Helper: compute penalised composite ───────────────────────────────────
    def _penalise(comp, dt):
        if mr_k <= 0: return comp
        cp = comp.copy()
        mr = _load_momentum_exclusions(dt)
        for tkr, sc in mr.items():
            if tkr in cp and sc > 0:
                cp[tkr] = cp[tkr] / np.exp(mr_k * sc)
        return cp

    # ── Helper: compute portfolio turnover ───────────────────────────────────
    def _turnover(w_new, w_old):
        if w_old.empty: return 1.0
        all_t = list(set(w_new.index) | set(w_old.index))
        return (w_new.reindex(all_t).fillna(0) -
                w_old.reindex(all_t).fillna(0)).abs().sum() / 2

    # ── Helper: drift weights ─────────────────────────────────────────────────
    def _drift(w, d_from, d_to):
        if w.empty or d_from not in Pxs_df.index or d_to not in Pxs_df.index:
            return w
        tks = [t for t in w.index if t in pxs_cols]
        if not tks: return w
        ratio = (Pxs_df.loc[d_to, tks] / Pxs_df.loc[d_from, tks]).fillna(1)
        wd    = w.reindex(tks) * ratio
        return wd / wd.sum() if wd.sum() > 0 else w.reindex(tks)

    # ── Helper: record rebalance cost ─────────────────────────────────────────
    def _record_cost(s_idx, w_new, w_old, dt=None):
        to  = _turnover(w_new, w_old)
        c   = to * trading_cost_bps / 10000
        state[s_idx]['costs'] += c
        if dt is not None:
            yr = dt.year
            state[s_idx]['costs_by_year'][yr] = \
                state[s_idx]['costs_by_year'].get(yr, 0.0) + c
        return c

    # ── Helper: determine portfolio regime from DD only ──────────────────────
    def _get_regime(dd, exog_regime=None):
        """
        Portfolio regime is determined solely by drawdown level.
        Rates regime (exog_regime) affects factor weights only, not portfolio type.
        """
        if dd <= -sh_dd_hybrid:
            return 'mvo'
        elif dd <= -sh_dd_alpha:
            return 'hybrid'
        else:
            return 'alpha'

    # ── Helper: get exogenous rates regime at date ───────────────────────────
    def _exog_regime(dt):
        idx = regime_s.index[regime_s.index <= dt]
        if idx.empty: return 'alpha'
        rv = regime_s.loc[idx[-1]]
        return 'mvo' if rv >= 1.0 else 'hybrid' if rv >= 0.5 else 'alpha'

    # ── Helper: select portfolio by regime ───────────────────────────────────
    def _regime_weights(dt, regime, cached):
        """Return the appropriate cached portfolio for given regime."""
        if regime == 'mvo':
            return cached.get('mvo', cached.get('alpha', pd.Series(dtype=float)))
        elif regime == 'hybrid':
            return cached.get('hybrid', cached.get('alpha', pd.Series(dtype=float)))
        else:
            return cached.get('alpha', pd.Series(dtype=float))

    # ── Helper: build baseline portfolio ─────────────────────────────────────
    def _build_baseline(dt):
        if dt not in quality_wide.index: return pd.Series(dtype=float)
        scores = quality_wide.loc[dt].dropna()
        scores = scores.loc[[t for t in scores.index if t in pxs_cols]]
        if scores.empty: return pd.Series(dtype=float)
        top    = scores.nlargest(top_n)
        return pd.Series(1.0 / len(top), index=top.index)

    # ── Helper: check dynamic trigger ────────────────────────────────────────
    def _check_trigger(s_idx, dt, w_hyp, regime):
        """
        Return True if rebalance should be triggered.
        Matches old run_mvo_backtest trigger logic exactly.
        """
        st = state[s_idx]
        if st['last_rebal'] is None: return True
        if w_hyp.empty: return False

        days_held     = (dt - st['last_rebal']).days
        deployed_reg  = st.get('deployed_regime', 'alpha')

        # DD-forced regime override: immediate rebalance, no min hold
        dd_regime_override = st.get('dd_regime_forced', False)
        if dd_regime_override:
            st['dd_regime_forced'] = False  # consume the flag
            return True

        # Drift current weights to today
        w_drift = _drift(st['weights'], st['last_rebal'], dt)

        # Compute turnover against drifted portfolio
        if w_drift.empty:
            to_val = 1.0
        else:
            all_t = list(set(w_hyp.index) | set(w_drift.index))
            to_val = (w_hyp.reindex(all_t).fillna(0) -
                      w_drift.reindex(all_t).fillna(0)).abs().sum() / 2

        # Vol diff: new portfolio vol minus deployed portfolio vol
        vol_new  = _portfolio_vol(w_hyp,   Pxs_df, dt)
        vol_live = _portfolio_vol(w_drift, Pxs_df, dt)
        vd_val   = vol_new - vol_live

        # TO trigger: TO > threshold AND vol not spiking
        to_thr     = (to_thresh_mvo    if deployed_reg == 'mvo' else
                      to_thresh_hybrid if deployed_reg == 'hybrid' else
                      to_thresh_alpha)
        to_trigger = (to_val > to_thr and
                      vd_val < voldiff_cap and
                      days_held >= min_hold_days)

        # Regime switch trigger
        regime_switch = (deployed_reg is not None and
                         regime != deployed_reg and
                         days_held >= min_hold_days)

        # De-risk trigger (vol spike independent of TO)
        derisk = (vd_val < DYN_VOLDIFF_DERISK and
                  days_held >= min_hold_days)

        return to_trigger or regime_switch or derisk

    # ── Helper: apply DD de-grossing ─────────────────────────────────────────
    def _apply_dd(s_idx, dt):
        """Update DD state — runs daily, matches old run_mvo_backtest DD logic exactly."""
        st       = state[s_idx]
        nav      = st['nav']
        hwm      = st['hwm']

        # Update theo_nav daily — tracks full-exposure portfolio (no de-gross costs)
        # Same as old code: theo_nav *= (1 + full_return) each day when dd_active
        if st['dd_active'] and prev_dt is not None:
            w   = st['weights']
            tks = [t for t in w.index if t in pxs_cols]
            if tks and prev_dt in Pxs_df.index and dt in Pxs_df.index:
                full_ret = (w.reindex(tks).fillna(0) *
                            (Pxs_df.loc[dt, tks] /
                             Pxs_df.loc[prev_dt, tks] - 1).fillna(0)).sum()
                st['theo_nav'] *= (1 + full_ret)

        # HWM only updates when not in a de-gross event
        if not st['dd_active']:
            st['hwm'] = max(hwm, nav)
        dd_current = nav / st['hwm'] - 1

        # Check next DD level (sequential — only next level can trigger)
        next_lvl = st['dd_level'] + 1
        if next_lvl < len(dd_levels):
            dd_thresh, cut_frac = dd_levels[next_lvl]
            if dd_current <= -dd_thresh:
                old_gross        = st['gross']
                st['gross']     *= (1 - cut_frac)
                st['dd_active']  = True
                st['trough']     = nav
                st['theo_nav']   = nav
                st['reentry_count'] = 0
                st['dd_level']   = next_lvl
                st['regime']         = dd_level_regime[next_lvl]
                st['dd_regime_forced'] = True   # override min hold on next check
                st['dd_log'].append({
                    'dt': dt, 'event': f'DE-GROSS lv{next_lvl+1}',
                    'dd': dd_current,
                    'exp_from': old_gross, 'exp_to': st['gross'],
                })

        # Re-entry logic (runs every day when de-grossed)
        if st['dd_active'] and st['gross'] < 1.0:
            st['trough']    = min(st['trough'], nav)
            theo_recovery   = st['theo_nav'] / st['trough'] - 1

            # Annual reset condition
            yr_navs = [v for d, v in nav_records[s_idx].items()
                       if d.year == dt.year]
            ytd_dd  = (nav / yr_navs[0] - 1) if yr_navs else 0
            new_year_ok = (dt.month == 1 and dt.day <= 10 and
                           ytd_dd >= -DD_ANNUAL_RESET_PCT)

            recovery_ok = theo_recovery >= dd_reentry_pct
            if recovery_ok or new_year_ok:
                st['reentry_count'] += 1
            else:
                st['reentry_count'] = 0

            if st['reentry_count'] >= dd_reentry_confirm:
                old_gross        = st['gross']
                st['gross']      = 1.0
                st['dd_active']  = False
                st['dd_level']   = -1
                st['reentry_count'] = 0
                st['hwm']        = nav   # reset HWM
                st['trough']     = nav
                trigger = "recovery" if recovery_ok else "annual reset"
                st['dd_log'].append({
                    'dt': dt, 'event': 'RE-ENTRY 100%',
                    'dd': theo_recovery,
                    'exp_from': old_gross, 'exp_to': 1.0,
                })

    # ── Helper: hedge selection and P&L ──────────────────────────────────────
    def _update_hedge(s_idx, dt, port_w, port_nav):
        """Check hedge triggers using macro signal from hedge_multi."""
        if not hedge_enabled: return
        st = state[s_idx]
        if dt not in Pxs_df.index: return

        prev_dates = [d for d in trading_days if d < dt]
        if not prev_dates: return

        # Macro signal trigger: 1-day lag from hedge_multi signal_df
        # hedge_multi structure: {'results': {ticker: {'signal_df': ...}}, ...}
        _hedge_results = hedge_multi.get('results', hedge_multi) if hedge_multi else {}
        trigger_on = False
        for ta in hedge_trigger_assets:
            if ta in _hedge_results:
                sig_df = _hedge_results[ta].get('signal_df', pd.DataFrame())
                if not sig_df.empty and dt in sig_df.index:
                    prev_sig_dates = sig_df.index[sig_df.index < dt]
                    if len(prev_sig_dates) > 0:
                        prev_sig = sig_df.loc[prev_sig_dates[-1], 'signal']
                        if prev_sig == 1:
                            trigger_on = True
                            break

        # Close hedges when trigger no longer on
        if st['active_hedges'] and not trigger_on:
            total_pnl = 0.0
            details   = {}
            for inst, h in st['active_hedges'].items():
                if inst in Pxs_df.columns:
                    pnl = -(Pxs_df.loc[dt, inst] / h['entry_px'] - 1) * h['weight']
                    st['hedge_acct'] += pnl
                    total_pnl += pnl
                    details[inst] = {
                        'weight':   h['weight'],
                        'entry_px': h['entry_px'],
                        'entry_dt': h.get('entry_dt', dt),
                        'beta':     h.get('beta', 0),
                        'eff':      h.get('eff', 0),
                    }
            days_held = (dt - min(h.get('entry_dt', dt)
                                  for h in st['active_hedges'].values())).days
            st['hedge_log'].append({
                'event':       'CLOSE',
                'date':        dt,
                'instruments': list(st['active_hedges'].keys()),
                'total_pnl':   total_pnl,
                'days_held':   days_held,
                'details':     details,
            })
            if s_idx == S_HEDGE:
                print(f"  [HEDGE {dt.date()}] Signal OFF → P&L={total_pnl:+.2%}  "
                      f"held={days_held}d", flush=True)
            st['active_hedges'] = {}

        # Open new hedges when trigger fires
        if trigger_on and not st['active_hedges'] and hedges_l:
            _wbd_dyn   = state[S_DYN]['weights_by_date']
            _ret_dates = pd.DatetimeIndex(prev_dates[-BETA_WINDOW:])
            port_ret_s = (_portfolio_returns(_wbd_dyn, Pxs_df, _ret_dates)
                          if _wbd_dyn else pd.Series(dtype=float))
            if not port_ret_s.empty:
                inst_ret = {inst: Pxs_df[inst].pct_change().dropna()
                            for inst in hedges_l if inst in Pxs_df.columns}
                selected = _select_hedge_instruments(
                    dt             = dt,
                    hedges_l       = hedges_l,
                    results        = _hedge_results,
                    port_ret_s     = port_ret_s,
                    inst_ret       = inst_ret,
                    beta_window    = BETA_WINDOW,
                    corr_window    = CORR_WINDOW,
                    eff_mav_win    = EFF_MAV_WINDOW,
                    eff_floor      = eff_floor,
                    corr_floor     = corr_floor,
                    trigger_assets = hedge_trigger_assets,
                )
                if not selected:
                    if s_idx == S_HEDGE:
                        print(f"  [HEDGE {dt.date()}] Signal ON — no qualifying instruments", flush=True)
                if selected:
                    n_inst      = len(selected)
                    total_hedge = min(n_inst * hedge_ratio, max_hedge)
                    raw_scores  = {i: d['beta'] * d['effectiveness']
                                   for i, d in selected.items()
                                   if not np.isnan(d.get('beta', np.nan)) and
                                      not np.isnan(d.get('effectiveness', np.nan))}
                    total_score = sum(raw_scores.values())
                    for inst, h in selected.items():
                        if inst in Pxs_df.columns:
                            if total_score > 0 and inst in raw_scores:
                                w_inst = total_hedge * raw_scores[inst] / total_score
                            else:
                                w_inst = total_hedge / n_inst
                            st['active_hedges'][inst] = {
                                'entry_px': Pxs_df.loc[dt, inst],
                                'entry_dt': dt,
                                'weight':   w_inst,
                                'beta':     h.get('beta', 0),
                                'eff':      h.get('effectiveness', 0),
                            }
                    st['hedge_log'].append({
                        'event':       'OPEN',
                        'date':        dt,
                        'instruments': list(st['active_hedges'].keys()),
                    })
                    if s_idx == S_HEDGE:
                        _inst_str = '  '.join(
                            f"{inst}(w={st['active_hedges'][inst]['weight']:.1%},"
                            f"eff={st['active_hedges'][inst]['eff']:.2f})"
                            for inst in st['active_hedges'])
                        print(f"  [HEDGE {dt.date()}] Signal ON  → {_inst_str}", flush=True)

    # ── Helper: compute daily NAV update ─────────────────────────────────────
    def _update_nav(s_idx, d_from, d_to, gross=None):
        """
        Update strategy NAV using price returns from d_from to d_to.
        gross: override gross exposure (for DD strategies)
        """
        st  = state[s_idx]
        w   = st['weights']
        if w.empty or d_from not in Pxs_df.index or d_to not in Pxs_df.index:
            nav_records[s_idx][d_to] = st['nav']
            return 0.0

        tks = [t for t in w.index if t in pxs_cols and t not in
               state[s_idx].get('active_hedges', {})]
        hedge_ret = 0.0

        # Hedge P&L
        for inst, h in st.get('active_hedges', {}).items():
            if inst in Pxs_df.columns:
                r = -(Pxs_df.loc[d_to, inst] / Pxs_df.loc[d_from, inst] - 1)
                hedge_ret += r * h['weight']

        port_ret = 0.0
        if tks:
            port_ret = (w.reindex(tks).fillna(0) *
                        (Pxs_df.loc[d_to, tks] /
                         Pxs_df.loc[d_from, tks] - 1).fillna(0)).sum()

        g         = gross if gross is not None else st.get('gross', 1.0)
        total_ret = (port_ret + hedge_ret) * g
        st['nav'] *= (1 + total_ret)
        if st['nav'] > st['hwm']:
            st['hwm'] = st['nav']
        st['dd'] = st['nav'] / st['hwm'] - 1
        nav_records[s_idx][d_to] = st['nav']
        return total_ret

    # ── 6. Main loop over all trading days ───────────────────────────────────
    print(f"  Running backtest: {len(trading_days)} days, {N_STRAT} strategies...")

    # Track last calc_date for static strategies
    last_calc_dt = None
    calc_date_set = set(calc_dates)   # only original calc_dates drive rebalancing

    for day_idx, dt in enumerate(trading_days):
        is_calc_date = dt in calc_date_set
        prev_dt = trading_days[day_idx - 1] if day_idx > 0 else None

        # -- Update days_held for dynamic strategies --------------------------
        for s_idx in DYNAMIC:
            if state[s_idx]['last_rebal'] is not None:
                state[s_idx]['days_held'] += 1

        # -- Static strategies: rebalance on calc_dates -----------------------
        if is_calc_date:
            # Build X snapshot on-demand if not yet cached
            n_before = len(X_snapshots)
            _ensure_x_snapshot(dt)
            if len(X_snapshots) > n_before:
                print(f"  [{day_idx+1}/{len(trading_days)}] {dt.date()}  "
                      f"X snapshot built  ({len(X_snapshots)} total)", flush=True)
            comp = composite_by_date.get(dt)
            if comp is not None:
                comp = comp.dropna()
                comp = comp.loc[[t for t in comp.index if t in pxs_cols]]

                # Prefilt candidates
                n_pf   = max(n_cands, int(np.ceil(len(comp) * prefilt_pct)))
                cands  = comp.nlargest(min(n_pf, len(comp)))

                # Check cache
                if dt not in cached_dates or mode == 'rebuild':
                    # Build alpha
                    w_alpha = _build_alpha(dt, cands)
                    # Build MVO
                    w_mvo   = _build_mvo(dt, cands, comp)
                    if w_mvo.empty: w_mvo = w_alpha.copy()
                    # Build hybrid
                    all_t   = list(set(w_alpha.index) | set(w_mvo.index))
                    w_hyb   = (w_alpha.reindex(all_t).fillna(0) +
                               w_mvo.reindex(all_t).fillna(0)) / 2
                    w_hyb   = w_hyb[w_hyb > 0]
                    if w_hyb.sum() > 0: w_hyb /= w_hyb.sum()
                    # Build penalised alpha (for S9)
                    comp_pen  = _penalise(comp, dt)
                    cands_pen = comp_pen.nlargest(min(n_pf, len(comp_pen)))
                    w_alpha_p = _build_alpha(dt, cands_pen)
                    w_mvo_p   = _build_mvo(dt, cands_pen, comp_pen)
                    if w_mvo_p.empty: w_mvo_p = w_alpha_p.copy()
                    all_tp    = list(set(w_alpha_p.index) | set(w_mvo_p.index))
                    w_hyb_p   = (w_alpha_p.reindex(all_tp).fillna(0) +
                                 w_mvo_p.reindex(all_tp).fillna(0)) / 2
                    w_hyb_p   = w_hyb_p[w_hyb_p > 0]
                    if w_hyb_p.sum() > 0: w_hyb_p /= w_hyb_p.sum()
                    # target_n for hybrid portfolios = union size (can exceed top_n)
                    # Save to cache
                    _save_portfolios(dt, model_version, ph, {
                        'alpha':        w_alpha,
                        'mvo':          w_mvo,
                        'hybrid':       w_hyb,
                        'alpha_excl':   w_alpha_p,
                        'mvo_excl':     w_mvo_p,
                        'hybrid_excl':  w_hyb_p,
                        'candidates':   cands,
                    })
                    cached_dates.add(dt)
                else:
                    # Load from cache
                    with ENGINE.connect() as conn:
                        rows = conn.execute(text(f"""
                            SELECT strategy, weights_json FROM {DAILY_PORT_TBL}
                            WHERE date=:dt AND model_version=:mv AND params_hash=:ph
                        """), {'dt': dt.strftime('%Y-%m-%d'),
                               'mv': model_version, 'ph': ph}).fetchall()
                    pc = {r[0]: pd.Series(json.loads(r[1])) for r in rows}
                    w_alpha   = pc.get('alpha',   pd.Series(dtype=float))
                    w_mvo     = pc.get('mvo',     pd.Series(dtype=float))
                    w_hyb     = pc.get('hybrid',  pd.Series(dtype=float))
                    w_alpha_p = pc.get('alpha_excl',  w_alpha.copy())
                    w_mvo_p   = pc.get('mvo_excl',    w_mvo.copy())
                    w_hyb_p   = pc.get('hybrid_excl', w_hyb.copy())
                    cands     = pc.get('candidates',  pd.Series(dtype=float))

                # -- Baseline (S0) ----------------------------------------
                w_bl = _build_baseline(dt)
                _record_cost(S_BASE, w_bl, state[S_BASE]['weights'], dt)
                state[S_BASE]['rebal_log'].append({'dt': dt, 'w': w_bl.copy()})
                state[S_BASE]['weights'] = w_bl
                state[S_BASE]['weights_by_date'][dt] = w_bl.copy()
                state[S_BASE]['last_rebal'] = dt

                # -- Pure Alpha (S1) --------------------------------------
                w_a1, _ = _apply_advp(w_alpha, cands, dt, S_ALPHA)
                if day_idx == len(trading_days) - 1 or (day_idx > len(trading_days) - 10):
                    print(f"  [AUM DIAG] {dt.date()}  "
                          f"Alpha=${aum*state[S_ALPHA]['nav']/1e6:.1f}M  "
                          f"MVO=${aum*state[S_MVO]['nav']/1e6:.1f}M  "
                          f"Hybrid=${aum*state[S_HYB]['nav']/1e6:.1f}M  "
                          f"Smart=${aum*state[S_SMART]['nav']/1e6:.1f}M  "
                          f"Dyn=${aum*state[S_DYN]['nav']/1e6:.1f}M  "
                          f"DynHdg=${aum*state[S_HEDGE]['nav']/1e6:.1f}M  "
                          f"DD=${aum*state[S_DD]['nav']/1e6:.1f}M", flush=True)
                _record_cost(S_ALPHA, w_a1, state[S_ALPHA]['weights'], dt)
                state[S_ALPHA]['rebal_log'].append({'dt': dt, 'w': w_a1.copy()})
                state[S_ALPHA]['weights'] = w_a1
                state[S_ALPHA]['weights_by_date'][dt] = w_a1.copy()
                state[S_ALPHA]['last_rebal'] = dt

                # -- MVO (S2) ---------------------------------------------
                w_m2, _ = _apply_advp(w_mvo, cands, dt, S_MVO)
                _record_cost(S_MVO, w_m2, state[S_MVO]['weights'], dt)
                state[S_MVO]['rebal_log'].append({'dt': dt, 'w': w_m2.copy()})
                state[S_MVO]['weights'] = w_m2
                state[S_MVO]['weights_by_date'][dt] = w_m2.copy()
                state[S_MVO]['last_rebal'] = dt

                # -- Hybrid (S3) ------------------------------------------
                w_h3, _ = _apply_advp(w_hyb, cands, dt, S_HYB,
                                      target_n=len(w_hyb))
                _record_cost(S_HYB, w_h3, state[S_HYB]['weights'], dt)
                state[S_HYB]['rebal_log'].append({'dt': dt, 'w': w_h3.copy()})
                state[S_HYB]['weights'] = w_h3
                state[S_HYB]['weights_by_date'][dt] = w_h3.copy()
                state[S_HYB]['last_rebal'] = dt

                # -- Smart Hybrid (S4) ------------------------------------
                reg4   = state[S_SMART]['regime']
                w_s4   = _regime_weights(dt, reg4, {
                    'alpha': w_alpha, 'mvo': w_mvo, 'hybrid': w_hyb})
                w_s4, _ = _apply_advp(w_s4, cands, dt, S_SMART,
                                      target_n=len(w_s4))
                _record_cost(S_SMART, w_s4, state[S_SMART]['weights'], dt)
                state[S_SMART]['rebal_log'].append({'dt': dt, 'w': w_s4.copy()})
                state[S_SMART]['weights'] = w_s4
                state[S_SMART]['weights_by_date'][dt] = w_s4.copy()
                state[S_SMART]['last_rebal'] = dt

            last_calc_dt = dt

        # -- Dynamic strategies (S5-S8): check triggers daily ----------------
        if last_calc_dt is not None and last_calc_dt in composite_by_date:
            comp_dyn  = composite_by_date[last_calc_dt].dropna()
            comp_dyn  = comp_dyn.loc[[t for t in comp_dyn.index if t in pxs_cols]]
            n_pf_dyn  = max(n_cands, int(np.ceil(len(comp_dyn) * prefilt_pct)))
            cands_dyn = comp_dyn.nlargest(min(n_pf_dyn, len(comp_dyn)))

            # Use in-memory portfolios if this is a calc_date, else load from DB
            if is_calc_date and 'w_alpha' in dir():
                _cached_dyn = {
                    'alpha':        w_alpha,
                    'mvo':          w_mvo,
                    'hybrid':       w_hyb,
                    'alpha_excl':   w_alpha_p,
                    'mvo_excl':     w_mvo_p,
                    'hybrid_excl':  w_hyb_p,
                    'candidates':   cands,
                }
            else:
                _cached_dyn = {}
                if last_calc_dt in cached_dates:
                    try:
                        with ENGINE.connect() as conn:
                            rows = conn.execute(text(f"""
                                SELECT strategy, weights_json FROM {DAILY_PORT_TBL}
                                WHERE date=:dt AND model_version=:mv AND params_hash=:ph
                            """), {'dt': last_calc_dt.strftime('%Y-%m-%d'),
                                   'mv': model_version, 'ph': ph}).fetchall()
                        _cached_dyn = {r[0]: pd.Series(json.loads(r[1])) for r in rows}
                    except Exception:
                        pass

            for s_idx in [S_DYN, S_HEDGE, S_DD, S_EXCL]:
                st   = state[s_idx]
                reg  = st['regime']

                # Select hypothetical new portfolio for trigger check
                if s_idx == S_EXCL:
                    comp_s  = _penalise(comp_dyn, last_calc_dt)
                    cands_s = comp_s.nlargest(min(n_pf_dyn, len(comp_s)))
                    w_hyp = _regime_weights(dt, reg, {
                        'alpha':  _cached_dyn.get('alpha_excl',
                                  _cached_dyn.get('alpha', pd.Series(dtype=float))),
                        'mvo':    _cached_dyn.get('mvo_excl',
                                  _cached_dyn.get('mvo',   pd.Series(dtype=float))),
                        'hybrid': _cached_dyn.get('hybrid_excl',
                                  _cached_dyn.get('hybrid',pd.Series(dtype=float))),
                    })
                else:
                    cands_s = cands_dyn
                    w_hyp = _regime_weights(dt, reg, {
                        'alpha':  _cached_dyn.get('alpha',  pd.Series(dtype=float)),
                        'mvo':    _cached_dyn.get('mvo',    pd.Series(dtype=float)),
                        'hybrid': _cached_dyn.get('hybrid', pd.Series(dtype=float)),
                    })

                if w_hyp.empty: continue

                # Apply ADVP using this strategy's own AUM and today's prices
                w_hyp, _advp_affected = _apply_advp(w_hyp, cands_s, dt, s_idx,
                                                     target_n=len(w_hyp))

                # Check rebalance trigger
                if _check_trigger(s_idx, dt, w_hyp, reg):
                    to   = _turnover(w_hyp, _drift(st['weights'],
                                                    st['last_rebal'] or dt, dt))
                    cost = _record_cost(s_idx, w_hyp, st['weights'], dt)
                    st['weights']             = w_hyp
                    st['weights_by_date'][dt] = w_hyp.copy()
                    st['last_rebal']          = dt
                    st['days_held']           = 0
                    st['deployed_regime']     = reg
                    st['rebal_log'].append({
                        'dt': dt, 'regime': reg, 'w': w_hyp.copy(),
                        'to': to, 'nav': st['nav'],
                    })

                    # -- Per-rebalance print for DD Policy (S_DD) ---------------
                    if s_idx == S_DD:
                        _cur_aum  = aum * st['nav'] * st.get('gross', 1.0)
                        _rebal_n  = len(st['rebal_log'])
                        _eff_n    = 1.0 / (w_hyp**2).sum() if not w_hyp.empty else 0
                        _pv       = _portfolio_vol(w_hyp, Pxs_df, dt)
                        _dd_str   = f"dd={st['dd']*100:+.1f}%"
                        _gr_str   = f"gross={st['gross']*100:.0f}%"
                        print(f"\n  -- {dt.date()}  [{_rebal_n}]  "
                              f"regime={reg}  {_dd_str}  {_gr_str}  "
                              f"n={len(w_hyp)}  eff_N={_eff_n:.1f}  "
                              f"TO={to*100:.0f}%  "
                              f"AUM=${_cur_aum/1e6:.1f}M  "
                              f"port_vol={_pv*100:.1f}%", flush=True)
                        # MR top scores
                        _mr_dt = _load_momentum_exclusions(last_calc_dt)
                        if _mr_dt and mr_k > 0:
                            _mr_top = sorted(_mr_dt.items(), key=lambda x: -x[1])[:10]
                            _mr_str = '  '.join(f"{t}({s:.2f})"
                                                for t, s in _mr_top if s > 0)
                            if _mr_str:
                                print(f"  MR top-10 (k={mr_k}): {_mr_str}", flush=True)
                        # Portfolio weights
                        print(f"  {'Ticker':<8} {'Weight%':>8}  {'Sector':<30}",
                              flush=True)
                        print("  " + "-"*50, flush=True)
                        # ADVP affected stocks (capped or excluded)
                        if _advp_affected:
                            print(f"  *** ADVP cap active "
                                  f"(AUM=${_cur_aum:,.0f}, cap={advp_cap:.1%} ADV): "
                                  f"{', '.join(sorted(_advp_affected))}", flush=True)
                        for _tkr in w_hyp.sort_values(ascending=False).index:
                            _flag = '***' if _tkr in _advp_affected else '   '
                            _sec  = sectors_s.get(_tkr, '')
                            print(f"  {_tkr:<8} {w_hyp[_tkr]*100:>7.2f}%  "
                                  f"{_sec:<30} {_flag}", flush=True)

        # -- Update all NAVs first (using current gross before any DD change) --
        if prev_dt is not None:
            for s_idx in range(N_STRAT):
                gross = state[s_idx].get('gross', 1.0)
                gross_records[s_idx][dt] = gross
                _update_nav(s_idx, prev_dt, dt, gross=gross)
        else:
            for s_idx in range(N_STRAT):
                nav_records[s_idx][dt]   = state[s_idx]['nav']
                gross_records[s_idx][dt] = state[s_idx].get('gross', 1.0)

        # -- Update regime for all strategies (uses updated dd) ---------------
        for s_idx in range(N_STRAT):
            state[s_idx]['regime'] = _get_regime(state[s_idx]['dd'])

        # -- Update hedge for S6/S7/S8 ----------------------------------------
        for s_idx in [S_HEDGE, S_DD, S_EXCL]:
            if prev_dt is not None and not state[s_idx]['weights'].empty:
                _update_hedge(s_idx, dt, state[s_idx]['weights'],
                              state[s_idx]['nav'])

        # -- Apply DD de-grossing (after NAV update, takes effect next day) ---
        for s_idx in [S_DD, S_EXCL]:
            _apply_dd(s_idx, dt)

        # -- Progress ----------------------------------------------------------
        if (day_idx + 1) % 50 == 0 or day_idx == len(trading_days) - 1:
            print(f"  [{day_idx+1}/{len(trading_days)}] {dt.date()}  "
                  f"nav_dyn={state[S_DYN]['nav']:.3f}  "
                  f"nav_dd={state[S_DD]['nav']:.3f}  "
                  f"AUM_dyn=${aum*state[S_DYN]['nav']/1e6:.0f}M",
                  end='\r', flush=True)

    print()  # newline after progress

    # ── 7. Build NAV series ───────────────────────────────────────────────────
    def _to_nav_series(s_idx):
        d = nav_records[s_idx]
        if not d: return pd.Series(dtype=float)
        return pd.Series(d).sort_index()

    nav_series = [_to_nav_series(s) for s in range(N_STRAT)]
    labels     = ['Baseline', 'Pure Alpha', 'MVO', 'Hybrid', 'Smart Hybrid',
                  'Dynamic', 'Dyn+Hedge', 'DD Policy', 'Excl']

    # ── 8. Performance summary ────────────────────────────────────────────────
    def _perf(nav_s):
        if nav_s is None or len(nav_s) < 2:
            return dict(cagr=np.nan, vol=np.nan, sharpe=np.nan,
                        mdd=np.nan, cagr_dd=np.nan)
        rets  = nav_s.pct_change().dropna()
        n_yrs = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
        cagr  = nav_s.iloc[-1] ** (1 / n_yrs) - 1 if n_yrs > 0 else np.nan
        vol   = rets.std() * np.sqrt(252)
        sharpe= cagr / vol if vol > 0 else np.nan
        roll_max = nav_s.cummax()
        mdd   = ((nav_s - roll_max) / roll_max).min()
        cagr_dd = cagr / abs(mdd) if mdd != 0 else np.nan
        return dict(cagr=cagr, vol=vol, sharpe=sharpe, mdd=mdd, cagr_dd=cagr_dd)

    print(f"\n  {'='*72}")
    print(f"  COMPARISON")
    print(f"  {'='*72}")
    print(f"  All returns net of {trading_cost_bps}bps trading costs")
    print(f"  {'Strategy':<42} {'CAGR':>7} {'Vol':>7} {'Sharpe':>8} "
          f"{'MDD':>8} {'CAGR/DD':>9}")
    print(f"  {'-'*70}")
    for s_idx, (nav_s, lbl) in enumerate(zip(nav_series, labels)):
        p = _perf(nav_s)
        if np.isnan(p['cagr']): continue
        print(f"  {lbl:<42} {p['cagr']*100:>6.1f}%  {p['vol']*100:>6.1f}%  "
              f"{p['sharpe']:>7.2f}  {p['mdd']*100:>7.1f}%  "
              f"{p['cagr_dd']:>7.2f}x")

    # Yearly returns — all columns use consistent 10-char width
    _aum_nav = nav_series[S_HEDGE]
    _COL  = 10   # fixed column width for all strategy columns
    _COLS = [
        (S_BASE,  'Baseline'),
        (S_ALPHA, 'Alpha'),
        (S_MVO,   'MVO'),
        (S_HYB,   'Hybrid'),
        (S_SMART, 'Smart'),
        (S_DYN,   'Dynamic'),
        (S_HEDGE, 'Dyn+Hdg'),
        (S_DD,    'DD Pol'),
        (S_EXCL,  'Excl'),
    ]
    _sep  = '  ' + '-' * (6 + (len(_COLS) + 1) * (_COL + 2))

    print(f"\n  Yearly returns  (starting AUM: ${aum/1e6:.1f}M)")
    # Header
    hdr  = f"  {'Year':<6}"
    for _, lbl in _COLS:
        hdr += f"  {lbl:>{_COL}}"
    hdr += f"  {'AUM($M)':>{_COL}}"
    print(hdr)
    # DD row
    dd_row = f"  {'DD now':<6}"
    for s_idx, _ in _COLS:
        if s_idx == S_BASE:
            dd_row += f"  {'':>{_COL}}"
        else:
            dd = state[s_idx]['dd'] * 100
            dd_row += f"  {dd:>+{_COL-1}.1f}%"
    dd_row += f"  {'':>{_COL}}"
    print(dd_row)
    print(_sep)

    def _fmt_ret(r):
        return f"{r:>+{_COL-1}.2f}%" if not np.isnan(r) else f"{'n/a':>{_COL}}"

    all_years = sorted(set(nav_series[0].index.year))
    for yr in all_years:
        def yr_ret(nav_s):
            yn = nav_s[nav_s.index.year == yr] if nav_s is not None else pd.Series()
            if len(yn) < 2: return np.nan
            return (yn.iloc[-1] / yn.iloc[0] - 1) * 100
        nav_to_yr = _aum_nav[_aum_nav.index.year <= yr]
        closing   = aum * float(nav_to_yr.iloc[-1]) if not nav_to_yr.empty else aum
        row = f"  {yr:<6}"
        for s_idx, _ in _COLS:
            row += f"  {_fmt_ret(yr_ret(nav_series[s_idx]))}"
        row += f"  ${closing/1e6:>{_COL-2}.1f}M"
        print(row)

    # Trading costs per strategy per year
    print(f"\n  Trading costs per year (as % of NAV at time of trade)")
    print(hdr.replace(f"{'AUM($M)':>{_COL}}", f"{'Total':>{_COL}}"))
    print(_sep)
    for yr in all_years:
        row = f"  {yr:<6}"
        for s_idx, _ in _COLS:
            c = state[s_idx]['costs_by_year'].get(yr, 0.0)
            row += f"  {c*100:>+{_COL-1}.2f}%" if c > 0 else f"  {'':>{_COL}}"
        total_c = sum(state[s]['costs_by_year'].get(yr, 0.0) for s in range(N_STRAT))
        row += f"  {total_c*100:>+{_COL-1}.2f}%" if total_c > 0 else f"  {'':>{_COL}}"
        print(row)
    # Totals row
    print(_sep)
    tot_row = f"  {'Total':<6}"
    for s_idx, _ in _COLS:
        c = state[s_idx]['costs']
        tot_row += f"  {c*100:>+{_COL-1}.2f}%" if c > 0 else f"  {'':>{_COL}}"
    tot_row += f"  {'':>{_COL}}"
    print(tot_row)

    # DD policy summary
    print(f"\n  DD Policy summary: {len(state[S_DD]['dd_log'])} events")
    if state[S_DD]['dd_log']:
        print(f"  {'Date':<12} {'Event':<24} {'DD':>8}  {'Exp From':>10}  {'Exp To':>8}")
        print("  " + "-" * 60)
        for ev in state[S_DD]['dd_log']:
            print(f"  {str(ev['dt'].date()):<12} {ev['event']:<24} "
                  f"{ev['dd']*100:>+7.1f}%  {ev['exp_from']*100:>9.1f}%  "
                  f"{ev['exp_to']*100:>7.1f}%")

    # ── 9. Live portfolio display ─────────────────────────────────────────────
    today_ts = trading_days[-1]

    # Unified current portfolio table
    _strat_display = [
        ('Baseline', S_BASE),
        ('Alpha',    S_ALPHA),
        ('MVO',      S_MVO),
        ('Hybrid',   S_HYB),
        ('Smart',    S_SMART),
        ('Dynamic',  S_DYN),
        ('Dyn+Hdg',  S_HEDGE),
        ('DD Pol',   S_DD),
        ('Excl',     S_EXCL),
    ]

    # Collect drifted weights per strategy
    def _curr_weights(s_idx):
        st  = state[s_idx]
        ldt = st['last_rebal']
        w   = st['weights']
        if w.empty or ldt is None: return pd.Series(dtype=float), ldt
        if ldt < today_ts:
            wd = _drift(w, ldt, today_ts)
        else:
            wd = w.copy()
        # Scale by gross exposure
        wd = wd * st.get('gross', 1.0)
        # Add hedge positions as negative weights
        for inst, h in st.get('active_hedges', {}).items():
            wd[inst] = wd.get(inst, 0.0) - h['weight']
        return wd, ldt

    _all_tks = set()
    _curr_w  = {}
    _curr_ldt= {}
    for lbl, s_idx in _strat_display:
        wd, ldt = _curr_weights(s_idx)
        _curr_w[lbl]   = wd
        _curr_ldt[lbl] = ldt
        _all_tks |= set(wd.index)

    # Last-day returns
    prev_td = trading_days[-2] if len(trading_days) >= 2 else today_ts
    _day_ret = {}
    for tkr in _all_tks:
        if tkr in Pxs_df.columns:
            p1 = Pxs_df.loc[prev_td, tkr]; p2 = Pxs_df.loc[today_ts, tkr]
            _day_ret[tkr] = (p2/p1 - 1)*100 if p1 > 0 else 0.0
        else:
            _day_ret[tkr] = 0.0

    # Sort by average weight
    _avg_w = {t: np.mean([_curr_w[l].get(t, 0.0) for l, _ in _strat_display])
              for t in _all_tks}
    _sorted_tks = sorted(_all_tks, key=lambda t: -_avg_w[t])
    _col_lbls = [l for l, _ in _strat_display]

    print(f"\n  {'='*72}")
    print("  CURRENT LIVE PORTFOLIOS — DRIFTED WEIGHTS (% of AUM)")
    print(f"  {'='*72}")
    _hdr = f"  {'Ticker':<8}  {'Day%':>6}" + "".join(f"  {l:>8}" for l in _col_lbls)
    print(_hdr)
    print(f"  {'Last reb:':8}  {'':6}" +
          "".join(f"  {(_curr_ldt[l].strftime('%m/%d') if _curr_ldt[l] else 'n/a'):>8}"
                  for l in _col_lbls))
    print("  " + "-"*8 + "  " + "-"*6 + ("  " + "-"*8)*len(_col_lbls))
    for tkr in _sorted_tks:
        dr  = _day_ret.get(tkr, 0.0)
        row = f"  {tkr:<8}  {dr:>+5.1f}%"
        any_nz = False
        for lbl in _col_lbls:
            wt = _curr_w[lbl].get(tkr, 0.0) * 100
            if abs(wt) >= 0.01:
                any_nz = True
                row += f"  {wt:>7.2f}%"
            else:
                row += f"  {'':>8}"
        if any_nz:
            print(row)

    # 10-day P&L table
    print(f"\n  {'='*72}")
    print("  DAILY P&L — LAST 10 TRADING DAYS (all strategies)")
    print(f"  {'='*72}")
    _last10 = list(trading_days[-10:])
    print(f"\n  {'Date':<12}" + "".join(f"  {l:>9}" for l in _col_lbls))
    print("  " + "-"*12 + ("  " + "-"*9)*len(_col_lbls))
    _cum = {l: 1.0 for l in _col_lbls}
    for dt_p in _last10:
        row = f"  {dt_p.strftime('%Y-%m-%d'):<12}"
        for lbl, s_idx in _strat_display:
            nav_s = nav_series[s_idx]
            prev_dates = [d for d in nav_s.index if d < dt_p]
            nav_at = nav_s[nav_s.index <= dt_p]
            if nav_at.empty or not prev_dates:
                row += f"  {'n/a':>9}"; continue
            r = (float(nav_at.iloc[-1]) / float(nav_s.loc[prev_dates[-1]]) - 1)*100
            _cum[lbl] *= (1 + r/100)
            row += f"  {r:>+8.2f}%"
        print(row)
    print("  " + "-"*12 + ("  " + "-"*9)*len(_col_lbls))
    cum_row = f"  {'Cumul':<12}"
    for lbl in _col_lbls:
        cum_row += f"  {(_cum[lbl]-1)*100:>+8.2f}%"
    print(cum_row)

    # Trade summary — all 9 strategies
    # Delta = fresh portfolio today - drifted weights from last rebalance
    # Fresh portfolios are computed using last calc_date scores but today's
    # ADVP filter (today's prices/volumes and each strategy's current AUM)
    print(f"\n  {'='*72}")
    print(f"  TRADE SUMMARY  —  {today_ts.date()}")
    print(f"  {'='*72}")

    _all_strats_summary = [
        ('Baseline', S_BASE),
        ('Alpha',    S_ALPHA),
        ('MVO',      S_MVO),
        ('Hybrid',   S_HYB),
        ('Smart',    S_SMART),
        ('Dynamic',  S_DYN),
        ('Dyn+Hdg',  S_HEDGE),
        ('DD Pol',   S_DD),
        ('Excl',     S_EXCL),
    ]

    # Find last calc_date with cached portfolios
    _last_cd = sorted([d for d in cached_dates if d <= today_ts])
    _last_cd = _last_cd[-1] if _last_cd else None
    _fresh_w = {}

    if _last_cd:
        # Load cached portfolios for last calc_date
        try:
            with ENGINE.connect() as conn:
                rows = conn.execute(text(f"""
                    SELECT strategy, weights_json FROM {DAILY_PORT_TBL}
                    WHERE date=:dt AND model_version=:mv AND params_hash=:ph
                """), {'dt': _last_cd.strftime('%Y-%m-%d'),
                       'mv': model_version, 'ph': ph}).fetchall()
            _cache_td = {r[0]: pd.Series(json.loads(r[1])) for r in rows}
        except Exception:
            _cache_td = {}

        _cands_td = _cache_td.get('candidates', pd.Series(dtype=float))
        def _fresh(s_idx, w_base):
            """Apply today's ADVP filter using this strategy's current AUM."""
            if w_base.empty: return w_base
            w, _ = _apply_advp(w_base, _cands_td, today_ts, s_idx,
                               target_n=len(w_base))
            return w

        for lbl, s_idx in _all_strats_summary:
            st  = state[s_idx]
            reg = _get_regime(st['dd'])
            if s_idx == S_BASE:
                _qual_dates   = [d for d in quality_wide.index if d <= today_ts]
                _qual_dt      = _qual_dates[-1] if _qual_dates else None
                _fresh_w[lbl] = _build_baseline(_qual_dt) if _qual_dt else pd.Series(dtype=float)
            elif s_idx == S_ALPHA:
                _fresh_w[lbl] = _fresh(s_idx, _cache_td.get('alpha',
                                        pd.Series(dtype=float)))
            elif s_idx == S_MVO:
                _fresh_w[lbl] = _fresh(s_idx, _cache_td.get('mvo',
                                        pd.Series(dtype=float)))
            elif s_idx == S_HYB:
                _fresh_w[lbl] = _fresh(s_idx, _cache_td.get('hybrid',
                                        pd.Series(dtype=float)))
            elif s_idx == S_SMART:
                _fresh_w[lbl] = _fresh(s_idx, _regime_weights(today_ts, reg, {
                    'alpha':  _cache_td.get('alpha',  pd.Series(dtype=float)),
                    'mvo':    _cache_td.get('mvo',    pd.Series(dtype=float)),
                    'hybrid': _cache_td.get('hybrid', pd.Series(dtype=float)),
                }))
            elif s_idx in (S_DYN, S_HEDGE, S_DD):
                _fresh_w[lbl] = _fresh(s_idx, _regime_weights(today_ts, reg, {
                    'alpha':  _cache_td.get('alpha',  pd.Series(dtype=float)),
                    'mvo':    _cache_td.get('mvo',    pd.Series(dtype=float)),
                    'hybrid': _cache_td.get('hybrid', pd.Series(dtype=float)),
                }))
            elif s_idx == S_EXCL:
                _fresh_w[lbl] = _fresh(s_idx, _regime_weights(today_ts, reg, {
                    'alpha':  _cache_td.get('alpha_excl',
                              _cache_td.get('alpha',  pd.Series(dtype=float))),
                    'mvo':    _cache_td.get('mvo_excl',
                              _cache_td.get('mvo',    pd.Series(dtype=float))),
                    'hybrid': _cache_td.get('hybrid_excl',
                              _cache_td.get('hybrid', pd.Series(dtype=float))),
                }))

    # Drifted current weights (B side of delta):
    # - if today IS a rebalancing date for this strategy: use log[-2] drifted to today
    # - if today is NOT a rebalancing date: use log[-1] drifted to today
    _drifted_w = {}
    for lbl, s_idx in _all_strats_summary:
        st  = state[s_idx]
        log = st['rebal_log']
        if not log:
            _drifted_w[lbl] = pd.Series(dtype=float)
            continue
        today_was_rebal = log[-1]['dt'] == today_ts
        if today_was_rebal:
            ref = log[-2] if len(log) >= 2 else None
        else:
            ref = log[-1]
        if ref is None:
            _drifted_w[lbl] = pd.Series(dtype=float)
        else:
            _drifted_w[lbl] = _drift(ref['w'], ref['dt'], today_ts)

    # Compute deltas
    all_trade_tks = set()
    for lbl, _ in _all_strats_summary:
        if lbl in _fresh_w:   all_trade_tks |= set(_fresh_w[lbl].index)
        if lbl in _drifted_w: all_trade_tks |= set(_drifted_w[lbl].index)

    deltas = {}
    for lbl, _ in _all_strats_summary:
        deltas[lbl] = {}
        if lbl not in _fresh_w: continue
        for tkr in all_trade_tks:
            n = _fresh_w[lbl].get(tkr, 0.0) * 100
            o = _drifted_w.get(lbl, pd.Series()).get(tkr, 0.0) * 100
            d = n - o
            if abs(d) >= 0.01:
                deltas[lbl][tkr] = d

    all_delta_tks = sorted(
        {t for d in deltas.values() for t in d},
        key=lambda t: -max(abs(deltas[l].get(t, 0)) for l in deltas))

    if all_delta_tks:
        col_w = 9
        print(f"\n  {'Ticker':<8}" +
              "".join(f"  {l:>{col_w}}" for l, _ in _all_strats_summary))
        print("  " + "-"*8 + ("  " + "-"*col_w)*len(_all_strats_summary))
        for tkr in all_delta_tks:
            row = f"  {tkr:<8}"
            for lbl, _ in _all_strats_summary:
                d = deltas[lbl].get(tkr, 0.0)
                row += f"  {d:>+8.2f}%" if abs(d) >= 0.01 else f"  {'':>9}"
            print(row)

    # ── 10. Hedge summary ────────────────────────────────────────────────────
    if hedge_enabled:
        hedge_log = state[S_HEDGE]['hedge_log']
        close_ev  = [e for e in hedge_log if e['event'] == 'CLOSE']
        print(f"\n  {'='*72}")
        print(f"  HEDGE SUMMARY")
        print(f"  {'='*72}")
        if not close_ev:
            print("\n  No completed hedge episodes.")
        else:
            all_pnl = [e['total_pnl'] for e in close_ev]
            n_pos   = sum(1 for p in all_pnl if p > 0)
            print(f"\n  Total episodes  : {len(close_ev)}")
            print(f"  Hit rate        : {n_pos/len(close_ev)*100:.1f}%  "
                  f"({n_pos} positive / {len(close_ev)-n_pos} negative)")
            print(f"  Total P&L       : {sum(all_pnl):+.4%}")
            print(f"  Avg P&L/episode : {np.mean(all_pnl):+.4%}")
            print(f"  Avg days held   : "
                  f"{np.mean([e.get('days_held',0) for e in close_ev]):.1f}")

            # Per-year
            print(f"\n  {'─'*68}")
            print(f"  PER-YEAR BREAKDOWN")
            print(f"  {'─'*68}")
            print(f"  {'Year':<6}  {'Episodes':>9}  {'Hit Rate':>9}  "
                  f"{'Total P&L':>10}  {'Avg P&L':>10}  {'Avg Days':>9}")
            print(f"  {'-'*60}")
            for yr in sorted(set(e['date'].year for e in close_ev)):
                ye  = [e for e in close_ev if e['date'].year == yr]
                yp  = [e['total_pnl'] for e in ye]
                ypos= sum(1 for p in yp if p > 0)
                yd  = np.mean([e.get('days_held', 0) for e in ye])
                print(f"  {yr:<6}  {len(ye):>9}  "
                      f"{ypos/len(ye)*100:>8.1f}%  "
                      f"{sum(yp):>+9.4%}  {np.mean(yp):>+9.4%}  {yd:>9.1f}")

            # Per-instrument
            inst_ev = {}
            for e in close_ev:
                for inst in e.get('instruments', []):
                    if inst not in inst_ev: inst_ev[inst] = []
                    d    = e['details'].get(inst, {})
                    w    = d.get('weight', 0)
                    epx  = d.get('entry_px', np.nan)
                    edt  = d.get('entry_dt', e['date'])
                    cdt  = e['date']
                    if (epx and not np.isnan(epx) and
                            inst in Pxs_df.columns and cdt in Pxs_df.index):
                        inst_pnl = -(Pxs_df.loc[cdt, inst] / epx - 1) * w
                    else:
                        inst_pnl = np.nan
                    inst_ev[inst].append({
                        'pnl': inst_pnl, 'weight': w,
                        'eff': d.get('eff', np.nan), 'beta': d.get('beta', np.nan),
                    })
            if inst_ev:
                print(f"\n  {'─'*68}")
                print(f"  PER-INSTRUMENT BREAKDOWN")
                print(f"  {'─'*68}")
                print(f"  {'Instrument':<12}  {'Uses':>5}  {'Hit Rate':>9}  "
                      f"{'Total P&L':>10}  {'Avg Weight':>11}  "
                      f"{'Avg Beta':>9}  {'Avg Eff':>8}")
                print(f"  {'-'*68}")
                for inst in sorted(inst_ev.keys()):
                    evs   = inst_ev[inst]
                    pnls  = [e['pnl'] for e in evs if not np.isnan(e['pnl'])]
                    pos   = sum(1 for p in pnls if p > 0)
                    avg_w = np.mean([e['weight'] for e in evs])
                    avg_b = np.nanmean([e['beta'] for e in evs])
                    avg_e = np.nanmean([e['eff']  for e in evs])
                    print(f"  {inst:<12}  {len(evs):>5}  "
                          f"{pos/len(pnls)*100:.1f}%  " if pnls else
                          f"  {inst:<12}  {len(evs):>5}  {'n/a':>9}  ", end='')
                    if pnls:
                        print(f"{sum(pnls):>+9.4%}  {avg_w:>10.2%}  "
                              f"{avg_b:>9.3f}  {avg_e:>8.3f}")

    # ── 11. Plots ─────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        GRAY  = '#888780'; TEAL = '#1D9E75'; BLUE = '#378ADD'; CORAL = '#D85A30'
        COLORS = {
            'Baseline':    GRAY,
            'Pure Alpha':  TEAL,
            'MVO':         BLUE,
            'Hybrid':      '#7F77DD',
            'Smart Hybrid':'#E8A838',
            'Dynamic':     '#9B59B6',
            'Dyn+Hedge':   '#2ECC71',
            'DD Policy':   '#E74C3C',
            'Excl':        '#F39C12',
        }

        fig, axes = plt.subplots(4, 1, figsize=(14, 16),
                                 gridspec_kw={'height_ratios': [3, 2, 2, 1]})
        fig.patch.set_facecolor('#FAFAF9')
        for ax in axes:
            ax.set_facecolor('#FAFAF9')

        # Common date index
        common = nav_series[0].index
        for ns in nav_series[1:]:
            if ns is not None and not ns.empty:
                common = common.intersection(ns.index)

        nav_r = [ns.reindex(common) if ns is not None and not ns.empty
                 else None for ns in nav_series]

        # Regime shading
        reg = regime_s.reindex(common, method='ffill').fillna(0)
        REGIME_BG = {0.0: '#E1F5EE', 0.5: '#F1EFE8', 1.0: '#FAECE7'}
        for ax in axes[:3]:
            prev_r, prev_d = float(reg.iloc[0]), common[0]
            for d, r in reg.items():
                r = float(r)
                if r != prev_r or d == common[-1]:
                    ax.axvspan(prev_d, d,
                               color=REGIME_BG.get(prev_r, '#F1EFE8'),
                               alpha=0.3, linewidth=0)
                    prev_r, prev_d = r, d

        # Panel 1: NAV
        ax = axes[0]
        for i, (ns, lbl) in enumerate(zip(nav_r, labels)):
            if ns is None: continue
            lw = 2.0 if lbl in ('Dyn+Hedge', 'DD Policy') else 1.2
            ax.plot(ns.index.to_numpy(), ns.values,
                    label=lbl, color=COLORS.get(lbl, GRAY), linewidth=lw)
        ax.set_ylabel("NAV", fontsize=10, color='#5F5E5A')
        ax.set_title("Strategy NAV Comparison",
                     fontsize=12, fontweight='500', color='#2C2C2A')
        ax.legend(fontsize=9, loc='upper left', framealpha=0.85)
        ax.grid(color='#D3D1C7', linewidth=0.5)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        # Panel 2: Relative to baseline
        ax2 = axes[1]
        nb = nav_r[S_BASE]
        for i, (ns, lbl) in enumerate(zip(nav_r, labels)):
            if ns is None or i == S_BASE or nb is None: continue
            rel = (ns / nb - 1) * 100
            ax2.plot(rel.index.to_numpy(), rel.values,
                     label=f'{lbl} vs baseline',
                     color=COLORS.get(lbl, GRAY), linewidth=1.2)
            ax2.fill_between(rel.index.to_numpy(), rel.values, 0,
                             where=(rel.values >= 0),
                             color=COLORS.get(lbl, GRAY), alpha=0.06)
        ax2.axhline(0, color=GRAY, linewidth=0.8, linestyle='--')
        ax2.set_ylabel("Relative to baseline (%)", fontsize=10, color='#5F5E5A')
        ax2.legend(fontsize=8, loc='upper left', framealpha=0.85)
        ax2.grid(color='#D3D1C7', linewidth=0.5)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

        # Panel 3: Drawdown
        ax3 = axes[2]
        for ns, lbl in zip(nav_r, labels):
            if ns is None: continue
            dd = (ns / ns.cummax() - 1) * 100
            ax3.plot(dd.index.to_numpy(), dd.values,
                     label=lbl, color=COLORS.get(lbl, GRAY), linewidth=1.0)
        ax3.axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
        ax3.set_ylabel("Drawdown (%)", fontsize=10, color='#5F5E5A')
        ax3.legend(fontsize=8, loc='lower left', framealpha=0.85)
        ax3.grid(color='#D3D1C7', linewidth=0.5)
        ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

        # Panel 4: Regime
        ax4 = axes[3]
        ax4.fill_between(reg.index.to_numpy(), reg.values, 0,
                         color=BLUE, alpha=0.25)
        ax4.plot(reg.index.to_numpy(), reg.values, color=BLUE, linewidth=1.0)
        ax4.set_yticks([0.0, 0.5, 1.0])
        ax4.set_yticklabels(['Easy', 'Neutral', 'Tight'], fontsize=8)
        ax4.set_ylabel("Regime", fontsize=10, color='#5F5E5A')
        ax4.grid(color='#D3D1C7', linewidth=0.5)
        ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()

        # DD exposure plot (if hedge enabled)
        if hedge_enabled:
            exposure_s = pd.Series(gross_records[S_DD]).sort_index()
            fig2, axes2 = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
            fig2.patch.set_facecolor('#FAFAF9')
            for ax in axes2:
                ax.set_facecolor('#FAFAF9')

            axes2[0].fill_between(exposure_s.index.to_numpy(),
                                  exposure_s.values, 0,
                                  where=(exposure_s.values < 1.0),
                                  color='#E74C3C', alpha=0.3, label='De-grossed')
            axes2[0].fill_between(exposure_s.index.to_numpy(),
                                  exposure_s.values, 0,
                                  where=(exposure_s.values >= 1.0),
                                  color='#2ECC71', alpha=0.2, label='Fully invested')
            axes2[0].set_ylim(0, 1.2)
            axes2[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            axes2[0].set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
            axes2[0].set_title('Gross Exposure (DD Policy)',
                               fontsize=11, fontweight='500', color='#2C2C2A')
            axes2[0].legend(fontsize=8)
            axes2[0].grid(color='#D3D1C7', linewidth=0.5)
            axes2[0].spines['top'].set_visible(False)
            axes2[0].spines['right'].set_visible(False)

            ndh = nav_series[S_HEDGE].reindex(common)
            ndd = nav_series[S_DD].reindex(common)
            axes2[1].plot(ndh.index.to_numpy(), ndh.values,
                          label='Dyn+Hedge', color='#2ECC71', linewidth=1.8)
            axes2[1].plot(ndd.index.to_numpy(), ndd.values,
                          label='DD Policy', color='#E74C3C', linewidth=1.8)
            axes2[1].set_title('NAV: Dyn+Hedge vs DD Policy',
                               fontsize=11, fontweight='500', color='#2C2C2A')
            axes2[1].legend(fontsize=8)
            axes2[1].grid(color='#D3D1C7', linewidth=0.5)
            axes2[1].spines['top'].set_visible(False)
            axes2[1].spines['right'].set_visible(False)

            plt.tight_layout()
            plt.show()

    except Exception as _plot_err:
        print(f"\n  (Plot skipped: {_plot_err})")

    # ── 12. Return results ────────────────────────────────────────────────────
    return {
        'nav_baseline':  nav_series[S_BASE],
        'nav_alpha':     nav_series[S_ALPHA],
        'nav_mvo':       nav_series[S_MVO],
        'nav_hybrid':    nav_series[S_HYB],
        'nav_smart':     nav_series[S_SMART],
        'nav_dynamic':   nav_series[S_DYN],
        'nav_dyn_hedged':nav_series[S_HEDGE],
        'nav_dd':        nav_series[S_DD],
        'nav_dd_excl':   nav_series[S_EXCL],
        'state':         state,
        'nav_series':    nav_series,
        'labels':        labels,
    }

