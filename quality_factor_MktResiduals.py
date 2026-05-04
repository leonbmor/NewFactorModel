#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
quality_factor.py
=================
Rate-conditioned quality factor: construction + evaluation + gridsearch.

Combines:
  - Growth Quality Factor (GQF): metrics weighted by t-stats in non-stress years
  - Conservative Quality Factor (CQF): metrics weighted by t-stats in 2021/2022
  - Rate momentum signal (USGG10YR vs rolling MAV) to blend the two

Usage:
    from quality_factor import run, gridsearch, get_quality_scores

    # Daily incremental update (fast — only computes missing dates)
    scores_df = get_quality_scores(calc_dates, universe, Pxs_df, sectors_s)

    # Full run with weight derivation + evaluation (slow — run occasionally)
    stats, annual, scores, gqf_w, cqf_w = run(Pxs_df, sectors_s)
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sqlalchemy import create_engine, text

# ==============================================================================
# CONFIG
# ==============================================================================
QUALITY_TABLE           = 'valuation_metrics_anchors'
RESIDUAL_TABLE          = 'factor_residuals_mkt'   # step 1 output — only mkt beta removed
QUALITY_SCORES_TBL      = 'quality_scores_df'
QUALITY_WEIGHTS_PIT_TBL = 'quality_weights_pit'
MAX_HORIZON             = 63   # max forward horizon (days) — anchor needs this many days before cutoff

# ==============================================================================
# POINT-IN-TIME WEIGHTS CACHE
# ==============================================================================

def _ensure_pit_table():
    """Create quality_weights_pit table if it doesn't exist."""
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {QUALITY_WEIGHTS_PIT_TBL} (
                cutoff_date  DATE         NOT NULL,
                regime       VARCHAR(20)  NOT NULL,
                metric       VARCHAR(50)  NOT NULL,
                weight       FLOAT        NOT NULL,
                PRIMARY KEY  (cutoff_date, regime, metric)
            )
        """))


def _load_pit_weights_cache():
    """Load all cached PIT weights. Returns dict {cutoff_date: {'gqf': {...}, 'cqf': {...}}}"""
    try:
        _ensure_pit_table()
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT cutoff_date, regime, metric, weight
                FROM {QUALITY_WEIGHTS_PIT_TBL} ORDER BY cutoff_date
            """)).fetchall()
        result = {}
        for cutoff, regime, metric, weight in rows:
            if regime == '_sentinel':   # skip sentinel rows
                continue
            dt = pd.Timestamp(cutoff)
            if dt not in result:
                result[dt] = {'gqf': {}, 'cqf': {}}
            key = 'gqf' if regime == 'growth' else 'cqf'
            result[dt][key][metric] = float(weight)
        return result
    except Exception as e:
        print(f"  WARNING: could not load PIT weights cache: {e}")
        return {}


def _save_pit_weights(cutoff_date, gqf_w, cqf_w):
    """Save PIT weights for a cutoff_date. Saves sentinel row if no weights derived."""
    rows = []
    for metric, weight in gqf_w.items():
        rows.append({'cutoff_date': cutoff_date.date(),
                     'regime': 'growth', 'metric': metric, 'weight': float(weight)})
    for metric, weight in cqf_w.items():
        rows.append({'cutoff_date': cutoff_date.date(),
                     'regime': 'conservative', 'metric': metric, 'weight': float(weight)})
    # Always save at least a sentinel row so this date appears in cached_dates
    if not rows:
        rows.append({'cutoff_date': cutoff_date.date(),
                     'regime': '_sentinel', 'metric': '_none', 'weight': 0.0})
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {QUALITY_WEIGHTS_PIT_TBL}
                (cutoff_date, regime, metric, weight)
            VALUES (:cutoff_date, :regime, :metric, :weight)
            ON CONFLICT (cutoff_date, regime, metric) DO NOTHING
        """), rows)


def _derive_weights_continuous(weighted_anchors, sectors_s, Pxs_df):
    """
    Like derive_weights but accepts {anchor_date: (snapshot, regime_weight)} dict.
    Each anchor's IC observations are scaled by its regime_weight (0-1).
    Allows continuous blending rather than binary growth/conservative classification.
    """
    eligible_metrics = [m for m in QUALITY_METRICS if m not in EXCLUDE_METRICS]
    metric_stats = {m: {h: [] for h in HORIZONS} for m in eligible_metrics}

    for anchor, (snap, reg_w) in weighted_anchors.items():
        if snap is None or snap.empty or reg_w <= 0:
            continue
        snap    = build_derived_metrics(snap)
        ranked  = rank_within_sector(snap, sectors_s, eligible_metrics)
        u_stats = {}
        for m in eligible_metrics:
            if m in ranked.columns:
                s = ranked[m].dropna()
                u_stats[m] = (float(s.mean()), float(s.std())) if len(s) > 1 else (np.nan, np.nan)
        for horizon in HORIZONS:
            resid = _compute_residuals(Pxs_df, sectors_s, None, anchor, horizon)
            if resid.empty or len(resid) < MIN_STOCKS:
                continue
            n_decile       = max(1, int(np.floor(len(resid) * TOP_PCTILE)))
            sorted_ret     = resid.sort_values(ascending=False)
            top_tickers    = sorted_ret.iloc[:n_decile].index
            bottom_tickers = sorted_ret.iloc[-n_decile:].index
            for m in eligible_metrics:
                if m not in ranked.columns:
                    continue
                u_mean, u_std = u_stats.get(m, (np.nan, np.nan))
                if not u_std or u_std <= 0:
                    continue
                top_med    = float(ranked[m].reindex(top_tickers).dropna().median())
                bottom_med = float(ranked[m].reindex(bottom_tickers).dropna().median())
                if np.isnan(top_med) or np.isnan(bottom_med):
                    continue
                # Scale IC observation by regime weight
                ic = (top_med - bottom_med) / u_std * reg_w
                metric_stats[m][horizon].append(ic)

    rows = []
    for m in eligible_metrics:
        sz_all, t_all = [], []
        for h in HORIZONS:
            vals = metric_stats[m][h]
            if len(vals) < 2:
                continue
            arr = np.array(vals)
            sz_all.append(float(arr.mean()))
            t_all.append(float(scipy_stats.ttest_1samp(arr, 0).statistic))
        if not sz_all or not t_all:
            continue
        rows.append({'metric': m, 'avg_sz': float(np.mean(sz_all)), 'avg_t': float(np.mean(t_all))})

    if not rows:
        return {}
    df       = pd.DataFrame(rows).set_index('metric')
    med_sz   = df['avg_sz'].median()
    med_t    = df['avg_t'].median()
    eligible = df[(df['avg_sz'] > 0) & (df['avg_sz'] > med_sz) & (df['avg_t'] > med_t)].copy()
    if eligible.empty:
        return {}
    eligible  = eligible.nlargest(MAX_COMPONENTS, 'avg_t')
    eligible['weight'] = eligible['avg_t'].clip(lower=0)
    total = eligible['weight'].sum()
    if total <= 0:
        return {}
    eligible['weight'] /= total
    return eligible['weight'].to_dict()


def run_pit_weights(Pxs_df, sectors_s, mode='incremental'):
    """
    Compute and cache point-in-time quality weights for each anchor date.

    For each anchor date T, weights are computed using only anchors where
    anchor + MAX_HORIZON < T — ensuring no forward bias.

    mode='incremental': only compute cutoff dates not yet in cache
    mode='rebuild'    : wipe cache and recompute all dates from scratch
    """
    _ensure_pit_table()

    if mode == 'rebuild':
        with ENGINE.begin() as conn:
            conn.execute(text(f"DELETE FROM {QUALITY_WEIGHTS_PIT_TBL}"))
        print(f"  PIT weights cache cleared (rebuild mode)")

    # Load existing cache — includes dates with real weights
    cached = _load_pit_weights_cache()

    # Track all cutoff dates that were ever computed (including equal-weight/sentinel)
    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT DISTINCT cutoff_date FROM {QUALITY_WEIGHTS_PIT_TBL}
            """)).fetchall()
        cached_dates = {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        cached_dates = set(cached.keys())

    # All anchor dates available
    anchor_dates = sorted(load_anchor_dates())
    if not anchor_dates:
        print("  No anchor dates found in DB.")
        return

    # Cutoff dates = anchor dates shifted by MAX_HORIZON trading days
    # (first date at which weights from that anchor are valid)
    all_px_dates = Pxs_df.index

    # For each anchor, compute weights at the first date MAX_HORIZON days after it
    # We compute weights at every anchor date as cutoff (using all prior anchors)
    cutoff_candidates = []
    for anchor in anchor_dates:
        # Find first trading day at least MAX_HORIZON days after anchor
        future = all_px_dates[all_px_dates > anchor]
        if len(future) < MAX_HORIZON:
            continue
        cutoff = future[MAX_HORIZON - 1]   # MAX_HORIZON-th trading day after anchor
        cutoff_candidates.append((anchor, cutoff))

    new_dates = [(a, c) for a, c in cutoff_candidates if c not in cached_dates]
    print(f"  PIT weights: {len(cached_dates)} cached, {len(new_dates)} to compute")

    if not new_dates:
        print("  All PIT weight dates already cached.")
        # Always recompute scores for the last anchor date (intraday price updates)
        last_anchor = pd.Timestamp(sorted(load_anchor_dates())[-1])
        print(f"  Refreshing quality scores for last anchor: {last_anchor.date()}")
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                DELETE FROM {QUALITY_SCORES_TBL} WHERE date = :dt
            """), {'dt': last_anchor.date()})
        get_quality_scores(
            calc_dates         = pd.DatetimeIndex([last_anchor]),
            universe           = list(sectors_s.index),
            Pxs_df             = Pxs_df,
            sectors_s          = sectors_s,
            use_cached_weights = False,
            force_recompute    = False,
            use_pit_weights    = True,
        )
        return

    # Load all snapshots once
    snapshots = load_all_snapshots(anchor_dates)

    for i, (trigger_anchor, cutoff) in enumerate(new_dates, 1):
        print(f"  [{i}/{len(new_dates)}] cutoff={cutoff.date()} "
              f"(triggered by anchor {trigger_anchor.date()})", end='', flush=True)

        # Only use anchors where anchor + MAX_HORIZON < cutoff (complete forward window)
        eligible_anchors = {}
        for a in anchor_dates:
            if a not in snapshots:
                continue
            future_a = all_px_dates[all_px_dates > a]
            if len(future_a) < MAX_HORIZON:
                continue
            if future_a[MAX_HORIZON - 1] < cutoff:
                eligible_anchors[a] = snapshots[a]

        if not eligible_anchors:
            print(" — no eligible anchors yet, using equal weights")
            _save_pit_weights(cutoff, {}, {})
            continue

        rate_signal = compute_rate_signal(Pxs_df, QF_MAV_WINDOW, QF_THRESHOLD)

        # Each eligible anchor contributes to GQF with weight (1-q) and CQF with weight q
        # where q is the rate signal at that anchor date (0=easing, 0.5=neutral, 1.0=tight)
        gqf_anchors = {}
        cqf_anchors = {}
        for a, snap in eligible_anchors.items():
            rate_dates = rate_signal.index[rate_signal.index <= a]
            q = float(rate_signal.loc[rate_dates[-1]]) if not rate_dates.empty else 0.0
            if (1 - q) > 0:
                gqf_anchors[a] = (snap, 1 - q)   # growth weight
            if q > 0:
                cqf_anchors[a] = (snap, q)         # conservative weight

        gqf_w = _derive_weights_continuous(gqf_anchors, sectors_s, Pxs_df) if gqf_anchors else {}
        cqf_w = _derive_weights_continuous(cqf_anchors, sectors_s, Pxs_df) if cqf_anchors else {}

        # If no conservative signal yet, use GQF as fallback for CQF
        if not cqf_w and gqf_w:
            cqf_w = gqf_w
            print(f" — GQF={len(gqf_w)} metrics, CQF=fallback(GQF)", flush=True)
        elif not gqf_w and not cqf_w:
            print(" — weight derivation failed, using equal weights", flush=True)
        else:
            print(f" — GQF={len(gqf_w)} metrics, CQF={len(cqf_w)} metrics", flush=True)

        _save_pit_weights(cutoff, gqf_w, cqf_w)

    print(f"\n  PIT weights computation complete.")

    if not new_dates:
        return

    # Delete and recompute quality scores only for anchor dates >= first new PIT cutoff
    new_cutoffs      = [c for _, c in new_dates]
    first_new_cutoff = min(new_cutoffs)
    anchor_dates_all = sorted(load_anchor_dates())
    affected = pd.DatetimeIndex(
        [pd.Timestamp(a) for a in anchor_dates_all
         if pd.Timestamp(a) >= first_new_cutoff])

    if not affected.empty:
        print(f"\n  Recomputing {len(affected)} quality score dates "
              f"from {first_new_cutoff.date()} onwards...")
        # Delete affected dates from cache so they get recomputed
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                DELETE FROM {QUALITY_SCORES_TBL}
                WHERE date = ANY(:dates)
            """), {'dates': [d.date() for d in affected]})
        get_quality_scores(
            calc_dates         = affected,
            universe           = list(sectors_s.index),
            Pxs_df             = Pxs_df,
            sectors_s          = sectors_s,
            use_cached_weights = False,
            force_recompute    = False,
            use_pit_weights    = True,
        )


def get_pit_weights_at(cutoff_date):
    """
    Return (gqf_weights, cqf_weights) for the most recent cutoff_date <= given date.
    Returns ({}, {}) if no prior weights exist (use equal weights).
    """
    cached = _load_pit_weights_cache()
    if not cached:
        return {}, {}
    prior = [d for d in sorted(cached.keys()) if d <= cutoff_date]
    if not prior:
        return {}, {}
    best = prior[-1]
    entry = cached[best]
    return entry.get('gqf', {}), entry.get('cqf', {})

QF_MAV_WINDOW      = 252
QF_THRESHOLD       = 50

# ==============================================================================
# PRE-DERIVED WEIGHTS  (update by running run() occasionally)
# ==============================================================================

GQF_WEIGHTS = {
  'GGP': 0.140102,
  'GS': 0.138517,
  'GS/S_Vol': 0.112783,
  'GS*r2_S': 0.105234,
  'ROId': 0.099669,
  'GGP/GP_Vol': 0.089854,
  'OMd*r2_S': 0.084534,
  'PSG': 0.081300,
  'FCF_PG': 0.076209,
  'OMd': 0.071798,
}
CQF_WEIGHTS = {
  'SGD*r2_S': 0.133990,
  'ISGD': 0.117356,
  'r&d': 0.112518,
  'GE': 0.102898,
  'OM': 0.095645,
  'LastSGD': 0.094775,
  'OMd*r2_S': 0.094254,
  'OMd': 0.089360,
  'GE*r2_E': 0.082863,
  'SGD': 0.076340,
}

GQF_WEIGHTS = {
  'GGP': 0.161202,
  'GS': 0.160172,
  'GS*r2_S': 0.126151,
  'PIG': 0.098813,
  'GS/S_Vol': 0.087437,
  'GGP*r2_GP': 0.080083,
  'PSG': 0.076695,
  'PIG*r2_E': 0.075812,
  'PIG/E_Vol': 0.068880,
  'ISGD': 0.064754,
}
CQF_WEIGHTS = {
  'SGD*r2_S': 0.137463,
  'ISGD': 0.116068,
  'r&d': 0.104249,
  'LastSGD': 0.100078,
  'GE': 0.098610,
  'OMd*r2_S': 0.095753,
  'OMd': 0.091794,
  'OM': 0.091036,
  'GE*r2_E': 0.088546,
  'GE/E_Vol': 0.076404,
}

CONSERVATIVE_YEARS = [2021, 2022]
MAX_COMPONENTS     = 10
EXCLUDE_METRICS    = ['ROE', 'ROE-P', 'ROEd']
HORIZONS           = [21, 63]
TOP_PCTILE         = 0.10
MIN_STOCKS         = 20
WINSOR             = (0.01, 0.99)
MAV_WINDOWS        = [63, 126, 252]
THRESHOLDS         = [25, 50, 75]
VOL_MIN            = 1.0

QUALITY_METRICS = [
    'HSG', 'GS', 'GE', 'GGP', 'SGD', 'LastSGD', 'PIG', 'PSG',
    'OM', 'ROI', 'FCF_PG', 'OMd', 'ROId', 'ISGD', 'r&d',
    'GS/S_Vol', 'HSG/S_Vol', 'PSG/S_Vol', 'GE/E_Vol', 'PIG/E_Vol', 'GGP/GP_Vol',
    'GS*r2_S', 'SGD*r2_S', 'OMd*r2_S', 'GE*r2_E', 'PIG*r2_E', 'GGP*r2_GP',
]
RAW_DB_COLS = [
    'HSG', 'GS', 'GE', 'GGP', 'SGD', 'LastSGD', 'PIG', 'PSG',
    'OM', 'ROI', 'FCF_PG', 'OMd', 'ROId', 'ISGD', 'r&d',
    'S Vol', 'E Vol', 'GP Vol', 'r2 S', 'r2 E', 'r2 GP',
]

DB_URL = 'postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db'
ENGINE = create_engine(DB_URL)

# ==============================================================================
# HELPERS
# ==============================================================================

def normalize_ticker(t):
    return str(t).split(' ')[0].strip().upper()

def normalize_index(idx):
    return pd.DatetimeIndex([pd.Timestamp(d) for d in idx])

def normalize_df(df, col='ticker'):
    df = df.copy()
    df[col] = df[col].apply(normalize_ticker)
    return df

def normalize_series(s):
    s = s.copy()
    s.index = [normalize_ticker(t) for t in s.index]
    return s

def winsorize(s):
    lo, hi = WINSOR
    return s.clip(lower=s.quantile(lo), upper=s.quantile(hi))

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_anchor_dates():
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT DISTINCT date FROM {QUALITY_TABLE} ORDER BY date
        """)).fetchall()
    return [pd.Timestamp(r[0]) for r in rows]


def load_all_snapshots(anchor_dates):
    """Load quality metric snapshots for given anchor dates."""
    if not anchor_dates:
        return {}
    fetch_cols = list(dict.fromkeys(['Size'] + RAW_DB_COLS))
    cols_sql   = ', '.join([f'"{m}"' for m in fetch_cols])
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT date, ticker, {cols_sql}
            FROM {QUALITY_TABLE}
            WHERE date = ANY(:dates) AND ticker IS NOT NULL
        """), conn, params={"dates": [d.date() for d in anchor_dates]})
    if df.empty:
        return {}
    df['date']   = pd.to_datetime(df['date'])
    df['ticker'] = df['ticker'].apply(normalize_ticker)
    snapshots = {}
    for date, grp in df.groupby('date'):
        snap = grp.drop(columns='date').drop_duplicates('ticker').set_index('ticker')
        for col in fetch_cols:
            if col in snap.columns:
                snap[col] = pd.to_numeric(snap[col], errors='coerce')
        snapshots[pd.Timestamp(date)] = snap
    return snapshots


def load_snapshots_for_dates(calc_dates, anchor_dates_sorted):
    """
    Load only the anchor snapshots actually needed for calc_dates.
    For each calc_date, the relevant anchor is the last one <= calc_date.
    Returns dict of only the needed snapshots.
    """
    needed_anchors = set()
    for dt in calc_dates:
        candidates = [a for a in anchor_dates_sorted if a <= dt]
        if candidates:
            needed_anchors.add(candidates[-1])
    return load_all_snapshots(sorted(needed_anchors))


# ==============================================================================
# DERIVED METRICS
# ==============================================================================

def build_derived_metrics(snap):
    s = snap.copy()

    def col(name):
        return s[name] if name in s.columns else pd.Series(np.nan, index=s.index)

    def safe_div(num, denom_name):
        return col(num) / col(denom_name).clip(lower=VOL_MIN)

    def safe_mul(base_name, r2_name):
        return col(base_name) * col(r2_name)

    s['GS/S_Vol']   = safe_div('GS',  'S Vol')
    s['HSG/S_Vol']  = safe_div('HSG', 'S Vol')
    s['PSG/S_Vol']  = safe_div('PSG', 'S Vol')
    s['GE/E_Vol']   = safe_div('GE',  'E Vol')
    s['PIG/E_Vol']  = safe_div('PIG', 'E Vol')
    s['GGP/GP_Vol'] = safe_div('GGP', 'GP Vol')
    s['GS*r2_S']    = safe_mul('GS',  'r2 S')
    s['SGD*r2_S']   = safe_mul('SGD', 'r2 S')
    s['OMd*r2_S']   = safe_mul('OMd', 'r2 S')
    s['GE*r2_E']    = safe_mul('GE',  'r2 E')
    s['PIG*r2_E']   = safe_mul('PIG', 'r2 E')
    s['GGP*r2_GP']  = safe_mul('GGP', 'r2 GP')
    return s

# ==============================================================================
# RATE REGIME
# ==============================================================================

def compute_rate_signal(Pxs_df, mav_window, threshold):
    if 'USGG10YR' not in Pxs_df.columns:
        raise ValueError("USGG10YR not found in Pxs_df columns")
    rate     = Pxs_df['USGG10YR'].dropna() * 100
    rate_mav = rate.rolling(mav_window, min_periods=mav_window // 2).mean()
    rate_mom = rate - rate_mav
    q = pd.Series(0.5, index=rate_mom.index)
    q[rate_mom >  threshold] = 1.0
    q[rate_mom < -threshold] = 0.0
    return q

# ==============================================================================
# WITHIN-SECTOR RANKING
# ==============================================================================

def rank_within_sector(snap, sectors_s, metrics):
    ranked = pd.DataFrame(index=snap.index)
    sec    = normalize_series(sectors_s).reindex(snap.index)
    for m in metrics:
        if m not in snap.columns:
            ranked[m] = np.nan
            continue
        col = snap[m].copy()
        out = pd.Series(np.nan, index=snap.index)
        for sector, grp_idx in sec.groupby(sec).groups.items():
            grp = col.reindex(grp_idx).dropna()
            if len(grp) < 3:
                continue
            grp_w = winsorize(grp)
            r     = grp_w.rank(method='average')
            out.loc[r.index] = (r - 1) / (len(r) - 1) if len(r) > 1 else 0.5
        ranked[m] = out
    return ranked

# ==============================================================================
# FACTOR WEIGHT DERIVATION
# ==============================================================================

def derive_weights(snapshots, sectors_s, Pxs_df, regime):
    eligible_metrics = [m for m in QUALITY_METRICS if m not in EXCLUDE_METRICS]
    anchor_dates     = sorted(snapshots.keys())
    if regime == 'conservative':
        dates_in_regime = [d for d in anchor_dates if d.year in CONSERVATIVE_YEARS]
    else:
        dates_in_regime = [d for d in anchor_dates if d.year not in CONSERVATIVE_YEARS]
    if not dates_in_regime:
        print(f"    WARNING: no anchor dates for regime='{regime}'")
        return {}
    print(f"    {len(dates_in_regime)} dates in regime '{regime}'", end='', flush=True)
    metric_stats = {m: {h: [] for h in HORIZONS} for m in eligible_metrics}
    for anchor in dates_in_regime:
        snap = snapshots.get(anchor)
        if snap is None or snap.empty:
            continue
        snap    = build_derived_metrics(snap)
        ranked  = rank_within_sector(snap, sectors_s, eligible_metrics)
        u_stats = {}
        for m in eligible_metrics:
            if m in ranked.columns:
                s = ranked[m].dropna()
                u_stats[m] = (float(s.mean()), float(s.std())) if len(s) > 1 else (np.nan, np.nan)
        for horizon in HORIZONS:
            resid = _compute_residuals(Pxs_df, sectors_s, None, anchor, horizon)
            if resid.empty or len(resid) < MIN_STOCKS:
                continue
            n_decile       = max(1, int(np.floor(len(resid) * TOP_PCTILE)))
            sorted_ret     = resid.sort_values(ascending=False)
            top_tickers    = sorted_ret.iloc[:n_decile].index
            bottom_tickers = sorted_ret.iloc[-n_decile:].index
            for m in eligible_metrics:
                if m not in ranked.columns:
                    continue
                u_mean, u_std = u_stats.get(m, (np.nan, np.nan))
                if not u_std or u_std <= 0:
                    continue
                top_med    = float(ranked[m].reindex(top_tickers).dropna().median())
                bottom_med = float(ranked[m].reindex(bottom_tickers).dropna().median())
                if np.isnan(top_med) or np.isnan(bottom_med):
                    continue
                metric_stats[m][horizon].append((top_med - bottom_med) / u_std)
    rows = []
    for m in eligible_metrics:
        sz_all, t_all = [], []
        for h in HORIZONS:
            vals = metric_stats[m][h]
            if len(vals) < 2:
                continue
            arr = np.array(vals)
            sz_all.append(float(arr.mean()))
            t_all.append(float(scipy_stats.ttest_1samp(arr, 0).statistic))
        if not sz_all or not t_all:
            continue
        rows.append({'metric': m, 'avg_sz': float(np.mean(sz_all)), 'avg_t': float(np.mean(t_all))})
    n_resid_ok = sum(1 for m in eligible_metrics for h in HORIZONS if metric_stats[m][h])
    print(f"  — {n_resid_ok} metric/horizon pairs with data")
    if not rows:
        return {}
    df       = pd.DataFrame(rows).set_index('metric')
    med_sz   = df['avg_sz'].median()
    med_t    = df['avg_t'].median()
    eligible = df[(df['avg_sz'] > 0) & (df['avg_sz'] > med_sz) & (df['avg_t'] > med_t)].copy()
    if eligible.empty:
        return {}
    eligible  = eligible.nlargest(MAX_COMPONENTS, 'avg_t')
    eligible['weight'] = eligible['avg_t'].clip(lower=0)
    total = eligible['weight'].sum()
    if total <= 0:
        return {}
    eligible['weight'] /= total
    return eligible['weight'].to_dict()

# ==============================================================================
# RESIDUALS
# ==============================================================================

_RESID_CACHE = None

def _load_resid_cache():
    global _RESID_CACHE
    if _RESID_CACHE is not None:
        return _RESID_CACHE
    print(f"  Loading {RESIDUAL_TABLE} from DB...", flush=True)
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(f"SELECT date, ticker, resid FROM {RESIDUAL_TABLE} ORDER BY date"), conn)
    df['date']   = pd.to_datetime(df['date'])
    df['resid']  = df['resid'].astype(float)
    df['ticker'] = df['ticker'].apply(normalize_ticker)
    pivot = df.pivot_table(index='date', columns='ticker', values='resid', aggfunc='last')
    _RESID_CACHE = pivot
    print(f"  Residuals loaded: {pivot.shape[0]} dates x {pivot.shape[1]} tickers")
    return pivot


def _compute_residuals(Pxs_df, sectors_s, size_s, anchor_date, horizon):
    resid_pivot  = _load_resid_cache()
    future_dates = resid_pivot.index[resid_pivot.index > anchor_date]
    if len(future_dates) < horizon:
        return pd.Series(dtype=float)
    window       = future_dates[:horizon]
    window_resid = resid_pivot.loc[window]
    compounded   = (1 + window_resid.fillna(0)).prod(axis=0) - 1
    valid_mask   = window_resid.notna().sum(axis=0) >= max(1, horizon // 2)
    compounded   = compounded[valid_mask].dropna()
    if len(compounded) < MIN_STOCKS:
        return pd.Series(dtype=float)
    return compounded

# ==============================================================================
# COMPOSITE SCORE COMPUTATION
# ==============================================================================

def compute_composite_scores(snap, sectors_s, gqf_weights, cqf_weights, q):
    all_metrics = list(set(list(gqf_weights.keys()) + list(cqf_weights.keys())))
    ranked      = rank_within_sector(snap, sectors_s, all_metrics)

    def weighted_score(weights):
        if not weights:
            return pd.Series(np.nan, index=ranked.index)
        score = pd.Series(0.0, index=ranked.index)
        total = 0.0
        for m, w in weights.items():
            if m not in ranked.columns:
                continue
            score = score.add(ranked[m] * w, fill_value=0)
            total += w
        return score / total if total > 0 else pd.Series(np.nan, index=ranked.index)

    return ((1 - q) * weighted_score(gqf_weights) + q * weighted_score(cqf_weights)).dropna()

# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_composite(scores_by_date, Pxs_df, sectors_s, snapshots):
    anchor_dates = sorted(scores_by_date.keys())
    raw = {}
    for anchor in anchor_dates:
        scores = scores_by_date[anchor]
        if scores.empty:
            continue
        u_std = float(scores.std())
        if u_std <= 0:
            continue
        for horizon in HORIZONS:
            resid = _compute_residuals(Pxs_df, sectors_s, None, anchor, horizon)
            if resid.empty or len(resid) < MIN_STOCKS:
                continue
            common = scores.index.intersection(resid.index)
            if len(common) < MIN_STOCKS:
                continue
            s_aligned = scores[common]
            r_aligned = resid[common]
            n_decile       = max(1, int(np.floor(len(common) * TOP_PCTILE)))
            sorted_ret     = r_aligned.sort_values(ascending=False)
            top_med    = float(s_aligned.reindex(sorted_ret.iloc[:n_decile].index).dropna().median())
            bottom_med = float(s_aligned.reindex(sorted_ret.iloc[-n_decile:].index).dropna().median())
            if not np.isnan(top_med) and not np.isnan(bottom_med):
                raw[(anchor, horizon)] = (top_med - bottom_med) / u_std
    if not raw:
        return pd.DataFrame(), {}
    summary_rows = []
    for h in HORIZONS:
        vals = [raw[(d, h)] for d in anchor_dates if (d, h) in raw]
        if len(vals) < 2:
            continue
        arr = np.array(vals)
        summary_rows.append({
            'horizon': h, 'mean_spread_z': round(float(arr.mean()), 3),
            't_stat': round(float(scipy_stats.ttest_1samp(arr, 0).statistic), 3),
            'consistency': round(float((arr > 0).mean() * 100), 1), 'n_dates': len(arr),
        })
    summary = pd.DataFrame(summary_rows).set_index('horizon')
    annual  = {}
    for yr in sorted(set(d.year for d, h in raw)):
        annual[yr] = {}
        for h in HORIZONS:
            vals = [raw[(d, h)] for d in anchor_dates if d.year == yr and (d, h) in raw]
            if vals:
                arr = np.array(vals)
                annual[yr][h] = {
                    'mean_spread_z': round(float(arr.mean()), 3),
                    'consistency': round(float((arr > 0).mean() * 100), 1),
                    'n_dates': len(arr),
                }
    return summary, annual

# ==============================================================================
# QUALITY SCORES CACHE
# ==============================================================================

def _ensure_scores_table():
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {QUALITY_SCORES_TBL} (
                date   DATE        NOT NULL,
                ticker VARCHAR(20) NOT NULL,
                score  NUMERIC,
                PRIMARY KEY (date, ticker)
            )
        """))


def _load_cached_dates():
    try:
        with ENGINE.connect() as conn:
            rows = conn.execute(text(
                f"SELECT DISTINCT date FROM {QUALITY_SCORES_TBL}"
            )).fetchall()
        return {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        return set()


def _save_scores(scores_by_date):
    if not scores_by_date:
        return
    _ensure_scores_table()
    rows = []
    for dt, scores in scores_by_date.items():
        for ticker, score in scores.items():
            if not np.isnan(float(score)):
                rows.append({'date': dt.date(), 'ticker': ticker, 'score': score})
    if not rows:
        return
    df    = pd.DataFrame(rows)
    dates = list({r['date'] for r in rows})
    with ENGINE.begin() as conn:
        conn.execute(text(
            f"DELETE FROM {QUALITY_SCORES_TBL} WHERE date = ANY(:dates)"
        ), {"dates": dates})
    df.to_sql(QUALITY_SCORES_TBL, ENGINE, if_exists='append', index=False)
    print(f"  Saved {len(df):,} quality score rows ({len(dates)} dates) to '{QUALITY_SCORES_TBL}'")


def _load_scores_from_db(calc_dates, universe):
    date_list = [d.date() for d in calc_dates]
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT date, ticker, score FROM {QUALITY_SCORES_TBL}
            WHERE date = ANY(:dates)
        """), conn, params={"dates": date_list})
    if df.empty:
        return pd.DataFrame(index=calc_dates, columns=universe, dtype=float)
    df['date']   = pd.to_datetime(df['date'])
    df['ticker'] = df['ticker'].apply(normalize_ticker)
    df['score']  = df['score'].astype(float)
    pivot = df.pivot_table(index='date', columns='ticker', values='score', aggfunc='last')
    return pivot.reindex(index=calc_dates, columns=universe)


def update_cached_weights(gqf_w, cqf_w):
    print("\n  Copy into quality_factor.py GQF_WEIGHTS / CQF_WEIGHTS:")
    print("  GQF_WEIGHTS = {")
    for m, w in sorted(gqf_w.items(), key=lambda x: -x[1]):
        print(f"      '{m}': {w:.6f},")
    print("  }")
    print("  CQF_WEIGHTS = {")
    for m, w in sorted(cqf_w.items(), key=lambda x: -x[1]):
        print(f"      '{m}': {w:.6f},")
    print("  }")


def get_quality_scores(calc_dates, universe, Pxs_df, sectors_s,
                        use_cached_weights=True, force_recompute=False,
                        use_pit_weights=False):
    """
    Main entry point for quality scores with DB caching.
    Incremental: only computes dates not already in cache.
    Fast daily path: loads only the anchor snapshot(s) needed for new dates.

    use_pit_weights=True: use point-in-time weights from quality_weights_pit table.
                          Automatically invalidates cache for dates with different weights.
    """
    _ensure_scores_table()

    if force_recompute:
        dates_to_compute = pd.DatetimeIndex(sorted(calc_dates))
        print(f"  Force recompute: computing all {len(dates_to_compute)} dates")
    else:
        cached_dates     = _load_cached_dates()
        dates_to_compute = pd.DatetimeIndex(
            sorted(d for d in calc_dates if d not in cached_dates)
        )
        print(f"  Quality scores: {len(cached_dates)} dates cached, "
              f"{len(dates_to_compute)} new dates to compute")

    if len(dates_to_compute) > 0:
        # ── Weight selection ──────────────────────────────────────────────────
        if use_pit_weights:
            print("  Using point-in-time weights from quality_weights_pit cache")
            _pit_cache = _load_pit_weights_cache()
            _pit_dates = sorted(_pit_cache.keys())
            # Weights will be looked up per calc_date below
            gqf_w = cqf_w = None   # sentinel — will be overridden per date
        elif use_cached_weights and GQF_WEIGHTS and CQF_WEIGHTS:
            gqf_w = GQF_WEIGHTS
            cqf_w = CQF_WEIGHTS
            print(f"  Using hardcoded weights: {len(gqf_w)} GQF, {len(cqf_w)} CQF components")
        else:
            print("  Deriving weights from IC analysis (slow)...")
            anchor_dates_all = load_anchor_dates()
            snapshots_all    = load_all_snapshots(anchor_dates_all)
            gqf_w = derive_weights(snapshots_all, sectors_s, Pxs_df, regime='growth')
            cqf_w = derive_weights(snapshots_all, sectors_s, Pxs_df, regime='conservative')
            print(f"  Derived: {len(gqf_w)} GQF, {len(cqf_w)} CQF components")

        # ── KEY OPTIMISATION: only load anchor snapshots needed for new dates ──
        anchor_dates_sorted = sorted(load_anchor_dates())
        snapshots_ = load_snapshots_for_dates(dates_to_compute, anchor_dates_sorted)
        print(f"  Loaded {len(snapshots_)} anchor snapshot(s) for {len(dates_to_compute)} new date(s)")

        rate_signal = compute_rate_signal(Pxs_df, QF_MAV_WINDOW, QF_THRESHOLD)

        new_scores = {}
        for calc_date in dates_to_compute:
            candidates = [a for a in anchor_dates_sorted if a <= calc_date]
            if not candidates:
                continue
            anchor = candidates[-1]
            snap   = snapshots_.get(anchor)
            if snap is None or snap.empty:
                continue
            snap = build_derived_metrics(snap)
            rate_dates = rate_signal.index[rate_signal.index <= calc_date]
            q = float(rate_signal.loc[rate_dates[-1]]) if not rate_dates.empty else 0.5

            # Look up PIT weights for this calc_date
            if use_pit_weights:
                prior_pit = [d for d in _pit_dates if d <= calc_date]
                if prior_pit:
                    entry  = _pit_cache[prior_pit[-1]]
                    gqf_w_ = entry.get('gqf', {})
                    cqf_w_ = entry.get('cqf', {})
                else:
                    # No PIT weights yet — equal weights
                    n = len([m for m in QUALITY_METRICS if m not in EXCLUDE_METRICS])
                    gqf_w_ = cqf_w_ = {m: 1.0/n for m in QUALITY_METRICS
                                        if m not in EXCLUDE_METRICS}
            else:
                gqf_w_ = gqf_w
                cqf_w_ = cqf_w

            scores = compute_composite_scores(snap, sectors_s, gqf_w_, cqf_w_, q)
            if not scores.empty:
                new_scores[calc_date] = scores

        if new_scores:
            _save_scores(new_scores)

    result = _load_scores_from_db(calc_dates, universe)
    print(f"  Quality scores loaded: "
          f"{result.notna().any(axis=1).sum()} dates | "
          f"{result.notna().any(axis=0).sum()} tickers")
    return result

# ==============================================================================
# MAIN RUN  (full rebuild — run occasionally to refresh weights)
# ==============================================================================

def run(Pxs_df, sectors_s, mav_window=252, threshold=50, verbose=True,
        use_cached_weights=True, force_recompute=False):
    """
    Build and evaluate rate-conditioned quality factor.

    Daily fast path (use_cached_weights=True, force_recompute=False):
      - Uses hardcoded GQF_WEIGHTS/CQF_WEIGHTS — no weight derivation
      - Only computes scores for dates not already in DB cache
      - Loads only the anchor snapshots needed for new dates

    Full rebuild (use_cached_weights=False or force_recompute=True):
      - Re-derives weights from IC analysis (slow)
      - Recomputes all scores
    """
    Pxs_df       = Pxs_df.copy()
    Pxs_df.index = normalize_index(Pxs_df.index)
    sectors_s    = normalize_series(sectors_s)

    if verbose:
        print("=" * 70)
        print(f"  QUALITY FACTOR  |  window={mav_window}d  |  threshold={threshold}bps")
        mode = "INCREMENTAL" if use_cached_weights and not force_recompute else "FULL REBUILD"
        print(f"  Mode: {mode}")
        print("=" * 70)

    # ── Resolve weights ────────────────────────────────────────────────────────
    if use_cached_weights and GQF_WEIGHTS and CQF_WEIGHTS and not force_recompute:
        gqf_weights = GQF_WEIGHTS
        cqf_weights = CQF_WEIGHTS
        if verbose:
            print(f"  Using hardcoded weights: {len(gqf_weights)} GQF, {len(cqf_weights)} CQF")
    else:
        anchor_dates = load_anchor_dates()
        if verbose:
            print(f"  Loading {len(anchor_dates)} anchor snapshots for weight derivation...",
                  end='', flush=True)
        snapshots = load_all_snapshots(anchor_dates)
        if verbose:
            print(f" done")
            print(f"\n  Deriving GQF weights (ex {CONSERVATIVE_YEARS})...")
        gqf_weights = derive_weights(snapshots, sectors_s, Pxs_df, regime='growth')
        if verbose:
            print(f"  Deriving CQF weights ({CONSERVATIVE_YEARS})...")
        cqf_weights = derive_weights(snapshots, sectors_s, Pxs_df, regime='conservative')
        if verbose:
            _print_weights(gqf_weights, cqf_weights)
            update_cached_weights(gqf_weights, cqf_weights)

    # ── Compute scores (incremental) ───────────────────────────────────────────
    _ensure_scores_table()
    if force_recompute:
        anchor_dates_sorted = sorted(load_anchor_dates())
        dates_to_compute    = pd.DatetimeIndex(sorted(anchor_dates_sorted))
    else:
        cached_dates        = _load_cached_dates()
        anchor_dates_sorted = sorted(load_anchor_dates())
        dates_to_compute    = pd.DatetimeIndex(
            sorted(d for d in anchor_dates_sorted if d not in cached_dates)
        )

    if verbose:
        print(f"  Anchor dates cached: {len(anchor_dates_sorted) - len(dates_to_compute)}  "
              f"| to compute: {len(dates_to_compute)}")

    new_scores = {}
    if len(dates_to_compute) > 0:
        # Only load snapshots needed for new dates
        snapshots_ = load_snapshots_for_dates(dates_to_compute, anchor_dates_sorted)
        if verbose:
            print(f"  Loaded {len(snapshots_)} anchor snapshot(s)")
        rate_signal = compute_rate_signal(Pxs_df, mav_window, threshold)
        for anchor in dates_to_compute:
            snap = snapshots_.get(anchor)
            if snap is None or snap.empty:
                continue
            snap = build_derived_metrics(snap)
            rate_dates = rate_signal.index[rate_signal.index <= anchor]
            q = float(rate_signal.loc[rate_dates[-1]]) if not rate_dates.empty else 0.5
            scores = compute_composite_scores(snap, sectors_s, gqf_weights, cqf_weights, q)
            if not scores.empty:
                new_scores[anchor] = scores
        if new_scores:
            _save_scores(new_scores)

    # ── Load all scores from DB for return value ───────────────────────────────
    all_anchor_dates = sorted(load_anchor_dates())
    all_scores_by_date = {}
    if all_anchor_dates:
        date_list = [d.date() for d in all_anchor_dates]
        with ENGINE.connect() as conn:
            df = pd.read_sql(text(f"""
                SELECT date, ticker, score FROM {QUALITY_SCORES_TBL}
                WHERE date = ANY(:dates)
            """), conn, params={"dates": date_list})
        if not df.empty:
            df['date']   = pd.to_datetime(df['date'])
            df['ticker'] = df['ticker'].apply(normalize_ticker)
            df['score']  = df['score'].astype(float)
            for dt, grp in df.groupby('date'):
                all_scores_by_date[pd.Timestamp(dt)] = grp.set_index('ticker')['score']

    if verbose:
        print(f"  Scores available: {len(all_scores_by_date)} dates")

    # ── Evaluation (only on full rebuild) ─────────────────────────────────────
    summary = pd.DataFrame()
    annual  = {}
    if not use_cached_weights or force_recompute:
        if verbose:
            print(f"  Evaluating composite...\n")
        snapshots_all = load_all_snapshots(all_anchor_dates)
        summary, annual = evaluate_composite(all_scores_by_date, Pxs_df,
                                              sectors_s, snapshots_all)
        if verbose:
            _print_results(summary, annual, mav_window, threshold)

    return summary, annual, all_scores_by_date, gqf_weights, cqf_weights


def gridsearch(Pxs_df, sectors_s):
    print("=" * 70)
    print("  QUALITY FACTOR GRIDSEARCH")
    print(f"  Windows: {MAV_WINDOWS}  Thresholds: {THRESHOLDS} bps")
    print("=" * 70 + "\n")
    rows = []
    for window in MAV_WINDOWS:
        for thresh in THRESHOLDS:
            print(f"\n--- window={window}  threshold={thresh}bps ---")
            summary, annual, _, _, _ = run(Pxs_df, sectors_s, mav_window=window,
                                            threshold=thresh, verbose=False)
            if summary.empty:
                continue
            for h in HORIZONS:
                if h not in summary.index:
                    continue
                r = summary.loc[h]
                rows.append({'mav_window': window, 'threshold': thresh, 'horizon': h,
                              'mean_spread_z': r['mean_spread_z'], 't_stat': r['t_stat'],
                              'consistency': r['consistency'], 'n_dates': r['n_dates']})
    grid_df = pd.DataFrame(rows)
    if not grid_df.empty:
        print("\n  GRIDSEARCH SUMMARY — sorted by t_stat (63d)")
        for h in HORIZONS:
            sub = grid_df[grid_df['horizon'] == h].sort_values('t_stat', ascending=False)
            print(f"\n  Horizon: {h}d")
            for _, row in sub.iterrows():
                print(f"  window={int(row['mav_window'])}  thresh={int(row['threshold'])}  "
                      f"spread_z={row['mean_spread_z']:+.3f}  t={row['t_stat']:+.3f}  "
                      f"cons={row['consistency']:.0f}%")
    return grid_df


def _print_weights(gqf_weights, cqf_weights):
    print("\n  GQF components (growth regime):")
    for m, w in sorted(gqf_weights.items(), key=lambda x: -x[1]):
        print(f"    {m:<12}  {w:.4f}")
    print("\n  CQF components (conservative regime):")
    for m, w in sorted(cqf_weights.items(), key=lambda x: -x[1]):
        print(f"    {m:<12}  {w:.4f}")


def _print_results(summary, annual, mav_window, threshold):
    print("\n" + "=" * 70)
    print(f"  COMPOSITE EVALUATION  |  window={mav_window}d  |  threshold={threshold}bps")
    print("=" * 70)
    print(summary.to_string())
    print("\n  Annual breakdown:")
    for h in HORIZONS:
        print(f"\n  Horizon {h}d:")
        print(f"  {'Year':<6} {'spread_z':>10} {'consistency%':>14} {'n':>4}")
        print("  " + "-" * 38)
        for yr in sorted(annual.keys()):
            if h not in annual[yr]:
                continue
            r = annual[yr][h]
            print(f"  {yr:<6} {r['mean_spread_z']:>+10.3f} {r['consistency']:>14.1f} {r['n_dates']:>4}")
    print("=" * 70)


if __name__ == "__main__":
    print("Usage: from quality_factor import run, get_quality_scores")
    # daily, fast:
    summary, annual, scores, gqf_w, cqf_w = run(Pxs_df, sectors_s)
    # occasional weight refresh:
    # summary, annual, scores, gqf_w, cqf_w = run(Pxs_df, sectors_s, use_cached_weights=False)
    # full rebuild:
    # summary, annual, scores, gqf_w, cqf_w = run(Pxs_df, sectors_s, force_recompute=True)

