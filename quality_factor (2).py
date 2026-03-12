"""
quality_factor.py
=================
Rate-conditioned quality factor: construction + evaluation + gridsearch.

Combines:
  - Growth Quality Factor (GQF): metrics weighted by t-stats in non-stress years
  - Conservative Quality Factor (CQF): metrics weighted by t-stats in 2021/2022
  - Rate momentum signal (USGG10YR vs rolling MAV) to blend the two

Usage:
    from quality_factor import run, gridsearch

    # Single run with specific hyperparameters
    stats, annual, scores = run(Pxs_df, sectors_s,
                                 mav_window=126, threshold=50)

    # Gridsearch over all hyperparameter combinations
    grid_results = gridsearch(Pxs_df, sectors_s)
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sqlalchemy import create_engine, text

# ==============================================================================
# CONFIG
# ==============================================================================
QUALITY_TABLE      = 'valuation_metrics_anchors'
CONSERVATIVE_YEARS = [2021, 2022]
MAX_COMPONENTS     = 10
EXCLUDE_METRICS    = ['ROE', 'ROE-P', 'ROEd']
HORIZONS           = [21, 63]
TOP_PCTILE         = 0.10
MIN_STOCKS         = 20
WINSOR             = (0.01, 0.99)

# Gridsearch hyperparameter space
MAV_WINDOWS  = [63, 126, 252]
THRESHOLDS   = [25, 50, 75]   # bps

QUALITY_METRICS = [
    # Growth — standalone
    'HSG', 'GS', 'GE', 'GGP', 'SGD', 'LastSGD', 'PIG', 'PSG',
    # Profitability level
    'OM', 'ROI', 'FCF_PG',
    # Profitability trend
    'OMd', 'ROId', 'ISGD',
    # R&D intensity
    'r&d',
    # Vol-adjusted growth (derived: base / max(Vol, 1))
    'GS/S_Vol', 'HSG/S_Vol', 'PSG/S_Vol',
    'GE/E_Vol', 'PIG/E_Vol',
    'GGP/GP_Vol',
    # R²-weighted growth/trend (derived: base * r2)
    'GS*r2_S', 'SGD*r2_S', 'OMd*r2_S',
    'GE*r2_E', 'PIG*r2_E',
    'GGP*r2_GP',
]

# Raw columns needed from DB to compute derived metrics
RAW_DB_COLS = [
    'HSG', 'GS', 'GE', 'GGP', 'SGD', 'LastSGD', 'PIG', 'PSG',
    'OM', 'ROI', 'FCF_PG', 'OMd', 'ROId', 'ISGD', 'r&d',
    'S Vol', 'E Vol', 'GP Vol',
    'r2 S', 'r2 E', 'r2 GP',
]

VOL_MIN = 1.0   # floor for Vol divisor to avoid division by near-zero

DB_URL = 'postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db'
ENGINE = create_engine(DB_URL)


# ==============================================================================
# HELPERS
# ==============================================================================

def normalize_ticker(t: str) -> str:
    return str(t).split(' ')[0].strip().upper()


def normalize_index(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(d) for d in idx])


def normalize_df(df: pd.DataFrame, col: str = 'ticker') -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].apply(normalize_ticker)
    return df


def normalize_series(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = [normalize_ticker(t) for t in s.index]
    return s


def winsorize(s: pd.Series) -> pd.Series:
    lo, hi = WINSOR
    return s.clip(lower=s.quantile(lo), upper=s.quantile(hi))


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


def load_all_snapshots(anchor_dates: list) -> dict:
    """
    Load quality metric snapshots for all anchor dates in one pass.
    Returns dict {date: DataFrame indexed by bare ticker}.
    """
    if not anchor_dates:
        return {}

    # Always include Size for residualization
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


# ==============================================================================
# DERIVED METRICS
# ==============================================================================

def build_derived_metrics(snap: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived Vol-adjusted and R²-weighted metrics to a snapshot DataFrame.
    Raw columns (S Vol, E Vol, GP Vol, r2 S, r2 E, r2 GP) remain in the
    DataFrame but are never included in QUALITY_METRICS so won't be ranked.

    Vol divisor is floored at VOL_MIN to avoid division by near-zero.
    R² is already 0-1 so multiplication is the correct operation.
    """
    s = snap.copy()

    def col(name):
        return s[name] if name in s.columns else pd.Series(np.nan, index=s.index)

    def safe_div(num, denom_name):
        d = col(denom_name).clip(lower=VOL_MIN)
        return col(num) / d

    def safe_mul(base_name, r2_name):
        return col(base_name) * col(r2_name)

    # Vol-adjusted growth
    s['GS/S_Vol']   = safe_div('GS',  'S Vol')
    s['HSG/S_Vol']  = safe_div('HSG', 'S Vol')
    s['PSG/S_Vol']  = safe_div('PSG', 'S Vol')
    s['GE/E_Vol']   = safe_div('GE',  'E Vol')
    s['PIG/E_Vol']  = safe_div('PIG', 'E Vol')
    s['GGP/GP_Vol'] = safe_div('GGP', 'GP Vol')

    # R²-weighted growth/trend
    s['GS*r2_S']   = safe_mul('GS',  'r2 S')
    s['SGD*r2_S']  = safe_mul('SGD', 'r2 S')
    s['OMd*r2_S']  = safe_mul('OMd', 'r2 S')
    s['GE*r2_E']   = safe_mul('GE',  'r2 E')
    s['PIG*r2_E']  = safe_mul('PIG', 'r2 E')
    s['GGP*r2_GP'] = safe_mul('GGP', 'r2 GP')

    return s


# ==============================================================================
# RATE REGIME
# ==============================================================================

def compute_rate_signal(Pxs_df: pd.DataFrame,
                         mav_window: int,
                         threshold: float) -> pd.Series:
    """
    Extract USGG10YR from Pxs_df and compute quantized rate momentum.

    q = 0.0 : rates below MAV by > threshold (falling → growth regime)
    q = 0.5 : rates within ±threshold of MAV (neutral)
    q = 1.0 : rates above MAV by > threshold (rising → conservative regime)

    Returns Series indexed by date.
    """
    if 'USGG10YR' not in Pxs_df.columns:
        raise ValueError("USGG10YR not found in Pxs_df columns")

    rate      = Pxs_df['USGG10YR'].dropna() * 100   # convert to bps if in %
    rate_mav  = rate.rolling(mav_window, min_periods=mav_window // 2).mean()
    rate_mom  = rate - rate_mav   # positive = rates above MAV

    q = pd.Series(0.5, index=rate_mom.index)
    q[rate_mom >  threshold] = 1.0
    q[rate_mom < -threshold] = 0.0

    return q


# ==============================================================================
# WITHIN-SECTOR RANKING
# ==============================================================================

def rank_within_sector(snap: pd.DataFrame,
                        sectors_s: pd.Series,
                        metrics: list) -> pd.DataFrame:
    """Rank each metric 0-1 within sector. Missing values stay NaN."""
    ranked  = pd.DataFrame(index=snap.index)
    sec     = normalize_series(sectors_s).reindex(snap.index)

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

def derive_weights(snapshots: dict,
                   sectors_s: pd.Series,
                   Pxs_df: pd.DataFrame,
                   regime: str) -> dict:
    """
    Derive metric weights for GQF or CQF from in-sample spread_z analysis.

    For each metric, compute avg spread_z and avg t-stat across:
      - regime='growth'       : all years except CONSERVATIVE_YEARS
      - regime='conservative' : CONSERVATIVE_YEARS only

    Eligibility: avg_spread_z > 0 AND above median spread_z
                 AND avg_t above median t-stat
    Cap at MAX_COMPONENTS, normalize weights to sum to 1.

    Returns dict {metric: weight}.
    """
    eligible_metrics = [m for m in QUALITY_METRICS if m not in EXCLUDE_METRICS]
    anchor_dates     = sorted(snapshots.keys())

    # Filter dates by regime
    if regime == 'conservative':
        dates_in_regime = [d for d in anchor_dates if d.year in CONSERVATIVE_YEARS]
    else:
        dates_in_regime = [d for d in anchor_dates if d.year not in CONSERVATIVE_YEARS]

    if not dates_in_regime:
        print(f"    WARNING: no anchor dates found for regime='{regime}'")
        return {}

    print(f"    {len(dates_in_regime)} dates in regime '{regime}'", end='', flush=True)

    # Compute spread_z per metric per date per horizon
    metric_stats = {m: {h: [] for h in HORIZONS} for m in eligible_metrics}

    for anchor in dates_in_regime:
        snap = snapshots.get(anchor)
        if snap is None or snap.empty:
            continue

        snap   = build_derived_metrics(snap)
        size_s = snap['Size'].dropna() if 'Size' in snap.columns else pd.Series(dtype=float)
        size_s = size_s[size_s > 0]

        if size_s.empty:
            continue   # can't residualize without size

        ranked  = rank_within_sector(snap, sectors_s, eligible_metrics)
        u_stats = {}
        for m in eligible_metrics:
            if m in ranked.columns:
                s = ranked[m].dropna()
                u_stats[m] = (float(s.mean()), float(s.std())) if len(s) > 1 else (np.nan, np.nan)

        for horizon in HORIZONS:
            resid = _compute_residuals(Pxs_df, sectors_s, size_s, anchor, horizon)
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

                spread_z = (top_med - bottom_med) / u_std
                metric_stats[m][horizon].append(spread_z)

    # Aggregate: avg spread_z and t-stat per metric (averaged across horizons)
    rows = []
    for m in eligible_metrics:
        sz_all = []
        t_all  = []
        for h in HORIZONS:
            vals = metric_stats[m][h]
            if len(vals) < 2:
                continue
            arr    = np.array(vals)
            sz_all.append(float(arr.mean()))
            t_all.append(float(scipy_stats.ttest_1samp(arr, 0).statistic))

        if not sz_all or not t_all:
            continue

        rows.append({
            'metric':     m,
            'avg_sz':     float(np.mean(sz_all)),
            'avg_t':      float(np.mean(t_all)),
        })

    n_resid_ok = sum(
        1 for m in eligible_metrics
        for h in HORIZONS
        if metric_stats[m][h]
    )
    print(f"  — {n_resid_ok} metric/horizon pairs with data")

    if not rows:
        return {}

    df      = pd.DataFrame(rows).set_index('metric')
    med_sz  = df['avg_sz'].median()
    med_t   = df['avg_t'].median()

    eligible = df[
        (df['avg_sz'] > 0) &
        (df['avg_sz'] > med_sz) &
        (df['avg_t']  > med_t)
    ].copy()

    if eligible.empty:
        return {}

    eligible  = eligible.nlargest(MAX_COMPONENTS, 'avg_t')
    eligible['weight'] = eligible['avg_t'].clip(lower=0)
    total     = eligible['weight'].sum()
    if total <= 0:
        return {}

    eligible['weight'] /= total
    return eligible['weight'].to_dict()


# ==============================================================================
# RESIDUALIZATION: mkt beta + sector dummies + log(size)
# ==============================================================================

def _compute_residuals(Pxs_df: pd.DataFrame,
                        sectors_s: pd.Series,
                        size_s: pd.Series,
                        anchor_date: pd.Timestamp,
                        horizon: int) -> pd.Series:
    """OLS residuals: mkt beta + sector dummies + log(size)."""
    # Exclude rate column from price universe
    eq_cols  = [c for c in Pxs_df.columns if c != 'USGG10YR']
    px_eq    = Pxs_df[eq_cols]

    future_dates = px_eq.index[px_eq.index > anchor_date]
    if len(future_dates) < horizon:
        return pd.Series(dtype=float)

    end_date   = future_dates[horizon - 1]
    past_dates = px_eq.index[px_eq.index <= anchor_date]
    if past_dates.empty:
        return pd.Series(dtype=float)
    start_date = past_dates[-1]

    px_start = px_eq.loc[start_date].dropna()
    px_end   = px_eq.loc[end_date].reindex(px_start.index).dropna()
    common   = px_start.index.intersection(px_end.index)
    if common.empty:
        return pd.Series(dtype=float)

    fwd_ret       = (px_end[common] / px_start[common] - 1)
    fwd_ret.index = [normalize_ticker(t) for t in fwd_ret.index]
    mkt_ret       = fwd_ret.mean()

    sec   = normalize_series(sectors_s).reindex(fwd_ret.index)
    sz    = normalize_series(size_s).reindex(fwd_ret.index)
    valid = fwd_ret.index[sec.notna() & sz.notna() & (sz > 0)]

    if len(valid) < MIN_STOCKS:
        return pd.Series(dtype=float)

    sec_val = sec[valid]
    sectors = sorted(sec_val.unique())
    n       = len(valid)

    # Trailing beta
    trail_start_idx = max(0, len(past_dates) - 63)
    trail_dates     = past_dates[trail_start_idx:]

    if len(trail_dates) < 20:
        betas = pd.Series(1.0, index=valid)
    else:
        cols     = [t + ' US' if t + ' US' in px_eq.columns else t for t in valid]
        px_trail = px_eq.loc[trail_dates, cols].copy()
        px_trail.columns = [normalize_ticker(c) for c in px_trail.columns]
        r_trail  = px_trail.pct_change().dropna()
        mkt_r    = r_trail.mean(axis=1)
        betas    = {}
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

    y_adj    = fwd_ret[valid] - betas.reindex(valid).fillna(1.0) * mkt_ret
    log_size = np.log(sz[valid].values)
    log_size = (log_size - log_size.mean()) / (log_size.std() + 1e-8)

    sec_dummies = np.column_stack([
        (sec_val == s).astype(float).values for s in sectors[:-1]
    ]) if len(sectors) > 1 else np.zeros((n, 0))

    X = np.hstack([np.ones((n, 1)), sec_dummies, log_size.reshape(-1, 1)])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y_adj.values, rcond=None)
        residuals        = y_adj.values - X @ coeffs
    except Exception:
        residuals = y_adj.values - y_adj.values.mean()

    return pd.Series(residuals, index=valid)


# ==============================================================================
# COMPOSITE SCORE COMPUTATION
# ==============================================================================

def compute_composite_scores(snap: pd.DataFrame,
                               sectors_s: pd.Series,
                               gqf_weights: dict,
                               cqf_weights: dict,
                               q: float) -> pd.Series:
    """
    Compute rate-conditioned composite quality score per stock.

    Steps:
    1. Rank all eligible metrics within sector (0-1)
    2. Weighted average → GQF score and CQF score
    3. composite = (1-q) * GQF + q * CQF

    Returns Series indexed by bare ticker.
    """
    all_metrics = list(set(list(gqf_weights.keys()) + list(cqf_weights.keys())))
    ranked      = rank_within_sector(snap, sectors_s, all_metrics)

    def weighted_score(weights: dict) -> pd.Series:
        if not weights:
            return pd.Series(np.nan, index=ranked.index)
        score = pd.Series(0.0, index=ranked.index)
        total = 0.0
        for m, w in weights.items():
            if m not in ranked.columns:
                continue
            col = ranked[m]
            score = score.add(col * w, fill_value=0)
            total += w
        return score / total if total > 0 else pd.Series(np.nan, index=ranked.index)

    gqf_scores = weighted_score(gqf_weights)
    cqf_scores = weighted_score(cqf_weights)

    composite = (1 - q) * gqf_scores + q * cqf_scores
    return composite.dropna()


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_composite(scores_by_date: dict,
                        Pxs_df: pd.DataFrame,
                        sectors_s: pd.Series,
                        snapshots: dict) -> tuple:
    """
    Evaluate composite quality scores using the same spread_z framework
    as quality_factor_diag.

    Returns:
        summary : DataFrame — mean_spread_z, t_stat, consistency, n_dates per horizon
        annual  : dict {year: {horizon: {mean_spread_z, consistency, n}}}
    """
    anchor_dates = sorted(scores_by_date.keys())
    raw          = {}   # (date, horizon) -> spread_z

    for anchor in anchor_dates:
        scores = scores_by_date[anchor]
        if scores.empty:
            continue

        snap   = snapshots.get(anchor)
        if snap is None or snap.empty:
            continue

        # No need to build derived metrics here — scores already computed
        size_s = snap['Size'].dropna() if 'Size' in snap.columns else pd.Series(dtype=float)
        size_s = size_s[size_s > 0]

        # Universe stats for spread_z (from ranked composite scores)
        u_mean = float(scores.mean())
        u_std  = float(scores.std())
        if u_std <= 0:
            continue

        for horizon in HORIZONS:
            resid = _compute_residuals(Pxs_df, sectors_s, size_s, anchor, horizon)
            if resid.empty or len(resid) < MIN_STOCKS:
                continue

            # Align scores and residuals
            common = scores.index.intersection(resid.index)
            if len(common) < MIN_STOCKS:
                continue

            s_aligned = scores[common]
            r_aligned = resid[common]

            n_decile       = max(1, int(np.floor(len(common) * TOP_PCTILE)))
            sorted_ret     = r_aligned.sort_values(ascending=False)
            top_tickers    = sorted_ret.iloc[:n_decile].index
            bottom_tickers = sorted_ret.iloc[-n_decile:].index

            top_med    = float(s_aligned.reindex(top_tickers).dropna().median())
            bottom_med = float(s_aligned.reindex(bottom_tickers).dropna().median())

            if np.isnan(top_med) or np.isnan(bottom_med):
                continue

            spread_z           = (top_med - bottom_med) / u_std
            raw[(anchor, horizon)] = spread_z

    if not raw:
        return pd.DataFrame(), {}

    # Summary
    summary_rows = []
    for h in HORIZONS:
        vals = [raw[(d, h)] for d in anchor_dates
                if (d, h) in raw and not np.isnan(raw[(d, h)])]
        if len(vals) < 2:
            continue
        arr = np.array(vals)
        summary_rows.append({
            'horizon':      h,
            'mean_spread_z': round(float(arr.mean()), 3),
            't_stat':        round(float(scipy_stats.ttest_1samp(arr, 0).statistic), 3),
            'consistency':   round(float((arr > 0).mean() * 100), 1),
            'n_dates':       len(arr),
        })

    summary = pd.DataFrame(summary_rows).set_index('horizon')

    # Annual breakdown
    annual = {}
    for yr in sorted(set(d.year for d, h in raw)):
        annual[yr] = {}
        for h in HORIZONS:
            vals = [raw[(d, h)] for d in anchor_dates
                    if d.year == yr and (d, h) in raw and not np.isnan(raw[(d, h)])]
            if not vals:
                continue
            arr = np.array(vals)
            annual[yr][h] = {
                'mean_spread_z': round(float(arr.mean()), 3),
                'consistency':   round(float((arr > 0).mean() * 100), 1),
                'n_dates':       len(arr),
            }

    return summary, annual


# ==============================================================================
# MAIN RUN
# ==============================================================================

def run(Pxs_df: pd.DataFrame,
        sectors_s: pd.Series,
        mav_window: int = 126,
        threshold: float = 50,
        verbose: bool = True) -> tuple:
    """
    Build and evaluate rate-conditioned quality factor.

    Args:
        Pxs_df     : Price DataFrame — must include 'USGG10YR' column.
        sectors_s  : Series mapping bare ticker -> sector.
        mav_window : Rolling window for rate MAV in trading days.
        threshold  : Rate momentum threshold in bps.
        verbose    : Print progress and results.

    Returns:
        summary         : DataFrame of evaluation stats per horizon.
        annual          : Dict of annual breakdown stats.
        scores_by_date  : Dict {date: Series of composite scores}.
        gqf_weights     : Dict {metric: weight} for GQF.
        cqf_weights     : Dict {metric: weight} for CQF.
    """
    # Normalize inputs
    Pxs_df       = Pxs_df.copy()
    Pxs_df.index = normalize_index(Pxs_df.index)
    sectors_s    = normalize_series(sectors_s)

    if verbose:
        print("=" * 70)
        print(f"  QUALITY FACTOR  |  window={mav_window}d  |  threshold={threshold}bps")
        print("=" * 70)

    # Load data
    anchor_dates = load_anchor_dates()
    if verbose:
        print(f"  Loading {len(anchor_dates)} anchor date snapshots...", end='', flush=True)
    snapshots = load_all_snapshots(anchor_dates)
    if verbose:
        print(f" done ({len(snapshots)} loaded)")

    # Rate signal
    rate_signal = compute_rate_signal(Pxs_df, mav_window, threshold)

    # Derive factor weights
    if verbose:
        print(f"\n  Deriving GQF weights (ex {CONSERVATIVE_YEARS})...")
    gqf_weights = derive_weights(snapshots, sectors_s, Pxs_df, regime='growth')

    if verbose:
        print(f"  Deriving CQF weights ({CONSERVATIVE_YEARS})...")
    cqf_weights = derive_weights(snapshots, sectors_s, Pxs_df, regime='conservative')

    if verbose:
        _print_weights(gqf_weights, cqf_weights)

    # Compute composite scores per anchor date
    scores_by_date = {}
    for anchor in anchor_dates:
        snap = snapshots.get(anchor)
        if snap is None or snap.empty:
            continue

        snap = build_derived_metrics(snap)

        # Get rate regime on anchor date (snap to nearest available)
        rate_dates = rate_signal.index[rate_signal.index <= anchor]
        q = float(rate_signal.loc[rate_dates[-1]]) if not rate_dates.empty else 0.5

        scores = compute_composite_scores(snap, sectors_s, gqf_weights, cqf_weights, q)
        if not scores.empty:
            scores_by_date[anchor] = scores

    if verbose:
        print(f"\n  Computed scores for {len(scores_by_date)} dates")
        print(f"  Evaluating composite...\n")

    # Evaluate
    summary, annual = evaluate_composite(scores_by_date, Pxs_df, sectors_s, snapshots)

    if verbose:
        _print_results(summary, annual, mav_window, threshold)

    return summary, annual, scores_by_date, gqf_weights, cqf_weights


# ==============================================================================
# GRIDSEARCH
# ==============================================================================

def gridsearch(Pxs_df: pd.DataFrame,
               sectors_s: pd.Series) -> pd.DataFrame:
    """
    Run gridsearch over all (mav_window, threshold) combinations.

    Returns DataFrame with mean_spread_z and t_stat per horizon per combination.
    """
    print("=" * 70)
    print("  QUALITY FACTOR GRIDSEARCH")
    print(f"  Windows    : {MAV_WINDOWS}")
    print(f"  Thresholds : {THRESHOLDS} bps")
    print(f"  Total runs : {len(MAV_WINDOWS) * len(THRESHOLDS)}")
    print("=" * 70 + "\n")

    rows = []
    for window in MAV_WINDOWS:
        for thresh in THRESHOLDS:
            print(f"\n--- window={window}  threshold={thresh}bps ---")
            summary, annual, _, _, _ = run(Pxs_df, sectors_s,
                                            mav_window=window,
                                            threshold=thresh,
                                            verbose=False)
            if summary.empty:
                continue
            for h in HORIZONS:
                if h not in summary.index:
                    continue
                r = summary.loc[h]
                rows.append({
                    'mav_window':   window,
                    'threshold':    thresh,
                    'horizon':      h,
                    'mean_spread_z': r['mean_spread_z'],
                    't_stat':        r['t_stat'],
                    'consistency':   r['consistency'],
                    'n_dates':       r['n_dates'],
                })
            # Print compact summary
            for h in HORIZONS:
                if h in summary.index:
                    r = summary.loc[h]
                    print(f"  h={h:>2}d  spread_z={r['mean_spread_z']:+.3f}  "
                          f"t={r['t_stat']:+.3f}  cons={r['consistency']:.0f}%")

    grid_df = pd.DataFrame(rows)
    if grid_df.empty:
        return grid_df

    print("\n" + "=" * 70)
    print("  GRIDSEARCH SUMMARY — sorted by t_stat (63d)")
    print("=" * 70)
    for h in HORIZONS:
        sub = (grid_df[grid_df['horizon'] == h]
               .sort_values('t_stat', ascending=False)
               .reset_index(drop=True))
        print(f"\n  Horizon: {h}d")
        print(f"  {'window':>8} {'threshold':>10} {'spread_z':>10} {'t_stat':>8} "
              f"{'cons%':>7} {'n':>5}")
        print("  " + "-" * 54)
        for _, row in sub.iterrows():
            print(f"  {int(row['mav_window']):>8} {int(row['threshold']):>10} "
                  f"{row['mean_spread_z']:>+10.3f} {row['t_stat']:>+8.3f} "
                  f"{row['consistency']:>7.1f} {int(row['n_dates']):>5}")

    return grid_df


# ==============================================================================
# PRINTING
# ==============================================================================

def _print_weights(gqf_weights: dict, cqf_weights: dict):
    print("\n  GQF components (growth regime):")
    for m, w in sorted(gqf_weights.items(), key=lambda x: -x[1]):
        print(f"    {m:<12}  {w:.4f}")
    print("\n  CQF components (conservative regime):")
    for m, w in sorted(cqf_weights.items(), key=lambda x: -x[1]):
        print(f"    {m:<12}  {w:.4f}")


def _print_results(summary: pd.DataFrame,
                    annual: dict,
                    mav_window: int,
                    threshold: float):
    print("\n" + "=" * 70)
    print(f"  COMPOSITE EVALUATION  |  window={mav_window}d  |  threshold={threshold}bps")
    print("=" * 70)
    print(summary.to_string())

    print("\n  Annual breakdown:")
    years = sorted(annual.keys())
    for h in HORIZONS:
        print(f"\n  Horizon {h}d:")
        print(f"  {'Year':<6} {'spread_z':>10} {'consistency%':>14} {'n':>4}")
        print("  " + "-" * 38)
        for yr in years:
            if h not in annual[yr]:
                continue
            r = annual[yr][h]
            print(f"  {yr:<6} {r['mean_spread_z']:>+10.3f} "
                  f"{r['consistency']:>14.1f} {r['n_dates']:>4}")
    print("=" * 70)


if __name__ == "__main__":
    print("Usage: from quality_factor import run, gridsearch")
    print("       summary, annual, scores, gqf_w, cqf_w = run(Pxs_df, sectors_s)")
    print("       grid = gridsearch(Pxs_df, sectors_s)")
