#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# CREATES THE VALUE FACTOR

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# CREATES THE VALUE FACTOR

"""
ic_study.py
===========
Information Coefficient (IC) study for valuation metrics.

Measures within-sector Spearman IC between each valuation metric rank
and forward idiosyncratic returns from factor_residuals_mom
(net of market, size, macro, sectors, quality, vol, SI, idio_mom).

Usage:
    from ic_study import run_ic_study
    ic_ts, ic_summary, ic_annual, weights = run_ic_study(Pxs_df, sectors_s)
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy import stats
from sqlalchemy import create_engine, text

# ==============================================================================
# CONFIG
# ==============================================================================
VALUATION_TABLE = 'valuation_consolidated'
RESIDUAL_SOURCE = 'factor_residuals_mom'
VALUE_SCORES_TBL = 'value_scores_df'
HORIZONS        = [21, 63]
MIN_STOCKS      = 10
WINSOR          = (0.01, 0.99)
METRICS         = ['P/S', 'P/Ee', 'P/Eo', 'sP/S', 'sP/E', 'sP/GP', 'P/GP']

# ==============================================================================
# PRE-DERIVED WEIGHTS  (update by running run_ic_study(use_cached_weights=False))
# Set to {} to force re-derivation from IC analysis
# ==============================================================================

VALUE_WEIGHTS = {
  'P/S': 0.1606,
  'P/Ee': 0.1512,
  'P/Eo': 0.1397,
  'sP/S': 0.1556,
  'sP/E': 0.1349,
  'sP/GP': 0.1507,
  'P/GP': 0.1073,
}
VALUE_WEIGHTS = {
  'P/S': 0.1587,
  'P/Ee': 0.1480,
  'P/Eo': 0.1455,
  'sP/S': 0.1522,
  'sP/E': 0.1312,
  'sP/GP': 0.1489,
  'P/GP': 0.1154,
}
# Set True once weights above are populated
_VALUE_WEIGHTS_READY = True

VALUE_WEIGHTS_PIT_TBL = 'value_weights_pit'
MAX_HORIZON_VALUE     = max(HORIZONS)   # 63 days

# ==============================================================================
# POINT-IN-TIME WEIGHTS CACHE
# ==============================================================================

def _ensure_value_pit_table():
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {VALUE_WEIGHTS_PIT_TBL} (
                cutoff_date  DATE        NOT NULL,
                metric       VARCHAR(20) NOT NULL,
                weight       FLOAT       NOT NULL,
                PRIMARY KEY  (cutoff_date, metric)
            )
        """))


def _load_value_pit_cache():
    """Returns dict {cutoff_date: {metric: weight}}"""
    try:
        _ensure_value_pit_table()
        with ENGINE.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT cutoff_date, metric, weight
                FROM {VALUE_WEIGHTS_PIT_TBL} ORDER BY cutoff_date
            """)).fetchall()
        result = {}
        for cutoff, metric, weight in rows:
            dt = pd.Timestamp(cutoff)
            if dt not in result:
                result[dt] = {}
            result[dt][metric] = float(weight)
        return result
    except Exception as e:
        print(f"  WARNING: could not load value PIT cache: {e}")
        return {}


def _save_value_pit_weights(cutoff_date, weights):
    rows = [{'cutoff_date': cutoff_date.date(), 'metric': m, 'weight': float(w)}
            for m, w in weights.items()]
    if not rows:
        return
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {VALUE_WEIGHTS_PIT_TBL} (cutoff_date, metric, weight)
            VALUES (:cutoff_date, :metric, :weight)
            ON CONFLICT (cutoff_date, metric) DO NOTHING
        """), rows)


def run_pit_weights_value(Pxs_df, sectors_s, mode='incremental'):
    """
    Compute and cache point-in-time value weights for each anchor date.
    At cutoff T, only uses anchors where anchor + MAX_HORIZON_VALUE trading days < T.

    mode='incremental': only new cutoff dates
    mode='rebuild'    : wipe and recompute all
    """
    _ensure_value_pit_table()

    if mode == 'rebuild':
        with ENGINE.begin() as conn:
            conn.execute(text(f"DELETE FROM {VALUE_WEIGHTS_PIT_TBL}"))
        print("  Value PIT weights cache cleared (rebuild mode)")

    cached       = _load_value_pit_cache()
    cached_dates = set(cached.keys())
    anchor_dates = load_valuation_dates()
    all_px_dates = Pxs_df.index

    # Build (trigger_anchor, cutoff) pairs
    cutoff_candidates = []
    for anchor in anchor_dates:
        future = all_px_dates[all_px_dates > anchor]
        if len(future) < MAX_HORIZON_VALUE:
            continue
        cutoff = future[MAX_HORIZON_VALUE - 1]
        cutoff_candidates.append((anchor, cutoff))

    new_dates = [(a, c) for a, c in cutoff_candidates if c not in cached_dates]
    print(f"  Value PIT weights: {len(cached_dates)} cached, {len(new_dates)} to compute")

    if not new_dates:
        print("  All value PIT weight dates already cached.")
        return

    for i, (trigger_anchor, cutoff) in enumerate(new_dates, 1):
        print(f"  [{i}/{len(new_dates)}] cutoff={cutoff.date()} "
              f"(anchor {trigger_anchor.date()})", end='', flush=True)

        # Only use anchors with complete forward windows before cutoff
        eligible = []
        for anchor in anchor_dates:
            fut = all_px_dates[all_px_dates > anchor]
            if len(fut) < MAX_HORIZON_VALUE:
                continue
            if fut[MAX_HORIZON_VALUE - 1] < cutoff:
                eligible.append(anchor)

        if not eligible:
            print(" — no eligible anchors, using equal weights")
            equal = {m: 1.0 / len(METRICS) for m in METRICS}
            _save_value_pit_weights(cutoff, equal)
            continue

        # Compute IC for each eligible anchor and aggregate
        ic_records = []
        for anchor in eligible:
            snap = load_valuation_snapshot(anchor)
            if snap.empty:
                continue
            row = {}
            for horizon in HORIZONS:
                resid = compute_residual_returns(Pxs_df, sectors_s, anchor, horizon)
                if resid.empty:
                    continue
                for m in METRICS:
                    ic = compute_ic_for_date(snap, resid, sectors_s, m)
                    if not np.isnan(ic):
                        row[(m, horizon)] = ic
            if row:
                ic_records.append(row)

        if not ic_records:
            print(" — no IC computed, using equal weights")
            equal = {m: 1.0 / len(METRICS) for m in METRICS}
            _save_value_pit_weights(cutoff, equal)
            continue

        # Aggregate IC across anchors and horizons — IC t-stat weighting
        metric_ics = {m: [] for m in METRICS}
        for rec in ic_records:
            for m in METRICS:
                vals = [rec[(m, h)] for h in HORIZONS if (m, h) in rec]
                if vals:
                    metric_ics[m].extend(vals)

        weights = {}
        for m in METRICS:
            ics = metric_ics[m]
            if len(ics) < 2:
                continue
            arr = np.array(ics)
            t_stat = float(scipy_stats.ttest_1samp(arr, 0).statistic)
            if t_stat > 0:
                weights[m] = t_stat

        if not weights:
            weights = {m: 1.0 / len(METRICS) for m in METRICS}
        else:
            total = sum(weights.values())
            weights = {m: w / total for m, w in weights.items()}

        print(f" — {len(weights)} metrics weighted")
        _save_value_pit_weights(cutoff, weights)

    print("  Value PIT weights computation complete.")

    # Recompute value scores using new PIT weights
    print("\n  Recomputing value scores with PIT weights...")
    _compute_and_save_value_scores(
        weights_norm    = VALUE_WEIGHTS,   # fallback, overridden per-date by PIT
        sectors_s       = sectors_s,
        force_recompute = True,
        use_pit_weights = True,
    )


def get_value_pit_weights_at(cutoff_date):
    """Return value weights at most recent cutoff_date <= given date. Equal weights if none."""
    cached = _load_value_pit_cache()
    if not cached:
        return {m: 1.0 / len(METRICS) for m in METRICS}
    prior = [d for d in sorted(cached.keys()) if d <= cutoff_date]
    if not prior:
        return {m: 1.0 / len(METRICS) for m in METRICS}
    return cached[prior[-1]]


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
    if RESIDUAL_SOURCE.startswith('factor_residuals'):
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
            SELECT date, ticker, resid AS residual
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

    sector_ics     = []
    sector_weights = []

    for sec, grp in df.groupby('sector'):
        if len(grp) < MIN_STOCKS:
            continue

        m = grp['metric'].copy()

        # Reflexivity treatment for negative valuations:
        # Within sector, negatives get adjusted = sector_max_positive + abs(value)
        # This ranks them as most expensive (worst value) without excluding them
        pos_mask = m > 0
        if pos_mask.any():
            sector_max = m[pos_mask].max()
            m[~pos_mask] = sector_max + m[~pos_mask].abs()
        else:
            # All negative in this sector — shift all to positive preserving order
            m = m.abs().max() - m  # most negative → highest rank (most expensive)

        # Winsorize metric within sector (after adjustment, all values positive)
        m_wins = winsorize_series(m)

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


# ==============================================================================
# VALUE SCORES CACHE — DB-backed persistence
# ==============================================================================

def _ensure_value_scores_table():
    """Create value_scores_df table if it does not exist."""
    with ENGINE.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {VALUE_SCORES_TBL} (
                date   DATE        NOT NULL,
                ticker VARCHAR(20) NOT NULL,
                score  NUMERIC,
                PRIMARY KEY (date, ticker)
            )
        """))


def _compute_and_save_value_scores(weights_norm: dict,
                                    sectors_s: pd.Series,
                                    force_recompute: bool = False,
                                    use_pit_weights: bool = False):
    """
    Compute IC-weighted value composite scores for all valuation dates
    and save to value_scores_df. Only computes missing dates unless
    force_recompute=True.

    use_pit_weights=True: use point-in-time weights from value_weights_pit table,
                          looked up per valuation date. No backfill — equal weights
                          if no prior PIT weights exist.
    """
    _ensure_value_scores_table()

    # Load PIT weights if needed
    if use_pit_weights:
        _pit_cache = _load_value_pit_cache()
        _pit_dates = sorted(_pit_cache.keys())
        print(f"  Value PIT weights: {len(_pit_cache)} cutoff dates available")

    # Get all available valuation dates
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT DISTINCT date FROM {VALUATION_TABLE} ORDER BY date
        """)).fetchall()
    all_val_dates = [pd.Timestamp(r[0]) for r in rows]

    if not all_val_dates:
        print("  WARNING: No valuation dates found.")
        return

    if force_recompute:
        dates_to_compute = all_val_dates
        print(f"  Force recompute: computing all {len(dates_to_compute)} valuation dates")
    else:
        with ENGINE.connect() as conn:
            cached_rows = conn.execute(text(
                f"SELECT DISTINCT date FROM {VALUE_SCORES_TBL}"
            )).fetchall()
        cached = {pd.Timestamp(r[0]) for r in cached_rows}
        dates_to_compute = [d for d in all_val_dates if d not in cached]
        print(f"  Value scores cache: {len(cached)} dates cached, "
              f"{len(dates_to_compute)} new dates to compute")

    if not dates_to_compute:
        print("  All valuation dates already cached — nothing to compute")
        return

    # Load raw data for missing dates
    with ENGINE.connect() as conn:
        df_all = pd.read_sql(text(f"""
            SELECT date, ticker, {', '.join(f'"{m}"' for m in METRICS)}
            FROM {VALUATION_TABLE}
            WHERE date = ANY(:dates) AND ticker IS NOT NULL
            ORDER BY date
        """), conn, params={"dates": [d.date() for d in dates_to_compute]})

    if df_all.empty:
        print("  WARNING: No valuation data found for missing dates.")
        return

    df_all['date']   = pd.to_datetime(df_all['date'])
    df_all['ticker'] = df_all['ticker'].apply(normalize_ticker)

    rows_to_save = []
    n_dates_done = 0

    for val_date, grp in df_all.groupby('date'):
        snap = grp.drop(columns='date').drop_duplicates('ticker').set_index('ticker')
        for m in METRICS:
            if m in snap.columns:
                snap[m] = pd.to_numeric(snap[m], errors='coerce')

        snap['_sector'] = snap.index.map(normalize_series_index(sectors_s))
        snap = snap[snap['_sector'].notna()]
        if snap.empty:
            continue

        # Select weights for this date
        if use_pit_weights:
            prior_pit = [d for d in _pit_dates if d <= val_date]
            if prior_pit:
                w_at_date = _pit_cache[prior_pit[-1]]
            else:
                # No PIT weights yet — equal weights, no backfill
                w_at_date = {m: 1.0 / len(METRICS) for m in METRICS}
        else:
            w_at_date = weights_norm

        metric_scores = {}
        for m in METRICS:
            if m not in snap.columns:
                continue
            scores = pd.Series(np.nan, index=snap.index)
            for sec, sec_grp in snap.groupby('_sector'):
                m_vals = sec_grp[m].copy()
                valid  = m_vals.dropna()
                if len(valid) < 3:
                    continue
                pos_mask = valid > 0
                if pos_mask.any():
                    sec_max  = valid[pos_mask].max()
                    adjusted = valid.copy()
                    adjusted[~pos_mask] = sec_max + valid[~pos_mask].abs()
                else:
                    adjusted = valid.abs().max() - valid
                asc_rank = adjusted.rank(method='average', ascending=True)
                val_rank = 1.0 - (asc_rank - 1) / (len(asc_rank) - 1) \
                           if len(asc_rank) > 1 \
                           else pd.Series(0.5, index=asc_rank.index)
                scores.loc[val_rank.index] = val_rank
            metric_scores[m] = scores

        if not metric_scores:
            continue

        composite = pd.Series(0.0, index=snap.index)
        total_w   = 0.0
        for m, w in w_at_date.items():
            if m in metric_scores:
                s = metric_scores[m].reindex(snap.index)
                valid_mask = s.notna()
                composite[valid_mask] += w * s[valid_mask]
                total_w += w
        if total_w > 0:
            composite /= total_w

        for ticker, score in composite.items():
            if pd.notna(score):
                rows_to_save.append({
                    'date': pd.Timestamp(val_date).date(),
                    'ticker': ticker,
                    'score': float(score)
                })
        n_dates_done += 1

    if not rows_to_save:
        print("  WARNING: No scores computed.")
        return

    df_save = pd.DataFrame(rows_to_save)
    dates_saved = list({r['date'] for r in rows_to_save})
    with ENGINE.begin() as conn:
        conn.execute(text(
            f"DELETE FROM {VALUE_SCORES_TBL} WHERE date = ANY(:dates)"
        ), {"dates": dates_saved})
    df_save.to_sql(VALUE_SCORES_TBL, ENGINE, if_exists='append', index=False)
    print(f"  Saved {len(df_save):,} value score rows "
          f"({n_dates_done} dates) to '{VALUE_SCORES_TBL}'")


def run_ic_study(Pxs_df: pd.DataFrame,
                 sectors_s: pd.Series,
                 force_recompute_cache: bool = False,
                 use_cached_weights: bool = True,
                 use_pit_weights: bool = False) -> tuple:
    """
    Run IC study for value factor with incremental caching.

    Daily fast path (use_cached_weights=True):
      - Uses hardcoded VALUE_WEIGHTS — skips IC computation entirely
      - Only computes value scores for dates not already in DB cache

    PIT mode (use_pit_weights=True):
      - Uses point-in-time weights from value_weights_pit table
      - No forward bias — weights at each date use only prior IC history
      - Requires run_pit_weights_value() to have been run first

    Full rebuild (use_cached_weights=False):
      - Runs full IC study over all anchor dates (slow)
    """
    Pxs_df            = Pxs_df.copy()
    Pxs_df.index      = normalize_index(Pxs_df.index)
    sectors_s         = normalize_series_index(sectors_s)

    # ── Fast daily path ────────────────────────────────────────────────────────
    if use_cached_weights and _VALUE_WEIGHTS_READY and not force_recompute_cache:
        weights_norm = VALUE_WEIGHTS
        print(f"  Value factor: using {'PIT' if use_pit_weights else 'hardcoded'} weights")
        print(f"  Populating value scores cache (incremental)...")
        _compute_and_save_value_scores(
            weights_norm    = weights_norm,
            sectors_s       = sectors_s,
            force_recompute = False,
            use_pit_weights = use_pit_weights,
        )
        return pd.DataFrame(), pd.DataFrame(), {}, weights_norm

    # ── Full IC study (run occasionally) ──────────────────────────────────────
    anchor_dates = load_valuation_dates()
    print(f"  {len(anchor_dates)} anchor dates in {VALUATION_TABLE}")
    print(f"  Metrics         : {METRICS}")
    print(f"  Horizons        : {HORIZONS} trading days")
    print(f"  Residual source : {RESIDUAL_SOURCE}")
    print(f"  Valuation table : {VALUATION_TABLE}\n")

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
                row[(m, horizon)] = compute_ic_for_date(snap, resid, sectors_s, m)
        sample_ics = [row[(m, HORIZONS[0])] for m in METRICS
                      if (m, HORIZONS[0]) in row and not np.isnan(row[(m, HORIZONS[0])])]
        if sample_ics:
            print(f"  IC({HORIZONS[0]}d) range [{min(sample_ics):.3f}, {max(sample_ics):.3f}]")
        else:
            print("  — no IC computed")
        records.append(row)

    if not records:
        print("No results.")
        return pd.DataFrame(), pd.DataFrame(), {}, {}

    ic_ts       = pd.DataFrame(records).set_index('date')
    ic_ts.index = normalize_index(ic_ts.index)
    ic_ts.columns = pd.MultiIndex.from_tuples(
        [(m, h) for m, h in ic_ts.columns], names=['metric', 'horizon'])

    summary_rows = []
    for m in METRICS:
        for h in HORIZONS:
            if (m, h) not in ic_ts.columns:
                continue
            s = ic_ts[(m, h)].dropna()
            n = len(s)
            if n < 2:
                continue
            mean_ic = s.mean()
            t_stat  = float(stats.ttest_1samp(s, 0).statistic)
            pct_pos = (s > 0).mean() * 100
            ic_ir   = mean_ic / s.std() if s.std() > 0 else np.nan
            summary_rows.append({
                'metric': m, 'horizon': h,
                'mean_IC': round(mean_ic, 4), 't_stat': round(t_stat, 3),
                'pct_positive': round(pct_pos, 1),
                'IC_IR': round(ic_ir, 3) if not np.isnan(ic_ir) else np.nan,
                'n_dates': n,
            })

    ic_summary = pd.DataFrame(summary_rows).set_index(['metric', 'horizon'])
    ic_annual  = _compute_annual_breakdown(ic_ts)

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

    # Derive weights
    weights_raw = {}
    for m in METRICS:
        t_vals = []
        for h in HORIZONS:
            if (m, h) in ic_summary.index:
                t_vals.append(abs(ic_summary.loc[(m, h), 't_stat']))
        if t_vals:
            weights_raw[m] = float(np.mean(t_vals))
    total        = sum(weights_raw.values())
    weights_norm = {m: round(w / total, 4) for m, w in weights_raw.items()}

    print("\n" + "=" * 65)
    print("  DERIVED VALUE FACTOR WEIGHTS (IC t-stat, avg 21d+63d)")
    print("=" * 65)
    print(f"  {'Metric':<12} {'Avg |t|':>10} {'Weight':>10}")
    print(f"  {'-'*35}")
    for m in METRICS:
        print(f"  {m:<12} {weights_raw.get(m,0):>10.3f} {weights_norm.get(m,0):>10.4f}")

    # Print copy-paste block for hardcoding
    print(f"\n  ── Copy into value_factor.py to enable fast daily path ──")
    print(f"  VALUE_WEIGHTS = {{")
    for m in METRICS:
        print(f"      '{m}': {weights_norm.get(m, 0):.4f},")
    print(f"  }}")
    print(f"  _VALUE_WEIGHTS_READY = True")

    print(f"\n  Copy into factor_model_step1.py _VALUE_TSTAT dict:")
    print(f"  _VALUE_TSTAT = {{")
    for m in METRICS:
        h0 = ic_summary.loc[(m, HORIZONS[0]), 't_stat'] if (m, HORIZONS[0]) in ic_summary.index else 0
        h1 = ic_summary.loc[(m, HORIZONS[1]), 't_stat'] if (m, HORIZONS[1]) in ic_summary.index else 0
        print(f"      '{m}': ({h0:.3f} + {h1:.3f}) / 2,   # {(h0+h1)/2:.3f}")
    print(f"  }}")

    print("\n" + "=" * 65)
    print("  POPULATING VALUE SCORES CACHE")
    print("=" * 65)
    _compute_and_save_value_scores(
        weights_norm    = weights_norm,
        sectors_s       = sectors_s,
        force_recompute = force_recompute_cache,
    )

    return ic_ts, ic_summary, ic_annual, weights_norm

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
    return ic_ts, ic_summary, ic_annual, weights_norm


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
    print("       ic_ts, ic_summary, ic_annual, weights = run_ic_study(Pxs_df, sectors_s)")
    print(f"\nCurrent VALUATION_TABLE = '{VALUATION_TABLE}'")
    print(f"Metrics  : {METRICS}")
    print(f"Horizons : {HORIZONS}")
    # daily fast path:
    ic_ts, ic_summary, ic_annual, weights = run_ic_study(Pxs_df, sectors_s)
    # occasional full rebuild:
    # ic_ts, ic_summary, ic_annual, weights = run_ic_study(Pxs_df, sectors_s, use_cached_weights=False)

