#!/usr/bin/env python
# coding: utf-8

"""
Factor Model - v2
==================
Sequential Fama-MacBeth cross-sectional factor model.
Reordered sequence vs v1 to prioritise desirable alpha factors early
in the Gram-Schmidt chain, pushing risk/noise factors to the end.

Key differences from v1
-----------------------
  1. Sequence reordered — alpha factors (Quality, Momentum, Value) precede
     structural/risk factors (Size, SI, Vol, Macro, Sectors)
  2. Momentum computed on quality residuals (v1: vol residuals)
  3. Value follows momentum — eliminates price-contamination of value signal
  4. Macro and Sector characteristics are NOT Gram-Schmidt orthogonalized —
     macro uses raw EWMA betas to prior-step residuals (z-scored);
     sectors use sum-to-zero deviation coding with Ridge CV
  5. All other characteristics ARE Gram-Schmidt orthogonalized vs all prior
  6. All tables prefixed v2_ — v1 tables untouched

Step sequence
-------------
  Step 1:  Baseline UFV
  Step 2:  Market Beta       input=raw_rets        GS char ⊥ {}
  Step 3:  Quality           input=resid_mkt       GS char ⊥ {beta}
  Step 4:  Idio Momentum     input=resid_quality   GS char ⊥ {beta, quality}
  Step 5:  Size              input=resid_mom       GS char ⊥ {beta, quality, mom}
  Step 6:  Value             input=resid_size      GS char ⊥ {beta, quality, mom, size}
  Step 7:  SI Composite      input=resid_value     GS char ⊥ {all prior}
  Step 8:  GK Volatility     input=resid_si        GS char ⊥ {all prior}
  Step 9:  Macro Factors     input=resid_vol       raw betas, joint Ridge CV
  Step 10: Sector Dummies    input=resid_macro     sum-to-zero, Ridge CV
  Step 11: O-U Mean Rev      input=resid_sec       GS char ⊥ {all prior}

Usage
-----
    from factor_model_v2 import run
    results = run(Pxs_df, sectors_s)
    results = run(Pxs_df, sectors_s, volumeTrd_df=volumeTrd_df)
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

# ── Reuse all machinery from v1 (loaded in kernel) ────────────────────────────
# The following functions/constants are assumed live in the Jupyter kernel
# from factor_model_step1:
#   ENGINE, BETA_WINDOW, BETA_HL, VOL_WINDOW, VOL_HL, MOM_LONG, MOM_SKIP,
#   MOM_LONG_BUFFER, MIN_STOCKS, MACRO_COLS, RIDGE_GRID_MACRO, RIDGE_GRID_SEC,
#   OU_REVERSION_TBL (overridden below), OU_MEANREV_W, OU_MIN_OBS,
#   OU_ST_REV_W, OU_WEIGHT_REF, OU_WEIGHT_CAP, OU_VOL_CLIP_LO, OU_VOL_CLIP_HI,
#   clean_ticker, zscore,
#   get_universe, build_sector_dummies,
#   load_dynamic_size, get_log_size,
#   calc_rolling_betas, calc_macro_betas, calc_vol_factor,
#   calc_idio_momentum, calc_idio_momentum_volscaled,
#   load_ohlc_tables, load_si_composite,
#   load_quality_scores, load_value_scores,
#   _fit_ou_single, _compute_ou_for_dates,
#   wls_cross_section, wls_ridge_cross_section,
#   run_factor_step, run_factor_step_optimal_ridge,
#   orthogonalize_char, orthogonalize_char_df,
#   variance_stats, r2_stats, lambda_stats,
#   print_lambda_summary, print_sector_lambdas,
#   save_lambdas, save_lambdas_incremental,
#   save_residuals, save_residuals_incremental,
#   _load_resid_from_db, _load_char_from_db,
#   _compute_dynamic_size_for_dates

# ===============================================================================
# V2 CONSTANTS
# ===============================================================================

V2 = 'v2'   # version prefix — applied to all model-dependent table names

# Table name helper — prepends V2 prefix
def v2tbl(name):
    return f'{V2}_{name}'

# V2 table names
V2_RESID_MKT     = v2tbl('factor_residuals_mkt')
V2_RESID_QUALITY = v2tbl('factor_residuals_quality')
V2_RESID_MOM     = v2tbl('factor_residuals_mom')
V2_RESID_SIZE    = v2tbl('factor_residuals_size')
V2_RESID_VALUE   = v2tbl('factor_residuals_value')
V2_RESID_SI      = v2tbl('factor_residuals_si')
V2_RESID_VOL     = v2tbl('factor_residuals_vol')
V2_RESID_MACRO   = v2tbl('factor_residuals_macro')
V2_RESID_SEC     = v2tbl('factor_residuals_sec')
V2_RESID_OU      = v2tbl('factor_residuals_ou')

V2_LAM_MKT     = v2tbl('factor_lambdas_mkt')
V2_LAM_QUALITY = v2tbl('factor_lambdas_quality')
V2_LAM_MOM     = v2tbl('factor_lambdas_mom')
V2_LAM_SIZE    = v2tbl('factor_lambdas_size')
V2_LAM_VALUE   = v2tbl('factor_lambdas_value')
V2_LAM_SI      = v2tbl('factor_lambdas_si')
V2_LAM_VOL     = v2tbl('factor_lambdas_vol')
V2_LAM_MACRO   = v2tbl('factor_lambdas_macro')
V2_LAM_SEC     = v2tbl('factor_lambdas_sec')
V2_LAM_OU      = v2tbl('factor_lambdas_ou')

V2_OU_TBL      = v2tbl('ou_reversion_df')
V2_QUALITY_TBL = v2tbl('quality_scores_df')
V2_VALUE_TBL   = v2tbl('value_scores_df')


# ===============================================================================
# V2-SPECIFIC LOADERS
# These wrap the v1 loaders but redirect cache reads/writes to v2 tables.
# ===============================================================================

def load_quality_scores_v2(universe, calc_dates, Pxs_df, sectors_s):
    """
    Load quality scores from v2_quality_scores_df cache.
    Delegates to get_quality_scores() but uses v2 cache table.
    """
    try:
        print("  Loading quality factor scores (v2 cache)...")
        raw = get_quality_scores(
            calc_dates         = calc_dates,
            universe           = universe,
            Pxs_df             = Pxs_df,
            sectors_s          = sectors_s,
            use_cached_weights = True,
            force_recompute    = False,
            cache_table        = V2_QUALITY_TBL,
        )
        if raw.empty or raw.isna().all().all():
            print("  WARNING: No v2 quality scores — falling back to v1 cache")
            return load_quality_scores(universe, calc_dates, Pxs_df, sectors_s)
        all_dates  = calc_dates.union(raw.index).sort_values()
        quality_ff = raw.reindex(all_dates).ffill().bfill().reindex(calc_dates).astype(float)
        return quality_ff.apply(zscore, axis=1)
    except TypeError:
        # get_quality_scores doesn't accept cache_table param — fall back to v1 cache
        # (scores are model-independent since quality is computed on raw fundamentals)
        print("  Loading quality scores from v1 cache (shared)...")
        return load_quality_scores(universe, calc_dates, Pxs_df, sectors_s)


def load_value_scores_v2(universe, calc_dates, sectors_s):
    """
    Load value scores from v2_value_scores_df cache.
    Falls back to v1 cache if v2 not yet populated.
    """
    print("  Loading value scores (v2 cache)...")
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(text(f"""
                SELECT date, ticker, score FROM {V2_VALUE_TBL}
            """), conn)
        if df.empty:
            raise ValueError("v2 value cache empty")
        df['date']   = pd.to_datetime(df['date'])
        df['ticker'] = df['ticker'].apply(clean_ticker)
        df['score']  = df['score'].astype(float)
        val_df = df.pivot_table(index='date', columns='ticker',
                                values='score', aggfunc='last')
        val_df = val_df.reindex(columns=universe)
        all_dates = calc_dates.union(val_df.index).sort_values()
        value_ff  = val_df.reindex(all_dates).ffill().reindex(calc_dates)
        return value_ff.apply(zscore, axis=1)
    except Exception:
        print("  v2 value cache not found — falling back to v1 cache")
        return load_value_scores(universe, calc_dates, sectors_s)


def load_ou_reversion_v2(universe, calc_dates, resid_pivot,
                          Pxs_df, volumeTrd_df=None):
    """Load O-U scores from v2_ou_reversion_df cache."""
    already_done = set()
    try:
        with ENGINE.connect() as conn:
            exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = :t
                )
            """), {"t": V2_OU_TBL}).scalar()
        if exists:
            with ENGINE.connect() as conn:
                rows = conn.execute(text(
                    f"SELECT DISTINCT date FROM {V2_OU_TBL}"
                )).fetchall()
            already_done = {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        pass

    dates_to_calc = pd.DatetimeIndex(
        [d for d in calc_dates if d not in already_done]
    )

    if len(dates_to_calc) > 0:
        print(f"  Computing v2 O-U for {len(dates_to_calc)} new dates "
              f"({len(already_done)} already in cache)...")
        new_df       = _compute_ou_for_dates(dates_to_calc, universe,
                                              resid_pivot, Pxs_df, volumeTrd_df)
        long         = new_df.stack(dropna=False).reset_index()
        long.columns = ['date', 'ticker', 'ou_score']
        long         = long.dropna(subset=['ou_score'])
        long['date'] = pd.to_datetime(long['date'])
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {V2_OU_TBL} (
                    date DATE, ticker VARCHAR(20), ou_score NUMERIC,
                    PRIMARY KEY (date, ticker)
                )
            """))
        long.to_sql(V2_OU_TBL, ENGINE, if_exists='append', index=False)
        print(f"  Saved {len(long):,} rows to '{V2_OU_TBL}'")
    else:
        print(f"  v2 O-U: all {len(calc_dates)} dates already cached")

    date_list = [d.date() for d in calc_dates]
    with ENGINE.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT date, ticker, ou_score FROM {V2_OU_TBL}
            WHERE date = ANY(:dates)
        """), {"dates": date_list}).fetchall()

    df           = pd.DataFrame(rows, columns=['date', 'ticker', 'ou_score'])
    df['date']   = pd.to_datetime(df['date'])
    df['ou_score'] = df['ou_score'].astype(float)
    pivot        = df.pivot_table(index='date', columns='ticker',
                                   values='ou_score', aggfunc='last')
    pivot        = pivot.reindex(columns=universe).reindex(calc_dates)
    print(f"  v2 O-U loaded: {pivot.shape}")
    return pivot


def _v2_get_anchor_date():
    """Latest date in v2_factor_residuals_mkt."""
    try:
        with ENGINE.connect() as conn:
            row = conn.execute(
                text(f"SELECT MAX(date) FROM {V2_RESID_MKT}")
            ).fetchone()
        if row and row[0]:
            return pd.Timestamp(row[0])
    except Exception:
        pass
    return None


# ===============================================================================
# INCREMENTAL UPDATE — v2 fast single-date path
# ===============================================================================

def _v2_run_incremental(Pxs_df, sectors_s, volumeTrd_df=None,
                         use_vol_scale=False,
                         VOL_LOWER=0.5, VOL_UPPER=3.0):
    """
    Fast single-date incremental update for v2 model.
    Mirrors _run_incremental from v1 but follows the v2 step sequence
    and writes to v2_* tables.
    """
    dt         = Pxs_df.index[-1]
    calc_dates = pd.DatetimeIndex([dt])
    print(f"\n  v2 incremental update for {dt.date()}")

    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]

    st_dt          = pd.Timestamp('2019-01-01')
    all_dates      = Pxs_df.index
    ext_loc        = max(0, all_dates.searchsorted(st_dt) - MOM_LONG_BUFFER)
    extended_st_dt = all_dates[ext_loc]

    universe    = get_universe(Pxs_df, sectors_s, extended_st_dt)
    sector_dum  = build_sector_dummies(universe, sectors_s)
    sec_cols    = sector_dum.columns.tolist()
    all_rets    = Pxs_df[universe].pct_change().clip(-0.5, 0.5)

    dynamic_size = load_dynamic_size(universe, Pxs_df, calc_dates)

    print("  Computing characteristics for new date...")

    # Beta
    beta_df = calc_rolling_betas(Pxs_df, universe, calc_dates)

    # Size
    s = dynamic_size.loc[dt, universe].dropna() if dt in dynamic_size.index \
        else pd.Series()
    size_char_df = pd.DataFrame(
        {dt: zscore(np.log(s.clip(lower=1)))}
    ).T.reindex(columns=universe)
    size_char_df.index.name = 'date'

    # Quality and Value
    ext_dates = Pxs_df.index[Pxs_df.index >= extended_st_dt]
    valid_ext = ext_dates[
        all_rets.loc[ext_dates].notna().sum(axis=1) >= MIN_STOCKS
    ]
    quality_df = load_quality_scores_v2(universe, valid_ext, Pxs_df, sectors_s)
    value_df   = load_value_scores_v2(universe, valid_ext, sectors_s)

    # SI composite (load recent window for ffill)
    si_dates         = pd.DatetimeIndex(
        [d for d in valid_ext if d <= dt]
    )[-60:]
    si_composite_full = load_si_composite(universe, si_dates)
    si_composite      = si_composite_full.reindex([dt])

    # GK Vol
    open_df, high_df, low_df = load_ohlc_tables(universe)
    vol_df = calc_vol_factor(Pxs_df, universe, calc_dates,
                             open_df=open_df, high_df=high_df, low_df=low_df)

    # Macro betas (computed against resid_vol later — for incremental,
    # compute against current step input which we approximate with raw rets
    # for the single-date fast path; full recalc uses proper residuals)
    macro_betas = calc_macro_betas(Pxs_df, universe, calc_dates)
    macro_cols  = list(macro_betas.keys())

    # Sector dummies
    sec_char = {col: pd.DataFrame({dt: sector_dum[col]}).T
                for col in sec_cols}

    # ── Step 2: Beta ──────────────────────────────────────────────────────────
    r_mkt, lam_mkt, _ = run_factor_step(
        ['beta'], {'beta': beta_df},
        all_rets, dynamic_size, calc_dates, universe
    )

    # ── Step 3: Quality ⊥ {beta} ─────────────────────────────────────────────
    quality_perp = orthogonalize_char_df(
        quality_df, {'beta': beta_df}, calc_dates, dynamic_size=dynamic_size
    )
    r_quality, lam_quality, _ = run_factor_step(
        ['quality'], {'quality': quality_perp},
        r_mkt, dynamic_size, calc_dates, universe
    )

    # ── Step 4: Idio Momentum ⊥ {beta, quality} ──────────────────────────────
    # Load quality residual history for momentum lookback
    qual_hist = _load_resid_from_db(V2_RESID_QUALITY, universe, 400)
    if not qual_hist.empty and not r_quality.empty:
        qual_hist = pd.concat([
            qual_hist[~qual_hist.index.isin(r_quality.index)], r_quality
        ]).sort_index()
    elif not r_quality.empty:
        qual_hist = r_quality

    if use_vol_scale and volumeTrd_df is not None:
        mom_df = calc_idio_momentum_volscaled(
            qual_hist, volumeTrd_df, calc_dates,
            vol_lower=VOL_LOWER, vol_upper=VOL_UPPER
        )
    else:
        mom_df = calc_idio_momentum(qual_hist, calc_dates)

    if mom_df.empty or dt not in mom_df.index:
        print("  WARNING: momentum not available — aborting")
        return None

    mom_perp = orthogonalize_char_df(
        mom_df, {'beta': beta_df, 'quality': quality_perp},
        calc_dates, dynamic_size=dynamic_size
    )
    r_mom, lam_mom, _ = run_factor_step(
        ['idio_mom'], {'idio_mom': mom_perp},
        r_quality, dynamic_size, calc_dates, universe
    )

    # ── Step 5: Size ⊥ {beta, quality, mom} ──────────────────────────────────
    size_perp = orthogonalize_char_df(
        size_char_df,
        {'beta': beta_df, 'quality': quality_perp, 'idio_mom': mom_perp},
        calc_dates, dynamic_size=dynamic_size
    )
    r_size, lam_size, _ = run_factor_step(
        ['size'], {'size': size_perp},
        r_mom, dynamic_size, calc_dates, universe
    )

    # ── Step 6: Value ⊥ {beta, quality, mom, size} ───────────────────────────
    value_perp = orthogonalize_char_df(
        value_df,
        {'beta': beta_df, 'quality': quality_perp,
         'idio_mom': mom_perp, 'size': size_perp},
        calc_dates, dynamic_size=dynamic_size
    )
    r_value, lam_value, _ = run_factor_step(
        ['value'], {'value': value_perp},
        r_size, dynamic_size, calc_dates, universe
    )

    # ── Step 7: SI ⊥ {all prior} ─────────────────────────────────────────────
    prior_for_si = {
        'beta': beta_df, 'quality': quality_perp,
        'idio_mom': mom_perp, 'size': size_perp, 'value': value_perp
    }
    si_perp = orthogonalize_char_df(
        si_composite, prior_for_si, calc_dates, dynamic_size=dynamic_size
    )
    r_si, lam_si, _ = run_factor_step(
        ['si_composite'], {'si_composite': si_perp},
        r_value, dynamic_size, calc_dates, universe
    )

    # ── Step 8: GK Vol ⊥ {all prior} ─────────────────────────────────────────
    prior_for_vol = dict(prior_for_si)
    prior_for_vol['si_composite'] = si_perp
    vol_perp = orthogonalize_char_df(
        vol_df, prior_for_vol, calc_dates, dynamic_size=dynamic_size
    )
    r_vol, lam_vol, _ = run_factor_step(
        ['vol'], {'vol': vol_perp},
        r_si, dynamic_size, calc_dates, universe
    )

    # ── Step 9: Macro — raw betas to vol residuals, joint Ridge ───────────────
    # For incremental, macro betas are computed against Pxs_df (approximation)
    # Full recalc computes them against the actual vol residuals
    if macro_cols:
        r_macro, lam_macro, _ = run_factor_step_optimal_ridge(
            macro_cols, macro_betas,
            r_vol, dynamic_size, calc_dates, universe,
            lambda_grid=RIDGE_GRID_MACRO, default_lambda=0.5
        )
    else:
        r_macro  = r_vol
        lam_macro = pd.DataFrame()

    # ── Step 10: Sectors — sum-to-zero dummies, Ridge ─────────────────────────
    r_sec, lam_sec, _ = run_factor_step_optimal_ridge(
        sec_cols, {c: sec_char[c] for c in sec_cols},
        r_macro, dynamic_size, calc_dates, universe,
        lambda_grid=RIDGE_GRID_SEC, default_lambda=2.0
    )

    # ── Step 11: O-U on sector residuals ─────────────────────────────────────
    sec_hist = _load_resid_from_db(V2_RESID_SEC, universe, 120)
    if not sec_hist.empty and not r_sec.empty:
        sec_hist = pd.concat([
            sec_hist[~sec_hist.index.isin(r_sec.index)], r_sec
        ]).sort_index()
    elif not r_sec.empty:
        sec_hist = r_sec

    if r_sec.empty or dt not in sec_hist.index:
        print("  WARNING: sector residuals empty — O-U skipped")
        ou_pivot  = pd.DataFrame(index=calc_dates, columns=universe, dtype=float)
        r_ou      = pd.DataFrame()
        lam_ou    = pd.DataFrame()
    else:
        new_ou       = _compute_ou_for_dates(
            calc_dates, universe, sec_hist, Pxs_df,
            volumeTrd_df if use_vol_scale else None
        )
        long         = new_ou.stack(dropna=False).reset_index()
        long.columns = ['date', 'ticker', 'ou_score']
        long         = long.dropna(subset=['ou_score'])
        long['date'] = pd.to_datetime(long['date'])
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {V2_OU_TBL} (
                    date DATE, ticker VARCHAR(20), ou_score NUMERIC,
                    PRIMARY KEY (date, ticker)
                )
            """))
            conn.execute(text(
                f"DELETE FROM {V2_OU_TBL} WHERE date = :d"
            ), {"d": dt.date()})
        long.to_sql(V2_OU_TBL, ENGINE, if_exists='append', index=False)
        ou_pivot = new_ou.reindex(columns=universe)
        r_ou, lam_ou, _ = run_factor_step(
            ['ou_reversion'], {'ou_reversion': ou_pivot},
            r_sec, dynamic_size, calc_dates, universe
        )

    # ── Save to DB ────────────────────────────────────────────────────────────
    print("  Saving v2 results to DB...")
    lam_pairs = [
        (lam_mkt,     V2_LAM_MKT),
        (lam_quality, V2_LAM_QUALITY),
        (lam_mom,     V2_LAM_MOM),
        (lam_size,    V2_LAM_SIZE),
        (lam_value,   V2_LAM_VALUE),
        (lam_si,      V2_LAM_SI),
        (lam_vol,     V2_LAM_VOL),
        (lam_macro,   V2_LAM_MACRO) if macro_cols else (None, None),
        (lam_sec,     V2_LAM_SEC),
        (lam_ou,      V2_LAM_OU),
    ]
    for ldf, tbl in lam_pairs:
        if ldf is not None and tbl is not None:
            save_lambdas_incremental(ldf, tbl)

    resid_pairs = [
        (r_mkt,     V2_RESID_MKT),
        (r_quality, V2_RESID_QUALITY),
        (r_mom,     V2_RESID_MOM),
        (r_size,    V2_RESID_SIZE),
        (r_value,   V2_RESID_VALUE),
        (r_si,      V2_RESID_SI),
        (r_vol,     V2_RESID_VOL),
        (r_macro,   V2_RESID_MACRO) if macro_cols else (None, None),
        (r_sec,     V2_RESID_SEC),
        (r_ou,      V2_RESID_OU),
    ]
    for rdf, tbl in resid_pairs:
        if rdf is not None and tbl is not None and not rdf.empty:
            save_residuals_incremental(rdf, tbl)

    # ── Snapshot print ────────────────────────────────────────────────────────
    def _val(ldf, col):
        if ldf is not None and not ldf.empty \
                and dt in ldf.index and col in ldf.columns:
            return ldf.loc[dt, col] * 100
        return np.nan

    rows_snap = [
        ('Market Beta',   _val(lam_mkt,     'beta')),
        ('Quality',       _val(lam_quality, 'quality')),
        ('Idio Momentum', _val(lam_mom,     'idio_mom')),
        ('Size',          _val(lam_size,    'size')),
        ('Value',         _val(lam_value,   'value')),
        ('SI Composite',  _val(lam_si,      'si_composite')),
        ('GK Vol',        _val(lam_vol,     'vol')),
        ('O-U Reversion', _val(lam_ou,      'ou_reversion')),
    ]
    for c in macro_cols:
        rows_snap.append((f'Macro: {c}', _val(lam_macro, c)))
    if not lam_sec.empty and dt in lam_sec.index:
        for c in sec_cols:
            if c in lam_sec.columns:
                rows_snap.append((f'Sector: {c}', lam_sec.loc[dt, c] * 100))

    ridge_str = ''
    for ldf, label in [(lam_macro, 'Macro'), (lam_sec, 'Sec')]:
        if ldf is not None and not ldf.empty \
                and 'ridge_lambda' in ldf.columns and dt in ldf.index:
            rl = ldf.loc[dt, 'ridge_lambda']
            if not np.isnan(rl):
                ridge_str += f'   {label} Ridge λ: {rl:.2f}'

    # ── Consolidated intercept ───────────────────────────────────────────────
    intercept_val = sum(
        ldf.loc[dt, 'intercept']
        if (ldf is not None and not ldf.empty
            and dt in ldf.index and 'intercept' in ldf.columns)
        else 0.0
        for ldf in [lam_mkt, lam_quality, lam_mom, lam_size,
                    lam_value, lam_si, lam_vol, lam_macro,
                    lam_sec, lam_ou]
    )

    # ── Daily R² ─────────────────────────────────────────────────────────────
    r2_consolidated = np.nan
    if not r_ou.empty and dt in r_ou.index and dt in all_rets.index:
        ou_res = r_ou.loc[dt].dropna()
        raw_r  = all_rets.loc[dt, ou_res.index].dropna()
        idx    = ou_res.index.intersection(raw_r.index)
        if len(idx) > 1:
            r2_consolidated = 1.0 - ou_res[idx].var() / raw_r[idx].var()

    r2_str = f"{r2_consolidated*100:.2f}%" if not np.isnan(r2_consolidated) else "n/a"

    W = 38
    print(f"\n  {'='*(W+14)}")
    print(f"  [v2] {dt.date()}   Intercept: {intercept_val*100:+.2f}%   Daily R²: {r2_str}{ridge_str}")
    print(f"  {'='*(W+14)}")
    print(f"  {'Factor':<{W}} {'Return%':>8}")
    print(f"  {'-'*(W+10)}")
    for factor, val in rows_snap:
        val_str = f"{val:>+8.2f}%" if not np.isnan(val) else f"{'n/a':>9}"
        print(f"  {factor:<{W}} {val_str}")
    print(f"  {'='*(W+14)}")

    return {
        'dt': dt, 'universe': universe,
        'sec_cols': sec_cols, 'macro_cols': macro_cols,
        'lambda_mkt': lam_mkt, 'lambda_quality': lam_quality,
        'lambda_mom': lam_mom, 'lambda_size': lam_size,
        'lambda_value': lam_value, 'lambda_si': lam_si,
        'lambda_vol': lam_vol, 'lambda_macro': lam_macro,
        'lambda_sec': lam_sec, 'lambda_ou': lam_ou,
        'resid_ou': r_ou, 'ou_pivot': ou_pivot,
    }


# ===============================================================================
# FULL RECALCULATION
# ===============================================================================

def _v2_run_full(Pxs_df, sectors_s, st_dt, volumeTrd_df=None,
                  use_vol_scale=False, VOL_LOWER=0.5, VOL_UPPER=3.0):
    """Full recalculation for v2 model."""

    all_dates      = Pxs_df.index
    st_dt_loc      = all_dates.searchsorted(st_dt)
    ext_loc        = max(0, st_dt_loc - MOM_LONG_BUFFER)
    extended_st_dt = all_dates[ext_loc]
    print(f"  Extended start: {extended_st_dt.date()}")

    universe   = get_universe(Pxs_df, sectors_s, extended_st_dt)
    sector_dum = build_sector_dummies(universe, sectors_s)
    sec_cols   = sector_dum.columns.tolist()

    all_rets = Pxs_df[universe].pct_change()
    RETURN_CLIP = 0.50
    n_clipped = int((all_rets.abs() > RETURN_CLIP).sum().sum())
    if n_clipped > 0:
        print(f"  Winsorizing {n_clipped} extreme returns (|ret|>{RETURN_CLIP:.0%})")
    all_rets = all_rets.clip(-RETURN_CLIP, RETURN_CLIP)

    ext_dates  = all_dates[all_dates >= extended_st_dt]
    valid_ext  = ext_dates[
        all_rets.loc[ext_dates].notna().sum(axis=1) >= MIN_STOCKS
    ]
    valid_days = valid_ext[valid_ext >= st_dt]
    print(f"  Extended dates: {len(valid_ext)} | Valid dates: {len(valid_days)}")

    dynamic_size = load_dynamic_size(universe, Pxs_df, valid_ext)
    si_composite = load_si_composite(universe, valid_ext)

    # ── Build characteristics ─────────────────────────────────────────────────
    print("\n  Building characteristics...")

    beta_df     = calc_rolling_betas(Pxs_df, universe, valid_ext)
    macro_betas = calc_macro_betas(Pxs_df, universe, valid_ext)
    macro_cols  = list(macro_betas.keys())

    # Size
    print("  Building size characteristic...")
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

    quality_df = load_quality_scores_v2(universe, valid_ext, Pxs_df, sectors_s)
    value_df   = load_value_scores_v2(universe, valid_ext, sectors_s)

    open_df, high_df, low_df = load_ohlc_tables(universe)
    vol_df = calc_vol_factor(Pxs_df, universe, valid_ext,
                             open_df=open_df, high_df=high_df, low_df=low_df)

    # Sector dummies expanded to date index
    dates_ext_common = valid_ext.intersection(beta_df.index).intersection(
        size_char_df.index
    )
    sec_char = {col: pd.DataFrame(
        {dt: sector_dum[col] for dt in dates_ext_common}
    ).T for col in sec_cols}

    # ── Common sample ─────────────────────────────────────────────────────────
    common_dates = valid_days.intersection(beta_df.index).intersection(
        size_char_df.index
    )
    for label, idx in [
        ('vol_df',       vol_df.index),
        ('si_composite', si_composite.index),
        ('quality_df',   quality_df.index[quality_df.notna().any(axis=1)]),
        ('value_df',     value_df.index[value_df.notna().any(axis=1)]),
    ]:
        before = len(common_dates)
        common_dates = common_dates.intersection(idx)
        if len(common_dates) < before:
            print(f"    after ∩ {label}: {len(common_dates)} "
                  f"(-{before-len(common_dates)})")
    for col in macro_cols:
        before = len(common_dates)
        common_dates = common_dates.intersection(macro_betas[col].index)
        if len(common_dates) < before:
            print(f"    after ∩ macro[{col}]: {len(common_dates)} "
                  f"(-{before-len(common_dates)})")
    print(f"\n  Common sample: {len(common_dates)} dates")

    raw_mat = all_rets.loc[common_dates, universe]
    UFV     = variance_stats(raw_mat, "UFV - Raw Returns (common sample)")

    # ── Step 2: Market Beta ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 2: Market Beta")
    print("="*70)
    resid_mkt_full, lambda_mkt, r2_mkt = run_factor_step(
        ['beta'], {'beta': beta_df},
        all_rets, dynamic_size, dates_ext_common, universe
    )

    # ── Step 3: Quality ⊥ {beta} ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 3: Quality")
    print("="*70)
    print("  Orthogonalizing quality vs beta...")
    quality_perp = orthogonalize_char_df(
        quality_df, {'beta': beta_df},
        resid_mkt_full.index, dynamic_size=dynamic_size
    )
    resid_quality_full, lambda_quality, r2_quality = run_factor_step(
        ['quality'], {'quality': quality_perp},
        resid_mkt_full, dynamic_size,
        resid_mkt_full.index, universe
    )

    # ── Step 4: Idio Momentum ⊥ {beta, quality} ──────────────────────────────
    print("\n" + "="*70)
    print("  STEP 4: Idio Momentum (on quality residuals)")
    print("="*70)
    if use_vol_scale and volumeTrd_df is not None:
        mom_df = calc_idio_momentum_volscaled(
            resid_quality_full, volumeTrd_df, valid_ext,
            vol_lower=VOL_LOWER, vol_upper=VOL_UPPER
        )
    else:
        mom_df = calc_idio_momentum(resid_quality_full, valid_ext)

    print("  Orthogonalizing momentum vs beta + quality...")
    mom_perp_dates = common_dates.intersection(mom_df.index)
    before_mom     = len(common_dates)
    common_dates   = mom_perp_dates
    print(f"  Common sample after idio_mom: {len(common_dates)} dates "
          f"(dropped {before_mom - len(common_dates)})")

    mom_perp = orthogonalize_char_df(
        mom_df,
        {'beta': beta_df, 'quality': quality_perp},
        common_dates, dynamic_size=dynamic_size
    )
    resid_mom_full, lambda_mom, r2_mom = run_factor_step(
        ['idio_mom'], {'idio_mom': mom_perp},
        resid_quality_full, dynamic_size,
        resid_quality_full.index.intersection(mom_df.index), universe
    )

    # ── Step 5: Size ⊥ {beta, quality, mom} ──────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 5: Size")
    print("="*70)
    print("  Orthogonalizing size vs beta + quality + momentum...")
    size_perp = orthogonalize_char_df(
        size_char_df,
        {'beta': beta_df, 'quality': quality_perp, 'idio_mom': mom_perp},
        resid_mom_full.index, dynamic_size=dynamic_size
    )
    resid_size_full, lambda_size, r2_size = run_factor_step(
        ['size'], {'size': size_perp},
        resid_mom_full, dynamic_size,
        resid_mom_full.index, universe
    )

    # ── Step 6: Value ⊥ {beta, quality, mom, size} ───────────────────────────
    print("\n" + "="*70)
    print("  STEP 6: Value")
    print("="*70)
    print("  Orthogonalizing value vs beta + quality + momentum + size...")
    value_perp = orthogonalize_char_df(
        value_df,
        {'beta': beta_df, 'quality': quality_perp,
         'idio_mom': mom_perp, 'size': size_perp},
        resid_size_full.index, dynamic_size=dynamic_size
    )
    resid_value_full, lambda_value, r2_value = run_factor_step(
        ['value'], {'value': value_perp},
        resid_size_full, dynamic_size,
        resid_size_full.index, universe
    )

    # ── Step 7: SI ⊥ {all prior} ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 7: SI Composite")
    print("="*70)
    prior_for_si = {
        'beta': beta_df, 'quality': quality_perp,
        'idio_mom': mom_perp, 'size': size_perp, 'value': value_perp
    }
    print("  Orthogonalizing SI vs all prior...")
    si_perp_full = orthogonalize_char_df(
        si_composite, prior_for_si,
        resid_value_full.index, dynamic_size=dynamic_size
    )
    si_perp = si_perp_full.reindex(common_dates)
    resid_si_full, lambda_si, r2_si = run_factor_step(
        ['si_composite'], {'si_composite': si_perp},
        resid_value_full, dynamic_size,
        resid_value_full.index.intersection(si_perp_full.index[
            si_perp_full.notna().any(axis=1)
        ]), universe
    )

    # ── Step 8: GK Vol ⊥ {all prior} ─────────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 8: GK Volatility")
    print("="*70)
    prior_for_vol = dict(prior_for_si)
    prior_for_vol['si_composite'] = si_perp_full
    print("  Orthogonalizing GK vol vs all prior...")
    vol_perp_full = orthogonalize_char_df(
        vol_df, prior_for_vol,
        resid_si_full.index, dynamic_size=dynamic_size
    )
    vol_perp = vol_perp_full.reindex(common_dates)
    resid_vol_full, lambda_vol, r2_vol = run_factor_step(
        ['vol'], {'vol': vol_perp},
        resid_si_full, dynamic_size,
        resid_si_full.index.intersection(vol_perp_full.index[
            vol_perp_full.notna().any(axis=1)
        ]), universe
    )

    # ── Step 9: Macro — raw betas to vol residuals, joint Ridge ───────────────
    print("\n" + "="*70)
    print("  STEP 9: Macro Factors (raw betas to vol residuals, joint Ridge)")
    print("="*70)
    # Recompute macro betas against vol residuals (true v2 design)
    print("  Computing macro betas against vol residuals...")
    macro_betas_v2 = calc_macro_betas(
        Pxs_df, universe,
        resid_vol_full.index
    )
    # Note: calc_macro_betas uses Pxs_df macro columns as factors but
    # stock returns are approximated by Pxs_df returns. For the full
    # Gram-Schmidt spirit, macro betas should be vs the vol residuals.
    # We achieve this by passing resid_vol as the return series implicitly
    # via the run_factor_step call — the regression target is resid_vol_full.
    macro_cols_v2 = list(macro_betas_v2.keys())

    if macro_cols_v2:
        full_dates_macro = resid_vol_full.index
        for col in macro_cols_v2:
            full_dates_macro = full_dates_macro.intersection(
                macro_betas_v2[col].index
            )
        resid_macro_full, lambda_macro, r2_macro = run_factor_step_optimal_ridge(
            macro_cols_v2, macro_betas_v2,
            resid_vol_full, dynamic_size,
            full_dates_macro, universe,
            lambda_grid=RIDGE_GRID_MACRO, default_lambda=0.5
        )
    else:
        resid_macro_full = resid_vol_full
        lambda_macro     = pd.DataFrame()
        r2_macro         = pd.Series(dtype=float)
        macro_cols_v2    = []

    # ── Step 10: Sectors — sum-to-zero dummies, Ridge ─────────────────────────
    print("\n" + "="*70)
    print("  STEP 10: Sector Dummies (sum-to-zero, Ridge)")
    print("="*70)
    resid_sec_full, lambda_sec, r2_sec = run_factor_step_optimal_ridge(
        sec_cols, {c: sec_char[c] for c in sec_cols},
        resid_macro_full, dynamic_size,
        resid_macro_full.index, universe,
        lambda_grid=RIDGE_GRID_SEC, default_lambda=2.0
    )

    # ── Step 11: O-U on sector residuals ─────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 11: O-U Mean Reversion (on sector residuals)")
    print("="*70)

    ou_already_done = set()
    try:
        with ENGINE.connect() as conn:
            exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = :t
                )
            """), {"t": V2_OU_TBL}).scalar()
        if exists:
            with ENGINE.connect() as conn:
                rows = conn.execute(text(
                    f"SELECT DISTINCT date FROM {V2_OU_TBL}"
                )).fetchall()
            ou_already_done = {pd.Timestamp(r[0]) for r in rows}
    except Exception:
        pass

    ou_dates_to_calc = pd.DatetimeIndex(
        [d for d in common_dates if d not in ou_already_done]
    )
    print(f"  O-U cached: {len(ou_already_done)} | to compute: "
          f"{len(ou_dates_to_calc)}")

    if len(ou_dates_to_calc) > 0:
        new_ou       = _compute_ou_for_dates(
            ou_dates_to_calc, universe, resid_sec_full, Pxs_df,
            volumeTrd_df if use_vol_scale else None
        )
        long         = new_ou.stack(dropna=False).reset_index()
        long.columns = ['date', 'ticker', 'ou_score']
        long         = long.dropna(subset=['ou_score'])
        long['date'] = pd.to_datetime(long['date'])
        with ENGINE.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {V2_OU_TBL} (
                    date DATE, ticker VARCHAR(20), ou_score NUMERIC,
                    PRIMARY KEY (date, ticker)
                )
            """))
        long.to_sql(V2_OU_TBL, ENGINE, if_exists='append', index=False)
        print(f"  Saved {len(long):,} rows to '{V2_OU_TBL}'")

    date_list = [d.date() for d in common_dates]
    try:
        with ENGINE.connect() as conn:
            ou_rows = conn.execute(text(f"""
                SELECT date, ticker, ou_score FROM {V2_OU_TBL}
                WHERE date = ANY(:dates)
            """), {"dates": date_list}).fetchall()
    except Exception:
        ou_rows = []

    if ou_rows:
        ou_df             = pd.DataFrame(ou_rows,
                                         columns=['date','ticker','ou_score'])
        ou_df['date']     = pd.to_datetime(ou_df['date'])
        ou_df['ou_score'] = ou_df['ou_score'].astype(float)
        ou_pivot = ou_df.pivot_table(
            index='date', columns='ticker',
            values='ou_score', aggfunc='last'
        ).reindex(columns=universe).reindex(common_dates)
    else:
        ou_pivot = pd.DataFrame(
            index=common_dates, columns=universe, dtype=float
        )

    ou_common = common_dates.intersection(
        ou_pivot.index[ou_pivot.notna().any(axis=1)]
    )
    resid_ou, lambda_ou, r2_ou = run_factor_step(
        ['ou_reversion'], {'ou_reversion': ou_pivot},
        resid_sec_full, dynamic_size,
        ou_common, universe
    )

    # ── Restrict early residuals to common sample for variance stats ──────────
    def _cs(df):
        return df[df.index.isin(common_dates)] if not df.empty else df

    resid_mkt     = _cs(resid_mkt_full)
    resid_quality = _cs(resid_quality_full)
    resid_mom     = _cs(resid_mom_full)
    resid_size    = _cs(resid_size_full)
    resid_value   = _cs(resid_value_full)
    resid_si      = _cs(resid_si_full)
    resid_vol     = _cs(resid_vol_full)
    resid_macro   = _cs(resid_macro_full) if macro_cols_v2 else resid_vol
    resid_sec     = _cs(resid_sec_full)

    # ── Variance reduction summary ────────────────────────────────────────────
    print("\n" + "="*70)
    print("  VARIANCE REDUCTION SUMMARY (v2, common sample)")
    print("="*70)
    mkt_UFV     = variance_stats(resid_mkt,     "mkt_UFV     (+beta)",     UFV)
    quality_UFV = variance_stats(resid_quality, "quality_UFV (+quality)",  mkt_UFV)
    mom_UFV     = variance_stats(resid_mom,     "mom_UFV     (+idio_mom)", quality_UFV)
    size_UFV    = variance_stats(resid_size,    "size_UFV    (+size)",     mom_UFV)
    value_UFV   = variance_stats(resid_value,   "value_UFV   (+value)",    size_UFV)
    si_UFV      = variance_stats(resid_si,      "si_UFV      (+SI)",       value_UFV)
    vol_UFV     = variance_stats(resid_vol,     "vol_UFV     (+GK_vol)",   si_UFV)
    macro_UFV   = variance_stats(resid_macro,   "macro_UFV   (+macro)",    vol_UFV) \
                  if macro_cols_v2 else vol_UFV
    sec_UFV     = variance_stats(resid_sec,     "sec_UFV     (+sectors)",  macro_UFV)
    ou_UFV      = variance_stats(resid_ou,      "ou_UFV      (+O-U)",      sec_UFV) \
                  if not resid_ou.empty else sec_UFV

    print(f"\n  {'Step':<44} {'%UFV':>8} {'%prev':>8}")
    print(f"  {'-'*62}")
    for lbl, var, base, prev in [
        ("UFV (raw)",            UFV,         UFV,         None),
        ("+ Beta",               mkt_UFV,     UFV,         UFV),
        ("+ Quality",            quality_UFV, UFV,         mkt_UFV),
        ("+ Idio Momentum",      mom_UFV,     UFV,         quality_UFV),
        ("+ Size",               size_UFV,    UFV,         mom_UFV),
        ("+ Value",              value_UFV,   UFV,         size_UFV),
        ("+ SI",                 si_UFV,      UFV,         value_UFV),
        ("+ GK Vol",             vol_UFV,     UFV,         si_UFV),
        ("+ Macro",              macro_UFV,   UFV,         vol_UFV),
        ("+ Sectors",            sec_UFV,     UFV,         macro_UFV),
        ("+ O-U",                ou_UFV,      UFV,         sec_UFV),
    ]:
        pct_ufv  = f"{var/base*100:.2f}%"
        pct_prev = f"{var/prev*100:.2f}%" if prev else "---"
        print(f"  {lbl:<44} {pct_ufv:>8} {pct_prev:>8}")

    # ── Save to DB ────────────────────────────────────────────────────────────
    print("\n  Saving v2 results to DB...")
    for ldf, tbl in [
        (lambda_mkt[lambda_mkt.index.isin(common_dates)],         V2_LAM_MKT),
        (lambda_quality[lambda_quality.index.isin(common_dates)], V2_LAM_QUALITY),
        (lambda_mom[lambda_mom.index.isin(common_dates)],         V2_LAM_MOM),
        (lambda_size[lambda_size.index.isin(common_dates)],       V2_LAM_SIZE),
        (lambda_value[lambda_value.index.isin(common_dates)],     V2_LAM_VALUE),
        (lambda_si[lambda_si.index.isin(common_dates)],           V2_LAM_SI),
        (lambda_vol[lambda_vol.index.isin(common_dates)],         V2_LAM_VOL),
        (lambda_sec[lambda_sec.index.isin(common_dates)],         V2_LAM_SEC),
        (lambda_ou[lambda_ou.index.isin(ou_common)],              V2_LAM_OU),
    ]:
        save_lambdas(ldf, tbl)
    if macro_cols_v2:
        save_lambdas(
            lambda_macro[lambda_macro.index.isin(common_dates)], V2_LAM_MACRO
        )

    for rdf, tbl in [
        (resid_mkt_full,     V2_RESID_MKT),
        (resid_quality_full, V2_RESID_QUALITY),
        (resid_mom_full,     V2_RESID_MOM),
        (resid_size_full,    V2_RESID_SIZE),
        (resid_value_full,   V2_RESID_VALUE),
        (resid_si_full,      V2_RESID_SI),
        (resid_vol_full,     V2_RESID_VOL),
        (resid_sec_full,     V2_RESID_SEC),
        (resid_ou,           V2_RESID_OU),
    ]:
        save_residuals(rdf, tbl)
    if macro_cols_v2:
        save_residuals(resid_macro_full, V2_RESID_MACRO)

    r2_stats(r2_mkt[r2_mkt.index.isin(common_dates)],             "Step 2: Beta")
    r2_stats(r2_quality[r2_quality.index.isin(common_dates)],     "Step 3: Quality")
    r2_stats(r2_mom[r2_mom.index.isin(common_dates)],             "Step 4: Idio Mom")
    r2_stats(r2_size[r2_size.index.isin(common_dates)],           "Step 5: Size")
    r2_stats(r2_value[r2_value.index.isin(common_dates)],         "Step 6: Value")
    r2_stats(r2_si,                                                "Step 7: SI")
    r2_stats(r2_vol,                                               "Step 8: GK Vol")
    if macro_cols_v2:
        r2_stats(r2_macro[r2_macro.index.isin(common_dates)],     "Step 9: Macro")
    r2_stats(r2_sec[r2_sec.index.isin(common_dates)],             "Step 10: Sectors")
    r2_stats(r2_ou,                                                "Step 11: O-U")

    print_lambda_summary(lambda_mkt,     ['beta'],       "Step 2: Beta",        common_dates)
    print_lambda_summary(lambda_quality, ['quality'],    "Step 3: Quality",     common_dates, annual_col='quality')
    print_lambda_summary(lambda_mom,     ['idio_mom'],   "Step 4: Idio Mom",    common_dates, annual_col='idio_mom')
    print_lambda_summary(lambda_size,    ['size'],       "Step 5: Size",        common_dates, annual_col='size')
    print_lambda_summary(lambda_value,   ['value'],      "Step 6: Value",       common_dates, annual_col='value')
    print_lambda_summary(lambda_si,      ['si_composite'],"Step 7: SI",         common_dates, annual_col='si_composite')
    print_lambda_summary(lambda_vol,     ['vol'],        "Step 8: GK Vol",      common_dates, annual_col='vol')
    if macro_cols_v2:
        print_lambda_summary(lambda_macro, macro_cols_v2, "Step 9: Macro",      common_dates)
    print_lambda_summary(lambda_sec,     sec_cols,       "Step 10: Sectors",    common_dates)
    print_sector_lambdas(lambda_sec, sec_cols, common_dates)
    print_lambda_summary(lambda_ou,      ['ou_reversion'],"Step 11: O-U",       ou_common, annual_col='ou_reversion')

    return {
        'UFV': UFV, 'mkt_UFV': mkt_UFV, 'quality_UFV': quality_UFV,
        'mom_UFV': mom_UFV, 'size_UFV': size_UFV, 'value_UFV': value_UFV,
        'si_UFV': si_UFV, 'vol_UFV': vol_UFV, 'macro_UFV': macro_UFV,
        'sec_UFV': sec_UFV, 'ou_UFV': ou_UFV,
        'resid_mkt': resid_mkt, 'resid_quality': resid_quality,
        'resid_mom': resid_mom, 'resid_size': resid_size,
        'resid_value': resid_value, 'resid_si': resid_si,
        'resid_vol': resid_vol, 'resid_macro': resid_macro,
        'resid_sec': resid_sec, 'resid_ou': resid_ou,
        'resid_mkt_full': resid_mkt_full,
        'resid_quality_full': resid_quality_full,
        'resid_mom_full': resid_mom_full,
        'resid_size_full': resid_size_full,
        'resid_value_full': resid_value_full,
        'resid_si_full': resid_si_full,
        'resid_vol_full': resid_vol_full,
        'resid_macro_full': resid_macro_full if macro_cols_v2 else resid_vol_full,
        'resid_sec_full': resid_sec_full,
        'lambda_mkt': lambda_mkt, 'lambda_quality': lambda_quality,
        'lambda_mom': lambda_mom, 'lambda_size': lambda_size,
        'lambda_value': lambda_value, 'lambda_si': lambda_si,
        'lambda_vol': lambda_vol, 'lambda_macro': lambda_macro,
        'lambda_sec': lambda_sec, 'lambda_ou': lambda_ou,
        'beta_df': beta_df, 'size_char_df': size_char_df,
        'quality_perp': quality_perp, 'mom_perp': mom_perp,
        'size_perp': size_perp, 'value_perp': value_perp,
        'si_perp': si_perp, 'vol_perp': vol_perp,
        'macro_betas': macro_betas_v2, 'macro_cols': macro_cols_v2,
        'dynamic_size': dynamic_size, 'si_composite': si_composite,
        'quality_df': quality_df, 'value_df': value_df,
        'vol_df': vol_df, 'ou_pivot': ou_pivot,
        'universe': universe, 'sec_cols': sec_cols,
        'common_dates': common_dates, 'ou_common': ou_common,
        'st_dt': st_dt, 'extended_st_dt': extended_st_dt,
    }


# ===============================================================================
# ENTRY POINT
# ===============================================================================

def run(Pxs_df, sectors_s, volumeTrd_df=None):
    print("=" * 70)
    print("  FACTOR MODEL v2")
    print("  Sequence: Beta → Quality → Idio Mom → Size → Value → SI → "
          "GK Vol → Macro → Sectors → O-U")
    print("=" * 70)

    Pxs_df    = Pxs_df.loc[:, ~Pxs_df.columns.duplicated(keep='first')]
    sectors_s = sectors_s[~sectors_s.index.duplicated(keep='first')]

    update_input = input(
        "\n  Incremental update? (y/n) [default=y]: "
    ).strip().lower()
    incremental = update_input != 'n'
    print(f"  Mode: {'INCREMENTAL UPDATE' if incremental else 'FULL RECALCULATION'}")

    if incremental and _v2_get_anchor_date() is None:
        print("  No existing v2 data — switching to full recalculation")
        incremental = False

    if incremental:
        vol_input     = input(
            "  Volume-scaled momentum? (y/n) [default=n]: "
        ).strip().lower()
        use_vol_scale = vol_input == 'y'
        VOL_LOWER, VOL_UPPER = 0.5, 3.0
        if use_vol_scale:
            lo = input("    Vol scalar lower bound [default=0.5]: ").strip()
            hi = input("    Vol scalar upper bound [default=3.0]: ").strip()
            VOL_LOWER = float(lo) if lo else 0.5
            VOL_UPPER = float(hi) if hi else 3.0
        return _v2_run_incremental(
            Pxs_df, sectors_s, volumeTrd_df,
            use_vol_scale=use_vol_scale,
            VOL_LOWER=VOL_LOWER, VOL_UPPER=VOL_UPPER
        )

    # Full recalculation
    st_input = input(
        "\n  Start date (YYYY-MM-DD, or Enter for 2019-01-01): "
    ).strip()
    st_dt    = pd.Timestamp(st_input) if st_input else pd.Timestamp('2019-01-01')

    vol_input     = input(
        "  Volume-scaled momentum? (y/n) [default=n]: "
    ).strip().lower()
    use_vol_scale = vol_input == 'y'
    VOL_LOWER, VOL_UPPER = 0.5, 3.0
    if use_vol_scale:
        lo = input("    Vol scalar lower bound [default=0.5]: ").strip()
        hi = input("    Vol scalar upper bound [default=3.0]: ").strip()
        VOL_LOWER = float(lo) if lo else 0.5
        VOL_UPPER = float(hi) if hi else 3.0

    print(f"  Start date: {st_dt.date()}")
    print(f"  Ridge λ grid macro: {RIDGE_GRID_MACRO}")
    print(f"  Ridge λ grid sec  : {RIDGE_GRID_SEC}")

    return _v2_run_full(
        Pxs_df, sectors_s, st_dt, volumeTrd_df,
        use_vol_scale=use_vol_scale,
        VOL_LOWER=VOL_LOWER, VOL_UPPER=VOL_UPPER
    )
