"""
ou_reversion_diag.py
====================
Standalone diagnostic for the O-U mean reversion signal.

For the last available trading day, fits an AR(1)/O-U process to each stock's
compounded residual price index (from factor_residuals_joint), computes -DistST
(standardized distance to LT mean, negated so positive = undervalued = buy),
and falls back to ST reversal rank for stocks where the O-U fit fails.

Outputs a DataFrame with one row per stock showing:
  - ou_dist     : raw DistST (positive = above LT mean = expensive)
  - neg_dist_st : -DistST (positive = cheap/undervalued)
  - lt_mean     : O-U long-term mean (in residual price index units)
  - mr_speed    : mean reversion speed k = -log(b)
  - mr_halflife : mean reversion half-life in days = 1/k
  - resid_vol   : residual std of AR(1) fit
  - st_reversal : 21d log return of compounded residual index (fallback signal, net of model factors)
  - ou_valid    : True if O-U fit succeeded
  - final_score : combined rank score (0-1, higher = more attractive)
  - final_rank  : rank by final_score descending

Usage (Jupyter):
    from ou_reversion_diag import run_ou_diag
    ou_df = run_ou_diag(Pxs_df, volumeTrd_df)

    # or without volume scaling:
    ou_df = run_ou_diag(Pxs_df)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine, text

ENGINE   = create_engine("postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db")
RESID_TABLE = 'factor_residuals_joint'

# OU fitting parameters (matching original code)
MEANREV_W  = 60     # lookback window for O-U fit (trading days)
MIN_OBS    = 30     # minimum observations for a valid fit
ST_REV_W   = 21     # ST reversal lookback (trading days)


# ==============================================================================
# HELPERS
# ==============================================================================

def clean_ticker(t: str) -> str:
    return str(t).strip().split(' ')[0].upper()


def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def rank_01(s: pd.Series, ascending: bool = True) -> pd.Series:
    """Rank series to 0-1 range. ascending=True: lowest value → rank 0."""
    valid = s.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=s.index)
    ranked = valid.rank(method='average', ascending=ascending)
    return ((ranked - 1) / (len(ranked) - 1)).reindex(s.index)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_residuals() -> pd.DataFrame:
    """
    Load factor_residuals_joint from DB.
    Returns pivot DataFrame (date x ticker), daily residual returns.
    """
    print("  Loading residuals from DB...")
    with ENGINE.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT date, ticker, resid
            FROM {RESID_TABLE}
            ORDER BY date
        """), conn)

    df['date']   = pd.to_datetime(df['date'])
    df['ticker'] = df['ticker'].apply(clean_ticker)
    pivot = df.pivot(index='date', columns='ticker', values='resid').sort_index()
    print(f"  Residuals loaded: {pivot.shape[0]} dates x {pivot.shape[1]} stocks")
    return pivot


# ==============================================================================
# O-U / AR(1) FIT
# ==============================================================================

def fit_ou(resid_series: pd.Series,
           px_series: pd.Series,
           asset: str) -> tuple:
    """
    Fit AR(1)/O-U process to compounded residual price index.

    Steps:
      1. Compound daily residuals into a price-like index anchored at
         the first available price in px_series
      2. Fit AR(1): P[t] = a + b * P[t-1] + eps
      3. Derive O-U parameters: mean m = a/(1-b), speed k = -log(b)
      4. Compute DistST = (last_price - m) / (std(eps) / sqrt(2k))

    Returns: (m, k, resid_std, dist_st) or (nan, nan, nan, nan) on failure.
    """
    # Compound residuals into price index
    resid_clean = resid_series.replace({np.inf: np.nan, -np.inf: np.nan}).dropna()
    if len(resid_clean) < MIN_OBS:
        return np.nan, np.nan, np.nan, np.nan

    # Anchor to first available price for the asset
    anchor_dates = px_series.index[px_series.index >= resid_clean.index[0]]
    if anchor_dates.empty:
        return np.nan, np.nan, np.nan, np.nan
    anchor_price = float(px_series.loc[anchor_dates[0]])
    if np.isnan(anchor_price) or anchor_price <= 0:
        return np.nan, np.nan, np.nan, np.nan

    px_idx = (1 + resid_clean).cumprod() * anchor_price

    sX1 = px_idx.iloc[:-1].values.reshape(-1, 1)
    sX2 = px_idx.iloc[1:].values

    try:
        mod = LinearRegression()
        mod.fit(sX1, sX2)
        a = float(mod.intercept_)
        b = float(mod.coef_[0])
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

    # Validate AR(1) coefficient — must be in (0, 1) for mean reversion
    if not (0 < b < 1):
        return np.nan, np.nan, np.nan, np.nan

    m = a / (1 - b)
    k = -np.log(b)

    residuals = sX2 - mod.predict(sX1).flatten()
    resid_std = float(np.std(residuals))

    if resid_std == 0 or k == 0:
        return np.nan, np.nan, np.nan, np.nan

    # Use the actual last price from Pxs_df for DistST
    # Scale lt_mean to actual price space: multiply by ratio of actual/index last price
    # so that DistST reflects distance in actual price units
    last_px_actual = float(px_series.iloc[-1]) if not px_series.empty else np.nan
    if np.isnan(last_px_actual) or last_px_actual <= 0:
        return np.nan, np.nan, np.nan, np.nan

    idx_last   = float(px_idx.iloc[-1])
    scale      = last_px_actual / idx_last if idx_last != 0 else 1.0
    m_scaled   = m * scale
    std_scaled = resid_std * scale

    # Discard degenerate fits — negative LT mean is economically meaningless
    if m_scaled <= 0:
        return np.nan, np.nan, np.nan, np.nan

    dist_st = (last_px_actual - m_scaled) / (std_scaled / np.sqrt(2 * k))

    return m_scaled, k, std_scaled, round(dist_st, 4)


# ==============================================================================
# ST REVERSAL FALLBACK
# ==============================================================================

def calc_st_reversal_resid(resid_pivot: pd.DataFrame,
                            tickers: list,
                            ref_date: pd.Timestamp) -> pd.Series:
    """
    21-day log return reversal computed from compounded residual price index.
    log(cum_resid[t-1] / cum_resid[t-22]) — idiosyncratic, net of all model factors.
    Lower (more negative) = recent idiosyncratic loser = higher reversal score.
    """
    past = resid_pivot.index[resid_pivot.index <= ref_date]
    if len(past) < ST_REV_W + 1:
        return pd.Series(np.nan, index=tickers)

    # Compounded residual index up to ref_date
    resid_past = resid_pivot.loc[past, tickers]
    cum_resid  = (1 + resid_past.fillna(0)).cumprod()

    cum_recent = cum_resid.iloc[-1]
    cum_old    = cum_resid.iloc[-ST_REV_W - 1]

    valid = (cum_recent > 0) & (cum_old > 0)
    rev   = np.log(cum_recent / cum_old).where(valid)
    return rev.reindex(tickers)


# ==============================================================================
# MAIN
# ==============================================================================

def run_ou_diag(Pxs_df: pd.DataFrame,
                volumeTrd_df: pd.DataFrame = None,
                vol_clip_lo: float = 0.5,
                vol_clip_hi: float = 3.0,
                ou_weight_cap: float = 10.0) -> pd.DataFrame:
    """
    Compute O-U mean reversion scores for the last available trading day.

    Args:
        Pxs_df        : Price DataFrame (dates x bare tickers).
        volumeTrd_df  : Volume scalars (already normalized, same shape as Pxs_df).
        vol_clip_lo   : Lower clip for volume scalar (default 0.5).
        vol_clip_hi   : Upper clip for volume scalar (default 3.0).
        ou_weight_cap : Maximum OU weight in blended score (default 10.0).
                        ou_weight = min(30 / Th, ou_weight_cap)
                        final_score = (ou_weight * ou_rank + st_rank) / (ou_weight + 1)

    Returns:
        DataFrame indexed by ticker with columns:
          ou_dist, neg_dist_st, lt_mean, mr_speed, mr_halflife,
          resid_vol, st_reversal, ou_valid, ou_weight,
          final_score, final_rank
    """
    resid_pivot = load_residuals()

    # Volume-scale residuals if provided
    # volumeTrd_df is assumed to already be normalized volume scalars
    # Divide residuals by clipped scalars: higher volume → smaller residual
    if volumeTrd_df is not None:
        print("  Applying volume scaling to residuals...")
        common_tickers = resid_pivot.columns.intersection(volumeTrd_df.columns)
        vol_ratio = (volumeTrd_df[common_tickers]
                     .reindex(resid_pivot.index)
                     .ffill()
                     .clip(vol_clip_lo, vol_clip_hi))
        resid_pivot[common_tickers] = resid_pivot[common_tickers] / vol_ratio
        print(f"  Volume scaling applied to {len(common_tickers)} tickers "
              f"(clip: [{vol_clip_lo}, {vol_clip_hi}])")
    else:
        print("  No volume scaling (volumeTrd_df not provided)")

    # Reference date: last available in residuals
    ref_date = resid_pivot.index[-1]
    print(f"\n  Reference date : {ref_date.date()}")
    print(f"  Fitting O-U for {resid_pivot.shape[1]} stocks (window={MEANREV_W}d)...")

    tickers = resid_pivot.columns.tolist()
    records = {}

    for i, ticker in enumerate(tickers):
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(tickers)}]", end='\r')

        # Get last MEANREV_W days of residuals for this ticker
        stock_resid = resid_pivot[ticker].dropna()
        if len(stock_resid) < MIN_OBS:
            records[ticker] = {'ou_valid': False}
            continue

        # Use last MEANREV_W observations
        stock_resid = stock_resid.iloc[-MEANREV_W:]

        # Get price series for anchor
        if ticker not in Pxs_df.columns:
            records[ticker] = {'ou_valid': False}
            continue
        px_series = Pxs_df[ticker].dropna()

        # Fit O-U
        m, k, resid_std, dist_st = fit_ou(stock_resid, px_series, ticker)

        ou_valid = not (np.isnan(dist_st) or np.isnan(m))
        records[ticker] = {
            'ou_dist':     dist_st if ou_valid else np.nan,
            'neg_dist_st': -dist_st if ou_valid else np.nan,
            'lt_mean':     m if ou_valid else np.nan,
            'mr_speed':    k if ou_valid else np.nan,
            'mr_halflife': round(np.log(2) / k, 1) if (ou_valid and k > 0) else np.nan,
            'resid_vol':   resid_std if ou_valid else np.nan,
            'ou_valid':    ou_valid,
        }

    print(f"\n  O-U fit complete.")

    df = pd.DataFrame(records).T
    df.index.name = 'ticker'

    # ST reversal from residual index for all stocks (fallback + comparison)
    print(f"  Computing ST reversal from residual index (fallback)...")
    st_rev = calc_st_reversal_resid(resid_pivot, tickers, ref_date)
    df['st_reversal'] = st_rev.reindex(df.index)

    # Summary stats
    n_valid   = df['ou_valid'].sum()
    n_invalid = (~df['ou_valid']).sum()
    print(f"\n  O-U valid   : {n_valid} stocks")
    print(f"  O-U invalid : {n_invalid} stocks (will use ST reversal rank)")

    # Cross-sectional ranks (0=worst, 1=best)
    # neg_dist_st: high = undervalued = good → ascending rank
    # st_reversal: low (negative return) = recent loser = buy → descending rank
    ou_rank  = rank_01(df['neg_dist_st'], ascending=True)
    rev_rank = rank_01(df['st_reversal'],  ascending=False)

    df['ou_rank']  = ou_rank
    df['rev_rank'] = rev_rank

    # OU weight per stock: min(30 / Th, ou_weight_cap)
    # Th = mr_halflife (ln(2)/k). Fast convergers get high OU weight.
    # NaN halflife (invalid fit) → ou_weight = 0 → pure ST reversal
    df['ou_weight'] = (30.0 / df['mr_halflife']).clip(upper=ou_weight_cap).where(
        df['ou_valid'], other=0.0
    )

    # Weighted blend: (ou_weight * ou_rank + rev_rank) / (ou_weight + 1)
    # When ou_weight = 0 (NaN/invalid): final_score = rev_rank
    df['final_score'] = (
        (df['ou_weight'] * df['ou_rank'].fillna(0) + df['rev_rank'])
        / (df['ou_weight'] + 1)
    )
    # For stocks where ou_rank is NaN, ou_weight is already 0 so formula reduces
    # to rev_rank / 1 = rev_rank — no further adjustment needed

    df['final_rank'] = df['final_score'].rank(ascending=False, method='min').astype('Int64')

    # Round for display
    float_cols = ['ou_dist', 'neg_dist_st', 'lt_mean', 'mr_speed',
                  'mr_halflife', 'resid_vol', 'st_reversal',
                  'ou_weight', 'ou_rank', 'rev_rank', 'final_score']
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').round(4)

    # Print summary table — top 30 and bottom 30 by final_score
    display_cols = ['neg_dist_st', 'mr_halflife', 'ou_weight', 'st_reversal',
                    'ou_valid', 'ou_rank', 'rev_rank', 'final_score', 'final_rank']

    print(f"\n{'='*70}")
    print(f"  O-U MEAN REVERSION SCORES — {ref_date.date()}")
    print(f"{'='*70}")
    print(f"\n  TOP 30 (most undervalued / mean-reversion buy):")
    print(df.sort_values('final_score', ascending=False)[display_cols].head(30).to_string())

    print(f"\n  BOTTOM 30 (most overvalued / mean-reversion sell):")
    print(df.sort_values('final_score', ascending=True)[display_cols].head(30).to_string())

    print(f"\n  Distribution of -DistST (O-U valid stocks only):")
    valid_dist = df.loc[df['ou_valid'] == True, 'neg_dist_st'].dropna()
    if not valid_dist.empty:
        print(f"  Mean  : {valid_dist.mean():.3f}")
        print(f"  Std   : {valid_dist.std():.3f}")
        print(f"  Min   : {valid_dist.min():.3f}")
        print(f"  25th  : {valid_dist.quantile(0.25):.3f}")
        print(f"  Median: {valid_dist.median():.3f}")
        print(f"  75th  : {valid_dist.quantile(0.75):.3f}")
        print(f"  Max   : {valid_dist.max():.3f}")

    print(f"\n  Mean reversion half-life distribution (O-U valid):")
    valid_hl = df.loc[df['ou_valid'] == True, 'mr_halflife'].dropna()
    if not valid_hl.empty:
        print(f"  Mean  : {valid_hl.mean():.1f} days")
        print(f"  Median: {valid_hl.median():.1f} days")
        print(f"  10th  : {valid_hl.quantile(0.10):.1f} days")
        print(f"  90th  : {valid_hl.quantile(0.90):.1f} days")

    return df


if __name__ == "__main__":
    print("Usage: from ou_reversion_diag import run_ou_diag")
    print("       ou_df = run_ou_diag(Pxs_df)")
    print("       ou_df = run_ou_diag(Pxs_df, volumeTrd_df)")
    print("       ou_df = run_ou_diag(Pxs_df, volumeTrd_df, vol_clip_lo=0.3, vol_clip_hi=5.0)")
    print("       ou_df = run_ou_diag(Pxs_df, volumeTrd_df, ou_weight_cap=10.0)")
