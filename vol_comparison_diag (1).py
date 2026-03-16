"""
vol_comparison_diag.py
======================
Diagnostic comparing close-to-close EWMA vol vs Garman-Klass EWMA vol.

Loads all data directly from DB — no inputs required.
Outputs two DataFrames (dates x tickers), annualized vol:
  - cc_vol_df  : close-to-close EWMA volatility
  - gk_vol_df  : Garman-Klass EWMA volatility

Usage (Jupyter):
    from vol_comparison_diag import run_vol_comparison
    cc_vol_df, gk_vol_df = run_vol_comparison()

    # Compare a specific stock:
    import matplotlib.pyplot as plt
    ticker = 'AAPL'
    cc_vol_df[ticker].plot(label='Close-to-Close')
    gk_vol_df[ticker].plot(label='Garman-Klass')
    plt.legend(); plt.title(f'{ticker} EWMA Vol'); plt.show()
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

ENGINE     = create_engine("postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db")
VOL_WINDOW = 84    # EWMA lookback window (trading days)
VOL_HL     = 42    # EWMA half-life (trading days)


# ==============================================================================
# HELPERS
# ==============================================================================

def clean_ticker(t: str) -> str:
    return str(t).strip().split(' ')[0].upper()


def ewma_weights(n: int, hl: int) -> np.ndarray:
    alpha   = 1 - np.exp(-np.log(2) / hl)
    weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
    return weights / weights.sum()


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_table(tbl: str, label: str) -> pd.DataFrame:
    """Load a price table from DB, normalize tickers, return dates x tickers."""
    print(f"  Loading '{tbl}'...")
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(text(f"SELECT * FROM {tbl}"), conn)
    except Exception as e:
        print(f"  ERROR loading '{tbl}': {e}")
        return pd.DataFrame()

    # Find date column
    date_col = [c for c in df.columns if 'date' in c.lower() or c.lower() == 'index']
    if not date_col:
        print(f"  ERROR: no date column in '{tbl}' — columns: {list(df.columns[:8])}")
        return pd.DataFrame()

    dc = date_col[0]
    df[dc] = pd.to_datetime(df[dc])
    df = df.set_index(dc).sort_index()

    # Normalize ticker columns
    orig_cols    = list(df.columns)
    df.columns   = [clean_ticker(c) for c in df.columns]
    dupes        = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        print(f"  WARNING: {len(dupes)} duplicate tickers after normalization in '{tbl}' "
              f"(e.g. {dupes[:5]}) — keeping first occurrence")
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    df = df.astype(float)
    print(f"  '{tbl}': {df.shape[0]} dates x {df.shape[1]} tickers "
          f"| date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Sample cols (raw → clean): "
          f"{list(zip(orig_cols[:4], df.columns[:4].tolist()))}")
    return df


def load_all_data() -> tuple:
    """Load close, open, high, range from DB. Return (close, open, high, low)."""
    close_df = load_table('prices_relation', 'close')
    open_df  = load_table('daily_open',      'open')
    high_df  = load_table('daily_high',      'high')
    range_df = load_table('daily_range',     'range')

    if any(df.empty for df in [close_df, open_df, high_df, range_df]):
        print("\n  ERROR: one or more tables failed to load")
        return None, None, None, None

    # Low = High - Range
    low_df = (high_df - range_df.reindex(index=high_df.index,
                                          columns=high_df.columns))

    # Common tickers across all four tables
    common_tickers = (set(close_df.columns)
                      .intersection(open_df.columns)
                      .intersection(high_df.columns)
                      .intersection(low_df.columns))
    common_tickers = sorted(common_tickers)

    print(f"\n  Common tickers across all tables: {len(common_tickers)}")
    missing_close = [t for t in open_df.columns if t not in close_df.columns]
    missing_ohlc  = [t for t in close_df.columns if t not in open_df.columns]
    if missing_close:
        print(f"  Tickers in OHLC but not in close prices: {len(missing_close)} "
              f"(e.g. {missing_close[:5]})")
    if missing_ohlc:
        print(f"  Tickers in close prices but not in OHLC: {len(missing_ohlc)} "
              f"(e.g. {missing_ohlc[:5]})")

    # Common dates
    common_dates = (close_df.index
                    .intersection(open_df.index)
                    .intersection(high_df.index))
    print(f"  Common dates: {len(common_dates)} "
          f"| {common_dates[0].date()} → {common_dates[-1].date()}")

    close_df = close_df.loc[common_dates, common_tickers]
    open_df  = open_df.loc[common_dates,  common_tickers]
    high_df  = high_df.loc[common_dates,  common_tickers]
    low_df   = low_df.loc[common_dates,   common_tickers]

    return close_df, open_df, high_df, low_df


# ==============================================================================
# VOL COMPUTATION
# ==============================================================================

def compute_cc_vol(close_df: pd.DataFrame) -> pd.DataFrame:
    """
    Close-to-close EWMA realized volatility.
    Returns annualized vol DataFrame (dates x tickers).
    """
    print(f"\n  Computing close-to-close EWMA vol (window={VOL_WINDOW}d, hl={VOL_HL}d)...")
    tickers   = close_df.columns.tolist()
    all_dates = close_df.index
    vol_dict  = {}

    for i, dt in enumerate(all_dates):
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(all_dates)}]", end='\r')

        past = all_dates[all_dates < dt]
        if len(past) < VOL_WINDOW // 2:
            continue

        window  = past[-VOL_WINDOW:]
        px_win  = close_df.loc[window]
        rets    = px_win.pct_change().dropna(how='all')
        if len(rets) < VOL_WINDOW // 2:
            continue

        w        = ewma_weights(len(rets), VOL_HL)
        ewma_var = (rets ** 2).mul(w, axis=0).sum(axis=0)
        vol_dict[dt] = np.sqrt(ewma_var * 252)

    print(f"\n  Close-to-close vol: {len(vol_dict)} dates computed")
    df            = pd.DataFrame(vol_dict).T.reindex(columns=tickers)
    df.index.name = 'date'
    return df


def compute_gk_vol(close_df: pd.DataFrame,
                   open_df: pd.DataFrame,
                   high_df: pd.DataFrame,
                   low_df: pd.DataFrame) -> pd.DataFrame:
    """
    Garman-Klass EWMA vol: σ²_GK = 0.5·(ln(H/L))² - (2·ln2-1)·(ln(C/O))²
    Returns annualized vol DataFrame (dates x tickers).
    """
    print(f"\n  Computing Garman-Klass EWMA vol (window={VOL_WINDOW}d, hl={VOL_HL}d)...")
    tickers   = close_df.columns.tolist()
    all_dates = close_df.index
    vol_dict  = {}

    for i, dt in enumerate(all_dates):
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(all_dates)}]", end='\r')

        past = all_dates[all_dates < dt]
        if len(past) < VOL_WINDOW // 2:
            continue

        window = past[-VOL_WINDOW:]
        C = close_df.loc[window]
        O = open_df.loc[window]
        H = high_df.loc[window]
        L = low_df.loc[window]

        valid_mask = (C > 0) & (O > 0) & (H > 0) & (L > 0) & (H >= L)
        log_hl     = np.log(H / L).where(valid_mask)
        log_co     = np.log(C / O).where(valid_mask)
        gk_var     = (0.5 * log_hl ** 2
                      - (2 * np.log(2) - 1) * log_co ** 2).clip(lower=0)

        w        = ewma_weights(len(window), VOL_HL)
        ewma_var = gk_var.mul(w, axis=0).sum(axis=0)
        vol_dict[dt] = np.sqrt(ewma_var * 252)

    print(f"\n  Garman-Klass vol: {len(vol_dict)} dates computed")
    df            = pd.DataFrame(vol_dict).T.reindex(columns=tickers)
    df.index.name = 'date'
    return df


# ==============================================================================
# MAIN
# ==============================================================================

def run_vol_comparison() -> tuple:
    """
    Load all data from DB and compute both vol estimates.

    Returns:
        cc_vol_df : DataFrame (dates x tickers) — close-to-close annualized vol
        gk_vol_df : DataFrame (dates x tickers) — Garman-Klass annualized vol
    """
    print("=" * 60)
    print("  VOL COMPARISON: Close-to-Close vs Garman-Klass")
    print("=" * 60)

    close_df, open_df, high_df, low_df = load_all_data()
    if close_df is None:
        return None, None

    cc_vol_df = compute_cc_vol(close_df)
    gk_vol_df = compute_gk_vol(close_df, open_df, high_df, low_df)

    # Summary comparison
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    cc_last = cc_vol_df.iloc[-1].dropna()
    gk_last = gk_vol_df.iloc[-1].dropna()

    print(f"\n  Last date: {cc_vol_df.index[-1].date()}")
    print(f"\n  Close-to-close vol (last date):")
    print(f"    Mean   : {cc_last.mean()*100:.1f}%")
    print(f"    Median : {cc_last.median()*100:.1f}%")
    print(f"    Min    : {cc_last.min()*100:.1f}%")
    print(f"    Max    : {cc_last.max()*100:.1f}%")

    print(f"\n  Garman-Klass vol (last date):")
    print(f"    Mean   : {gk_last.mean()*100:.1f}%")
    print(f"    Median : {gk_last.median()*100:.1f}%")
    print(f"    Min    : {gk_last.min()*100:.1f}%")
    print(f"    Max    : {gk_last.max()*100:.1f}%")

    # Ratio GK/CC — should be < 1 since GK is more efficient (lower noise)
    common = cc_last.index.intersection(gk_last.index)
    ratio  = (gk_last[common] / cc_last[common]).dropna()
    print(f"\n  GK / CC ratio (last date, {len(ratio)} tickers):")
    print(f"    Mean   : {ratio.mean():.3f}")
    print(f"    Median : {ratio.median():.3f}")
    print(f"    <1.0   : {(ratio < 1).mean()*100:.1f}% of tickers")

    # Flag any suspicious GK values
    neg_gk = (gk_vol_df < 0).any(axis=1).sum()
    nan_ratio = gk_vol_df.isna().mean().mean()
    print(f"\n  GK diagnostics:")
    print(f"    Dates with any negative GK vol : {neg_gk}")
    print(f"    Overall NaN rate               : {nan_ratio*100:.1f}%")

    return cc_vol_df, gk_vol_df


if __name__ == "__main__":
    print("Usage: from vol_comparison_diag import run_vol_comparison")
    print("       cc_vol_df, gk_vol_df = run_vol_comparison()")
