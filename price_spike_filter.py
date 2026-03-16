"""
price_spike_filter.py
=====================
Filters single or double-day price spikes from a price DataFrame.

Logic:
  A price at date t is flagged as a spike if:
    1. P[t] / P[t-1] > 3.0  OR  P[t] / P[t-1] < 0.2
       (price more than triples or drops more than 80% in one day)
    AND
    2. P[t+1] / P[t-1] is within [0.85, 1.15]  [1-day reversion]
       OR P[t+2] / P[t-1] is within [0.85, 1.15]  [2-day reversion]
       (price reverts to within ±15% of pre-spike level within 2 days)

  Flagged values are replaced with NaN, then forward-filled.

Usage:
    from price_spike_filter import filter_price_spikes
    Pxs_clean = filter_price_spikes(Pxs_df)
    Pxs_clean = filter_price_spikes(Pxs_df, spike_up=3.0, spike_dn=0.2, revert_band=0.15)
"""

import numpy as np
import pandas as pd


def filter_price_spikes(Pxs_df: pd.DataFrame,
                        spike_up: float = 3.0,
                        spike_dn: float = 0.2,
                        revert_band: float = 0.15) -> pd.DataFrame:
    """
    Remove single/double-day price spikes that revert within 2 days.

    Parameters
    ----------
    Pxs_df      : DataFrame of prices, dates x tickers.
    spike_up    : Flag if P[t]/P[t-1] > spike_up.    Default 3.0.
    spike_dn    : Flag if P[t]/P[t-1] < spike_dn.    Default 0.2.
    revert_band : Confirm spike if P[t+k]/P[t-1] in  Default 0.15
                  [1 - revert_band, 1 + revert_band]  (i.e. ±15%).

    Returns
    -------
    DataFrame with spike values replaced by NaN then forward-filled.
    """
    df     = Pxs_df.copy().astype(float)
    prices = df.values  # (dates x tickers)
    n_dates, n_tickers = prices.shape
    n_flagged = 0

    revert_lo = 1.0 - revert_band   # 0.85
    revert_hi = 1.0 + revert_band   # 1.15

    for j in range(n_tickers):
        col = prices[:, j].copy()

        for i in range(1, n_dates):
            p_prev = col[i - 1]
            p_curr = col[i]

            # Skip if either price is NaN or zero
            if np.isnan(p_prev) or np.isnan(p_curr) or p_prev == 0:
                continue

            ratio = p_curr / p_prev

            # Step 1: Is this an extreme move?
            if spike_dn <= ratio <= spike_up:
                continue

            # Step 2: Does price revert to within ±revert_band of pre-spike within 2 days?
            reverted = False
            for k in [1, 2]:
                if i + k < n_dates:
                    p_future = col[i + k]
                    if np.isnan(p_future) or p_future == 0:
                        continue
                    revert_ratio = p_future / p_prev
                    if revert_lo <= revert_ratio <= revert_hi:
                        reverted = True
                        break

            if reverted:
                prices[i, j] = np.nan
                n_flagged += 1

    print(f"  Price spike filter: {n_flagged} values flagged and NaN'd "
          f"({n_flagged / (n_dates * n_tickers) * 100:.3f}% of all observations)")

    # Rebuild DataFrame and forward-fill
    df_clean = pd.DataFrame(prices, index=Pxs_df.index, columns=Pxs_df.columns)
    df_clean = df_clean.ffill()

    # Report tickers with most spikes removed
    spike_mask  = pd.DataFrame(prices, index=Pxs_df.index,
                                columns=Pxs_df.columns).isna() & Pxs_df.notna()
    spike_counts = spike_mask.sum()
    top_spikes   = spike_counts[spike_counts > 0].sort_values(ascending=False)
    if not top_spikes.empty:
        print(f"  Tickers with spikes removed ({len(top_spikes)} total):")
        for ticker, count in top_spikes.head(10).items():
            print(f"    {ticker}: {count} spike(s)")

    return df_clean


if __name__ == "__main__":
    print("Usage: from price_spike_filter import filter_price_spikes")
    print("       Pxs_clean = filter_price_spikes(Pxs_df)")
    print("       Pxs_clean = filter_price_spikes(Pxs_df, spike_up=3.0, "
          "spike_dn=0.2, revert_band=0.15)")
