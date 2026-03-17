"""
plot_factor_returns.py
======================
Visualizes cumulative Fama-MacBeth lambda returns for all factors
in the 9-step factor model. Organized into 5 figures:

  Fig 1 — Market & Structural   : intercept, beta, size
  Fig 2 — Macro Factors         : rates, spread, breakevens, crude, gold
  Fig 3 — Sector Dummies        : all 12 sectors
  Fig 4 — Alpha Factors         : quality, vol, SI, idio_mom, value, O-U
  Fig 5 — Rolling 252d t-stats  : statistical significance over time for alpha factors

Usage:
    from plot_factor_returns import plot_all
    plot_all(results)

    # Or plot individual figures:
    from plot_factor_returns import (
        plot_structural, plot_macro, plot_sectors, plot_alpha, plot_rolling_tstats
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator


# ==============================================================================
# HELPERS
# ==============================================================================

def _prep(lambda_df: pd.DataFrame, col: str, st_dt: pd.Timestamp) -> tuple:
    """Return (dates_array, cumsum_array) for a given lambda column."""
    lm = lambda_df[lambda_df.index >= st_dt]
    if col not in lm.columns:
        return None, None
    s    = lm[col].fillna(0)
    cum  = s.cumsum().to_numpy()
    dates = lm.index.to_numpy()
    return dates, cum


def _fill(ax, dates, cum, color='steelblue'):
    """Fill between cumsum line and zero."""
    ax.fill_between(dates, cum, 0, where=(cum >= 0), alpha=0.12, color='green')
    ax.fill_between(dates, cum, 0, where=(cum <  0), alpha=0.12, color='red')


def _style(ax, title: str, ylabel: str = 'Cumulative Lambda', legend_cols: int = 1):
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, ncol=legend_cols, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())


def _collect(results: dict, table: str) -> pd.DataFrame:
    """Pull a lambda DataFrame from results dict, returning empty DF if missing."""
    df = results.get(table, pd.DataFrame())
    if df is None:
        return pd.DataFrame()
    return df


# ==============================================================================
# FIG 1 — MARKET & STRUCTURAL (intercept, beta, size)
# ==============================================================================

def plot_structural(results: dict):
    st_dt    = results['st_dt']
    lm_mkt   = _collect(results, 'lambda_mkt')
    lm_size  = _collect(results, 'lambda_size')

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Fig 1 — Market & Structural Factor Returns',
                 fontsize=13, fontweight='bold', y=0.99)

    # Intercept
    ax = axes[0]
    d, c = _prep(lm_mkt, 'intercept', st_dt)
    if d is not None:
        ax.plot(d, c, color='black', linewidth=1.2, label='Intercept (EW market)')
        _fill(ax, d, c, 'black')
    _style(ax, 'Intercept (Equal-Weighted Market Return)')

    # Market beta
    ax = axes[1]
    d, c = _prep(lm_mkt, 'beta', st_dt)
    if d is not None:
        ax.plot(d, c, color='steelblue', linewidth=1.2, label='Market Beta')
        _fill(ax, d, c)
    _style(ax, 'Market Beta Lambda')

    # Size
    ax = axes[2]
    d, c = _prep(lm_size, 'size', st_dt)
    if d is not None:
        ax.plot(d, c, color='darkgreen', linewidth=1.2, label='Size (log mkt cap z-score)')
        _fill(ax, d, c, 'darkgreen')
    _style(ax, 'Size Lambda')

    plt.tight_layout()
    plt.show()


# ==============================================================================
# FIG 2 — MACRO FACTORS
# ==============================================================================

def plot_macro(results: dict):
    st_dt     = results['st_dt']
    lm_macro  = _collect(results, 'lambda_macro')
    macro_cols = results.get('macro_cols', [])

    if lm_macro.empty or not macro_cols:
        print("No macro factor data available.")
        return

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    labels = {
        'USGG2YR':            '2Y Rate',
        'US10Y2Y_SPREAD_CHG': '2Y/10Y Spread',
        'T10YIE':             '10Y B/E Inflation',
        'T5YIFR':             '5y5y Inflation',
        'Crude':              'WTI Crude',
        'XAUUSD':             'Gold',
    }

    n     = len(macro_cols)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=True)
    axes = np.array(axes).flatten()
    fig.suptitle('Fig 2 — Macro Factor Returns (EWMA Betas)',
                 fontsize=13, fontweight='bold', y=0.99)

    for i, col in enumerate(macro_cols):
        ax  = axes[i]
        lbl = labels.get(col, col)
        d, c = _prep(lm_macro, col, st_dt)
        if d is not None:
            ax.plot(d, c, color=colors[i % len(colors)], linewidth=1.2, label=lbl)
            _fill(ax, d, c, colors[i % len(colors)])
        _style(ax, lbl)

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# FIG 3 — SECTOR DUMMIES
# ==============================================================================

def plot_sectors(results: dict):
    st_dt    = results['st_dt']
    lm_sec   = _collect(results, 'lambda_sec')
    sec_cols = results.get('sec_cols', [])

    if lm_sec.empty or not sec_cols:
        print("No sector data available.")
        return

    colors = plt.cm.tab20.colors
    lm     = lm_sec[lm_sec.index >= st_dt]
    dates  = lm.index.to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Fig 3 — Sector Dummy Factor Returns',
                 fontsize=13, fontweight='bold', y=0.99)

    # Top: all sectors cumulative
    ax = axes[0]
    for i, col in enumerate(sorted(sec_cols)):
        if col not in lm.columns:
            continue
        cum = lm[col].fillna(0).cumsum().to_numpy()
        ax.plot(dates, cum, label=col, color=colors[i % len(colors)], linewidth=1.0)
    _style(ax, 'All Sectors — Cumulative Lambda', legend_cols=4)

    # Bottom: spread (best minus worst)
    ax = axes[1]
    cum_all = {}
    for col in sec_cols:
        if col in lm.columns:
            cum_all[col] = lm[col].fillna(0).cumsum().to_numpy()
    if len(cum_all) >= 2:
        mat   = np.array(list(cum_all.values()))  # (n_sectors x dates)
        best  = mat.max(axis=0)
        worst = mat.min(axis=0)
        spread = best - worst
        ax.plot(dates, spread, color='purple', linewidth=1.2, label='Best − Worst sector')
        ax.fill_between(dates, spread, 0, alpha=0.15, color='purple')
    _style(ax, 'Cross-Sector Spread (Best − Worst)')

    plt.tight_layout()
    plt.show()


# ==============================================================================
# FIG 4 — ALPHA FACTORS
# ==============================================================================

def plot_alpha(results: dict):
    st_dt      = results['st_dt']
    lm_quality = _collect(results, 'lambda_quality')
    lm_mom     = _collect(results, 'lambda_mom')
    lm_joint   = _collect(results, 'lambda_joint')
    lm_ou      = _collect(results, 'lambda_ou')

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=True)
    fig.suptitle('Fig 4 — Alpha Factor Returns',
                 fontsize=13, fontweight='bold', y=0.99)

    specs = [
        (lm_quality, 'quality',      'Quality Factor',           'darkorchid',  axes[0, 0]),
        (lm_mom,     'vol',          'GK Volatility',            'coral',       axes[0, 1]),
        (lm_mom,     'si_composite', 'Short Interest Composite', 'crimson',     axes[1, 0]),
        (lm_mom,     'idio_mom',     'Idio Momentum (252/21)',   'darkorange',  axes[1, 1]),
        (lm_joint,   'value',        'Value Factor',             'teal',        axes[2, 0]),
        (lm_ou,      'ou_reversion', 'O-U Mean Reversion',       'navy',        axes[2, 1]),
    ]

    for ldf, col, title, color, ax in specs:
        d, c = _prep(ldf, col, st_dt)
        if d is not None:
            ax.plot(d, c, color=color, linewidth=1.2, label=title)
            _fill(ax, d, c, color)
        _style(ax, title)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# FIG 5 — ROLLING 252D T-STATS (alpha factors)
# ==============================================================================

def plot_rolling_tstats(results: dict, window: int = 252):
    st_dt      = results['st_dt']
    lm_quality = _collect(results, 'lambda_quality')
    lm_mom     = _collect(results, 'lambda_mom')
    lm_joint   = _collect(results, 'lambda_joint')
    lm_ou      = _collect(results, 'lambda_ou')
    lm_macro   = _collect(results, 'lambda_macro')
    macro_cols = results.get('macro_cols', [])

    def rolling_tstat(ldf: pd.DataFrame, col: str) -> pd.Series:
        if ldf.empty or col not in ldf.columns:
            return pd.Series(dtype=float)
        s = ldf[col].dropna()
        rm = s.rolling(window).mean()
        rs = s.rolling(window).std()
        rn = s.rolling(window).count()
        return (rm / (rs / np.sqrt(rn))).reindex(s.index)

    alpha_specs = [
        (lm_quality, 'quality',      'Quality',       'darkorchid'),
        (lm_mom,     'vol',          'GK Vol',        'coral'),
        (lm_mom,     'si_composite', 'SI',            'crimson'),
        (lm_mom,     'idio_mom',     'Idio Mom',      'darkorange'),
        (lm_joint,   'value',        'Value',         'teal'),
        (lm_ou,      'ou_reversion', 'O-U',           'navy'),
    ]
    macro_specs = [(lm_macro, col, col, f'C{i}') for i, col in enumerate(macro_cols)]

    # Two subplots: alpha factors + macro factors
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Fig 5 — Rolling {window}d t-statistics',
                 fontsize=13, fontweight='bold', y=0.99)

    for ax, specs, title in zip(axes,
                                 [alpha_specs, macro_specs],
                                 ['Alpha Factors', 'Macro Factors']):
        for ldf, col, label, color in specs:
            ts = rolling_tstat(ldf, col)
            ts = ts[ts.index >= st_dt]
            if ts.empty:
                continue
            ax.plot(ts.index.to_numpy(), ts.to_numpy(),
                    label=label, color=color, linewidth=1.0)
        ax.axhline( 2.0, color='green', linewidth=0.8, linestyle=':', alpha=0.7, label='+2 t-stat')
        ax.axhline(-2.0, color='red',   linewidth=0.8, linestyle=':', alpha=0.7, label='−2 t-stat')
        ax.axhline( 0.0, color='grey',  linewidth=0.5, linestyle='--')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Rolling t-stat', fontsize=9)
        ax.legend(fontsize=8, ncol=4, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.yaxis.set_minor_locator(MultipleLocator(1))

    plt.tight_layout()
    plt.show()


# ==============================================================================
# PLOT ALL
# ==============================================================================

def plot_all(results: dict, rolling_window: int = 252):
    """Plot all 5 figures."""
    plot_structural(results)
    plot_macro(results)
    plot_sectors(results)
    plot_alpha(results)
    plot_rolling_tstats(results, window=rolling_window)


if __name__ == '__main__':
    print("Usage: from plot_factor_returns import plot_all")
    print("       plot_all(results)")
