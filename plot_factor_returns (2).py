"""
plot_factor_returns.py
======================
Visualizes cumulative Fama-MacBeth lambda returns for all factors
in the 9-step factor model. Organized into 6 figures:

  Fig 1 — Market & Structural   : intercept, beta, size
  Fig 2 — Macro Factors         : rates, spread, breakevens, crude, gold
  Fig 3 — Sector Dummies        : all 12 sectors
  Fig 4 — Alpha Factors         : quality, SI, vol, idio_mom, value, O-U (Steps 6-11)
  Fig 5 — Rolling 252d t-stats  : statistical significance over time
  Fig 6 — Optimal Ridge λ       : chosen λ over time for macro and step 7

Usage — from results dict (full recalculation):
    from plot_factor_returns import plot_all
    plot_all(results)

Usage — from DB (incremental or standalone):
    from plot_factor_returns import load_lambdas_from_db, plot_all
    results = load_lambdas_from_db(st_date='2019-01-01')
    plot_all(results)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from sqlalchemy import create_engine, text

ENGINE = create_engine(
    "postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db"
)

MACRO_COLS = [
    'USGG2YR', 'US10Y2Y_SPREAD_CHG', 'T10YIE', 'T5YIFR', 'Crude', 'XAUUSD'
]
SECTOR_COLS = [
    'IGV', 'REZ', 'SOXX', 'XHB', 'XLB', 'XLC',
    'XLE', 'XLF', 'XLI', 'XLU', 'XLV', 'XLY'
]


# ==============================================================================
# DB LOADER
# ==============================================================================

def _load_table(table, st_date):
    try:
        with ENGINE.connect() as conn:
            df = pd.read_sql(
                text(f"SELECT * FROM {table} WHERE date >= :st ORDER BY date"),
                conn, params={"st": st_date}
            )
        if df.empty:
            return pd.DataFrame()
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        if date_col is None:
            return pd.DataFrame()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        df = df.apply(pd.to_numeric, errors='coerce')
        return df
    except Exception as e:
        print(f"  WARNING: could not load '{table}' from DB: {e}")
        return pd.DataFrame()


def load_lambdas_from_db(st_date='2019-01-01'):
    """
    Load all lambda tables from DB and return a results-compatible dict.
    Use this after an incremental update to plot without a full run.
    """
    print(f"  Loading lambda tables from DB (from {st_date})...")
    tables = {
        'lambda_mkt':     'factor_lambdas_mkt',
        'lambda_size':    'factor_lambdas_size',
        'lambda_macro':   'factor_lambdas_macro',
        'lambda_sec':     'factor_lambdas_sec',
        'lambda_quality': 'factor_lambdas_quality',
        'lambda_si':      'factor_lambdas_si',
        'lambda_vol':     'factor_lambdas_vol',
        'lambda_mom':     'factor_lambdas_mom',
        'lambda_joint':   'factor_lambdas_joint',
        'lambda_ou':      'factor_lambdas_ou',
    }
    results = {}
    for key, tbl in tables.items():
        df = _load_table(tbl, st_date)
        results[key] = df
        if not df.empty:
            print(f"    {tbl}: {len(df)} dates")
    results['st_dt']      = pd.Timestamp(st_date)
    results['macro_cols'] = [c for c in MACRO_COLS if c in results['lambda_macro'].columns]
    results['sec_cols']   = [c for c in SECTOR_COLS if c in results['lambda_sec'].columns]
    print(f"  Done.")
    return results


# ==============================================================================
# HELPERS
# ==============================================================================

def _prep(lambda_df, col, st_dt):
    if lambda_df is None or lambda_df.empty:
        return None, None
    lm = lambda_df[lambda_df.index >= st_dt]
    if col not in lm.columns:
        return None, None
    s     = lm[col].fillna(0)
    cum   = s.cumsum().to_numpy()
    dates = lm.index.to_numpy()
    return dates, cum


def _fill(ax, dates, cum, color='steelblue'):
    ax.fill_between(dates, cum, 0, where=(cum >= 0), alpha=0.12, color='green')
    ax.fill_between(dates, cum, 0, where=(cum <  0), alpha=0.12, color='red')


def _style(ax, title, ylabel='Cumulative Lambda', legend_cols=1):
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, ncol=legend_cols, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())


def _collect(results, key):
    df = results.get(key, pd.DataFrame())
    return pd.DataFrame() if df is None else df


# ==============================================================================
# FIG 1 — MARKET & STRUCTURAL
# ==============================================================================

def plot_structural(results):
    st_dt   = results['st_dt']
    lm_mkt  = _collect(results, 'lambda_mkt')
    lm_size = _collect(results, 'lambda_size')

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Fig 1 — Market & Structural Factor Returns',
                 fontsize=13, fontweight='bold', y=0.99)

    for ax, ldf, col, label, color in [
        (axes[0], lm_mkt,  'intercept', 'Intercept (EW market)', 'black'),
        (axes[1], lm_mkt,  'beta',      'Market Beta',           'steelblue'),
        (axes[2], lm_size, 'size',      'Size (log mkt cap)',    'darkgreen'),
    ]:
        d, c = _prep(ldf, col, st_dt)
        if d is not None:
            ax.plot(d, c, color=color, linewidth=1.2, label=label)
            _fill(ax, d, c, color)
        _style(ax, label)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# FIG 2 — MACRO FACTORS
# ==============================================================================

def plot_macro(results):
    st_dt      = results['st_dt']
    lm_macro   = _collect(results, 'lambda_macro')
    macro_cols = results.get('macro_cols', MACRO_COLS)

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
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4 * nrows), sharex=True)
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

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# FIG 3 — SECTOR DUMMIES
# ==============================================================================

def plot_sectors(results):
    st_dt    = results['st_dt']
    lm_sec   = _collect(results, 'lambda_sec')
    sec_cols = results.get('sec_cols', SECTOR_COLS)

    if lm_sec.empty:
        print("No sector data available.")
        return

    sec_cols = [c for c in sec_cols if c in lm_sec.columns]
    if not sec_cols:
        print("No sector columns found in lambda_sec.")
        return

    colors = plt.cm.tab20.colors
    lm     = lm_sec[lm_sec.index >= st_dt]
    dates  = lm.index.to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Fig 3 — Sector Dummy Factor Returns',
                 fontsize=13, fontweight='bold', y=0.99)

    ax = axes[0]
    for i, col in enumerate(sorted(sec_cols)):
        cum = lm[col].fillna(0).cumsum().to_numpy()
        ax.plot(dates, cum, label=col, color=colors[i % len(colors)], linewidth=1.0)
    _style(ax, 'All Sectors — Cumulative Lambda', legend_cols=4)

    ax  = axes[1]
    mat = np.array([lm[c].fillna(0).cumsum().to_numpy() for c in sec_cols])
    spread = mat.max(axis=0) - mat.min(axis=0)
    ax.plot(dates, spread, color='purple', linewidth=1.2, label='Best − Worst sector')
    ax.fill_between(dates, spread, 0, alpha=0.15, color='purple')
    _style(ax, 'Cross-Sector Spread (Best − Worst)')

    plt.tight_layout()
    plt.show()


# ==============================================================================
# FIG 4 — ALPHA FACTORS
# ==============================================================================

def plot_alpha(results):
    st_dt      = results['st_dt']
    lm_quality = _collect(results, 'lambda_quality')
    lm_si      = _collect(results, 'lambda_si')
    lm_vol     = _collect(results, 'lambda_vol')
    lm_mom     = _collect(results, 'lambda_mom')
    lm_joint   = _collect(results, 'lambda_joint')
    lm_ou      = _collect(results, 'lambda_ou')

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=True)
    fig.suptitle('Fig 4 — Alpha Factor Returns (Steps 6-11)',
                 fontsize=13, fontweight='bold', y=0.99)

    specs = [
        (lm_quality, 'quality',      'Step 6: Quality',                    'darkorchid',  axes[0, 0]),
        (lm_si,      'si_composite', 'Step 7: SI Composite',               'crimson',     axes[0, 1]),
        (lm_vol,     'vol',          'Step 8: GK Volatility',              'coral',       axes[1, 0]),
        (lm_mom,     'idio_mom',     'Step 9: Idio Momentum (on vol resid)','darkorange',  axes[1, 1]),
        (lm_joint,   'value',        'Step 10: Value',                      'teal',        axes[2, 0]),
        (lm_ou,      'ou_reversion', 'Step 11: O-U Mean Reversion',         'navy',        axes[2, 1]),
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
# FIG 5 — ROLLING 252D T-STATS
# ==============================================================================

def plot_rolling_tstats(results, window=252):
    st_dt      = results['st_dt']
    lm_quality = _collect(results, 'lambda_quality')
    lm_mom     = _collect(results, 'lambda_mom')
    lm_joint   = _collect(results, 'lambda_joint')
    lm_ou      = _collect(results, 'lambda_ou')
    lm_macro   = _collect(results, 'lambda_macro')
    macro_cols = results.get('macro_cols', MACRO_COLS)

    def rolling_tstat(ldf, col):
        if ldf.empty or col not in ldf.columns:
            return pd.Series(dtype=float)
        s  = ldf[col].dropna()
        rm = s.rolling(window).mean()
        rs = s.rolling(window).std()
        rn = s.rolling(window).count()
        return (rm / (rs / np.sqrt(rn))).reindex(s.index)

    lm_si  = _collect(results, 'lambda_si')
    lm_vol = _collect(results, 'lambda_vol')

    alpha_specs = [
        (lm_quality, 'quality',      'Quality',    'darkorchid'),
        (lm_si,      'si_composite', 'SI',         'crimson'),
        (lm_vol,     'vol',          'GK Vol',     'coral'),
        (lm_mom,     'idio_mom',     'Idio Mom',   'darkorange'),
        (lm_joint,   'value',        'Value',      'teal'),
        (lm_ou,      'ou_reversion', 'O-U',        'navy'),
    ]
    macro_specs = [(lm_macro, col, col, f'C{i}') for i, col in enumerate(macro_cols)]

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
        ax.axhline( 2.0, color='green', linewidth=0.8, linestyle=':', alpha=0.7, label='+2')
        ax.axhline(-2.0, color='red',   linewidth=0.8, linestyle=':', alpha=0.7, label='−2')
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
# FIG 6 — OPTIMAL RIDGE λ OVER TIME
# ==============================================================================

def plot_ridge_lambdas(results):
    st_dt    = results['st_dt']
    lm_macro = _collect(results, 'lambda_macro')
    lm_mom   = _collect(results, 'lambda_mom')

    has_macro = not lm_macro.empty and 'ridge_lambda' in lm_macro.columns

    if not has_macro:
        print("No ridge_lambda data available (macro step only).")
        return

    fig, axes = plt.subplots(1, 1, figsize=(14, 5))
    fig.suptitle('Fig 6 — Optimal Ridge λ Over Time (5-fold CV, Step 4: Macro)',
                 fontsize=13, fontweight='bold', y=1.01)

    s  = lm_macro[lm_macro.index >= st_dt]['ridge_lambda'].dropna()
    axes.plot(s.index.to_numpy(), s.to_numpy(),
              color='steelblue', linewidth=0.8, alpha=0.5, label='Chosen λ')
    rm = s.rolling(63).mean()
    axes.plot(rm.index.to_numpy(), rm.to_numpy(),
              color='steelblue', linewidth=2.0, label='63d rolling mean')
    axes.axhline(s.median(), color='grey', linewidth=0.8,
                 linestyle='--', label=f'Median={s.median():.2f}')
    axes.set_title('Step 4: Macro Factors (default=0.5)', fontsize=11, fontweight='bold')
    axes.set_ylabel('Ridge λ', fontsize=9)
    axes.legend(fontsize=8, loc='upper left')
    axes.grid(True, alpha=0.3)
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.show()


# ==============================================================================
# PLOT ALL
# ==============================================================================

def plot_all(results, rolling_window=252):
    """
    Plot all 6 figures. Accepts either:
      - results dict from run()
      - dict loaded from DB via load_lambdas_from_db()
    """
    plot_structural(results)
    plot_macro(results)
    plot_sectors(results)
    plot_alpha(results)
    plot_rolling_tstats(results, window=rolling_window)
    plot_ridge_lambdas(results)


if __name__ == '__main__':
    print("Usage (from run results):")
    print("    from plot_factor_returns import plot_all")
    print("    plot_all(results)")
    print("")
    print("Usage (from DB, e.g. after incremental update):")
    print("    from plot_factor_returns import load_lambdas_from_db, plot_all")
    print("    results = load_lambdas_from_db('2019-01-01')")
    print("    plot_all(results)")
