"""
plot_factor_returns.py
======================
Visualizes cumulative Fama-MacBeth lambda returns for all factors.
Supports both the original model (v1, unversioned tables) and v2
(v2_* prefixed tables) via the model_version parameter.

Organized into 6 figures:
  Fig 1 — Market & Structural   : intercept, beta, size
  Fig 2 — Macro Factors         : rates, spread, breakevens, crude, gold
  Fig 3 — Sector Dummies        : all sectors
  Fig 4 — Alpha Factors         : quality, SI, vol, idio_mom, value, O-U
  Fig 5 — Rolling 252d t-stats  : statistical significance over time
  Fig 6 — Optimal Ridge λ       : chosen λ over time for macro and sectors

Usage — from results dict (full recalculation):
    from plot_factor_returns import plot_all
    plot_all(results)                          # v1 (default)
    plot_all(results, model_version='v2')      # v2

Usage — from DB (incremental or standalone):
    from plot_factor_returns import load_lambdas_from_db, plot_all
    results_v1 = load_lambdas_from_db(st_date='2019-01-01')
    results_v2 = load_lambdas_from_db(st_date='2019-01-01', model_version='v2')
    plot_all(results_v1)
    plot_all(results_v2, model_version='v2')
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
    'USGG2YR', 'US10Y2Y_SPREAD_CHG', 'US10YREAL',
    'BE5Y5YFWD', 'MOVE', 'Crude', 'XAUUSD'
]
SECTOR_COLS = pd.unique(sectors_s).tolist()


# ===============================================================================
# TABLE NAME RESOLUTION
# ===============================================================================

# v1: unversioned table names (original model)
_V1_TABLES = {
    'lambda_mkt':     'factor_lambdas_mkt',
    'lambda_size':    'factor_lambdas_size',
    'lambda_macro':   'factor_lambdas_macro',
    'lambda_sec':     'factor_lambdas_sec',
    'lambda_quality': 'factor_lambdas_quality',
    'lambda_si':      'factor_lambdas_si',
    'lambda_vol':     'factor_lambdas_vol',
    'lambda_mom':     'factor_lambdas_mom',
    'lambda_joint':   'factor_lambdas_joint',   # value step in v1
    'lambda_ou':      'factor_lambdas_ou',
}

# v2: prefixed table names, value step renamed
_V2_TABLES = {
    'lambda_mkt':     'v2_factor_lambdas_mkt',
    'lambda_size':    'v2_factor_lambdas_size',
    'lambda_macro':   'v2_factor_lambdas_macro',
    'lambda_sec':     'v2_factor_lambdas_sec',
    'lambda_quality': 'v2_factor_lambdas_quality',
    'lambda_si':      'v2_factor_lambdas_si',
    'lambda_vol':     'v2_factor_lambdas_vol',
    'lambda_mom':     'v2_factor_lambdas_mom',
    'lambda_joint':   'v2_factor_lambdas_value',  # value step in v2
    'lambda_ou':      'v2_factor_lambdas_ou',
}

# Step labels differ between v1 and v2
_V1_ALPHA_LABELS = {
    'quality':      'Step 6: Quality',
    'si_composite': 'Step 7: SI Composite',
    'vol':          'Step 8: GK Volatility',
    'idio_mom':     'Step 9: Idio Momentum (on vol resid)',
    'value':        'Step 10: Value',
    'ou_reversion': 'Step 11: O-U Mean Reversion',
}

_V2_ALPHA_LABELS = {
    'quality':      'Step 3: Quality',
    'si_composite': 'Step 7: SI Composite',
    'vol':          'Step 8: GK Volatility',
    'idio_mom':     'Step 4: Idio Momentum (on quality resid)',
    'value':        'Step 6: Value',
    'ou_reversion': 'Step 11: O-U Mean Reversion',
}

_V1_RIDGE_LABELS = {
    'macro': 'Step 4: Macro Factors — Ridge λ  (grid floor=0.15)',
    'sec':   'Step 5: Sector Dummies — Ridge λ  (grid floor=0.10, sum-to-zero)',
}

_V2_RIDGE_LABELS = {
    'macro': 'Step 9: Macro Factors — Ridge λ  (grid floor=0.15)',
    'sec':   'Step 10: Sector Dummies — Ridge λ  (grid floor=0.10, sum-to-zero)',
}

def _get_tables(model_version):
    return _V2_TABLES if model_version == 'v2' else _V1_TABLES

def _get_alpha_labels(model_version):
    return _V2_ALPHA_LABELS if model_version == 'v2' else _V1_ALPHA_LABELS

def _get_ridge_labels(model_version):
    return _V2_RIDGE_LABELS if model_version == 'v2' else _V1_RIDGE_LABELS

def _version_tag(model_version):
    return ' [v2]' if model_version == 'v2' else ' [v1]'


# ===============================================================================
# DB LOADER
# ===============================================================================

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


def load_lambdas_from_db(st_date='2019-01-01', model_version='v1'):
    """
    Load all lambda tables from DB and return a results-compatible dict.

    Parameters
    ----------
    st_date       : str  — start date for loading (default '2019-01-01')
    model_version : str  — 'v1' (default) or 'v2'
    """
    tag    = _version_tag(model_version)
    tables = _get_tables(model_version)

    print(f"  Loading lambda tables from DB{tag} (from {st_date})...")
    results = {}
    for key, tbl in tables.items():
        df = _load_table(tbl, st_date)
        results[key] = df
        if not df.empty:
            print(f"    {tbl}: {len(df)} dates")

    results['st_dt']         = pd.Timestamp(st_date)
    results['model_version'] = model_version
    results['macro_cols']    = [
        c for c in MACRO_COLS
        if c in results.get('lambda_macro', pd.DataFrame()).columns
    ]
    results['sec_cols'] = [
        c for c in SECTOR_COLS
        if c in results.get('lambda_sec', pd.DataFrame()).columns
    ]
    print("  Done.")
    return results


# ===============================================================================
# HELPERS
# ===============================================================================

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


def _mv(results):
    """Extract model_version from results dict, default to v1."""
    return results.get('model_version', 'v1')


# ===============================================================================
# FIG 1 — MARKET & STRUCTURAL
# ===============================================================================

def plot_structural(results):
    st_dt   = results['st_dt']
    mv      = _mv(results)
    lm_mkt  = _collect(results, 'lambda_mkt')
    lm_size = _collect(results, 'lambda_size')

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Fig 1 — Market & Structural Factor Returns{_version_tag(mv)}',
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


# ===============================================================================
# FIG 2 — MACRO FACTORS
# ===============================================================================

def plot_macro(results):
    st_dt      = results['st_dt']
    mv         = _mv(results)
    lm_macro   = _collect(results, 'lambda_macro')
    macro_cols = results.get('macro_cols', MACRO_COLS)

    if lm_macro.empty or not macro_cols:
        print("No macro factor data available.")
        return

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    labels = {
        'USGG2YR':            '2Y Rate',
        'US10Y2Y_SPREAD_CHG': '2Y/10Y Spread',
        'US10YREAL':          '10Y Real Yield',
        'BE5Y5YFWD':          '5y5y Fwd Breakeven',
        'MOVE':               'MOVE (Rate Vol)',
        'Crude':              'WTI Crude',
        'XAUUSD':             'Gold',
    }

    n     = len(macro_cols)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4 * nrows), sharex=True)
    axes = np.array(axes).flatten()

    # Step number differs between versions
    step_num = '9' if mv == 'v2' else '4'
    fig.suptitle(
        f'Fig 2 — Macro Factor Returns (Step {step_num}, EWMA Betas){_version_tag(mv)}',
        fontsize=13, fontweight='bold', y=0.99
    )

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


# ===============================================================================
# FIG 3 — SECTOR DUMMIES
# ===============================================================================

def plot_sectors(results):
    st_dt    = results['st_dt']
    mv       = _mv(results)
    lm_sec   = _collect(results, 'lambda_sec')
    sec_cols = results.get('sec_cols', SECTOR_COLS)

    if lm_sec.empty:
        print("No sector data available.")
        return

    sec_cols = [c for c in sec_cols if c in lm_sec.columns]
    if not sec_cols:
        print("No sector columns found in lambda_sec.")
        return

    step_num = '10' if mv == 'v2' else '5'
    colors   = plt.cm.tab20.colors
    lm       = lm_sec[lm_sec.index >= st_dt]
    dates    = lm.index.to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f'Fig 3 — Sector Dummy Factor Returns (Step {step_num}){_version_tag(mv)}',
        fontsize=13, fontweight='bold', y=0.99
    )

    ax = axes[0]
    for i, col in enumerate(sorted(sec_cols)):
        cum = lm[col].fillna(0).cumsum().to_numpy()
        ax.plot(dates, cum, label=col,
                color=colors[i % len(colors)], linewidth=1.0)
    _style(ax, 'All Sectors — Cumulative Lambda', legend_cols=4)

    ax  = axes[1]
    mat = np.array([lm[c].fillna(0).cumsum().to_numpy() for c in sec_cols])
    spread = mat.max(axis=0) - mat.min(axis=0)
    ax.plot(dates, spread, color='purple', linewidth=1.2,
            label='Best − Worst sector')
    ax.fill_between(dates, spread, 0, alpha=0.15, color='purple')
    _style(ax, 'Cross-Sector Spread (Best − Worst)')

    plt.tight_layout()
    plt.show()


# ===============================================================================
# FIG 4 — ALPHA FACTORS
# ===============================================================================

def plot_alpha(results):
    st_dt      = results['st_dt']
    mv         = _mv(results)
    alpha_lbl  = _get_alpha_labels(mv)

    lm_quality = _collect(results, 'lambda_quality')
    lm_si      = _collect(results, 'lambda_si')
    lm_vol     = _collect(results, 'lambda_vol')
    lm_mom     = _collect(results, 'lambda_mom')
    lm_joint   = _collect(results, 'lambda_joint')
    lm_ou      = _collect(results, 'lambda_ou')

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=True)
    fig.suptitle(
        f'Fig 4 — Alpha Factor Returns{_version_tag(mv)}',
        fontsize=13, fontweight='bold', y=0.99
    )

    specs = [
        (lm_quality, 'quality',      alpha_lbl['quality'],      'darkorchid', axes[0, 0]),
        (lm_si,      'si_composite', alpha_lbl['si_composite'], 'crimson',    axes[0, 1]),
        (lm_vol,     'vol',          alpha_lbl['vol'],          'coral',      axes[1, 0]),
        (lm_mom,     'idio_mom',     alpha_lbl['idio_mom'],     'darkorange', axes[1, 1]),
        (lm_joint,   'value',        alpha_lbl['value'],        'teal',       axes[2, 0]),
        (lm_ou,      'ou_reversion', alpha_lbl['ou_reversion'], 'navy',       axes[2, 1]),
    ]

    for ldf, col, title, color, ax in specs:
        d, c = _prep(ldf, col, st_dt)
        if d is not None:
            ax.plot(d, c, color=color, linewidth=1.2, label=title)
            _fill(ax, d, c, color)
        _style(ax, title)

    plt.tight_layout()
    plt.show()


# ===============================================================================
# FIG 5 — ROLLING 252D T-STATS
# ===============================================================================

def plot_rolling_tstats(results, window=252):
    st_dt      = results['st_dt']
    mv         = _mv(results)
    alpha_lbl  = _get_alpha_labels(mv)
    macro_cols = results.get('macro_cols', MACRO_COLS)

    lm_quality = _collect(results, 'lambda_quality')
    lm_si      = _collect(results, 'lambda_si')
    lm_vol     = _collect(results, 'lambda_vol')
    lm_mom     = _collect(results, 'lambda_mom')
    lm_joint   = _collect(results, 'lambda_joint')
    lm_ou      = _collect(results, 'lambda_ou')
    lm_macro   = _collect(results, 'lambda_macro')

    def rolling_tstat(ldf, col):
        if ldf.empty or col not in ldf.columns:
            return pd.Series(dtype=float)
        s  = ldf[col].dropna()
        rm = s.rolling(window).mean()
        rs = s.rolling(window).std()
        rn = s.rolling(window).count()
        return (rm / (rs / np.sqrt(rn))).reindex(s.index)

    alpha_specs = [
        (lm_quality, 'quality',      alpha_lbl['quality'],      'darkorchid'),
        (lm_si,      'si_composite', alpha_lbl['si_composite'], 'crimson'),
        (lm_vol,     'vol',          alpha_lbl['vol'],          'coral'),
        (lm_mom,     'idio_mom',     alpha_lbl['idio_mom'],     'darkorange'),
        (lm_joint,   'value',        alpha_lbl['value'],        'teal'),
        (lm_ou,      'ou_reversion', alpha_lbl['ou_reversion'], 'navy'),
    ]
    macro_specs = [
        (lm_macro, col, col, f'C{i}')
        for i, col in enumerate(macro_cols)
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f'Fig 5 — Rolling {window}d t-statistics{_version_tag(mv)}',
        fontsize=13, fontweight='bold', y=0.99
    )

    for ax, specs, title in zip(
        axes,
        [alpha_specs, macro_specs],
        ['Alpha Factors', 'Macro Factors']
    ):
        for ldf, col, label, color in specs:
            ts = rolling_tstat(ldf, col)
            ts = ts[ts.index >= st_dt]
            if ts.empty:
                continue
            ax.plot(ts.index.to_numpy(), ts.to_numpy(),
                    label=label, color=color, linewidth=1.0)
        ax.axhline( 2.0, color='green', linewidth=0.8,
                    linestyle=':', alpha=0.7, label='+2')
        ax.axhline(-2.0, color='red',   linewidth=0.8,
                    linestyle=':', alpha=0.7, label='−2')
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


# ===============================================================================
# FIG 6 — OPTIMAL RIDGE λ OVER TIME
# ===============================================================================

def plot_ridge_lambdas(results):
    st_dt      = results['st_dt']
    mv         = _mv(results)
    ridge_lbl  = _get_ridge_labels(mv)
    lm_macro   = _collect(results, 'lambda_macro')
    lm_sec     = _collect(results, 'lambda_sec')

    has_macro = not lm_macro.empty and 'ridge_lambda' in lm_macro.columns
    has_sec   = not lm_sec.empty   and 'ridge_lambda' in lm_sec.columns

    if not has_macro and not has_sec:
        print("No ridge_lambda data available.")
        return

    nrows = int(has_macro) + int(has_sec)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 5 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]
    fig.suptitle(
        f'Fig 6 — Optimal Ridge λ Over Time (5-fold CV per date){_version_tag(mv)}',
        fontsize=13, fontweight='bold', y=1.01
    )

    plot_idx = 0

    if has_macro:
        ax = axes[plot_idx]
        s  = lm_macro[lm_macro.index >= st_dt]['ridge_lambda'].dropna()
        ax.plot(s.index.to_numpy(), s.to_numpy(),
                color='steelblue', linewidth=0.8, alpha=0.5, label='Chosen λ')
        rm = s.rolling(63).mean()
        ax.plot(rm.index.to_numpy(), rm.to_numpy(),
                color='steelblue', linewidth=2.0, label='63d rolling mean')
        ax.axhline(s.median(), color='grey', linewidth=0.8,
                   linestyle='--', label=f'Median={s.median():.2f}')
        ax.set_title(ridge_lbl['macro'], fontsize=11, fontweight='bold')
        ax.set_ylabel('Ridge λ', fontsize=9)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plot_idx += 1

    if has_sec:
        ax = axes[plot_idx]
        s  = lm_sec[lm_sec.index >= st_dt]['ridge_lambda'].dropna()
        ax.plot(s.index.to_numpy(), s.to_numpy(),
                color='darkorange', linewidth=0.8, alpha=0.5, label='Chosen λ')
        rm = s.rolling(63).mean()
        ax.plot(rm.index.to_numpy(), rm.to_numpy(),
                color='darkorange', linewidth=2.0, label='63d rolling mean')
        ax.axhline(s.median(), color='grey', linewidth=0.8,
                   linestyle='--', label=f'Median={s.median():.2f}')
        high_reg = s[s >= 10.0]
        for dt in high_reg.index:
            ax.axvspan(dt, dt, color='red', alpha=0.15)
        ax.set_title(ridge_lbl['sec'], fontsize=11, fontweight='bold')
        ax.set_ylabel('Ridge λ', fontsize=9)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.show()


# ===============================================================================
# PLOT ALL
# ===============================================================================

def plot_all(results, model_version=None, rolling_window=252):
    """
    Plot all 6 figures.

    Parameters
    ----------
    results       : dict from run() or load_lambdas_from_db()
    model_version : 'v1' or 'v2'. If None, inferred from results dict
                    (falls back to 'v1' if not set).
    rolling_window: int, window for rolling t-stat (default 252)
    """
    # Allow caller to override model_version, otherwise read from results
    if model_version is not None:
        results = dict(results)   # shallow copy — don't mutate caller's dict
        results['model_version'] = model_version

    plot_structural(results)
    plot_macro(results)
    plot_sectors(results)
    plot_alpha(results)
    plot_rolling_tstats(results, window=rolling_window)
    plot_ridge_lambdas(results)


if __name__ == '__main__':
    print("Usage (from run results):")
    print("    from plot_factor_returns import plot_all")
    print("    plot_all(results)                      # v1")
    print("    plot_all(results, model_version='v2')  # v2")
    print("")
    print("Usage (from DB, e.g. after incremental update):")
    print("    from plot_factor_returns import load_lambdas_from_db, plot_all")
    print("    results_v1 = load_lambdas_from_db('2019-01-01')")
    print("    results_v2 = load_lambdas_from_db('2019-01-01', model_version='v2')")
    print("    plot_all(results_v1)")
    print("    plot_all(results_v2)")
    try:
        plot_all(results)
    except Exception:
        results = load_lambdas_from_db('2019-01-01')
        plot_all(results)
