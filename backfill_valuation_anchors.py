"""
backfill_valuation_anchors.py
─────────────────────────────
Backfills valuation_metrics_anchors for monthly dates from a user-supplied
start date up to (but NOT including) the first date already in the table.

Usage (in notebook):
    from backfill_valuation_anchors import run_backfill
    run_backfill(Pxs_df, start_date='2017-01-01')

Or run as a script:
    python backfill_valuation_anchors.py
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

DB_NAME = "factormodel_db"
ENGINE  = create_engine(f"postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/{DB_NAME}")
ANCHOR_TABLE = 'valuation_metrics_anchors'


# ── Import metric functions from calc_val_metrics ────────────────────────────
# These must already be loaded in the notebook session, or imported here.
# The functions needed: calculate_all_metrics_for_stock, normalize_ticker
try:
    from calc_val_metrics import (
        calculate_all_metrics_for_stock,
        normalize_ticker,
    )
except ImportError:
    # If running in notebook where these are already defined, they're in scope
    pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_first_existing_anchor():
    """Return the earliest date already in valuation_metrics_anchors."""
    with ENGINE.connect() as conn:
        row = conn.execute(text(
            f"SELECT MIN(date) FROM {ANCHOR_TABLE}"
        )).fetchone()
    return pd.Timestamp(row[0]) if row and row[0] else None


def _get_existing_anchor_dates():
    """Return set of all dates already in valuation_metrics_anchors."""
    with ENGINE.connect() as conn:
        rows = conn.execute(text(
            f"SELECT DISTINCT date FROM {ANCHOR_TABLE}"
        )).fetchall()
    return {pd.Timestamp(r[0]) for r in rows}


def _first_trading_days(Pxs_df, start_date, end_date_exclusive):
    """
    Return the first available trading day in each calendar month
    from start_date up to (but not including) end_date_exclusive.
    """
    px_dates = Pxs_df.index
    px_dates = px_dates[(px_dates >= start_date) & (px_dates < end_date_exclusive)]
    if px_dates.empty:
        return pd.DatetimeIndex([])

    months = px_dates.to_period('M').unique()
    result = []
    for m in months:
        days_in_month = px_dates[px_dates.to_period('M') == m]
        if not days_in_month.empty:
            result.append(days_in_month[0])
    return pd.DatetimeIndex(sorted(result))


def _save_anchor_only(metrics_df):
    """
    Save metrics to valuation_metrics_anchors only.
    Uses INSERT ... ON CONFLICT DO NOTHING to never overwrite existing data.
    """
    def sanitise(col):
        return col.replace(' ', '_').replace('/', '_').replace('&', '_').replace('-', '_')

    saved = 0
    for _, row in metrics_df.iterrows():
        metric_cols = [col for col in metrics_df.columns
                       if col not in ['date', 'ticker'] and pd.notna(row[col])]
        if not metric_cols:
            continue

        all_cols     = ['date', 'ticker'] + metric_cols
        columns_str  = ', '.join([f'"{col}"' for col in all_cols])
        placeholders = ', '.join([f':{sanitise(col)}' for col in all_cols])

        upsert_query = text(f"""
            INSERT INTO {ANCHOR_TABLE} ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT (date, ticker) DO NOTHING
        """)

        row_dict = {sanitise(col): (None if pd.isna(row[col]) else row[col])
                    for col in all_cols}
        with ENGINE.begin() as conn:
            conn.execute(upsert_query, row_dict)
        saved += 1

    calc_date = metrics_df['date'].iloc[0]
    print(f"  Saved {saved} rows for {calc_date.date()} → {ANCHOR_TABLE}")


# ── Main backfill function ────────────────────────────────────────────────────

def run_backfill(Pxs_df, start_date=None):
    """
    Backfill valuation_metrics_anchors with monthly snapshots.

    Parameters
    ----------
    Pxs_df     : pd.DataFrame  price data (dates × tickers)
    start_date : str or Timestamp, e.g. '2017-01-01'
                 If None, prompts the user.
    """
    print("=" * 72)
    print("  BACKFILL: valuation_metrics_anchors")
    print("=" * 72)

    # ── 1. Determine start date ───────────────────────────────────────────────
    if start_date is None:
        user_input = input("  Enter start date (YYYY-MM-DD): ").strip()
        start_date = pd.Timestamp(user_input)
    else:
        start_date = pd.Timestamp(start_date)

    # ── 2. Determine stop date ────────────────────────────────────────────────
    first_existing = _get_first_existing_anchor()
    if first_existing is None:
        print("  WARNING: valuation_metrics_anchors is empty — will run all dates up to today.")
        stop_date = pd.Timestamp.today()
    else:
        stop_date = first_existing
        print(f"  First existing anchor date : {first_existing.date()}")

    print(f"  Backfill range             : {start_date.date()} → {stop_date.date()} (exclusive)")

    # ── 3. Generate target dates ──────────────────────────────────────────────
    target_dates = _first_trading_days(Pxs_df, start_date, stop_date)
    if target_dates.empty:
        print("  No dates to compute in this range. Exiting.")
        return

    # Skip dates already computed (in case of partial reruns)
    existing = _get_existing_anchor_dates()
    target_dates = pd.DatetimeIndex([d for d in target_dates if d not in existing])

    print(f"  Dates to compute           : {len(target_dates)}")
    for d in target_dates:
        print(f"    {d.date()}")

    confirm = input("\n  Proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("  Cancelled.")
        return

    # ── 4. Run for each date ──────────────────────────────────────────────────
    for i, calc_date in enumerate(target_dates, 1):
        print(f"\n  [{i}/{len(target_dates)}] {calc_date.date()} ", end='', flush=True)

        # Tickers with price data on this date
        tickers = Pxs_df.loc[calc_date].dropna().index.tolist()
        # Exclude non-stock columns (SPX, sector ETFs etc.)
        tickers = [t for t in tickers if ' ' not in t or t.endswith(' US')]
        print(f"({len(tickers)} tickers)")

        all_results = []
        errors      = 0
        for j, ticker in enumerate(tickers, 1):
            if j % 100 == 0:
                print(f"    {j}/{len(tickers)}...", flush=True)
            try:
                r = calculate_all_metrics_for_stock(ticker, calc_date, Pxs_df)
                all_results.append(r)
            except Exception as e:
                errors += 1

        if not all_results:
            print(f"  WARNING: no results for {calc_date.date()}, skipping.")
            continue

        metrics_df = pd.DataFrame(all_results)
        _save_anchor_only(metrics_df)
        print(f"  {len(all_results)} stocks saved ({errors} errors)")

    print("\n" + "=" * 72)
    print(f"  Backfill complete. {len(target_dates)} monthly snapshots added.")
    print("=" * 72)


if __name__ == '__main__':
    print("Run via notebook: from backfill_valuation_anchors import run_backfill")
    print("Then call: run_backfill(Pxs_df, start_date='2017-01-01')")
