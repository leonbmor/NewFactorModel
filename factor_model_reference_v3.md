# Factor Model & Scripts — Comprehensive Reference
*Last updated: March 2026 (v3)*

---

## 1. OVERVIEW

A sequential Fama-MacBeth cross-sectional factor model implemented in Python, running in a Jupyter notebook kernel. The model strips systematic return sources one by one in a true Gram-Schmidt orthogonalization sequence, producing clean residuals at each step. All data is stored in a PostgreSQL database (`factormodel_db`). The universe consists of ~662–679 US large-cap stocks (varies with sector mapping updates).

**Database connection:**
```
postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db
```

**Key principle:** Both return residuals AND characteristics themselves are orthogonalized at each step using WLS (weighted by log market cap). Before entering any regression, each new characteristic is regressed cross-sectionally against all prior characteristics. Only the residual enters the regression. This is true Gram-Schmidt in the weighted inner-product space defined by cap weights.

---

## 2. FACTOR MODEL — 11-STEP ARCHITECTURE

### Step Sequence

```
Step 1:  Baseline UFV          Raw return variance (benchmark)
Step 2:  Market Beta           EWMA beta vs SPX, OLS cross-section
Step 3:  Size                  Z-scored log market cap, OLS; size ⊥ {beta}
Step 4:  Macro Factors         7 factors, joint Ridge (k-fold CV per date); each ⊥ {beta, size}
Step 5:  Sector Dummies        sum-to-zero coding, Ridge CV; each ⊥ {beta, size, macro}
Step 6:  Quality               OLS; quality ⊥ {beta, size, macro, sectors}
Step 7:  SI Composite          OLS; SI ⊥ {beta, size, macro, sectors, quality}
Step 8:  GK Volatility         OLS; vol ⊥ {beta, size, macro, sectors, quality, SI}
Step 9:  Idio Momentum         OLS (on vol residuals); mom ⊥ all prior
Step 10: Value                 OLS; value ⊥ all prior
Step 11: O-U Mean Reversion    OLS; final alpha step
```

### Key Constants

```python
BETA_WINDOW      = 252       # rolling window for beta/macro betas (trading days)
BETA_HL          = 126       # EWMA half-life for beta/macro betas
VOL_WINDOW       = 84        # shorter window for vol factor
VOL_HL           = 42        # EWMA half-life for vol factor
MOM_LONG         = 252       # momentum lookback
MOM_SKIP         = 21        # momentum skip period (avoid reversal contamination)
OU_MEANREV_W     = 60        # O-U AR(1) fitting window
OU_MIN_OBS       = 30        # minimum observations for valid O-U fit
OU_ST_REV_W      = 21        # ST reversal fallback window
OU_WEIGHT_REF    = 30.0      # reference half-life for O-U/reversal blend weight
OU_WEIGHT_CAP    = 10.0      # maximum O-U blend weight
MIN_STOCKS       = 150       # minimum stocks required for cross-sectional regression

RIDGE_GRID_MACRO = [0.15, 0.3, 0.75, 1.5, 3.0, 5.0, 10.0, 20.0, 40.0]
RIDGE_GRID_SEC   = [0.1, 0.2, 0.4, 0.75, 1.5, 3.0, 5.0, 10.0, 20.0, 40.0]
```

### Macro Factors (Step 4)

All are pre-computed daily changes passed in via `Pxs_df` columns:

| Column | Description |
|--------|-------------|
| `USGG2YR` | 2Y nominal rate daily change (bps) |
| `US10Y2Y_SPREAD_CHG` | 10Y-2Y spread daily change (bps) |
| `US10YREAL` | 10Y real yield / inflation breakeven daily change |
| `BE5Y5YFWD` | 5y5y forward breakeven inflation daily change |
| `MOVE` | Interest rate volatility index (MOVE) daily change |
| `Crude` | WTI crude oil daily change |
| `XAUUSD` | Gold daily change |

Each macro beta: `β_im = Cov_EWMA(r_i, d_m) / Var_EWMA(d_m)`, window=252, hl=126, z-scored cross-sectionally. All 7 enter a single joint ridge regression (no natural ordering; ridge handles collinearity).

**Ridge CV (Step 4):** 5-fold CV per date. Grid floor 0.15. λ=0.15–0.30 dominates (~60% of dates); λ=40 on collinear days (~12%). Mean λ ≈ 5.5, median λ ≈ 0.30.

### Sector Dummies (Step 5)

**Sum-to-zero deviation coding:** for K sectors, each dummy = +1 own sector, -1/(K-1) all others. All sectors included — no reference dropped. Intercept = true equal-weighted market return.

**Ridge CV (Step 5):** Grid floor 0.10. ~88.5% of dates select λ=0.10, ~3% select λ=40. Without ridge, sector lambdas showed ±7% artefacts on low-dispersion days.

Note: sector mapping is passed as `sectors_s` input — the number of sectors varies with the mapping (currently 17 sub-sectors after latest update).

### WLS Regression

All cross-sectional regressions: `w_i = log(market_cap_i)`, normalized to sum to 1.

### Characteristic Orthogonalization

```
new_char_perp = new_char - Proj_{prior_chars}(new_char)    [WLS]
```

Falls back to OLS if market cap unavailable. Full-history versions computed over extended date range for momentum lookback chain.

---

## 3. FACTOR DETAILS

### Market Beta (Step 2)
EWMA beta vs SPX. `β_i = Cov_EWMA(r_i, r_SPX) / Var_EWMA(r_SPX)`, window=252, hl=126.

### Size (Step 3)
`size_i = log(shares × price)` — dynamic daily. Cached in `dynamic_size_df`.

### Quality Factor (Step 6)
Rate-conditioned composite. Loaded from `quality_scores_df` cache.
- **GQF** (Growth Quality): non-2021/22 regime
- **CQF** (Conservative Quality): 2021/22 high-rate regime
- Blend: `(1-q)×GQF + q×CQF`, q from USGG10YR vs 252d MAV, threshold=50bps
- Optimal params: `mav_window=252, threshold=50`

```python
GQF_WEIGHTS = {
    'GGP': 0.140405, 'GS': 0.130709, 'GS/S_Vol': 0.118369,
    'GS*r2_S': 0.106796, 'GGP/GP_Vol': 0.097482, 'ROId': 0.090544,
    'GGP*r2_GP': 0.087003, 'FCF_PG': 0.085453, 'HSG': 0.072741, 'PSG': 0.070498,
}
CQF_WEIGHTS = {
    'OM': 0.128893, 'GE/E_Vol': 0.123484, 'ISGD': 0.120801,
    'GE*r2_E': 0.115882, 'OMd*r2_S': 0.113548, 'r&d': 0.091381,
    'LastSGD': 0.083599, 'SGD*r2_S': 0.082127, 'OMd': 0.072732, 'GE': 0.067552,
}
```

ROE/ROEd excluded (price contamination via market cap denominator).

### SI Composite (Step 7)
Short interest composite. Cached in `si_composite_df`.

### GK Volatility (Step 8)
`σ²_GK = 0.5·(ln(H/L))² - (2·ln2-1)·(ln(C/O))²`. Window=84d, hl=42d. Annualized. Positive lambda (t≈+4.1) — volatility risk premium.

### Idiosyncratic Momentum (Step 9)
On `factor_residuals_vol`. Volume-scaled: `mom_i = Σ r_resid × vol_scalar`, clipped [0.5, 3.0]. Window: [t-252, t-21].

### Value Factor (Step 10)
IC-weighted composite from `value_scores_df` cache.
```python
_VALUE_TSTAT = {
    'P/S':   (-4.088 + -6.591) / 2,   # -5.340
    'P/Ee':  (-4.307 + -5.428) / 2,   # -4.867
    'P/Eo':  (-3.975 + -5.550) / 2,   # -4.763
    'sP/S':  (-4.329 + -6.353) / 2,   # -5.341
    'sP/E':  (-3.688 + -5.282) / 2,   # -4.485
    'sP/GP': (-4.817 + -5.861) / 2,   # -5.339
    'P/GP':  (-3.491 + -4.329) / 2,   # -3.910
}
VALUE_WEIGHTS = {m: abs(w) / total for m, w in _VALUE_TSTAT.items()}
```

### O-U Mean Reversion (Step 11)
AR(1) fit to compounded residual price index. `half_life = ln(2)/(-ln(b))`. Requires `0 < b < 1`. Falls back to 21d ST reversal on residual index if fit invalid. Blended: `ou_weight = min(30/half_life, 10)`.

---

## 4. RUN MODES

### Full Recalculation (`n`)
Prompts: start date (default 2018-01-01), volume scaling. Extended start = st_dt - 252 trading days.

### Incremental Update (`y`, default)
Single-date fast path. Loads residual histories from DB, runs one cross-section per step. Upserts to DB.

### Snapshot Display
```
2026-03-23  |  Intercept: -0.09%  |  Daily R²: nan%  |  Macro Ridge λ: 0.30  |  Sec Ridge λ: 0.10
```

---

## 5. DATABASE TABLES

### Residual Tables
| Table | Step |
|-------|------|
| `factor_residuals_mkt` | Step 2 |
| `factor_residuals_size` | Step 3 |
| `factor_residuals_macro` | Step 4 |
| `factor_residuals_sec` | Step 5 |
| `factor_residuals_quality` | Step 6 |
| `factor_residuals_si` | Step 7 |
| `factor_residuals_vol` | Step 8 |
| `factor_residuals_mom` | Step 9 |
| `factor_residuals_joint` | Step 10 |
| `factor_residuals_ou` | Step 11 |

### Lambda Tables
`factor_lambdas_mkt/size/macro/sec/quality/si/vol/mom/joint/ou`

### Characteristic / Score Tables
| Table | Contents |
|-------|----------|
| `dynamic_size_df` | Daily market cap |
| `si_composite_df` | SI composite scores |
| `quality_scores_df` | Quality composite (cached) |
| `value_scores_df` | Value composite (cached) |
| `ou_reversion_df` | O-U scores (cached) |
| `valuation_consolidated` | Raw quarterly fundamentals |
| `valuation_metrics_anchors` | Anchor date snapshots |
| `income_data` | Ortex income fundamentals (ticker, download_date, period, metric_name, value, estimated_values) |
| `summary_data` | Ortex summary fundamentals (same schema, includes ebitda) |
| `estimation_status` | FEQ tracking (ticker, category, first_estimated_period, last_checked) |
| `daily_open/high/low` | OHLC prices |

### Sector Metrics Cache Tables
Produced by `sector_metrics.py`:
- `sector_metric_{metric_tag}` — cap-weighted metric per sector
- `index_metric_{metric_tag}` — cap-weighted metric for SPX/QQQ

Produced by `sector_fundamentals.py`:
- `sector_valuation_{metric}_{basis}` — e.g. `sector_valuation_sales_ltm`
- `sector_growth_{metric}_{basis}` — e.g. `sector_growth_ni_ntm`

---

## 6. LATEST FACTOR PERFORMANCE (March 2026)

| Step | % UFV | % prev |
|------|-------|--------|
| Beta | 71.66% | 71.66% |
| Size | 60.69% | 84.70% |
| Macro | 54.73% | 90.18% |
| Sectors | 51.93% | 94.88% |
| Quality | 51.58% | 99.33% |
| SI | 51.40% | 99.65% |
| GK Vol | 50.79% | 98.81% |
| Idio Mom | 50.16% | 98.75% |
| Value | 50.42% | 100.53%* |
| O-U | 47.61% | 98.95% |

*Value stale — needs ic_study re-run. **Consolidated R² = 52.39%**

| Factor | t-stat |
|--------|--------|
| SI Composite | +5.34 |
| GK Vol | +4.10 |
| Size | +3.79 |
| Quality | +2.73 |
| O-U | +1.47 |
| Idio Mom | +1.63 |
| Value | ~0.00 (stale) |

---

## 7. FEQ (FIRST ESTIMATED QUARTER) MAPPING PROCEDURE

This procedure is used by `sector_fundamentals.py` whenever a back-date calculation is needed and the correct quarter alignment must be determined. It is critical for any script that reads from `income_data` or `summary_data` for historical dates.

### Background
The Ortex fundamentals DB stores data with `download_date` (the date data was fetched) and `period` (the fiscal quarter, e.g. `2025Q3`). The `estimation_status` table records the `first_estimated_period` (FEQ) — the first quarter that was estimated (not yet reported) as of the most recent download.

### Algorithm

**Step 1 — Anchor from today's FEQ:**
```python
current_feq, last_checked = get from estimation_status WHERE ticker=t AND category='income'
# e.g. current_feq='2026Q1', last_checked=2026-02-15
```

**Step 2 — Find straddling download dates for the back-date:**
```python
update_before = latest  download_date <= calc_date   # last snapshot before calc_date
update_after  = earliest download_date > calc_date   # first snapshot after calc_date
# If update_before doesn't exist: skip this stock/date entirely
```

**Step 3 — Estimate past FEQ by walking back:**
```python
days_delta    = (last_checked - update_before).days
quarters_back = round(days_delta / 90)
est_past_feq  = add_quarters(current_feq, -quarters_back)
# e.g. last_checked=Feb-15-2026, update_before=Jul-10-2025 → 220 days → 2 quarters
# est_past_feq = 2026Q1 - 2 = 2025Q3
```

**Step 4 — Verify by comparing totalRevenues across update dates:**
- Build 4 candidate quarters: `[est_past_feq-1, est_past_feq, est_past_feq+1, est_past_feq+2]`
- For each candidate, compare `totalRevenues` between `update_before` and `update_after`
- The most recent candidate where the value **changed** between the two snapshots is the confirmed past FEQ
- If none changed: fall back to `est_past_feq`
- If `update_after` doesn't exist: trust `est_past_feq` directly

**Step 5 — Use confirmed FEQ:**
- **LTM actuals:** quarters `[feq-4, feq-3, feq-2, feq-1]`
- **NTM estimates:** quarters `[feq, feq+1, feq+2, feq+3]`
- **Shares:** `dilutedAverageShares` at `feq-1` (most recent actual)
- **Prior LTM (for growth):** quarters `[feq-8, feq-7, feq-6, feq-5]`

### Helper Function
```python
def add_quarters(quarter: str, n: int) -> str:
    year, q = int(quarter[:4]), int(quarter[5])
    q += n
    while q > 4: q -= 4; year += 1
    while q < 1: q += 4; year -= 1
    return f"{year}Q{q}"
```

### Key Tables Used
- `estimation_status`: `(ticker, category, first_estimated_period, last_checked)`
- `income_data`: `(ticker, download_date, period, metric_name, value, estimated_values)`
- `summary_data`: same schema, includes `ebitda`

### Important Notes
- All metric queries use `download_date <= calc_date` (forward-fill semantics) to get the most recent snapshot available as of each back-date
- `dilutedAverageShares` and all metrics use the **same FEQ mapping** — critical for consistency
- The verification step (Step 4) uses strict `download_date = update_before` (no ffill) to detect genuine changes between snapshots

---

## 8. SCRIPT: `factor_model_step1.py`

**Location:** `/mnt/user-data/outputs/factor_model_step1.py`
**Entry point:** `run(Pxs_df, sectors_s, volumeTrd_df=None)`

### Key Functions

| Function | Purpose |
|----------|---------|
| `get_universe()` | Filters stocks: in DB + sector-mapped + sufficient history |
| `load_dynamic_size()` | Loads/computes market cap from DB cache |
| `build_sector_dummies()` | Sum-to-zero deviation coding |
| `calc_rolling_betas()` | EWMA beta vs SPX |
| `calc_macro_betas()` | EWMA betas vs 7 macro factors, z-scored |
| `calc_idio_momentum_volscaled()` | Volume-weighted cumulative idio residuals |
| `calc_vol_factor()` | Garman-Klass EWMA vol |
| `wls_cross_section()` | Single-date WLS regression |
| `wls_ridge_cross_section()` | WLS + ridge, 5-fold CV lambda selection |
| `run_factor_step()` | Loops wls_cross_section over all dates |
| `run_factor_step_optimal_ridge()` | Loops with CV per date |
| `orthogonalize_char()` | Single-date WLS characteristic orthogonalization |
| `orthogonalize_char_df()` | Across all dates |
| `_fit_ou_single()` | AR(1) fit to residual price index |
| `_compute_ou_for_dates()` | O-U + ST reversal blend |
| `load_ou_reversion()` | Cache-aware O-U loader |
| `load_quality_scores()` | Loads from quality_scores_df cache |
| `load_value_scores()` | Loads from value_scores_df cache |
| `_run_incremental()` | Single-date fast path |
| `run()` | Master entry point |

**Known namespace issue:** `_load_cached_dates()` in `quality_factor.py` (no args) can be overwritten in the Jupyter kernel if `sector_fundamentals.py` is run first (its version takes `table_name` arg). Fixed in `sector_fundamentals.py` by renaming to `_load_cached_dates_sf()`.

---

## 9. SCRIPT: `quality_factor.py`

**Location:** `/mnt/user-data/outputs/quality_factor.py`
**Entry point:** `run(Pxs_df, sectors_s, mav_window=252, threshold=50)`

### Cache Refresh Workflow
```python
summary, annual, scores, gqf_w, cqf_w = run(Pxs_df, sectors_s,
                                              mav_window=252, threshold=50)
update_cached_weights(gqf_w, cqf_w)
# Copy printed GQF_WEIGHTS / CQF_WEIGHTS into quality_factor.py
```

---

## 10. SCRIPT: `ic_study.py` (Value Factor)

**Location:** `/mnt/user-data/outputs/ic_study.py`
**Entry point:** `run_ic_study(Pxs_df, sectors_s, force_recompute_cache=False)`

Targets `factor_residuals_mom`. Horizons: 21d and 63d.

### Cache Refresh Workflow
```python
ic_ts, ic_summary, ic_annual, weights = run_ic_study(Pxs_df, sectors_s,
                                                       force_recompute_cache=True)
# Copy printed _VALUE_TSTAT into factor_model_step1.py
# Re-run full factor model recalculation
```

---

## 11. SCRIPT: `primary_factor_backtest.py`

**Location:** `/mnt/user-data/outputs/primary_factor_backtest.py`
**Entry point:** `run(Pxs_df, sectors_s, volumeTrd_df=None)`

Long-only, bi-monthly rebalancing, TOP_N=20. Primary factor = quality composite.
Volume scaling available for both 12m1 and idio momentum when `volumeTrd_df` provided.

---

## 12. SCRIPT: `plot_factor_returns.py`

**Location:** `/mnt/user-data/outputs/plot_factor_returns.py`
**Entry point:** `plot_all(results)` or `plot_all(load_lambdas_from_db())`

6 figures: structural, macro (7 factors), sectors, alpha factors, rolling t-stats, ridge λ (macro + sector panels).
`MACRO_COLS` updated to: `['USGG2YR', 'US10Y2Y_SPREAD_CHG', 'US10YREAL', 'BE5Y5YFWD', 'MOVE', 'Crude', 'XAUUSD']`
`SECTOR_COLS` updated to 13 sectors including XLP.

---

## 13. SCRIPT: `sector_fundamentals.py`

**Location:** `/mnt/user-data/outputs/sector_fundamentals.py`
**Entry point:** `run(Pxs_df, sectors_s, override=False, spx_df=None, qqq_df=None)`

Calculates cap-weighted sector and index valuation/growth metrics from Ortex fundamentals DB using FEQ mapping (see Section 7).

### Prompts
1. Metric type: valuation (v) or growth (g)
2. Metric: Sales (s), Net Income (ni), EBITDA (e)
3. Basis: LTM or NTM
4. Lookback period in years

### Cache Tables
`sector_valuation_{metric}_{basis}` and `sector_growth_{metric}_{basis}`, e.g.:
- `sector_valuation_sales_ltm`
- `sector_growth_ni_ntm`

### Key Design
- FEQ mapping resolves correct quarter alignment for each back-date (see Section 7)
- Growth formula: `np.median((num - den) / ((num + den) / 2))` per quarter
- Cap-weighting: `Σ(val × mcap) / Σ(mcap)`, requires MIN_STOCKS=3
- Progress printed as `[date] — sector (n stocks)`
- SPX/QQQ index metrics computed alongside sectors using constituent lists from `spx_df`/`qqq_df`
- Growth rates output as percentages rounded to 2 decimal places

### Namespace Note
Uses `_load_cached_dates_sf()` (not `_load_cached_dates()`) to avoid collision with `quality_factor.py`.

---

## 14. SCRIPT: `sector_metrics.py`

**Location:** `/mnt/user-data/outputs/sector_metrics.py`
**Entry point:** `run(Pxs_df, sectors_s, spx_df=None, qqq_df=None, override=False, source_table='valuation_metrics_anchors')`

Cap-weighted aggregation of pre-computed valuation metrics directly from `valuation_metrics_anchors` or `valuation_consolidated` tables. Much faster than `sector_fundamentals.py` since no FEQ mapping needed — metrics are already computed as point-in-time snapshots.

### Available Metrics (33 total)
```
P/S, P/Ee, P/Eo, OM-t0, OM, OMd, GS, GE, r2 S, r2 E, GGP, r2 GP, Size,
ROI-P, ROI, ROId, ROE-P, ROE, ROEd, sP/S, sP/E, sP/GP, P/GP, S Vol, E Vol,
GP Vol, r&d, HSG, SGD, LastSGD, PIG, PSG, ISGD, FCF_PG
```

### Key Design

**Valuation multiples** (`P/S, P/Ee, P/Eo, sP/S, sP/E, sP/GP, P/GP`): denominator aggregation to handle negative earnings correctly:
```
sector_multiple = Σ(Size_i) / Σ(Size_i / multiple_i)
```
Reverse-engineers implied denominator (earnings/sales) from Size and ratio. Stocks with zero multiple excluded; negative denominators kept (correctly push multiple higher).

**All other metrics**: cap-weighted arithmetic mean after cross-sectional winsorization at `WINSOR_BOUNDS = (0.02, 0.98)`.

**Last-date mkt cap update:**
```
mkt_cap_updated = Size_last_snapshot × (Px_today / Px_last_snapshot)
```

**Time-series outlier filter** (controlled by `FILTER_TS_OUTLIERS = True`):
- Flags dates where `|pct_change| > TS_JUMP_THRESHOLD (0.25)`
- If next date is any closer to the pre-jump level: replace with average of two neighbours
- Applied iteratively — handles consecutive bad dates
- Raw DB cache untouched; filter runs fresh each time

### Cache Tables
`sector_metric_{metric_tag}` and `index_metric_{metric_tag}`, e.g.:
- `sector_metric_p_s`
- `index_metric_gs`

### Global Config
```python
DEFAULT_TABLE        = 'valuation_metrics_anchors'
MIN_STOCKS           = 3
WINSOR_BOUNDS        = (0.02, 0.98)
FILTER_TS_OUTLIERS   = True
TS_JUMP_THRESHOLD    = 0.25
VALUATION_MULTIPLES  = {'P/S', 'P/Ee', 'P/Eo', 'sP/S', 'sP/E', 'sP/GP', 'P/GP'}
```

### Output
Two DataFrames + two-panel figure (sectors top, indexes bottom):
```python
df_sectors, df_indexes, fig = run(Pxs_df, sectors_s, spx_df=spx_df, qqq_df=qqq_df)
```

---

## 15. SCRIPT: `load_sector_metrics.py`

**Location:** `/mnt/user-data/outputs/load_sector_metrics.py`
**Entry point:** `load_all(verbose=True)`

Scans DB for `sector_metric_*` and `index_metric_*` tables (from `sector_metrics.py`), loads all into a library dict.

```python
lib = load_all()
lib['sector']['p_s']        # sector P/S DataFrame
lib['index']['p_s']         # index P/S DataFrame
lib['available']            # list of metric tags
```

---

## 16. SCRIPT: `load_sector_fundamentals.py`

**Location:** `/mnt/user-data/outputs/load_sector_fundamentals.py`
**Entry point:** `load_all(verbose=True)`

Scans DB for `sector_valuation_*` and `sector_growth_*` tables (from `sector_fundamentals.py`), loads all into a nested library dict.

```python
lib = load_all()
lib['valuation']['sales']['ltm']    # P/Sales LTM by sector
lib['growth']['ni']['ntm']          # NI growth NTM by sector
lib['flat']['valuation_sales_ltm']  # same, flat access
lib['available']                    # list of full tags
```

---

## 17. PENDING WORK

### Immediate
1. **Value cache refresh** — `run_ic_study(..., force_recompute_cache=True)` then full recalculation. Value t-stat ~0.00 with stale weights.
2. **Quality cache** — 1966 stale dates with old weights. Run `quality_factor.run(force_recompute=True)` if needed.

### Next Project: Portfolio Risk Decomposition

Goal: given a portfolio with dollar weights, decompose daily dollar variance by factor.

**Key design decisions (agreed):**
- Use **raw (non-orthogonalized)** stock characteristics as factor exposures
- Factor covariance matrix is **sparse but not diagonal** — off-diagonal terms must be estimated

**Structure:**
```
Portfolio variance = w'X · F · X'w + w'Ω·w

where:
  w = dollar weight vector (N×1)
  X = raw characteristic matrix (N×K), one column per factor
  F = factor return covariance matrix (K×K), from lambda time series
  Ω = diagonal idiosyncratic variance matrix (from factor_residuals_ou)
```

---

## 18. NOTES AND CONVENTIONS

- **Ticker format:** bare tickers (no `' US'`). `clean_ticker()` strips suffix.
- **DB writes:** always upsert.
- **Common sample:** intersection of all dates/stocks where every characteristic available.
- **Extended dates:** factor model runs from `st_dt - 252` trading days; variance stats from `st_dt`.
- **O-U cache:** clearing takes ~30 min for 2143 dates × 662 stocks.
- **Jupyter kernel:** all scripts run in same kernel. Namespace collision risk between scripts defining same function names (e.g. `_load_cached_dates`, `run`, `clean_ticker`) — mitigated by using `_sf` suffix in `sector_fundamentals.py`.
- **Volume scalars:** `volumeTrd_df` = pre-computed `vol(t)/mean_vol[t-10, t-1]`, clipped `[0.5, 3.0]`.
- **Size in valuation tables:** stored as market cap in $millions.
- **Ortex data pipeline:** `income_data` and `summary_data` tables. `estimated_values=True` for estimates, `False` for actuals. `normalizedNetIncome` copied to `netIncome` for all estimate rows. Non-GAAP adjustments applied via AlphaVantage EPS on earnings release dates.
