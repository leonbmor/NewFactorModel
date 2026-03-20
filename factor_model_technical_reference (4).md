# Factor Model — Step 1: Technical Reference

**File:** `factor_model_step1.py`  
**Last updated:** March 2026  
**Database:** `postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db`

---

## Overview

Sequential Fama-MacBeth cross-sectional factor residualization across **11 steps**. At each step, a WLS regression is run cross-sectionally on every trading date. The residuals from each step become the input returns (target) for the next step.

### True Gram-Schmidt Orthogonalization

Both the **return residuals** and the **characteristics themselves** are orthogonalized at each step. Before entering any regression, each new characteristic is regressed cross-sectionally against **all prior characteristics** (full cross-section OLS), and only the residual — the component of the characteristic unexplained by everything that came before — enters the regression.

This ensures:
- Each step's lambda measures the **pure marginal contribution** of that factor, net of all prior factors
- Factor lambdas are non-redundant and non-overlapping
- Final residuals have zero projection onto any prior characteristic — regressing `factor_residuals_ou` against the original market beta would yield ~0

Without characteristic orthogonalization, sequential residualization of returns alone is insufficient: correlated characteristics contaminate each step's residuals with prior-factor information.

**Characteristic orthogonalization function:** `orthogonalize_char_df(char_df, prior_chars, calc_dates)` — runs cross-sectional OLS of the new characteristic on all prior characteristics and returns the residuals.

**Regression weights:** `log(dynamic_market_cap)`, normalized to sum to 1 within each cross-section.

**Regularization:** Ridge (L2) applied only at **Step 4 (Macro)** — the only step with multiple potentially correlated features (6 macro factors). All other steps use plain OLS (λ=0). Lambda selected per date via **5-fold cross-validation on the stock dimension** — fits on 4 folds, evaluates OOS residual variance on the held-out fold, picks λ minimizing mean OOS variance. Grid: `[0, 0.1, 0.25, 0.75, 1.5, 3.0, 5.0, 10.0]`. Fallback default: λ=0.5.

**Universe:** ~662 US equity stocks present in the fundamentals DB, mapped to a sector ETF, with sufficient price history.

**Run modes:** Full recalculation (complete history) or incremental update (single new date, fast path).

---

## Pre-Processing

### Return Computation
`Pxs_df[universe].pct_change()` using closing prices from `prices_relation`.

### Return Winsorization
Clipped at **±50%** before any regression. ~810 obs (~0.05%) affected.

### Price Spike Filter (upstream, one-off `price_spike_filter.py`)
Flags P[t]/P[t-1] > 3.0 or < 0.2, confirmed spike if P[t+k]/P[t-1] reverts to ±15% within 2 days. NaN'd and forward-filled.

### Dynamic Market Cap
Shares (from `valuation_consolidated`) × daily price. Cached in `dynamic_size_df`.

### Sample Periods
- **Extended start:** 252 trading days before `st_dt` — for beta/momentum lookback
- **`st_dt`:** user-prompted (default/current: 2018-01-01)
- **`common_dates`:** intersection of all dates where every characteristic is available

---

## Step 1 (Baseline): UFV

Pooled variance of raw winsorized returns over `common_dates`. Benchmark for all steps.

---

## Step 2: Market Beta

**Target:** Raw daily returns.  
**Orthogonalization:** None (first step).

**Feature:** EWMA beta vs SPX. Window=252d, hl=126d (strictly `< dt`):
```
β_i = Cov_EWMA(r_i, r_SPX) / Var_EWMA(r_SPX)
```
Entered raw (not z-scored). **OLS (λ=0).**

**Output:** `factor_residuals_mkt`, `factor_lambdas_mkt`

---

## Step 3: Size

**Target:** Step 2 residuals.  
**Orthogonalization:** size ⊥ {beta}

**Feature:** Cross-sectional z-score of log(dynamic_market_cap), residualized vs beta. **OLS (λ=0).**

**Output:** `factor_residuals_size`, `factor_lambdas_size`

---

## Step 4: Macro Factors (Joint)

**Target:** Step 3 residuals.  
**Orthogonalization:** each macro beta ⊥ {beta, size}

**Purpose:** Strip macro sensitivities before sectors so sector dummies capture pure sector effects net of rates, inflation, and commodity exposures.

**Features:** EWMA rolling betas vs 6 macro factor changes (window=252d, hl=126d), each z-scored cross-sectionally per date, then orthogonalized vs beta and size.

| Column | Description |
|--------|-------------|
| `USGG2YR` | 2Y nominal rate daily change (bps) |
| `US10Y2Y_SPREAD_CHG` | 2Y/10Y spread daily change (bps) |
| `US10YREAL` | 10Y real yield / inflation breakeven daily change (was T10YIE) |
| `BE5Y5YFWD` | 5y5y forward breakeven inflation daily change (was T5YIFR) |
| `MOVE` | Interest rate volatility index daily change |
| `Crude` | WTI crude daily pct change |
| `XAUUSD` | Gold daily pct change |

All 6 series provided pre-computed as daily changes in `Pxs_df`.

**Ridge regularization:** per-date 5-fold CV on stock dimension. Grid: `[0, 0.1, 0.25, 0.75, 1.5, 3.0, 5.0, 10.0]`. Default fallback: λ=0.5. Empirical distribution: ~20% at 0, ~20% at ceiling of 10, median ~0.25.

**Output:** `factor_residuals_macro`, `factor_lambdas_macro` (includes `ridge_lambda` column)

---

## Step 5: Sector Dummies

**Target:** Step 4 residuals.  
**Orthogonalization:** each sector dummy ⊥ {beta, size, macro betas}

**Feature:** One-hot encoding of sector ETF membership. 12 sectors (IGV, REZ, SOXX, XHB, XLB, XLC, XLE, XLF, XLI, XLU, XLV, XLY). XLP is the reference sector (dropped). Each dummy orthogonalized vs all prior continuous characteristics. **OLS (λ=0).**

**Output:** `factor_residuals_sec`, `factor_lambdas_sec`

---

## Step 6: Quality Factor

**Target:** Step 5 residuals.  
**Orthogonalization:** quality ⊥ {beta, size, macro betas, sector dummies}

**Feature:** Rate-conditioned composite of fundamental quality metrics:
- **GQF** (Growth Quality Factor): t-stat weighted from non-2021/22 anchor dates
- **CQF** (Conservative Quality Factor): t-stat weighted from 2021/22 dates
- **Rate signal:** `USGG10YR - USGG10YR.rolling(252).mean()`, quantized q∈[0,1]
- **Composite:** `(1-q) × GQF + q × CQF`

Computed at anchor dates (~monthly) from `valuation_metrics_anchors`, forward-filled to daily (with `bfill` for pre-first-anchor dates), z-scored cross-sectionally, then orthogonalized. **OLS (λ=0).**

**Output:** `factor_residuals_quality`, `factor_lambdas_quality`

---

## Step 7: SI Composite

**Target:** Step 6 residuals.  
**Orthogonalization:** SI ⊥ {beta, size, macro betas, sector dummies, quality}

**Feature:** From `short_interest_data`: SI % Free Float + Utilization, each z-scored, equally averaged. Forward-filled (weekly frequency). Cached in `si_composite_df`. Neutral score (~0) for missing stocks.

SI enters first among the alpha factors because it carries genuinely external information (borrowing demand, utilization) with no price-based analog. Stripping it first ensures vol and momentum capture distinct, non-SI-contaminated effects. **OLS (λ=0).**

**Output:** `factor_residuals_si`, `factor_lambdas_si`

---

## Step 8: GK Volatility

**Target:** Step 7 residuals.  
**Orthogonalization:** vol ⊥ {beta, size, macro betas, sector dummies, quality, SI}

**Feature:** Garman-Klass realized volatility. Window=84d, hl=42d. Uses `daily_open`, `daily_high`, `daily_low`:
```
σ²_GK = 0.5·(ln(H/L))² - (2·ln2-1)·(ln(C/O))²
```
~8x more efficient than close-to-close. Falls back to close-to-close if OHLC unavailable. Z-scored cross-sectionally, then orthogonalized. **OLS (λ=0).**

**Output:** `factor_residuals_vol`, `factor_lambdas_vol`

---

## Step 9: Idiosyncratic Momentum

**Target:** Step 8 residuals (GK vol residuals).  
**Orthogonalization:** momentum ⊥ {beta, size, macro betas, sector dummies, quality, SI, vol}

**Feature:** Computed from **GK vol residuals** (`factor_residuals_vol`) — not quality or sector residuals as in prior versions. This gives genuinely idiosyncratic momentum: price drift unexplained by market, size, macro, sectors, quality, short interest, AND realized volatility.

- **Standard:** cumulative sum of vol residuals over `[t-252, t-21]`
- **Volume-scaled:** `Σ(resid_vol_it × vol_scalar_it)` using pre-normalized volume scalars clipped to `[0.5, 3.0]`

Z-scored cross-sectionally, then orthogonalized. **OLS (λ=0).**

**Note on prior version:** momentum was previously computed on quality residuals and entered jointly with vol and SI under Ridge regularization. The prior joint step had high optimal λ (37.8% at ceiling of 50) reflecting structural collinearity among the three. Disaggregating eliminates this entirely.

**Output:** `factor_residuals_mom`, `factor_lambdas_mom`

---

## Step 10: Value Factor

**Target:** Step 9 residuals.  
**Orthogonalization:** value ⊥ {beta, size, macro betas, sector dummies, quality, SI, vol, momentum}

**Feature:** IC-weighted composite of 7 valuation metrics from `valuation_consolidated`: P/S, P/Ee, P/Eo, sP/S, sP/E, sP/GP, P/GP.

**Construction:**
1. Reflexivity treatment within each sector: negatives → `sector_max_positive + abs(metric)`
2. Ranked ascending within sector (low P/x = cheap), inverted so cheap scores HIGH
3. IC-derived weights from avg absolute t-stat (21d+63d horizons vs `factor_residuals_mom`)
4. Forward-filled from valuation dates, z-scored cross-sectionally, orthogonalized

**IC weights derived against `factor_residuals_mom`** — the actual regression target for this step. Refresh via `ic_study.py` after any upstream model change.

**OLS (λ=0).**

**Output:** `factor_residuals_joint` (value residuals, O-U input), `factor_lambdas_joint`

---

## Step 11: O-U Mean Reversion

**Target:** Step 10 residuals (`factor_residuals_joint`).

Mean reversion measured in fully idiosyncratic space, net of all 10 prior factors.

**Residual Price Index:**
```
px_idx[t] = (1 + resid[t]).cumprod() × P[t_0]
```

**Volume Scaling (optional):** residuals divided by volume scalars `[0.5, 3.0]` before fitting.

**AR(1) Fitting** on last `OU_MEANREV_W=60` observations (strictly `< dt`):
```
px_idx[t] = a + b·px_idx[t-1] + ε
```

Valid only if `b ∈ (0,1)` and `m = a/(1-b) > 0`. Parameters:
- `k = -ln(b)` — mean reversion speed
- `T_h = ln(2)/k` — half-life in trading days
- `DistST = (P_current - m) / (σ_resid / √(2k))`
- Factor score: **`-DistST`** (positive = below LT mean = buy)

**ST Reversal Fallback** (invalid fits): `log(cum_resid[t-1] / cum_resid[t-22])` from residual index.

**Weighted Blend:**
```
final_score = (ou_weight × ou_rank + rev_rank) / (ou_weight + 1)
ou_weight   = min(30 / T_h, 10.0)
```

Z-scored cross-sectionally. **OLS (λ=0).**

**Caching:** `ou_reversion_df`. Must be dropped and recomputed whenever any upstream factor changes.

**Output:** `factor_residuals_ou`, `factor_lambdas_ou`

---

## Consolidated Summary

**Consolidated R²:** `1 - ou_UFV / UFV_ou` (both on `ou_common` subsample).

**Consolidated intercept:** sum of all 11 step intercepts per date.

**Daily R²:** `1 - var(ou_residuals[t]) / var(raw_returns[t])` per date cross-section.

---

## DB Tables Summary

| Table | Contents | Step |
|-------|----------|------|
| `factor_residuals_mkt` | After market beta | 2 |
| `factor_residuals_size` | After size | 3 |
| `factor_residuals_macro` | After macro factors | 4 |
| `factor_residuals_sec` | After sectors | 5 |
| `factor_residuals_quality` | After quality | 6 |
| `factor_residuals_si` | After SI composite | 7 |
| `factor_residuals_vol` | After GK vol | 8 |
| `factor_residuals_mom` | After idio momentum | 9 |
| `factor_residuals_joint` | After value (O-U input) | 10 |
| `factor_residuals_ou` | Full model residuals | 11 |
| `factor_lambdas_mkt` | Beta lambda series | 2 |
| `factor_lambdas_size` | Size lambda series | 3 |
| `factor_lambdas_macro` | Macro lambdas + `ridge_lambda` col | 4 |
| `factor_lambdas_sec` | Sector lambda series | 5 |
| `factor_lambdas_quality` | Quality lambda series | 6 |
| `factor_lambdas_si` | SI composite lambda series | 7 |
| `factor_lambdas_vol` | GK vol lambda series | 8 |
| `factor_lambdas_mom` | Idio momentum lambda series | 9 |
| `factor_lambdas_joint` | Value lambda series | 10 |
| `factor_lambdas_ou` | O-U lambda series | 11 |
| `ou_reversion_df` | O-U scores cache | 11 |
| `dynamic_size_df` | Daily market cap | Pre-proc |
| `si_composite_df` | SI composite scores | 7 |

---

## Run Modes

### Full Recalculation (answer `n` to incremental prompt)
Computes all 11 steps over full history from `st_dt`. Required on first run and after any model change. Prompts for start date, volume scaling, and O-U override.

### Incremental Update (answer `y`, default)
Fast single-date update for `Pxs_df.index[-1]`.

**What is recomputed:** betas (market + macro), size char, vol factor, SI (last 60 dates for ffill), quality/value scores, characteristic orthogonalization, single-date regressions through all 11 steps.

**What is loaded from DB:** quality+SI+vol residual history (last 400d) for momentum lookback, value residual history (last 120d) for O-U fitting.

**Save pattern:** upsert — delete existing date then reinsert.

---

## Key Design Principles

**True Gram-Schmidt:** both return residuals and characteristics are orthogonalized. Each characteristic enters its regression already purged of all prior factor information, making variance attribution genuinely additive.

**Ordering rationale:**
- Macro before sectors — rates/commodities stripped so XLE etc. capture pure sector idiosyncrasy
- SI before vol — external information first, price-derived signals after
- Vol before momentum — genuine momentum = price drift beyond realized vol
- Value after momentum — avoids inverse causality
- O-U from value residuals — mean reversion in maximally purified idiosyncratic space

**No look-ahead bias:** all characteristics use strictly `< dt` data.

**Consistent stock universe:** all variance stats computed on `ou_common` stocks for fair comparison.

**Volume scaling consistency:** same flag applied to idio momentum (Step 9) and O-U fitting (Step 11).

**IC weights for value:** derived from IC study against `factor_residuals_mom`. Refresh via `ic_study.py` after any upstream model change.
