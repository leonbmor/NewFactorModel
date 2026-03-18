# Factor Model — Step 1: Technical Reference

**File:** `factor_model_step1.py`  
**Last updated:** March 2026  
**Database:** `postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db`

---

## Overview

Sequential Fama-MacBeth cross-sectional factor residualization across 9 steps. At each step, a weighted least squares (WLS) regression is run cross-sectionally on every trading date. The **residuals** from each step become the **input returns (target)** for the next step. Each step's regression uses **only its own new factor(s)** as features — prior factors are not repeated, since by construction the residuals used as targets are already orthogonal to all previously included characteristics.

This sequential orthogonalization (Gram-Schmidt approach) ensures each factor's contribution is measured **net of all factors added before it**, providing clean, non-redundant variance attribution.

**Regression weights:** `log(dynamic_market_cap)`, normalized to sum to 1 within each cross-section.

**Regularization:** Ridge (L2) with penalty λ (user-prompted, default=0.1). Applied **only at Step 4 (Macro)** and **Step 7 (Vol+SI+IdioMom)** — the only steps with multiple potentially correlated features. All other steps use plain OLS (λ=0). Macro ridge = `min(2λ, max(λ, 0.5))`.

**Universe:** ~662 US equity stocks present in the fundamentals DB, mapped to a sector ETF, with sufficient price history.

**Run modes:** Full recalculation (complete history) or incremental update (single new date, fast path loading residual histories from DB).

---

## Pre-Processing

### Return Computation
`Pxs_df[universe].pct_change()` using closing prices from `prices_relation`.

### Return Winsorization
Clipped at **±50%** before any regression — absolute threshold, no look-ahead bias. ~810 obs (~0.05%) affected.

### Price Spike Filter (upstream, one-off `price_spike_filter.py`)
Flags P[t]/P[t-1] > 3.0 or < 0.2, confirmed spike if P[t+k]/P[t-1] reverts to ±15% within 2 days. NaN'd and forward-filled.

### Dynamic Market Cap
Shares (from `valuation_consolidated`) × daily price. Cached in `dynamic_size_df`.

### Sample Periods
- **Extended start:** 252 trading days before `st_dt` — for beta/momentum lookback
- **`st_dt`:** user-prompted (default 2019-01-01, current runs 2018-01-01)
- **`common_dates`:** intersection of all dates where every characteristic is available

---

## Step 1 (Baseline): UFV

Pooled variance of raw winsorized returns over `common_dates`. Benchmark for all steps.

---

## Step 2: Market Beta

**Target:** Raw daily returns.

**Feature:** EWMA beta vs SPX. Window=252d, hl=126d (strictly `< dt`):
```
β_i = Cov_EWMA(r_i, r_SPX) / Var_EWMA(r_SPX)
```
Entered raw (not z-scored). **OLS (λ=0).**

**Output:** `factor_residuals_mkt`, `factor_lambdas_mkt`

---

## Step 3: Size

**Target:** Step 2 residuals.

**Feature:** Cross-sectional z-score of log(dynamic_market_cap) per date:
```
size_i = zscore(log(shares_i × price_i))
```
**OLS (λ=0).**

**Output:** `factor_residuals_size`, `factor_lambdas_size`

---

## Step 4: Macro Factors (Joint)

**Target:** Step 3 residuals.

**Purpose:** Strip macro sensitivities before sectors, so sector dummies capture pure sector effects net of rates, inflation, and commodity exposures.

**Features:** EWMA rolling betas vs 6 macro factor changes. Same structure as market beta (window=252d, hl=126d). Each macro beta **z-scored cross-sectionally** per date before regression.

| Column | Description | Type |
|--------|-------------|------|
| `USGG2YR` | 2Y nominal rate daily change (bps) | Change |
| `US10Y2Y_SPREAD_CHG` | 2Y/10Y spread daily change (bps) | Change |
| `T10YIE` | 10Y inflation breakeven daily change | Change |
| `T5YIFR` | 5y5y implied inflation daily change | Change |
| `Crude` | WTI crude daily pct change | Change |
| `XAUUSD` | Gold daily pct change | Change |

All 6 series provided pre-computed as daily changes in `Pxs_df`. The spread change (`US10Y2Y_SPREAD_CHG`) is provided directly — not computed internally.

**Note on VIX Mom and M2MP:** Initially included, removed after observing near-zero t-stats with no directional consistency. VIX Mom is partially redundant with market beta; M2MP suffers from weekly forward-fill stale signal.

**Ridge regularization:** `MACRO_RIDGE = min(2λ, max(λ, 0.5))` applied here due to potential collinearity among rate and inflation factors. For λ=0.1: MACRO_RIDGE=0.2; for λ≥0.25: MACRO_RIDGE=max(λ, 0.5).

**Output:** `factor_residuals_macro`, `factor_lambdas_macro`

---

## Step 5: Sector Dummies

**Target:** Step 4 residuals (macro-adjusted).

**Feature:** One-hot encoding of sector ETF membership. 12 sectors (IGV, REZ, SOXX, XHB, XLB, XLC, XLE, XLF, XLI, XLU, XLV, XLY). **XLP is the reference sector** (dropped). Static assignment. **OLS (λ=0).**

Sectors come after macro so that each sector's lambda captures pure sector-specific effects after rates, inflation, and commodities are stripped out — e.g. XLE's lambda is energy-sector idiosyncrasy, not oil sensitivity.

**Output:** `factor_residuals_sec`, `factor_lambdas_sec`

---

## Step 6: Quality Factor (Standalone)

**Target:** Step 5 residuals.

**Feature:** Rate-conditioned composite of fundamental quality metrics:
- **GQF** (Growth Quality Factor): t-stat weighted from non-2021/22 anchor dates
- **CQF** (Conservative Quality Factor): t-stat weighted from 2021/22 dates
- **Rate signal:** `USGG10YR - USGG10YR.rolling(252).mean()`, quantized: q=0 (falling, <-15bps), q=0.5 (neutral), q=1.0 (rising)
- **Composite:** `(1-q) × GQF + q × CQF`

Computed at anchor dates (~monthly) from `valuation_metrics_anchors`, **forward-filled** to daily, then **z-scored cross-sectionally** each date. **OLS (λ=0).**

Isolated standalone to get a clean uncontaminated read on quality's standalone predictive power, and because quality is correlated with vol, SI, and momentum.

**Output:** `factor_residuals_quality`, `factor_lambdas_quality`

---

## Step 7: GK Vol + SI Composite + Idiosyncratic Momentum (Joint)

**Target:** Step 6 residuals. All three features enter simultaneously.

**Feature 1: Garman-Klass Volatility**
Window=84d, hl=42d. Uses `daily_open`, `daily_high`, `daily_low` tables:
```
σ²_GK = 0.5·(ln(H/L))² - (2·ln2-1)·(ln(C/O))²
```
~8x more efficient than close-to-close. Clips negative GK estimates to 0. Falls back to close-to-close if OHLC unavailable. **Z-scored cross-sectionally.**

**Feature 2: SI Composite**
From `short_interest_data`: SI % Free Float + Utilization, each z-scored, equally averaged. Forward-filled (weekly frequency). Cached in `si_composite_df`. Neutral score (~0) for missing stocks.

**Feature 3: Idiosyncratic Momentum**
Computed from **quality residuals** (Step 6 output) — not sector residuals. Captures price momentum orthogonal to market, size, sectors, and quality:

- **Standard:** cumulative sum of quality residuals over `[t-252, t-21]`
- **Volume-scaled:** `Σ(resid_quality_it × vol_scalar_it)` using pre-normalized volume scalars clipped to `[0.5, 3.0]`

Z-scored cross-sectionally. Uses strictly `< dt` data.

**Ridge regularization:** `RIDGE_LAMBDA` (user-prompted, default=0.1) — the only other step besides macro where ridge is applied, due to correlation among vol, SI, and momentum.

**Output:** `factor_residuals_mom`, `factor_lambdas_mom`

---

## Step 8: Value Factor

**Target:** Step 7 residuals. Placed after momentum to avoid inverse causality trap.

**Feature:** IC-weighted composite of 7 valuation metrics from `valuation_consolidated`: P/S, P/Ee, P/Eo, sP/S, sP/E, sP/GP, P/GP.

**Construction:**
1. Reflexivity treatment within each sector: negatives → `sector_max_positive + abs(metric)`
2. Ranked ascending within sector (low P/x = cheap), inverted so cheap scores HIGH
3. IC-derived weights (from avg absolute t-stat, 21d+63d horizons vs `factor_residuals_joint`)
4. Forward-filled from valuation dates, z-scored cross-sectionally

**OLS (λ=0).**

**Output:** `factor_residuals_joint` (= value residuals, used as O-U input), `factor_lambdas_joint`

---

## Step 9: O-U Mean Reversion (Final Step)

**Target:** Step 8 residuals (`factor_residuals_joint`). Mean reversion measured in purely idiosyncratic space, net of all 8 prior factors.

**Residual Price Index:**
```
px_idx[t] = (1 + resid[t]).cumprod() × P[t_0]
```

**Volume Scaling (optional):** residuals divided by volume scalars `[0.5, 3.0]` before fitting (same flag as idio momentum).

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

**Consolidated intercept:** sum of all 9 step intercepts per date. Persistent non-zero mean would indicate an unexplained systematic return.

**Daily R²:** computed per date as `1 - var(ou_residuals[t]) / var(raw_returns[t])` on the cross-section — varies by date unlike the pooled R².

---

## DB Tables Summary

| Table | Contents | Step |
|-------|----------|------|
| `factor_residuals_mkt` | After market beta | 2 |
| `factor_residuals_size` | After size | 3 |
| `factor_residuals_macro` | After macro factors | 4 |
| `factor_residuals_sec` | After sectors | 5 |
| `factor_residuals_quality` | After quality | 6 |
| `factor_residuals_mom` | After vol+SI+idio_mom | 7 |
| `factor_residuals_joint` | After value (O-U input) | 8 |
| `factor_residuals_ou` | Full model residuals | 9 |
| `factor_lambdas_mkt` | Beta lambda series | 2 |
| `factor_lambdas_size` | Size lambda series | 3 |
| `factor_lambdas_macro` | Macro lambda series | 4 |
| `factor_lambdas_sec` | Sector lambda series | 5 |
| `factor_lambdas_quality` | Quality lambda series | 6 |
| `factor_lambdas_mom` | Vol+SI+mom lambda series | 7 |
| `factor_lambdas_joint` | Value lambda series | 8 |
| `factor_lambdas_ou` | O-U lambda series | 9 |
| `ou_reversion_df` | O-U scores cache | 9 |
| `dynamic_size_df` | Daily market cap | Pre-proc |
| `si_composite_df` | SI composite scores | 7 |

---

## Run Modes

### Full Recalculation (answer `n` to incremental prompt)
Computes all steps over full history from `st_dt`. Required on first run and after any model change. Prompts for start date, ridge λ, volume scaling, and O-U override.

### Incremental Update (answer `y`, default)
Fast single-date update for `Pxs_df.index[-1]`. Principle: load all previously computed residuals and characteristics from DB, compute only what is new.

**What is recomputed:** betas (market + macro), size char, vol factor, SI composite (last 60 dates for ffill), quality/value scores, single-date regressions through all 9 steps.

**What is loaded from DB:** quality residual history (last 400d) for momentum lookback, value residual history (last 120d) for O-U fitting.

**Save pattern:** upsert — delete existing date then reinsert, ensuring reruns of the same date always produce fresh results.

**O-U:** always computed incrementally in update mode (no override prompt).

---

## Key Design Principles

**Sequential orthogonalization:** each step regresses only on its own new factor. Prior factors not repeated.

**Macro before sectors:** macro sensitivities stripped before sector dummies, so XLE etc. capture pure idiosyncratic sector effects rather than oil/rate sensitivity.

**Quality before momentum:** idio momentum computed from quality residuals, capturing price trends beyond what fundamental quality justifies.

**Value after momentum:** avoids inverse causality where high valuations proxy for expected momentum continuation.

**O-U from value residuals:** mean reversion in the most purified idiosyncratic return series.

**No look-ahead bias:** all characteristics use strictly `< dt` data. O-U patched after look-ahead bug (was `<= dt`).

**Consistent stock universe in variance stats:** all steps evaluated on the `ou_common` stock universe for fair comparison.

**Volume scaling consistency:** same flag and scalars applied to both idio momentum (Step 7) and O-U fitting (Step 9).
