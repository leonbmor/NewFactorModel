# Factor Model — Step 1: Technical Reference

**File:** `factor_model_step1.py`  
**Last updated:** March 2026  
**Database:** `postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db`

---

## Overview

Sequential Fama-MacBeth cross-sectional factor residualization. At each step, a weighted least squares (WLS) regression is run cross-sectionally on every trading date. The **residuals** from each step become the **input returns (target)** for the next step. Each step's regression uses **only its own new factor(s)** as features — prior factors are not repeated, since by construction the residuals used as targets are already orthogonal to all previously included characteristics.

This sequential orthogonalization (Gram-Schmidt approach) ensures each factor's contribution is measured **net of all factors added before it**, providing clean, non-redundant variance attribution. Each lambda table contains only the factor(s) added at that step.

**Regression weights at every step:** `log(dynamic_market_cap)`, normalized to sum to 1 within each cross-section. This down-weights small and micro-cap stocks.

**Regularization:** Ridge (L2) regression with penalty λ (user-prompted, default=0.1) applied **only at Step 6** (Vol + SI + IdioMom) — the only step with multiple features that may be correlated. All other steps use plain OLS (λ=0). Applying ridge to a single-feature regression would introduce pure shrinkage bias with no variance reduction benefit.

**Universe:** ~662 US equity stocks present in the fundamentals DB, mapped to a sector ETF, with sufficient price history.

---

## Pre-Processing

### Return Computation
Daily returns: `Pxs_df[universe].pct_change()` using closing prices from `prices_relation`.

### Return Winsorization
Returns clipped at **±50%** before any regression. Absolute threshold — no look-ahead bias. ~810 observations (~0.05% of data) affected. Acts as last-resort backstop for any data errors not caught upstream.

### Price Spike Filter (upstream, one-off script `price_spike_filter.py`)
Flags and NaN's single/double-day price spikes where:
- `P[t]/P[t-1] > 3.0` or `P[t]/P[t-1] < 0.2`
- **AND** `P[t+k]/P[t-1]` reverts to within ±15% of pre-spike level within k=1 or 2 days

Flagged values forward-filled. Applied to `Pxs_df` before passing to the factor model.

### Dynamic Market Cap
Shares outstanding (`valuation_consolidated`) × daily price (`Pxs_df`). Cached in `dynamic_size_df`. Only missing dates computed on each run.

### Sample Periods
- **Extended start:** 252 trading days before `st_dt` — for beta/momentum lookback history
- **`st_dt`:** user-prompted (default 2019-01-01, current runs use 2018-01-01)
- **`common_dates`:** intersection of all dates where every characteristic is available — used for all variance stats and lambda distributions

---

## Step 1 (Baseline): UFV

**Unexplained Factor Variance** on raw winsorized returns over `common_dates`. Benchmark for all subsequent steps. Pooled variance: `var(returns.flatten())` across all stock-date observations.

---

## Step 2: Market Beta

### Target
Raw daily stock returns.

### Feature: EWMA Beta
Estimated for each stock on each date using strictly prior `BETA_WINDOW=252` trading days (no look-ahead).

```
β_i = Cov_EWMA(r_i, r_SPX) / Var_EWMA(r_SPX)
```

EWMA weights: `w_t = (1-α)^(n-1-t)` normalized, with `α = 1 - exp(-ln(2)/BETA_HL)`, `BETA_HL=126`. Entered raw (not z-scored) — already dimensionless.

### Regression
```
r_i = λ_0 + λ_beta · β_i + ε_i
```

**Output:** `factor_residuals_mkt`, `factor_lambdas_mkt`

---

## Step 3: Size

### Target
Market beta residuals from Step 2.

### Feature: Size
Cross-sectional z-score of log(dynamic_market_cap) computed fresh each date:
```
size_i = zscore(log(shares_i × price_i))
```
Mean-zero, unit variance cross-sectionally. Lambda interpretable as return per σ of log market cap.

### Regression
```
resid_mkt_i = λ_0 + λ_size · size_i + ε_i
```

**Output:** `factor_residuals_size`, `factor_lambdas_size`

---

## Step 4: Sector Dummies

### Target
Size residuals from Step 3.

### Feature: Sector Dummies
One-hot encoding of sector ETF membership. 12 sectors (IGV, REZ, SOXX, XHB, XLB, XLC, XLE, XLF, XLI, XLU, XLV, XLY). **XLP is the reference sector** (dropped). Static assignment from `sectors_s`.

### Regression
```
resid_size_i = λ_0 + Σ_k λ_k · D_ik + ε_i
```

**Output:** `factor_residuals_sec`, `factor_lambdas_sec`

---

## Step 5: Quality Factor (Standalone)

### Target
Sector residuals from Step 4. Clean read on quality beyond market, size, and sector effects.

### Feature: Quality Composite
Rate-conditioned composite of fundamental quality metrics. Key points:

- **Two sub-factors:** GQF (Growth Quality Factor, t-stat weighted from non-2021/22 anchor dates) and CQF (Conservative Quality Factor, t-stat weighted from 2021/22)
- **Rate signal:** `USGG10YR - USGG10YR.rolling(252).mean()` quantized: q=0 (falling, <-15bps), q=0.5 (neutral), q=1.0 (rising, >+15bps)
- **Composite:** `(1-q) × GQF + q × CQF`
- Computed at anchor dates (~monthly) from `valuation_metrics_anchors`, **forward-filled** to daily
- **Z-scored cross-sectionally** each date before entering regression

### Regression
```
resid_sec_i = λ_0 + λ_quality · quality_i + ε_i
```

**Output:** `factor_residuals_quality`, `factor_lambdas_quality`

**Why standalone:** quality is correlated with vol, SI, and momentum. Isolating it provides a clean uncontaminated read on its standalone predictive power. The quality residuals produced here are then used to compute idio momentum in Step 6.

---

## Step 6: GK Vol + SI Composite + Idiosyncratic Momentum (Joint)

### Target
Quality residuals from Step 5. All three factors compete simultaneously in a single joint regression.

### Feature 1: Garman-Klass Volatility

Short-window EWMA realized vol. Window: `VOL_WINDOW=84` days, half-life: `VOL_HL=42` days.

**Garman-Klass estimator** (using `daily_open`, `daily_high`, `daily_low` tables):
```
σ²_GK = 0.5·(ln(H/L))² - (2·ln2-1)·(ln(C/O))²
```
~8x more efficient than close-to-close. Negative GK estimates clipped to 0. Falls back to close-to-close EWMA if OHLC unavailable. EWMA weights same structure as beta. Annualized: `σ = sqrt(σ²_daily × 252)`. **Z-scored cross-sectionally** each date. Strictly `< dt` data.

### Feature 2: SI Composite

Combines from `short_interest_data`:
1. SI % Free Float (shares short / freely tradeable shares)
2. Utilization (shares on loan / shares available to borrow)

Each z-scored cross-sectionally, equally averaged:
```
si_composite_i = (z(si_pct_float_i) + z(utilization_i)) / 2
```
Forward-filled to daily. Stocks missing SI data receive neutral score (~0). Cached in `si_composite_df`.

### Feature 3: Idiosyncratic Momentum

**Key design:** computed from **quality residuals** (`factor_residuals_quality`), not sector residuals. Captures price momentum orthogonal to market, size, sectors, *and quality* — stocks trending beyond what their quality level justifies.

**Standard version:** cumulative sum of quality residuals over `[t-252, t-21]` trading days. Z-scored cross-sectionally. Strictly `< dt` data.

**Volume-scaled version** (when `volumeTrd_df` provided):
```
idio_mom_i = Σ_{t-252}^{t-21} (resid_quality_it × vol_scalar_it)
```
Pre-computed volume scalars clipped to `[VOL_LOWER, VOL_UPPER]` = `[0.5, 3.0]`.

### Regression (with L2 ridge regularization, λ user-prompted default=0.1)
```
resid_quality_i = λ_0 + λ_vol·vol_i + λ_si·si_composite_i + λ_idio_mom·idio_mom_i + ε_i
```
Ridge applied here because vol, SI, and idio_mom may be correlated (e.g. high-vol stocks tend to have high momentum). Ridge shrinks all three coefficients toward zero jointly, stabilizing the estimates without biasing the residuals used as input to Step 7.

**Output:** `factor_residuals_mom`, `factor_lambdas_mom`

---

## Step 7: Value Factor

### Target
Step 6 residuals (quality + vol + SI + momentum all removed). Placement after momentum avoids the inverse causality trap where high-valuation stocks appear to predict returns simply because valuations reflect expected momentum continuation.

### Feature: Value Composite

Seven metrics from `valuation_consolidated`: P/S, P/Ee, P/Eo, sP/S, sP/E, sP/GP, P/GP.

**Negative valuation treatment:** within each sector, negative ratios receive `adjusted = sector_max_positive + abs(metric)`, placing them at the expensive end without exclusion.

**Sector ranking:** ranked ascending within sector (low P/x = cheap), inverted so cheap → rank near 1.

**IC-derived weights** (from IC study vs `factor_residuals_joint`, avg of 21d and 63d absolute t-stats):

| Metric | Approx. Weight |
|--------|----------------|
| P/S | 17.0% |
| sP/S | 16.0% |
| P/Ee | 15.9% |
| sP/GP | 14.7% |
| P/Eo | 14.1% |
| sP/E | 13.7% |
| P/GP | 13.6% |

Forward-filled to daily. **Z-scored cross-sectionally** each date.

### Regression
```
resid_mom_i = λ_0 + λ_value · value_i + ε_i
```

**Output:** `factor_residuals_joint` (= value residuals, used as O-U input), `factor_lambdas_joint`

---

## Step 8: O-U Mean Reversion (Final Step)

### Target
Value residuals from Step 7. O-U mean reversion measured in purely idiosyncratic space, net of all 7 prior factors.

### Residual Price Index
```
px_idx[t] = (1 + resid[t]).cumprod() × P[t_0]
```
Anchored to actual price, then scaled to actual price space for interpretability.

### Volume Scaling
If `volumeTrd_df` provided (same flag as idio momentum — consistent), residuals divided by volume scalars (clipped `[0.5, 3.0]`) before O-U fitting. Strictly `< dt` residuals.

### O-U / AR(1) Fitting

AR(1) fitted to last `OU_MEANREV_W=60` observations:
```
px_idx[t] = a + b · px_idx[t-1] + ε
```

**Fit discarded (→ NaN → ST reversal fallback) if:**
- `b ∉ (0, 1)` — non-stationary or oscillating
- `m = a/(1-b) ≤ 0` — negative LT mean
- `resid_std = 0` or `k = 0`

**Parameters:**
- `m`: long-term mean (actual price scale)
- `k = -ln(b)`: mean reversion speed
- `T_h = ln(2)/k`: half-life in trading days
- `σ_resid`: AR(1) residual std (actual price scale)

**Standardized distance:**
```
DistST = (P_current - m) / (σ_resid / sqrt(2k))
```
Factor score: **`-DistST`** (positive = below LT mean = buy signal).

### ST Reversal Fallback

For invalid fits, computed from compounded residual index:
```
st_reversal_i = log(cum_resid[t-1] / cum_resid[t-22])
```
Low = recent idiosyncratic loser = buy. Ranked descending.

### Weighted Blend
```
final_score_i = (ou_weight_i × ou_rank_i + rev_rank_i) / (ou_weight_i + 1)
ou_weight_i   = min(30 / T_h_i, 10.0)
```

| Half-life T_h | ou_weight | O-U weight |
|--------------|-----------|------------|
| 3d | 10.0 (capped) | 91% |
| 30d | 1.0 | 50% |
| 60d | 0.5 | 33% |
| 120d | 0.25 | 20% |
| Invalid | 0.0 | 0% |

Z-scored cross-sectionally. Cached in `ou_reversion_df`. **Must be overridden whenever any upstream step changes.**

### Regression
```
resid_joint_i = λ_0 + λ_ou · ou_reversion_i + ε_i
```

**Output:** `factor_residuals_ou` (full model residuals), `factor_lambdas_ou`

---

## DB Tables Summary

| Table | Contents | Step |
|-------|----------|------|
| `factor_residuals_mkt` | After market beta | 2 |
| `factor_residuals_size` | After size | 3 |
| `factor_residuals_sec` | After sectors | 4 |
| `factor_residuals_quality` | After quality | 5 |
| `factor_residuals_mom` | After vol+SI+idio_mom | 6 |
| `factor_residuals_joint` | After value (O-U input) | 7 |
| `factor_residuals_ou` | Full model residuals | 8 |
| `factor_lambdas_mkt` | Beta lambda series | 2 |
| `factor_lambdas_size` | Size lambda series | 3 |
| `factor_lambdas_sec` | Sector lambda series | 4 |
| `factor_lambdas_quality` | Quality lambda series | 5 |
| `factor_lambdas_mom` | Vol+SI+mom lambda series | 6 |
| `factor_lambdas_joint` | Value lambda series | 7 |
| `factor_lambdas_ou` | O-U lambda series | 8 |
| `ou_reversion_df` | O-U scores cache | 8 |
| `dynamic_size_df` | Daily market cap | Pre-proc |
| `si_composite_df` | SI composite scores | 6 |

---

## Key Design Principles

**True sequential orthogonalization:** each step uses only its own new factor as a feature. Prior factors not repeated.

**Quality before momentum:** quality (Step 5) precedes momentum (Step 6) so idio momentum captures price trends *beyond what quality justifies* — computed from quality residuals.

**Value after momentum:** avoids inverse causality trap where high valuations proxy for momentum persistence.

**O-U from value residuals:** mean reversion measured in the most purified idiosyncratic return series, net of all 7 prior factors.

**No look-ahead bias:** all characteristics use strictly `< dt` data. O-U patched after look-ahead bug found (was using `<= dt`, causing spurious t=-35).

**Volume scaling consistency:** same flag and same scalars applied to both idio momentum and O-U fitting.

**Full history vs common sample:** two regressions at each step — full extended dates for DB storage, common dates only for variance stats and lambda distributions.
