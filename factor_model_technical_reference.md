# Factor Model — Step 1: Technical Reference

**File:** `factor_model_step1.py`  
**Last updated:** March 2026  
**Database:** `postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db`

---

## Overview

Sequential Fama-MacBeth cross-sectional factor residualization. At each step, a weighted least squares (WLS) regression is run cross-sectionally on each trading date. The residuals from each step become the input returns for the next step. This sequential structure means each factor's contribution is measured **net of all factors added before it**, providing clean, non-redundant variance attribution.

**Regression target at every step:** daily stock returns (or residuals from the prior step) for the universe of stocks on that date.

**Regression weights at every step:** `log(dynamic_market_cap)`, normalized to sum to 1 within each cross-section. This down-weights small and micro-cap stocks that would otherwise dominate the pooled variance.

**Regularization:** Ridge regression with penalty λ (user-prompted, default=0.1) applied uniformly to all factor steps.

**Universe:** ~662 US equity stocks present in the fundamentals DB, mapped to a sector ETF, and with sufficient price history (≥126 non-NaN observations in the pre-sample beta window).

---

## Pre-Processing

### Return Computation
Daily returns computed as `Pxs_df[universe].pct_change()` using closing prices from `prices_relation`.

### Return Winsorization
Before any regression, returns are clipped at **±50%** to remove residual data errors not caught by the price spike filter. This is an absolute threshold (not percentile-based), so it introduces no look-ahead bias. Approximately 810 observations (~0.05% of all data) are affected.

### Price Spike Filter (upstream, applied to Pxs_df before model entry)
Separate one-off script (`price_spike_filter.py`) flags and NaN's single/double-day price spikes where:
- `P[t]/P[t-1] > 3.0` or `P[t]/P[t-1] < 0.2` (price more than triples or drops >80% in one day)
- **AND** `P[t+k]/P[t-1]` reverts to within ±15% of pre-spike level within k=1 or k=2 days

Flagged values are forward-filled. This removes genuine data errors while preserving legitimate large moves (e.g. acquisitions, earnings gaps) that do not reverse.

### Dynamic Market Cap
Shares outstanding from `valuation_consolidated` × daily price from `Pxs_df`. Cached in DB table `dynamic_size_df`. Only missing dates computed on each run.

### Sample Periods
- **Extended start:** 252 trading days before `st_dt` — used for beta estimation and idio momentum lookback
- **`st_dt`:** user-prompted start date (default 2019-01-01, current runs use 2018-01-01)
- **`common_dates`:** intersection of all dates where every characteristic is available — used for variance stats and lambda distributions to ensure apples-to-apples comparison across steps

---

## Step 1: Baseline UFV

**Unexplained Factor Variance (UFV)** computed on raw winsorized returns over `common_dates`. This is the benchmark against which all subsequent steps are measured.

Computed as pooled variance across all stock-date observations: `var(returns.flatten())`.

---

## Step 2: Market Beta

### Feature Construction
**EWMA market beta** estimated for each stock on each date using the prior `BETA_WINDOW=252` trading days (strictly before the calculation date — no look-ahead).

**Formula:**
```
β_i = Cov_EWMA(r_i, r_SPX) / Var_EWMA(r_SPX)
```

EWMA weights: `w_t = (1-α)^(n-1-t)`, normalized to sum to 1, with `α = 1 - exp(-ln(2)/BETA_HL)` and `BETA_HL=126` (half-life of 126 days, so observations ~6 months ago receive half the weight of today's).

Requires at least `BETA_WINDOW/2 = 126` non-null observations. Beta is entered into the regression **as-is** (raw, not z-scored) since it's already dimensionless and on a natural scale.

### Cross-Sectional Regression (per date)
```
r_i = λ_0 + λ_beta · β_i + ε_i
```
- **Target:** raw daily stock return `r_i`
- **Feature:** EWMA beta `β_i` (raw)
- **Weights:** log(dynamic_market_cap)
- **Output:** `factor_residuals_mkt` (residuals), `factor_lambdas_mkt` (daily lambdas)

---

## Step 3: Size

### Feature Construction
**Cross-sectional z-score of log(dynamic_market_cap)** computed fresh each date.

```
size_i = zscore(log(shares_i × price_i))
```

Z-scoring is done cross-sectionally (across all stocks on that date), so size is always mean-zero with unit variance relative to the current universe. This means the size lambda `λ_size` measures return per standard deviation of log market cap, not return per dollar.

### Cross-Sectional Regression (per date)
```
resid_mkt_i = λ_0 + λ_beta · β_i + λ_size · size_i + ε_i
```
- **Target:** market beta residual from Step 2
- **Features:** beta (same as Step 2) + size z-score
- **Output:** `factor_residuals_size`, `factor_lambdas_size`

---

## Step 4: Sector Dummies

### Feature Construction
One-hot encoded sector ETF membership. 12 sectors used (IGV, REZ, SOXX, XHB, XLB, XLC, XLE, XLF, XLI, XLU, XLV, XLY). **XLP is the reference sector** (dropped to avoid multicollinearity). Each stock receives a 1 for its sector and 0 for all others.

Sector assignment is static (from `sectors_s` input Series) — it does not change over time.

### Cross-Sectional Regression (per date)
```
resid_size_i = λ_0 + λ_beta · β_i + λ_size · size_i + Σ_k λ_k · D_ik + ε_i
```
- **Target:** size residual from Step 3
- **Features:** beta + size + 11 sector dummies
- **Output:** `factor_residuals_sec`, `factor_lambdas_sec`

---

## Step 5: Idiosyncratic Momentum + Garman-Klass Vol + SI Composite + Quality (Joint)

All four factors entered simultaneously in a single multivariate regression. This is a **joint** step — all four compete for explanatory power simultaneously, so none gets a sequential first-mover advantage over the others.

### Feature 1: Idiosyncratic Momentum

**Standard version:** cumulative sum of sector residuals (`factor_residuals_sec`) over the window `[t - MOM_LONG, t - MOM_SKIP]` = `[t-252, t-21]` trading days. The 21-day skip avoids the short-term reversal effect. Z-scored cross-sectionally.

**Volume-scaled version (used when `volumeTrd_df` provided):** each daily sector residual is multiplied by the volume scalar for that day before cumulating:
```
idio_mom_i = Σ_{t-252}^{t-21} (resid_sec_it × vol_scalar_it)
```
Volume scalars are provided pre-computed (e.g. `volume(t) / rolling_mean_volume(t-10, t-1)`), clipped to `[VOL_LOWER, VOL_UPPER]` = `[0.5, 3.0]`. High-volume days get more weight (their idiosyncratic moves are more informative). Result z-scored cross-sectionally.

**Key design choice:** idio momentum is computed from **sector residuals** (Step 4 output), not raw returns. This strips out market and sector momentum, leaving only the stock-specific momentum component.

**Look-ahead check:** window uses strictly `< dt` dates — no look-ahead.

### Feature 2: Garman-Klass Volatility

**Short-window EWMA realized volatility** using intraday OHLC data. Window: `VOL_WINDOW=84` trading days, EWMA half-life: `VOL_HL=42` days. Shorter than the beta window (252/hl=126) to capture a distinct, more recent volatility signal.

**Garman-Klass estimator** (when OHLC available from `daily_open`, `daily_high`, `daily_low`):
```
σ²_GK = 0.5·(ln(H/L))² - (2·ln2-1)·(ln(C/O))²
```
~8x more statistically efficient than close-to-close. The `ln(H/L)` term captures intraday range; the `ln(C/O)` term adjusts for the open-to-close directional component. Negative GK estimates (when the C/O term dominates) clipped to 0.

**Fallback:** if OHLC tables unavailable, falls back to close-to-close EWMA: `σ² = EWMA(r²)`.

EWMA weights applied same way as beta: `w_t = (1-α)^(n-1-t)`, annualized as `σ_annual = sqrt(σ²_daily × 252)`. Result z-scored cross-sectionally.

**Look-ahead check:** window uses strictly `< dt` dates.

### Feature 3: Short Interest (SI) Composite

Combines two SI metrics from `short_interest_data`:
1. **SI % Free Float** — shares sold short as % of freely tradeable shares
2. **Utilization** — shares on loan as % of shares available to borrow

Each metric is z-scored cross-sectionally per date, then equally averaged:
```
si_composite_i = (z(si_pct_float_i) + z(utilization_i)) / 2
```

SI data is lower frequency than daily — forward-filled to daily dates. Stocks with no SI data receive a neutral score (~0 after z-scoring). Cached in DB table `si_composite_df`.

High SI composite = heavily shorted stock = negative return predictor (expected negative lambda).

### Feature 4: Quality Factor

Rate-conditioned composite built from fundamental metrics in `valuation_metrics_anchors`. Full construction detailed in the quality factor documentation. Key points:

- **Two sub-factors:** GQF (Growth Quality Factor, weighted by t-stats in non-2021/22 years) and CQF (Conservative Quality Factor, weighted by 2021/22 t-stats)
- **Rate signal:** `USGG10YR - USGG10YR.rolling(252).mean()`, quantized to q=0 (falling), q=0.5 (neutral), q=1.0 (rising) with 15bps threshold
- **Composite:** `(1-q) × GQF + q × CQF`
- Scores computed at anchor dates (~monthly), **forward-filled** to daily, then **z-scored cross-sectionally** each date
- Forward-filling means quality is a slow-moving signal — updated ~monthly but used daily

### Cross-Sectional Regression (per date)
```
resid_sec_i = λ_0 + λ_beta·β_i + λ_size·size_i + Σ_k λ_k·D_ik
            + λ_idio_mom·idio_mom_i + λ_vol·vol_i
            + λ_si·si_composite_i + λ_quality·quality_i + ε_i
```
- **Target:** sector residual from Step 4
- **Features:** all prior factors + idio_mom + vol + SI + quality
- **Output:** `factor_residuals_mom`, `factor_lambdas_mom`

---

## Step 6: Value Factor

Added **after** idio momentum to avoid the inverse causality trap: high-valuation stocks outperform partially because their high valuations anticipate continued momentum. By controlling for momentum first, the value lambda measures genuine fundamental cheapness orthogonal to price trends.

### Feature Construction

Seven valuation metrics from `valuation_consolidated`: P/S, P/Ee, P/Eo, sP/S, sP/E, sP/GP, P/GP.

**Negative valuation treatment (reflexivity):** within each sector, stocks with negative valuation ratios (loss-making, negative book) are assigned `adjusted_metric = sector_max_positive + abs(metric)`, placing them at the expensive end without excluding them from the universe.

**Sector ranking:** after adjustment, each metric is ranked **ascending** within sector (low P/x = cheap), then inverted so cheap stocks receive rank close to 1.

**IC-derived weights:** metrics weighted by average absolute t-stat from the IC study (avg of 21d and 63d horizons, run against `factor_residuals_joint`):

| Metric | Weight |
|--------|--------|
| P/S | 17.0% |
| P/Ee | 15.9% |
| sP/S | 16.0% |
| P/Eo | 14.1% |
| sP/GP | 14.7% |
| sP/E | 13.7% |
| P/GP | 13.6% |

**Composite:** weighted average of sector ranks across all metrics. Forward-filled from valuation dates to daily, then **z-scored cross-sectionally** each date.

### Cross-Sectional Regression (per date)
```
resid_mom_i = λ_0 + [all prior factors] + λ_value·value_i + ε_i
```
- **Target:** Step 5 residual (idio_mom + vol + SI + quality already removed)
- **Output:** `factor_residuals_joint`, `factor_lambdas_joint`

---

## Step 7: O-U Mean Reversion (Final Step)

AR(1)/Ornstein-Uhlenbeck process fitted to the **compounded residual price index** of each stock, using **Step 6 residuals** (`factor_residuals_joint`) as the input. This ensures mean reversion is measured in the idiosyncratic component after all systematic factors (market, size, sectors, momentum, vol, SI, quality, value) have been removed.

### Residual Price Index
Daily residuals are compounded into a price-like index anchored at the first available actual price:
```
px_idx[t] = (1 + resid[t]).cumprod() × P[t_0]
```

### Volume Scaling (optional)
Before fitting, residuals are divided by volume scalars (clipped to `[OU_VOL_CLIP_LO, OU_VOL_CLIP_HI]` = `[0.5, 3.0]`). High-volume days get more weight (residuals divided by a larger scalar → shrunk), reflecting that high-volume price moves are more informative.

**Look-ahead check:** only residuals strictly `< dt` are used for fitting.

### O-U / AR(1) Fitting

For each stock on each date, AR(1) fitted to the last `OU_MEANREV_W=60` observations of the residual price index:
```
px_idx[t] = a + b · px_idx[t-1] + ε
```

**Validity conditions (if any fail → NaN, fallback to ST reversal):**
- `b ∈ (0, 1)` — stationarity and mean reversion (b≥1 = non-stationary/trending; b≤0 = oscillating)
- `m = a/(1-b) > 0` — LT mean must be positive (negative LT mean = degenerate fit)
- `resid_std > 0` and `k > 0`

**O-U parameters:**
- `m` = long-term mean (scaled to actual price space)
- `k = -ln(b)` = mean reversion speed
- `T_h = ln(2)/k` = half-life (time for gap to LT mean to halve, in trading days)
- `σ_resid` = residual std of AR(1) fit (scaled to actual price space)

**Standardized distance:**
```
DistST = (P_current - m) / (σ_resid / sqrt(2k))
```
Negative DistST = stock below LT mean = undervalued on idiosyncratic basis = expected positive future return.

Factor score used: **`-DistST`** (negated so positive = buy signal).

### ST Reversal Fallback

For stocks where O-U fit is invalid, a fallback signal is computed from the **compounded residual price index** (not raw prices):
```
st_reversal_i = log(cum_resid[t-1] / cum_resid[t-22])
```
Low (negative) = recent idiosyncratic loser = mean reversion buy. Ranked descending (low reversal → rank 1).

### Weighted Blend

Both signals ranked cross-sectionally to `[0,1]`. Final score:
```
final_score_i = (ou_weight_i × ou_rank_i + rev_rank_i) / (ou_weight_i + 1)
```
where:
```
ou_weight_i = min(OU_WEIGHT_REF / T_h_i, OU_WEIGHT_CAP) = min(30 / T_h, 10.0)
```

- Fast mean reversion (T_h=3d) → ou_weight=10 → O-U dominates (91% weight)
- Reference speed (T_h=30d) → ou_weight=1 → equal blend (50/50)
- Slow mean reversion (T_h=120d) → ou_weight=0.25 → mostly ST reversal (80%)
- Invalid fit → ou_weight=0 → pure ST reversal

**Z-scored cross-sectionally** before entering the regression.

**Caching:** O-U scores cached in DB table `ou_reversion_df`. User prompted to override on each run (default: no, only compute missing dates). Should be overridden whenever any upstream factor changes.

### Cross-Sectional Regression (per date, on `ou_common` subsample)
```
resid_joint_i = λ_0 + [all prior factors] + λ_ou · ou_reversion_i + ε_i
```
- **Target:** Step 6 residual
- **Output:** `factor_residuals_ou`, `factor_lambdas_ou`

---

## Variance Reduction Summary

At each step, pooled variance is computed on the **common sample** (intersection of all dates where every characteristic is available). Step 7 uses a slightly smaller subsample (`ou_common`) since O-U requires sufficient residual history.

| Step | Factors Added | Residual Table | Lambda Table |
|------|--------------|----------------|--------------|
| Baseline | — | — | — |
| Step 2 | Market beta | `factor_residuals_mkt` | `factor_lambdas_mkt` |
| Step 3 | Size | `factor_residuals_size` | `factor_lambdas_size` |
| Step 4 | Sector dummies | `factor_residuals_sec` | `factor_lambdas_sec` |
| Step 5 | IdioMom + GK Vol + SI + Quality | `factor_residuals_mom` | `factor_lambdas_mom` |
| Step 6 | Value | `factor_residuals_joint` | `factor_lambdas_joint` |
| Step 7 | O-U Mean Reversion | `factor_residuals_ou` | `factor_lambdas_ou` |

---

## Key Design Principles

### No Look-Ahead Bias
All characteristic computations use strictly `< dt` data (not `<= dt`). The O-U fix was specifically patched after a look-ahead bug was discovered (was using `<= dt` residuals, producing t-stat of -35 before fix).

### Common Sample Integrity
All variance statistics and lambda distributions reported on the identical set of dates (`common_dates`) regardless of which step is being evaluated. This ensures that differences in explained variance across steps reflect genuine factor contributions, not changes in the sample composition.

### Sequential vs Joint
Steps 2-4 are **sequential** (each step's residuals feed the next) reflecting a causal hierarchy: market → size → sectors. Steps 5-7 are **partially joint**: within Step 5, idio_mom + vol + SI + quality compete simultaneously; Step 6 (value) comes after momentum to avoid reverse causality; Step 7 (O-U) comes last as a price-dynamics signal net of all fundamentals.

### Full History for DB, Common Sample for Stats
Two separate regressions are run at steps 5 and 6:
1. **Full extended dates** → residuals saved to DB (backtest needs maximum history)
2. **Common dates only** → variance stats and lambda distributions (clean comparison)

### WLS Weights
`log(dynamic_market_cap)` normalized to sum to 1 within each cross-section. Ensures large-cap stocks dominate the regression, consistent with their economic importance and liquidity. Dynamic market cap recomputed daily (price × shares) rather than using a static size measure.
