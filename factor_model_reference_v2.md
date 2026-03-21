# Factor Model & Scripts — Comprehensive Reference
*Last updated: March 2026*

---

## 1. OVERVIEW

A sequential Fama-MacBeth cross-sectional factor model implemented in Python, running in a Jupyter notebook kernel. The model strips systematic return sources one by one in a true Gram-Schmidt orthogonalization sequence, producing clean residuals at each step. All data is stored in a PostgreSQL database (`factormodel_db`). The universe consists of ~662 US large-cap stocks.

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
Step 5:  Sector Dummies        13 sectors, sum-to-zero coding, Ridge CV; each ⊥ {beta, size, macro}
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

Each macro beta is computed as: `β_im = Cov_EWMA(r_i, d_m) / Var_EWMA(d_m)` using the same EWMA structure as market beta (window=252, hl=126), then z-scored cross-sectionally per date. All 7 macros enter a single joint ridge regression (not sequential), because there is no natural economic ordering among them and ridge handles their mutual collinearity.

**Ridge CV (Step 4):** 5-fold cross-validation on the stock dimension per date. Grid floor at 0.15 — no OLS fallback. On collinear days (high macro correlation), λ=40 is selected; on normal days, λ=0.15–0.30 dominates (~60% of dates). Mean λ ≈ 5.5, median λ ≈ 0.30.

### Sector Dummies (Step 5)

**Sum-to-zero deviation coding:** for K=13 sectors, each dummy = +1 for own sector, -1/(K-1) for all others. All 13 sectors included — no reference sector dropped. This makes the intercept equal to the true equal-weighted market return (not the XLP sector return as in reference coding). XLP is now visible as its own sector lambda.

Sectors: IGV, REZ, SOXX, XHB, XLB, XLC, XLE, XLF, XLI, XLP, XLU, XLV, XLY.

**Ridge CV (Step 5):** Same 5-fold CV as macro. Grid floor at 0.10. Empirically ~88.5% of dates select λ=0.10, ~3% select λ=40 (macro-driven high-collinearity days). Without ridge, sector lambdas showed ±7% artefacts on low-dispersion days — ridge eliminates this entirely. Sector NOT sequentially orthogonalized among themselves (ridge handles it).

### WLS Regression

All cross-sectional regressions use WLS weighted by log(market_cap):

```python
w_i = log(market_cap_i)  →  normalized to sum to 1
```

This overweights large-cap stocks which have cleaner data and are more relevant for institutional portfolios.

### Characteristic Orthogonalization

Before every step, the new characteristic is projected out of all prior characteristics using WLS (same cap-weights as the factor regressions):

```
new_char_perp = new_char - Proj_{prior_chars}(new_char)    [WLS]
```

This ensures the Gram-Schmidt property holds in the weighted inner-product space. Falls back to OLS if market cap data unavailable.

**Full-history versions** (`si_perp_full`, `vol_perp_full`) are computed over the extended date range for the momentum lookback chain — momentum needs residuals from ~252 days before the first valid date.

---

## 3. FACTOR DETAILS

### Market Beta (Step 2)

EWMA rolling beta vs SPX. For each date t:
- Window: 252 trading days before t
- Weights: `α = 1 - exp(-ln(2)/126)`, decaying exponentially
- `β_i = Cov_EWMA(r_i, r_SPX) / Var_EWMA(r_SPX)`
- OLS cross-section (no WLS at this step — single factor, weights absorbed into residual scaling)

### Size (Step 3)

- `size_i = log(shares_outstanding × price)` — dynamic, updated daily
- Market cap from: `shares_outstanding (valuation_consolidated) × Pxs_df price`
- Cached in DB table `dynamic_size_df` (computed only for missing dates)
- Z-scored cross-sectionally per date
- Orthogonalized vs beta before regression

### Quality Factor (Step 6)

Loaded from `quality_scores_df` cache (see Section 5). Rate-conditioned composite:
- **GQF** (Growth Quality): used in non-2021/22 regimes
- **CQF** (Conservative Quality): used in 2021/22 high-rate period
- Blend: `(1-q) × GQF + q × CQF` where `q ∈ {0, 0.5, 1.0}` from USGG10YR vs 252d MAV
- Threshold: 50bps above/below MAV → q=0 or q=1; within threshold → q=0.5

Current hardcoded weights:
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

### SI Composite (Step 7)

Short interest composite signal. Cached in `si_composite_df`. Loaded and orthogonalized vs all prior factors before regression.

### GK Volatility (Step 8)

Garman-Klass realized volatility estimator:
```
σ²_GK = 0.5·(ln(H/L))² - (2·ln2-1)·(ln(C/O))²
```
~8x more efficient than close-to-close. Uses OHLC from DB tables `daily_open`, `daily_high`, `daily_low`. Window=84d, EWMA hl=42d. Annualized. Z-scored cross-sectionally. Falls back to close-to-close if OHLC unavailable.

**Sign:** positive lambda (t≈+4.1) — high-vol stocks **outperform** in this universe after controlling for all prior factors. This is a volatility risk premium, not the low-vol anomaly.

### Idiosyncratic Momentum (Step 9)

Computed on **vol residuals** (`factor_residuals_vol`) not raw returns. This ensures it captures truly idiosyncratic price drift, not market/sector/vol-driven momentum.

**Volume-scaled variant** (default when `volumeTrd_df` provided):
```
mom_i = Σ_{t in window} r_resid_i,t × vol_scalar_i,t
```
where `vol_scalar = vol(t) / mean(vol[t-10, t-1])`, clipped to `[0.5, 3.0]`.

Window: `[t-252, t-21]` (252 days, skip last 21).

**Full-history chain:** `resid_vol_full` computed over extended dates for lookback; `si_perp_full` and `vol_perp_full` are orthogonalized over full history for the same reason.

### Value Factor (Step 10)

IC-weighted composite of 7 valuation metrics, loaded from `value_scores_df` cache (computed by `ic_study.py`). Metrics are P/x ratios (negative IC — expensive stocks underperform). Within-sector reflexivity treatment applied.

Current IC t-stat weights (all negative — higher P/x = more expensive = underperforms):
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

AR(1)/Ornstein-Uhlenbeck fit to the **compounded residual price index** (cumulative product of `factor_residuals_joint` returns for each stock).

For each stock on each date:
1. Build residual price index: `P_t = Π(1 + r_resid)`
2. Fit AR(1): `P_{t+1} = a + b·P_t + ε`
3. Require `0 < b < 1` (mean-reverting); reject if oscillating (`b≤0`) or explosive (`b≥1`)
4. `mean_reversion_level = a / (1-b)`, `half_life = ln(2) / (-ln(b))`
5. `distance_to_mean = (P_current - mean_level) / σ_ε·√(2k)` — standardized

**Invalid fit fallback:** ST reversal on residual index = `log(cum_resid[t-1] / cum_resid[t-22])`, sign-flipped (recent losers get positive score).

**Blending:**
```
ou_weight = min(30 / half_life, 10.0)    # high weight for short half-lives
final_score = (ou_weight × ou_rank + rev_rank) / (ou_weight + 1)
```
So stocks with short half-lives are dominated by O-U signal; stocks with long half-lives or invalid fits get pure reversal.

**Volume scaling (optional):** residuals can be downscaled on high-volume days (normalizing by `vol(t)/mean_vol`) before fitting, so the O-U process is less influenced by conviction-driven moves.

---

## 4. RUN MODES

### Full Recalculation (`n` at prompt)

Prompts for start date (default: 2018-01-01), volume scaling option and clip bounds. Runs all 11 steps end-to-end, computing fresh residuals and lambdas for all dates. Extended start = 252 trading days before `st_dt` (for momentum lookback). Saves all residuals and lambdas to DB, overwriting existing tables. O-U cache can optionally be cleared and recomputed.

### Incremental Update (`y` at prompt, default)

Loads residual history from DB. Orthogonalizes characteristics for the new date using WLS. Runs a single cross-sectional regression per step. Upserts new rows to DB tables. Fast — typically seconds per date. Ridge lambda printed in snapshot header.

### Snapshot Display

After each incremental date:
```
2026-03-19  |  Intercept: -0.97%  |  Daily R²: 20.66%  |  Macro Ridge λ: 0.15  |  Sec Ridge λ: 0.10
Factor                    Lambda%   Intcpt%    R²%
beta                        +1.04    -0.96     9.43
size                        +0.10    +0.00     0.20
USGG2YR                     +0.11    +0.00     2.70
...
```

---

## 5. DATABASE TABLES

### Residual Tables (date × ticker, long format)

| Table | Step | Contents |
|-------|------|----------|
| `factor_residuals_mkt` | Step 2 | After market beta |
| `factor_residuals_size` | Step 3 | After size |
| `factor_residuals_macro` | Step 4 | After macro |
| `factor_residuals_sec` | Step 5 | After sectors |
| `factor_residuals_quality` | Step 6 | After quality |
| `factor_residuals_si` | Step 7 | After SI composite |
| `factor_residuals_vol` | Step 8 | After GK vol |
| `factor_residuals_mom` | Step 9 | After idio momentum |
| `factor_residuals_joint` | Step 10 | After value (O-U input) |
| `factor_residuals_ou` | Step 11 | Full model residuals |

### Lambda Tables (date-indexed)

`factor_lambdas_mkt/size/macro/sec/quality/si/vol/mom/joint/ou`

Each contains columns for the step's factors plus `intercept`, `r2`, and `ridge_lambda` (where applicable).

### Characteristic / Score Tables

| Table | Contents |
|-------|----------|
| `dynamic_size_df` | Daily market cap (shares × price) |
| `si_composite_df` | SI composite scores |
| `quality_scores_df` | Quality composite scores (cached) |
| `value_scores_df` | Value composite scores (cached) |
| `ou_reversion_df` | O-U mean reversion scores (cached) |
| `valuation_consolidated` | Raw quarterly fundamental data |
| `valuation_metrics_anchors` | Anchor dates for quality/value derivation |
| `daily_open/high/low` | OHLC prices for Garman-Klass vol |

---

## 6. LATEST FACTOR PERFORMANCE (as of March 2026 full run)

### Variance Reduction Summary

| Step | % UFV | % prev step |
|------|-------|------------|
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

*Value adds noise with stale IC weights — needs ic_study re-run.

**Consolidated R² = 52.39%**

### Factor t-stats

| Factor | t-stat | Notes |
|--------|--------|-------|
| SI Composite | +5.34 | Strongest and most consistent |
| GK Vol | +4.10 | Risk premium (high vol outperforms) |
| Size | +3.79 | Consistent large-cap premium |
| Quality | +2.73 | Positive every year except 2026 YTD |
| O-U | +1.47 | Fading in recent years |
| Idio Mom | +1.63 | Moderate; vol-scaled version |
| Value | ~0.00 | Stale weights — needs refresh |

---

## 7. SCRIPT: `factor_model_step1.py`

**Location:** `/mnt/user-data/outputs/factor_model_step1.py`
**Entry point:** `run(Pxs_df, sectors_s, volumeTrd_df=None)`

### Key Functions

| Function | Purpose |
|----------|---------|
| `get_universe()` | Filters stocks: in DB + sector-mapped + sufficient price history |
| `load_dynamic_size()` | Loads/computes market cap from DB cache |
| `build_sector_dummies()` | Sum-to-zero deviation coding, 13 sectors |
| `calc_rolling_betas()` | EWMA beta vs SPX, window=252, hl=126 |
| `calc_macro_betas()` | EWMA betas vs 7 macro factors, z-scored |
| `calc_idio_momentum_volscaled()` | Volume-weighted cumulative idio residuals |
| `calc_vol_factor()` | Garman-Klass EWMA vol, window=84, hl=42 |
| `wls_cross_section()` | Single-date WLS regression, returns (lambdas, residuals, R²) |
| `wls_ridge_cross_section()` | WLS with ridge penalty, 5-fold CV lambda selection |
| `run_factor_step()` | Loops wls_cross_section over all dates |
| `run_factor_step_optimal_ridge()` | Loops wls_ridge_cross_section, CV per date |
| `orthogonalize_char()` | Single-date WLS characteristic orthogonalization |
| `orthogonalize_char_df()` | Applies orthogonalize_char across all dates |
| `_fit_ou_single()` | AR(1) fit to residual price index for one stock |
| `_compute_ou_for_dates()` | O-U + ST reversal blend, all dates and stocks |
| `load_ou_reversion()` | Cache-aware O-U loader |
| `load_quality_scores()` | Loads from quality_scores_df, calls get_quality_scores() |
| `load_value_scores()` | Loads from value_scores_df cache |
| `_run_incremental()` | Single-date fast path for daily updates |
| `run()` | Master entry point — full recalc or incremental |
| `print_lambda_summary()` | Stats + annual breakdown + ridge distribution |
| `print_sector_lambdas()` | Sector-by-sector t-stat table |
| `variance_stats()` | Pooled variance with % reduction reporting |

### `run()` return value

Returns a `results` dict containing all residual DataFrames, lambda DataFrames, R² series, and configuration used. Passed directly to `plot_factor_returns.plot_all(results)`.

---

## 8. SCRIPT: `quality_factor.py`

**Location:** `/mnt/user-data/outputs/quality_factor.py`
**Entry point:** `run(Pxs_df, sectors_s, mav_window=252, threshold=50)`

### Architecture

Quality is a rate-conditioned composite of two sub-factors:
- **GQF** (Growth Quality Factor): derived from non-2021/22 anchor dates
- **CQF** (Conservative Quality Factor): derived from 2021/22 anchor dates

The rate signal `q ∈ {0, 0.5, 1.0}` blends them: when rates are well above their 252-day MAV (+50bps), CQF dominates; when below, GQF dominates.

### Key Functions

| Function | Purpose |
|----------|---------|
| `load_anchor_dates()` | Gets valuation snapshot dates from DB |
| `load_all_snapshots()` | Loads all fundamental snapshots from DB |
| `build_derived_metrics()` | Computes GGP, GS, GS/S_Vol, ROId, OM, etc. from raw fundamentals |
| `compute_rate_signal()` | USGG10YR vs rolling MAV → q ∈ {0, 0.5, 1.0} |
| `rank_within_sector()` | Sector-relative percentile ranking, handles reflexivity |
| `derive_weights()` | IC-based weight derivation against `factor_residuals_sec` |
| `compute_composite_scores()` | Applies GQF_WEIGHTS + CQF_WEIGHTS with rate blend |
| `evaluate_composite()` | Cross-validates composite vs forward residual returns |
| `get_quality_scores()` | Cache-aware entry point (loads from DB or computes) |
| `_ensure_scores_table()` | Creates `quality_scores_df` table if missing |
| `_save_scores()` | Saves date-batch of scores to DB |
| `update_cached_weights()` | Prints copy-paste code for GQF/CQF weight constants |
| `run()` | Full weight derivation + cache population + weight printing |
| `gridsearch()` | Exhaustive search over mav_window × threshold grid |

### Exclusions

`ROE`, `ROE-P`, `ROEd` excluded — ROE uses market cap as denominator, creating price contamination. ROI (uses operating costs) kept.

### Cache Refresh Workflow

```python
# Step 1: Re-derive weights
summary, annual, scores, gqf_w, cqf_w = run(Pxs_df, sectors_s,
                                              mav_window=252, threshold=50)
update_cached_weights(gqf_w, cqf_w)

# Step 2: Copy printed GQF_WEIGHTS / CQF_WEIGHTS into quality_factor.py

# Step 3: Factor model loads instantly from quality_scores_df cache
```

Optimal hyperparameters: `mav_window=252, threshold=50bps` (gridsearch result against `factor_residuals_sec`, 63d horizon).

---

## 9. SCRIPT: `ic_study.py` (Value Factor)

**Location:** `/mnt/user-data/outputs/ic_study.py`
**Entry point:** `run_ic_study(Pxs_df, sectors_s, force_recompute_cache=False)`

### Purpose

Measures within-sector Spearman IC between each valuation metric rank and forward idiosyncratic returns from `factor_residuals_mom` (returns net of all factors through Step 9). Derives IC-weighted composite for the value factor.

### Metrics

`P/S, P/Ee, P/Eo, sP/S, sP/E, sP/GP, P/GP`
(prefix `s` = sector-relative; `e/o` = earnings estimate/observed)

### Key Functions

| Function | Purpose |
|----------|---------|
| `load_valuation_dates()` | Valuation snapshot dates from DB |
| `load_valuation_snapshot()` | Raw valuation data for one date |
| `compute_residual_returns()` | Forward returns from `factor_residuals_mom` |
| `_residuals_from_db()` | Loads residuals from DB for a date range |
| `run_ic_study()` | Full IC study + weight derivation + cache population |
| `_ensure_value_scores_table()` | Creates `value_scores_df` if missing |
| `_compute_and_save_value_scores()` | Builds and saves IC-weighted composite scores |

### Horizons

21d and 63d forward returns. Weights = average absolute IC t-stat across both horizons.

### Cache Refresh Workflow

```python
# Targets factor_residuals_mom (Steps 2-9 removed)
ic_ts, ic_summary, ic_annual, weights = run_ic_study(Pxs_df, sectors_s,
                                                       force_recompute_cache=True)
# Prints new _VALUE_TSTAT — copy into factor_model_step1.py
# Re-run factor model full recalculation to pick up new weights
```

---

## 10. SCRIPT: `primary_factor_backtest.py`

**Location:** `/mnt/user-data/outputs/primary_factor_backtest.py`
**Entry point:** `run(Pxs_df, sectors_s, volumeTrd_df=None)`

### Architecture

Long-only, rebalancing every `STEP_DAYS=60` days (bi-monthly). Default portfolio size: `TOP_N=20`. Equal-weight baseline, optional concentration weighting.

The primary factor score is the quality composite (from `quality_scores_df` cache), rank-based within-sector. All metrics cross-sectionally ranked within each GICS sector to remove sector bias.

### Interactive Prompts

```
Volume filter?     (y/n) — exclude top vol quintile at each rebalance
Momentum?          (12m1/idio/n) — momentum overlay type
Volume-scaled?     (y/n) — volume-weight momentum calculation
Min market cap?    ($M or Enter)
Max stocks/sector? (int or Enter)
Portfolio size?    (int or Enter, default 20)
Rebal frequency?   (days or Enter, default 30)
Pre-filter?        (fraction 0-1 by quality, or Enter)
Concentration?     (factor ≥1.0 or Enter for equal-weight)
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `calc_metrics_for_date()` | Recomputes P/E, P/S etc. from 8Q backward fundamentals |
| `build_factor()` | Applies quality scores + sector ranking → primary factor |
| `calc_momentum_12m1()` | 12M-1M price momentum, optionally volume-scaled |
| `calc_idio_momentum_score()` | Cumulative idio residuals [t-252, t-21], optionally vol-scaled |
| `load_idio_momentum_db()` | Loads `factor_residuals_joint` from DB as pivot |
| `select_with_sector_cap()` | Sector cap with gradual relaxation (cap+1, cap+2...) |
| `run_backtest()` | Core backtest loop: rebalance → rank → select → track NAV |
| `print_performance()` | CAGR, Sharpe, max drawdown, Calmar |
| `run()` | Master entry: runs baseline + optional alternative, prints comparison |

### Volume Scaling

When `volumeTrd_df` provided and volume scaling selected:
- **12m1:** daily returns weighted by `vol_scalar = vol(t)/mean_vol`, clipped `[0.5, 3.0]`
- **idio:** cumulative residuals weighted by same scalars
- High-volume days carry more weight → conviction-backed moves amplified

### Performance Output

```
NAV comparison: baseline vs alternative
CAGR, Sharpe ratio, max drawdown, Calmar ratio
Annual return breakdown
Side-by-side portfolio holdings
```

---

## 11. SCRIPT: `plot_factor_returns.py`

**Location:** `/mnt/user-data/outputs/plot_factor_returns.py`
**Entry point:** `plot_all(results)` or `plot_all(load_lambdas_from_db())`

### Six Figures

| Figure | Content |
|--------|---------|
| Fig 1 | Structural: intercept, beta, size cumulative lambdas |
| Fig 2 | Macro: USGG2YR, US10Y2Y_SPREAD_CHG, US10YREAL, BE5Y5YFWD, MOVE, Crude, XAUUSD |
| Fig 3 | Sectors: all 13 sector cumulative lambdas |
| Fig 4 | Alpha factors: quality, SI, vol, idio_mom, value, O-U (Steps 6-11) |
| Fig 5 | Rolling 252d t-stats for all alpha factors |
| Fig 6 | Ridge λ selection over time (macro and sector steps) |

### Usage

```python
# From results dict (after full recalculation in same kernel)
plot_all(results)

# Standalone from DB
results = load_lambdas_from_db(st_date='2019-01-01')
plot_all(results)
```

**Note:** `MACRO_COLS` in plot script still lists old names (`T10YIE`, `T5YIFR`) — needs updating to `US10YREAL`, `BE5Y5YFWD`, `MOVE`.

---

## 12. PENDING WORK

### Immediate
1. **Value cache refresh** — run `run_ic_study(..., force_recompute_cache=True)` then full model recalculation. Value t-stat currently ~0.00 with stale weights.
2. **Quality cache stale dates** — 1966 of 2396 dates cached with old weights. Run `quality_factor.run(force_recompute=True)` if exact current-weight scores needed.
3. **Plot script macro names** — update `MACRO_COLS` list to new names.

### Next Project: Portfolio Risk Decomposition

Goal: given a portfolio of stocks with dollar weights, decompose daily dollar variance by factor.

**Key design decision (agreed):** use **raw (non-orthogonalized)** stock characteristics as factor exposures for risk attribution. The orthogonalization in the factor model is a statistical device for clean regression — it does not reflect the economic reality of what factors mean for risk.

**Factor covariance matrix:** expected to be sparse but not diagonal. Raw characteristics are correlated (e.g. high-quality stocks tend to be large-cap), so off-diagonal terms in the factor return covariance matrix are non-zero and must be estimated.

**Structure:**
```
Portfolio variance = w'X · F · X'w + w'Ω·w

where:
  w = dollar weight vector (N×1)
  X = raw characteristic matrix (N×K), one column per factor
  F = factor return covariance matrix (K×K), estimated from lambda time series
  Ω = diagonal idiosyncratic variance matrix (from factor_residuals_ou)
```

Factor returns (`F`) estimated from `factor_lambdas_*` tables — rolling or full-history covariance.

---

## 13. NOTES AND CONVENTIONS

- **Ticker format:** bare tickers throughout (no `' US'` suffix). `clean_ticker()` strips the suffix wherever needed.
- **DB writes:** always upsert (insert on conflict update) to handle reruns safely.
- **Common sample:** variance stats computed on intersection of all dates/stocks where every characteristic is available — ensures apples-to-apples comparisons.
- **Extended dates:** factor model computes from `st_dt - 252 trading days` for the momentum lookback chain; variance stats only from `st_dt`.
- **O-U cache:** `ou_reversion_df` can be cleared and recomputed by answering `y` to the override prompt. Takes ~30 minutes for 2143 dates × 662 stocks.
- **Jupyter kernel:** all scripts run in same kernel — `quality_factor` functions available in namespace without import, enabling `load_quality_scores()` to call `get_quality_scores()` from the kernel.
- **Volume scalars:** `volumeTrd_df` passed to `run()` is a DataFrame of pre-computed `vol(t)/mean_vol[t-10, t-1]` scalars, clipped to `[VOL_LOWER, VOL_UPPER]` = `[0.5, 3.0]`.
