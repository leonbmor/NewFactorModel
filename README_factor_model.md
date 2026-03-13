# Factor Model & Quality Factor — Technical Reference

**Last updated:** March 2026  
**Database:** `postgresql+psycopg2://postgres:akf7a7j5@localhost:5432/factormodel_db`

---

## Table of Contents

1. [Repository Overview](#1-repository-overview)
2. [Database Schema](#2-database-schema)
3. [Factor Model — Step 1](#3-factor-model--step-1)
4. [Quality Factor](#4-quality-factor)
5. [Quality Factor Diagnostic](#5-quality-factor-diagnostic)
6. [Primary Factor Backtest](#6-primary-factor-backtest)
7. [Supporting Scripts](#7-supporting-scripts)
8. [Key Design Decisions & Rationale](#8-key-design-decisions--rationale)
9. [Pending Items](#9-pending-items)

---

## 1. Repository Overview

| File | Purpose |
|------|---------|
| `factor_model_step1.py` | Fama-MacBeth sequential factor residualization (mkt → size → sectors → joint) |
| `quality_factor.py` | Rate-conditioned quality composite (GQF/CQF blend) |
| `quality_factor_diag.py` | Diagnostic: per-metric IC stats across dates and years |
| `primary_factor_backtest.py` | Portfolio backtest using quality factor + optional momentum blend |
| `calc_valuation_anchors.py` | Computes and stores quality metrics snapshots to `valuation_metrics_anchors` |
| `calc_valuation_daily.py` | Daily valuation metrics computation |
| `ortex_fetcher_v2.py` | Fetches fundamentals from Ortex + AlphaVantage, stores to DB |
| `ic_study.py` | IC analysis: valuation metrics vs full model residuals |

**Entry point for most workflows:**
```python
from primary_factor_backtest import run, show_top_stocks
nav_base, nav_alt, port_base, port_alt, factor_by_date, all_metrics, gqf_w, cqf_w, snapshots = run(Pxs_df, sectors_s)
top50 = show_top_stocks(factor_by_date, all_metrics, sectors_s,
                        gqf_weights=gqf_w, cqf_weights=cqf_w, snapshots=snapshots)
```

---

## 2. Database Schema

### Core Fundamentals Tables

| Table | Contents | Key Columns |
|-------|----------|-------------|
| `income_data` | Quarterly income statement snapshots | `ticker`, `date`, `download_date`, `FEQ`, `revenue`, `gross_profit`, `ebitda`, `net_income`, `normalized_net_income` |
| `cash_data` | Cash flow statement snapshots | `ticker`, `date`, `download_date`, `FEQ`, `capex`, `fcf` |
| `summary_data` | Price, shares, market cap | `ticker`, `date`, `price`, `shares_out`, `mkt_cap` |
| `estimation_status` | FEQ tracking per ticker | `ticker`, `cFEQ`, `last_fund_date` |
| `short_interest_data` | SI % float, utilization | `ticker`, `date`, `si_pct_float`, `utilization` |
| `trading_volume` | Daily volume | `ticker`, `date`, `volume` |

### Computed / Derived Tables

| Table | Contents | Updated By |
|-------|----------|------------|
| `valuation_metrics_anchors` | Quality metrics at anchor dates (~monthly) | `calc_valuation_anchors.py` |
| `key_valuation_metrics` | GS, GS_vol, GS_adj, P/S, P/Ee, P/GP per stock per calc_date | `primary_factor_backtest.py` internally |
| `dynamic_size_df` | Daily dynamic market cap (shares × price) | `factor_model_step1.py` |
| `si_composite_df` | Normalized short interest composite score | `factor_model_step1.py` |
| `factor_lambdas_mkt` | Daily market beta lambda | `factor_model_step1.py` |
| `factor_lambdas_size` | Daily size lambda | `factor_model_step1.py` |
| `factor_lambdas_sec` | Daily sector lambdas | `factor_model_step1.py` |
| `factor_lambdas_joint` | Daily lambdas for all joint factors | `factor_model_step1.py` |
| `factor_residuals_mkt` | Returns after removing market beta | `factor_model_step1.py` |
| `factor_residuals_size` | Returns after removing mkt + size | `factor_model_step1.py` |
| `factor_residuals_sec` | Returns after removing mkt + size + sectors | `factor_model_step1.py` |
| `factor_residuals_joint` | Full model residuals (all factors) | `factor_model_step1.py` |

### Important Notes on Table Independence

- `key_valuation_metrics` and `valuation_metrics_anchors` are **completely independent** — populated by different scripts with no dependency on each other.
- Adding new dates to `valuation_metrics_anchors` has **no effect** on `key_valuation_metrics`. They just happen to share some metrics (e.g. GS) computed independently.
- The quality factor in `primary_factor_backtest.py` reads from `valuation_metrics_anchors` directly — new anchor dates are picked up automatically on each run.

---

## 3. Factor Model — Step 1

**File:** `factor_model_step1.py`

**Usage:**
```python
from factor_model_step1 import run
results = run(Pxs_df, sectors_s)
results = run(Pxs_df, sectors_s, volumeTrd_df=volumeTrd_df)  # vol-scaled momentum
```

### Architecture: Sequential Fama-MacBeth Residualization

Factors are added sequentially, each step explaining residual variance from the prior:

```
Step 1: Raw returns (UFV baseline)
Step 2: + Market beta          → mkt_UFV
Step 3: + Size                 → size_UFV
Step 4: + Sector dummies       → sec_UFV
Step 5: + IdioMom + Reversal + SI + Quality (joint OLS) → joint_UFV
```

Each step: cross-sectional WLS regression at every date, weights = log(dynamic_size).

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `BETA_WINDOW` | 252 | EWMA beta lookback |
| `BETA_HL` | 126 | EWMA half-life |
| `MOM_LONG` | 252 | Momentum lookback |
| `MOM_SKIP` | 21 | Skip last 21 days (reversal avoidance) |
| `SECTOR_REF` | `XLP` | Reference sector (omitted dummy) |
| `SI_HORIZON` | 21 | Forward return horizon for SI signal |
| `RIDGE_LAMBDA` | 0.1 (default, user-prompted) | Ridge regularization |

### Market Beta

EWMA beta estimated from prior 252 days (half-life=126), re-estimated daily. Stocks require ≥126 non-null observations.

### Size Factor

Z-scored log(dynamic_size) computed cross-sectionally each date. Dynamic size = shares_outstanding × daily price (cached in `dynamic_size_df`).

### Sector Dummies

One-hot encoded sector ETF membership (XLK, XLV, XLI, etc.), XLP as reference. Sectors assigned from `sectors_s` Series.

### Joint Step: Idio Momentum

Idiosyncratic momentum computed from sector residuals (`factor_residuals_sec`):
- Lookback: t-252 to t-21 (skip last 21 days)
- Standard: cumulative log return over the window
- Vol-scaled variant: volume-weighted cumulative return, normalized by rolling vol, clipped at [VOL_LOWER, VOL_UPPER]

### Joint Step: 21d Reversal

`log(P[t-1] / P[t-22])` — short-term reversal from prices directly.

### Joint Step: SI Composite

From `short_interest_data`: combines SI % float and utilization, both z-scored and averaged. Forward-filled to daily. Stocks missing SI data get neutral score (~0 after z-scoring).

### Joint Step: Quality Factor

Added in March 2026. Loaded from `valuation_metrics_anchors` via `quality_factor.py`, forward-filled to daily, z-scored cross-sectionally. See Section 4 for full construction details.

### Common Sample Logic

`common_dates` = intersection of dates where ALL characteristics are available (from `st_dt`). All variance stats and lambda distributions are computed on this common sample for clean apples-to-apples comparison. Full extended history is saved to DB for backtest use.

### Outputs

```python
results = {
    'UFV', 'mkt_UFV', 'size_UFV', 'sec_UFV', 'joint_UFV',    # variance stats
    'resid_mkt', 'resid_size', 'resid_sec', 'resid_joint',    # common sample residuals
    'resid_sec_full', 'resid_full',                            # full history residuals
    'lambda_mkt', 'lambda_size', 'lambda_sec', 'lambda_joint', # lambda time series
    'beta_df', 'size_char_df', 'dynamic_size', 'si_composite',
    'mom_df', 'rev_df', 'quality_df',
    'universe', 'sec_cols', 'common_dates', 'st_dt', 'extended_st_dt'
}
```

---

## 4. Quality Factor

**File:** `quality_factor.py` (~835 lines)

**Usage:**
```python
from quality_factor import run, gridsearch
summary, annual, scores, gqf_weights, cqf_weights = run(Pxs_df, sectors_s)
grid = gridsearch(Pxs_df, sectors_s)
```

### Conceptual Design

The quality factor is **rate-conditioned**: it blends two sub-factors based on where interest rates are relative to their recent average:

```
composite = (1 - q) × GQF + q × CQF

where q = 0.0  (rates falling  → pure Growth Quality Factor)
          0.5  (rates neutral  → equal blend)
          1.0  (rates rising   → pure Conservative Quality Factor)
```

### Rate Signal

```python
rate_mom = USGG10YR - USGG10YR.rolling(QF_MAV_WINDOW).mean()
# Quantized to 3 states:
q = 0.0  if rate_mom < -QF_THRESHOLD   # falling
q = 1.0  if rate_mom > +QF_THRESHOLD   # rising
q = 0.5  otherwise                      # neutral
```

`USGG10YR` is a column in `Pxs_df` (US 10-year yield × 100, i.e. in percent). The quantization prevents churning on small rate fluctuations.

**Rationale:** The 2021/22 rate cycle produced a clean regime flip — growth metrics became strongly negative while current profitability metrics (OM, ROE) surged. This is a well-documented economic mechanism (discount rate → duration of growth stocks) not a COVID-specific artifact. The rate signal encodes this rotation without any special-case logic.

### Hyperparameters (gridsearch optimal)

| Parameter | Optimal | Grid Tested |
|-----------|---------|-------------|
| `QF_MAV_WINDOW` | 252 | {63, 126, 252} |
| `QF_THRESHOLD` | 15 bps | {15, 25, 50, 75} bps |

Gridsearch metric: spread_z and t-stat of composite scores vs 21d and 63d forward returns, averaged across all anchor dates.

### Quality Metrics

Raw metrics fetched from `valuation_metrics_anchors`:

| Category | Metrics |
|----------|---------|
| Growth — standalone | HSG, GS, GE, GGP, SGD, LastSGD, PIG, PSG |
| Profitability — level | OM, ROI, FCF_PG |
| Profitability — trend | OMd, ROId, ISGD |
| R&D intensity | r&d |

Derived metrics computed on-the-fly via `build_derived_metrics()`:

| Derived | Formula | Rationale |
|---------|---------|-----------|
| `GS/S_Vol`, `HSG/S_Vol`, `PSG/S_Vol` | base / max(S_Vol, 1.0) | Vol-adjust revenue growth |
| `GE/E_Vol`, `PIG/E_Vol` | base / max(E_Vol, 1.0) | Vol-adjust earnings growth |
| `GGP/GP_Vol` | base / max(GP_Vol, 1.0) | Vol-adjust gross profit growth |
| `GS*r2_S`, `SGD*r2_S`, `OMd*r2_S` | base × r2_S | Revenue trend consistency |
| `GE*r2_E`, `PIG*r2_E` | base × r2_E | Earnings trend consistency |
| `GGP*r2_GP` | base × r2_GP | GP trend consistency |

`Vol` floored at `VOL_MIN = 1.0` to prevent division instability on very stable series.

**Excluded metrics:** `ROE`, `ROE-P`, `ROEd` — contain price in denominator (book value per share × price), contaminated by valuation effects.

### Stock Scoring

Each stock is ranked 0-1 within its sector on each metric (sector-neutral ranking). Derived metrics are computed from the raw values. Final score:

```python
score_i = sum(weight_m × sector_rank_im  for m in eligible_metrics)
```

### GQF Weight Derivation

Weights derived from diagnostic stats on all anchor dates **excluding 2021 and 2022**:

1. Compute avg t-stat (21d + 63d horizons averaged) and avg spread_z for each metric
2. Eligibility: avg_spread_z > 0 AND above median spread_z AND avg_t above median
3. Cap at top 10 by avg_t
4. Weights = normalized avg_t (negatives clipped to 0)

**GQF components (window=252, threshold=15):**
HSG(0.119), GGP(0.113), GS(0.104), GS/S_Vol(0.101), GS*r2_S(0.100), PSG(0.099), PIG*r2_E(0.093), GGP/GP_Vol(0.093), PIG/E_Vol(0.092), GGP*r2_GP(0.088)

### CQF Weight Derivation

Same procedure but using **only 2021 and 2022** anchor dates:

**CQF components:**
OM(0.144), LastSGD(0.135), SGD*r2_S(0.107), SGD(0.101), PIG/E_Vol(0.096), ROI(0.089), GE/E_Vol(0.088), ISGD(0.085), PIG*r2_E(0.081), GE*r2_E(0.076)

### Diagnostic Stats (84 anchor dates as of March 2026)

Top metrics at 21d horizon (t-stat):
- ROId: t=3.30, consistency=63.1%
- OMd*r2_S: t=2.76, consistency=66.7%
- PIG/E_Vol: t=2.52, consistency=56.0%
- GS/S_Vol: t=2.40, consistency=60.7%
- OMd: t=2.37, consistency=67.9%
- GS: t=2.15, consistency=64.3%
- HSG: t=2.29, consistency=63.1%

Top metrics at 63d horizon (t-stat):
- PIG/E_Vol: t=3.86, consistency=67.9%
- PIG*r2_E: t=3.55, consistency=63.0%
- ROId: t=3.39, consistency=70.4%
- GE*r2_E: t=2.75, consistency=69.1%
- GS/S_Vol: t=2.68, consistency=60.5%

---

## 5. Quality Factor Diagnostic

**File:** `quality_factor_diag.py` (~599 lines)

**Usage:**
```python
from quality_factor_diag import run_quality_diag
summary, annual, raw = run_quality_diag(Pxs_df, sectors_s)
```

Computes per-metric predictive power statistics across anchor dates:

- **spread_z**: (top 10% median score − bottom 10% median score) / cross-sectional std — all stocks sector-ranked 0-1 before computing
- **t-stat**: t-test of spread_z time series vs zero
- **consistency**: % of dates where spread_z > 0

Outputs:
- Overall summary (all dates)
- Annual breakdown (per year, per horizon)
- Bottom table: 2021 only (to characterize conservative regime)

Strips `USGG10YR` from price columns before residualization to avoid contamination.

---

## 6. Primary Factor Backtest

**File:** `primary_factor_backtest.py` (~1187 lines)

**Usage:**
```python
from primary_factor_backtest import run, show_top_stocks
nav_base, nav_alt, port_base, port_alt, factor_by_date, all_metrics, gqf_w, cqf_w, snapshots = run(Pxs_df, sectors_s)
top50 = show_top_stocks(factor_by_date, all_metrics, sectors_s,
                        gqf_weights=gqf_w, cqf_weights=cqf_w, snapshots=snapshots)
```

### Workflow

1. User selects enhancement options interactively
2. Calc dates generated at chosen rebalancing frequency
3. `key_valuation_metrics` populated for any missing dates (mkt_cap, GS, GS_vol, etc.)
4. Quality factor scores loaded from `valuation_metrics_anchors`, GQF/CQF weights derived
5. Factor scores snapped to nearest anchor ≤ each rebalance date and forward-filled
6. Baseline backtest run (pure quality, equal weight, N=20)
7. Alternative backtest run with all selected enhancements

### User Input Options

| Option | Default | Notes |
|--------|---------|-------|
| Vol filter | off | Removes top 20% by 6m realized vol |
| Momentum blend | none | `12m1` or `idio` |
| Momentum weight | 1.0 | Combined score = (factor_z + w × mom_z) / (1 + w) |
| Volume-scaled idio | off | Volume-weighted idio momentum |
| Min market cap floor | none | In $M |
| Max stocks per sector | none | Sector cap with gradual relaxation |
| Number of stocks (N) | 20 | Portfolio size |
| Rebalancing frequency | 30 days | Calendar days between rebalances |
| Pre-filter fraction | 1.0 (no filter) | Keeps top X% by quality score before momentum blend |
| Concentration factor | 1.0 (equal weight) | Top ceil(N/2) stocks get `c/(c+1)` allocation |

### Pre-filter Logic

Applied after vol filter and market cap filter, **before** momentum blend. Ranks by pure quality composite score and keeps top X% of the eligible universe (always at least N stocks):

```python
n_keep = max(top_n, int(ceil(len(fdf) × prefilt_pct)))
fdf = fdf.nlargest(n_keep, 'factor')
```

This ensures momentum is applied within a quality-screened universe, not the full stock universe.

### Concentration Weighting

```
Top ceil(N/2) stocks → allocation = c / (c + 1)   (equal weight within group)
Bottom floor(N/2) stocks → allocation = 1 / (c + 1)  (equal weight within group)
```

At c=1.0 → equal weight throughout (no concentration). At c=2.0 → top half gets 2/3, bottom half gets 1/3.

### Return Calculation

**Buy-and-hold** between rebalance dates. Entry and exit both at rebalance date close:

```python
px_start = Pxs_df.loc[rebal_date, portfolio]
px_end   = Pxs_df.loc[next_rebal_date, portfolio]
stk_rets = px_end / px_start - 1
period_ret = (stk_rets × weights).sum()
```

Intra-period NAV is tracked daily for vol/drawdown calculation by computing cumulative return from entry price each day — but the terminal period return is always point-to-point, with no intra-period rebalancing.

**Important:** Earlier versions used daily mean returns (implicit daily rebalancing), which systematically understated performance and introduced a spurious relationship where higher rebalancing frequency appeared to hurt performance. The buy-and-hold implementation corrects this.

### `factor_by_date` Structure

```python
factor_by_date[rebal_date]  # DataFrame, index = bare tickers
# Columns: 'factor' (quality composite 0-1), 'Sector', 'mkt_cap'
```

`factor` is the raw quality composite score before any momentum blend. The momentum blend happens inside `run_backtest()` and is not stored back.

### Baseline Performance (as of March 2026, 88 rebalance dates)

```
CAGR: 28.6% | Vol: 25.2% | Sharpe: 1.14 | MDD: -35.6%
```

### Best Observed Enhancement Combination

```
Mom(12M1, w=5.0) + MktCap≥$1000M + SectorCap=10 + N=30 + PreFilt=40% + Conc=2.0x
CAGR: 46.7% | Vol: 32.8% | Sharpe: 1.43 | MDD: -38.5%
```

### `show_top_stocks()` Signature

```python
top50 = show_top_stocks(factor_by_date, all_metrics, sectors_s)
top50 = show_top_stocks(factor_by_date, all_metrics, sectors_s, ref_date='2024-06-01')
top50 = show_top_stocks(factor_by_date, all_metrics, sectors_s,
                        gqf_weights=gqf_w, cqf_weights=cqf_w, snapshots=snapshots)
```

With decomposition, shows per-metric contribution to each stock's quality score.

---

## 7. Supporting Scripts

### `calc_valuation_anchors.py`

Computes quality metrics snapshots and stores to `valuation_metrics_anchors`. Key functions:

- `recalc_historical(Pxs_df, recalc_before='2026-02-06')` — overwrites all dates before cutoff
- `fill_anchor_gaps(Pxs_df, min_gap_days=20)` — fills in midpoint dates between existing anchors
- `to_pxs_ticker(ticker, pxs_columns)` — normalizes ticker format (tries bare and 'AAPL US' format)

Metrics stored: Size, HSG, GS, GE, GGP, SGD, LastSGD, PIG, PSG, OM, ROI, FCF_PG, OMd, ROId, ISGD, r&d, S Vol, E Vol, GP Vol, r2 S, r2 E, r2 GP

### `ortex_fetcher_v2.py` (~1172 lines)

Fetches quarterly fundamentals from Ortex + AlphaVantage. Critical design details:

- **Non-GAAP blending:** `adj = adj_I × w + adj_II × (1-w)` where `w` from AV
- **`GROSS_UP_FACTOR = 0.75`** applied to non-GAAP adjustments
- **`ref_NNI`** uses `normalizedNetIncome` from FED estimates (not GAAP NI)
- **FEQ estimation:** robust historical matching using snapshot comparison with >1% threshold

### `ic_study.py` (~500 lines)

IC analysis: valuation metrics (P/S, P/Ee, P/GP) vs full model residuals from `factor_residuals_joint`. All ICs negative (confirms genuine value signal). Toggle `RESIDUAL_SOURCE` config to switch between full model residuals and raw returns.

---

## 8. Key Design Decisions & Rationale

### Why sector-rank before scoring?

Sector-ranking 0-1 within each sector neutralizes sector-level differences in growth rates. A software company with 15% revenue growth should be compared to other software companies, not to utilities. This is applied to every metric before computing the quality composite.

### Why z-score in the factor model but sector-rank in the backtest?

In the backtest, sector-ranking preserves the cross-sector quality signal in a way that's portfolio-ready (scores are directly comparable within sector). In the factor model regression, z-scoring the cross-sectional quality score gives the lambda an interpretable return-per-unit-of-quality interpretation, consistent with how other factors (size, momentum) are treated.

### Why exclude ROE?

ROE = Net Income / Book Value. Book value per share is closely related to price/share through accounting identities and retained earnings. This creates a contamination channel where ROE encodes valuation information — high ROE can simply mean high P/B rather than genuine profitability efficiency. Empirically, ROE shows strongly negative t-stats in the growth regime and highly variable behavior overall. Excluded alongside ROE-P and ROEd.

### Why buy-and-hold rather than daily rebalancing in backtest?

A monthly-rebalancing strategy should hold each portfolio unchanged between dates. Daily mean returns imply daily rebalancing back to equal weight, which: (a) incorrectly suppresses momentum winners, (b) understates returns, and (c) creates a spurious artifact where higher rebalancing frequency appears to hurt performance (more rebalances → more implicit daily rebalancing → more return suppression). The buy-and-hold implementation uses point-to-point returns: entry at rebalance date close, exit at next rebalance date close.

### Why quantize the rate signal rather than use a continuous lambda?

Continuous lambda creates churn — tiny day-to-day rate movements would continuously shift factor weights, creating spurious turnover and overfitting to noise. Quantizing to three states (falling / neutral / rising) ensures the regime only changes when rates have moved meaningfully (beyond the threshold) and hold there for at least one day. The 15bps threshold was selected via gridsearch.

### Why use t-stat as metric weight rather than spread_z or IC?

T-stat accounts for both the magnitude and the consistency of the predictive signal across dates. A metric with high spread_z but high variance in that spread will get a lower t-stat. This naturally penalizes metrics that work in some years but fail in others — which is exactly the kind of regime-sensitivity you want to screen out. Spread_z and avg_t are both required for eligibility but avg_t drives the weighting.

### Why cap at 10 components per factor?

Diminishing returns from additional metrics beyond ~8-10. More components increase the risk of including noise metrics that happen to pass the eligibility filter. 10 is a practical cap that keeps the factor interpretable while still diversifying across signal sources.

---

## 9. Pending Items

| Item | Priority | Notes |
|------|----------|-------|
| Add `ROEd` back to eligible pool | Medium | Confirmed t=1.62/2.63 — different from ROE (delta, no price contamination) |
| Test Option B: `factor = quality / val_q` | High | Quality × value combo |
| Test Option C: `factor = z(quality) + z(1/val_q)` | High | Additive quality + value blend |
| Walk-forward weight derivation | Medium | Derive GQF/CQF weights only from anchor dates strictly before each rebalance date. Currently uses full history for weights |
| Run `factor_model_step1` with vol scaling | Medium | Refresh `factor_residuals_joint` with vol-scaled idio momentum |
| Portfolio construction using full model residuals | Medium | Use `factor_residuals_joint` as the alpha signal |
| Universe restriction fix | High | New backtest script pulls ~492 stocks vs old script's 369. Gap is because `factor_by_date` comes from broader anchor universe vs `key_valuation_metrics` universe. Fix: restrict `fdf` to tickers present in `key_valuation_metrics` for that calc_date |
| Lambda winsorization | Low | Consider ±3σ clipping for reversal outlier dates |
| Gridsearch re-run with expanded dataset | Low | 84 dates now vs 69 at last gridsearch — optimal hyperparameters likely stable |

---

## 10. Input Data Conventions

- **`Pxs_df`**: DataFrame, index = trading dates, columns = bare tickers (e.g. `AAPL`, not `AAPL US`) plus `USGG10YR` (10Y yield × 100) and `SPX`. No NaN values for relevant stocks.
- **`sectors_s`**: Series, index = bare tickers, values = sector ETF strings (e.g. `XLK`, `XLV`). Also includes the ETF tickers themselves mapping to their own names.
- **`volumeTrd_df`**: Optional. Same shape as `Pxs_df`, values = daily dollar trading volume. Required for vol-scaled idio momentum.
- All tickers are **bare** throughout the codebase (no ` US` suffix). Normalization applied at DB load boundaries.
