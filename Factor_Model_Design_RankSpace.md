# Factor Model Design: Rank-Space & Sector-Neutral Methodology

## Overview

This document outlines a multi-factor equity model combining fundamental factors (valuation, growth, quality), price-based factors (momentum, volatility), and alternative data (short interest). 

**Core Innovation:** All testing and validation uses **rank-space transformations** with **sector-neutral returns** to isolate stock-level alpha from sector timing effects.

---

## Why Rank Space + Sector Neutrality?

### **Problem 1: Raw Values Are Non-Stationary**

```python
# P/E ratio distribution shifts across regimes
2018 (rate hikes):   Mean P/E = 18, Std = 8
2021 (zero rates):   Mean P/E = 28, Std = 15

# A P/E of 22 means different things in different regimes
```

### **Problem 2: Outliers Dominate**

```python
# One extreme value drives the regression
TSLA P/E = 800 (outlier)
Most stocks P/E = 15-25

# TSLA dominates coefficient estimation
```

### **Problem 3: Sector Effects Confound Stock Selection**

```python
# 2023 example (AI boom)
NVDA (Tech):  +240% return
XOM (Energy): +10% return

# Model thinks: "High momentum predicts returns"
# Reality: "Tech sector membership predicts returns"
```

### **Solution: Rank Space + Sector Neutrality**

```python
# Step 1: Rank all factors [0, 1]
Value_rank = Value.rank(pct=True)

# Step 2: Rank returns WITHIN each sector
sector_neutral_returns = returns.groupby(sector).rank(pct=True)

# Now we test: "Within Tech, do high-value stocks outperform?"
#              "Within Energy, do high-value stocks outperform?"
```

---

## Data Preparation Pipeline

### **Step 1: Rank Transform All Factors**

```python
def rank_transform(factor_series):
    """
    Convert factor to fractional ranks [0, 1].
    
    Benefits:
    - Stationary: uniform [0,1] distribution every period
    - Robust: outliers compressed to [0.99, 1.0]
    - Interpretable: top decile = ranks > 0.9
    """
    return factor_series.rank(pct=True)

# Example
Value_raw = pd.Series({'AAPL': 0.8, 'MSFT': 1.2, 'TSLA': 0.3})
Value_rank = rank_transform(Value_raw)
# Result: {'AAPL': 0.67, 'MSFT': 1.0, 'TSLA': 0.33}
```

---

### **Step 2: Sector-Neutral Returns**

```python
def sector_neutral_returns(returns, sector_mapping):
    """
    Rank returns within each sector.
    
    This isolates stock selection from sector timing:
    - NVDA ranks 1.0 within Tech (top tech stock)
    - XOM ranks 1.0 within Energy (top energy stock)
    
    Even though NVDA had higher absolute return, both get same rank
    within their respective sectors.
    """
    sector_neutral = pd.Series(index=returns.index, dtype=float)
    
    for sector in sector_mapping.unique():
        sector_mask = sector_mapping == sector
        sector_stocks = returns[sector_mask]
        
        # Rank within sector
        sector_ranks = sector_stocks.rank(pct=True)
        sector_neutral[sector_mask] = sector_ranks
    
    return sector_neutral

# Usage
sector_map = pd.Series({
    'NVDA': 'Technology', 
    'AAPL': 'Technology',
    'XOM': 'Energy', 
    'CVX': 'Energy'
})

returns = pd.Series({
    'NVDA': 0.15,  # Top in Tech
    'AAPL': 0.08,  # Bottom in Tech
    'XOM': 0.02,   # Top in Energy
    'CVX': -0.01   # Bottom in Energy
})

sector_neutral_ret = sector_neutral_returns(returns, sector_map)
# Result: {'NVDA': 1.0, 'AAPL': 0.0, 'XOM': 1.0, 'CVX': 0.0}
```

---

### **Step 3: Complete Preparation Function**

```python
def prepare_data_for_modeling(returns, factors, sector_mapping):
    """
    Prepare data in rank space with sector-neutral returns.
    
    Returns:
        dict with 'returns' and 'factors' ready for IC/regression
    """
    # 1. Sector-neutral returns
    returns_sector_neutral = sector_neutral_returns(returns, sector_mapping)
    
    # 2. Rank all style factors
    factors_ranked = {
        name: rank_transform(factor) 
        for name, factor in factors.items()
    }
    
    # 3. Add sector dummies (keep as 0/1, don't rank)
    sectors = pd.get_dummies(sector_mapping, drop_first=True)
    for col in sectors.columns:
        factors_ranked[f'Sector_{col}'] = sectors[col]
    
    return {
        'returns': returns_sector_neutral,
        'factors': factors_ranked
    }
```

---

## Information Coefficient (IC) in Rank Space

**IC = Correlation between factor rank and sector-neutral return rank**

Since both are already ranks, Pearson correlation = Spearman rank correlation:

```python
def calculate_ic(factor_rank, return_rank):
    """
    Calculate Information Coefficient.
    
    Interpretation:
    - IC > 0.05: Strong predictive power
    - IC > 0.03: Good
    - IC > 0.01: Weak but usable
    - IC < 0.01: Noise
    """
    aligned = pd.DataFrame({
        'factor': factor_rank,
        'return': return_rank
    }).dropna()
    
    return aligned['factor'].corr(aligned['return'])

# Usage
data = prepare_data_for_modeling(returns, factors, sector_map)
ic_value = calculate_ic(data['factors']['Value'], data['returns'])
print(f"Value IC: {ic_value:.3f}")
```

---

### **Rolling IC for Stability**

```python
def calculate_rolling_ic(factor_ts, returns_ts, sector_ts, window=252):
    """
    Track IC stability over time.
    
    Good factor characteristics:
    - Mean IC > 0.03
    - IC Std < 0.05
    - Consistency (% positive) > 70%
    """
    ics = []
    
    for i in range(window, len(factor_ts)):
        window_data = slice(i-window, i)
        
        # Rank and sector-neutralize
        factor_rank = rank_transform(factor_ts.iloc[i])
        return_rank = sector_neutral_returns(
            returns_ts.iloc[i], 
            sector_ts.iloc[i]
        )
        
        ic = calculate_ic(factor_rank, return_rank)
        ics.append(ic)
    
    return pd.Series(ics, index=factor_ts.index[window:])

# Usage
rolling_ic = calculate_rolling_ic(value_ts, returns_ts, sector_ts)

print(f"Mean IC: {rolling_ic.mean():.3f}")
print(f"IC Std: {rolling_ic.std():.3f}")
print(f"Consistency: {(rolling_ic > 0).sum() / len(rolling_ic):.1%}")
```

---

## Factor Architecture

### **Layer 1: Core Factors**

Build 7 interpretable factors (all will be ranked):

#### 1. **Market**
- Equal-weight portfolio return or SPY
- Captures systematic risk
- **Exception: Keep as raw return** (used for beta, not ranked)

#### 2. **Size**  
```python
Size_raw = np.log(market_caps)  # Log first
Size_rank = rank_transform(Size_raw)  # Then rank
```
- Log captures proportional differences
- Rank makes it stationary
- No orthogonalization (fundamental factor)

#### 3. **Value**
```python
# Build composite from valuation metrics
value_metrics = ['sP/S', 'sP/E', 'P/S', 'P/Ee', 'P/GP']
Value_composite = ic_weighted_avg(value_metrics)  # See below
Value_rank = rank_transform(Value_composite)
```

#### 4. **Growth**
```python
growth_metrics = ['GS', 'GE', 'HSG', 'SGD', 'GGP', 'PIG']
Growth_composite = ic_weighted_avg(growth_metrics)

# Size-neutralize BEFORE ranking
Growth_resid = residualize(Growth_composite, Size_raw)
Growth_rank = rank_transform(Growth_resid)
```
- Residualize vs Size (small caps grow faster structurally)

#### 5. **Quality**
```python
quality_metrics = ['OM', 'ROI', 'ROE', 'r2_S', 'r2_E', 'ISGD']
Quality_composite = ic_weighted_avg(quality_metrics)

# Size-neutralize BEFORE ranking
Quality_resid = residualize(Quality_composite, Size_raw)
Quality_rank = rank_transform(Quality_resid)
```
- Residualize vs Size (large caps have stable margins)

#### 6. **Momentum**
```python
Momentum_rank = rank_transform(Momentum_composite)
```
- No orthogonalization (behavioral factor)

#### 7. **Short Interest**
```python
SI_rank = rank_transform(SI_composite)
```

#### 8-18. **Sector Dummies**
- Binary 0/1 indicators for each sector (GICS)
- Drop one as reference
- **Keep as 0/1, do NOT rank**

---

### **IC-Weighted Composite Construction**

```python
def ic_weighted_composite(metrics_dict, returns_ts, sector_ts, window=252):
    """
    Weight metrics by their trailing IC.
    
    Args:
        metrics_dict: {name: pd.Series} of raw metric values
        returns_ts: time series of returns
        sector_ts: time series of sector mappings
        window: lookback for IC calculation
    
    Returns:
        Weighted composite (raw values, not yet ranked)
    """
    # Calculate IC for each metric
    ics = {}
    for name, metric in metrics_dict.items():
        ic = calculate_ic(
            rank_transform(metric),
            sector_neutral_returns(returns_ts, sector_ts)
        )
        ics[name] = ic
    
    # Weight by absolute IC
    abs_ics = {k: abs(v) for k, v in ics.items()}
    total = sum(abs_ics.values())
    
    if total == 0:
        # Equal weight fallback
        weights = {k: 1/len(metrics_dict) for k in metrics_dict}
    else:
        weights = {k: v/total for k, v in abs_ics.items()}
    
    # Composite
    composite = sum(weights[k] * metrics_dict[k] for k in metrics_dict)
    
    print("\nIC-Weighted Composite:")
    for name, w in weights.items():
        print(f"  {name}: {w:.1%} (IC={ics[name]:.3f})")
    
    return composite

# Usage
value_metrics = {
    'sP/S': trailing_ps_series,
    'sP/E': trailing_pe_series,
    'P/S': forward_ps_series,
    'P/Ee': forward_pe_series,
    'P/GP': price_to_gp_series
}

Value_composite = ic_weighted_composite(
    value_metrics, 
    returns_ts, 
    sector_ts
)

# Then rank it
Value_rank = rank_transform(Value_composite)
```

---

### **Targeted Orthogonalization**

Only remove **structural** dependencies (Size effects):

```python
from sklearn.linear_model import LinearRegression

def residualize(y, X):
    """Regress y on X, return residuals."""
    model = LinearRegression().fit(
        X.values.reshape(-1, 1), 
        y.values.reshape(-1, 1)
    )
    residuals = y - model.predict(X.values.reshape(-1, 1)).flatten()
    return pd.Series(residuals, index=y.index)

# Apply before ranking
Size_raw = np.log(market_caps)

Growth_resid = residualize(Growth_composite, Size_raw)
Quality_resid = residualize(Quality_composite, Size_raw)

# Now rank the residualized factors
Growth_rank = rank_transform(Growth_resid)
Quality_rank = rank_transform(Quality_resid)
```

**Do NOT orthogonalize:**
- Value vs Growth (value stocks grow slower - economically meaningful)
- Quality vs Growth (sustainable growers - real relationship)
- Momentum vs Volatility (trending stocks are volatile - behavioral)

---

## Interaction Term Validation (Rank Space)

Test interactions in rank space for robustness:

### **Stage 1: Construct in Rank Space**

```python
# Example: Crowded Long = high momentum + low short interest
Crowded_Long_rank = Momentum_rank * (1 - SI_rank)

# Interpretation:
# - Momentum_rank = 0.9, SI_rank = 0.1 → Crowded_Long = 0.81 (high crowding)
# - Momentum_rank = 0.9, SI_rank = 0.9 → Crowded_Long = 0.09 (low crowding)
```

### **Stage 2: Statistical Validation**

```python
def validate_interaction_rank_space(returns_rank, base_factors_rank, 
                                    interaction_rank, interaction_name):
    """
    Test if interaction adds value in rank space.
    
    Returns:
        dict with t-stat, incremental R², p-value
    """
    from sklearn.linear_model import LinearRegression
    from scipy import stats
    import numpy as np
    
    # Prepare data
    X_base = np.column_stack([base_factors_rank[f] for f in base_factors_rank])
    X_full = np.column_stack([X_base, interaction_rank.values])
    y = returns_rank.values
    
    # Models
    model_base = LinearRegression().fit(X_base, y)
    model_full = LinearRegression().fit(X_full, y)
    
    r2_base = model_base.score(X_base, y)
    r2_full = model_full.score(X_full, y)
    
    # T-stat for interaction coefficient
    n, k = X_full.shape
    residuals = y - model_full.predict(X_full)
    mse = np.sum(residuals**2) / (n - k - 1)
    
    X_centered = X_full - X_full.mean(axis=0)
    cov_matrix = mse * np.linalg.inv(X_centered.T @ X_centered)
    
    coef = model_full.coef_[-1]
    se = np.sqrt(cov_matrix[-1, -1])
    t_stat = coef / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 1))
    
    return {
        'name': interaction_name,
        'coef': coef,
        't_stat': t_stat,
        'p_value': p_value,
        'incremental_r2': r2_full - r2_base,
        'is_significant': abs(t_stat) > 2 and (r2_full - r2_base) > 0.005
    }

# Usage
data_rank = prepare_data_for_modeling(returns, factors, sector_map)

validation = validate_interaction_rank_space(
    returns_rank=data_rank['returns'],
    base_factors_rank=data_rank['factors'],
    interaction_rank=Crowded_Long_rank,
    interaction_name='Crowded_Long'
)

print(f"t-stat: {validation['t_stat']:.2f}")
print(f"Incremental R²: {validation['incremental_r2']:.4f}")
print(f"Significant: {validation['is_significant']}")
```

**Criteria for inclusion:**
- |t-stat| > 2 (5% significance)
- Incremental R² > 0.5%
- Clear economic story
- Robust out-of-sample

---

## Complete Workflow

```python
# ============================================================================
# COMPLETE FACTOR ENGINEERING WORKFLOW (RANK SPACE + SECTOR NEUTRAL)
# ============================================================================

# Step 1: Raw data
returns = cross_sectional_returns  # pd.Series
sector_map = stock_sectors  # pd.Series
market_caps = stock_market_caps  # pd.Series

# Step 2: Build raw composites (IC-weighted)
Value_raw = ic_weighted_composite(
    {'sP/S': sps, 'sP/E': spe, 'P/S': ps, 'P/Ee': pee, 'P/GP': pgp},
    returns_ts, sector_ts
)

Growth_raw = ic_weighted_composite(
    {'GS': gs, 'GE': ge, 'HSG': hsg, 'SGD': sgd, 'GGP': ggp, 'PIG': pig},
    returns_ts, sector_ts
)

Quality_raw = ic_weighted_composite(
    {'OM': om, 'ROI': roi, 'ROE': roe, 'r2_S': r2s, 'r2_E': r2e, 'ISGD': isgd},
    returns_ts, sector_ts
)

# Step 3: Size-neutralize Growth and Quality (in raw space)
Size_raw = np.log(market_caps)
Growth_resid = residualize(Growth_raw, Size_raw)
Quality_resid = residualize(Quality_raw, Size_raw)

# Step 4: Rank ALL factors
factors_rank = {
    'Size': rank_transform(Size_raw),
    'Value': rank_transform(Value_raw),
    'Growth': rank_transform(Growth_resid),
    'Quality': rank_transform(Quality_resid),
    'Momentum': rank_transform(Momentum_raw),
    'ShortInterest': rank_transform(SI_raw)
}

# Step 5: Add sector dummies (0/1, don't rank)
sectors = pd.get_dummies(sector_map, drop_first=True)
for col in sectors.columns:
    factors_rank[f'Sector_{col}'] = sectors[col]

# Step 6: Sector-neutral returns
returns_rank = sector_neutral_returns(returns, sector_map)

# Step 7: Test interactions (optional)
Crowded_Long_rank = factors_rank['Momentum'] * (1 - factors_rank['ShortInterest'])

validation = validate_interaction_rank_space(
    returns_rank, factors_rank, Crowded_Long_rank, 'Crowded_Long'
)

if validation['is_significant']:
    factors_rank['Crowded_Long'] = Crowded_Long_rank

# Step 8: Ready for modeling!
# All factors in [0,1] rank space
# Returns sector-neutralized
# Ready for regression or portfolio construction
```

---

## Train-Test Split

### **Recommended: 65-35 Split (2017-2025 data)**

```python
TRAIN_START = '2017-12-31'
TRAIN_END = '2022-06-30'     # ~4.5 years
TEST_START = '2022-07-01'
TEST_END = '2025-02-18'       # ~2.5 years

# Covers multiple regimes:
# Train: 2018-2022 (includes COVID, recovery)
# Test: 2022-2025 (rate hikes, AI boom, current)
```

### **Walk-Forward for Final Validation**

```python
INITIAL_TRAIN = '2017-12-31' to '2021-12-31'  # 4 years
FIRST_TEST = '2022-01-01'
REFIT_EVERY = 3 months  # Quarterly

# Gives ~12-13 out-of-sample test periods
```

---

## Key Takeaways

1. **Always rank factors** → Stationarity, robustness to outliers
2. **Always rank returns within sectors** → Isolates stock selection from sector timing
3. **IC measured in rank space** → Clean signal-to-noise measure
4. **Orthogonalize only Size effects** → Preserve economic meaning
5. **Test interactions in rank space** → More robust validation
6. **Sector dummies stay binary** → Don't rank 0/1 indicators

---

## References

- Fama-French (1993): Common risk factors in returns
- Barra Risk Models: Industry standard for factor construction
- Grinold & Kahn (2000): Active Portfolio Management
- AQR Factor Research: White papers on quality, value, momentum

---

*Document version: 2.0 - Rank Space & Sector Neutral*  
*Last updated: February 2025*
