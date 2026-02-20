#!/usr/bin/env python
# coding: utf-8

"""
IC Measurement Tool: Sector-Neutral Rank-Space Information Coefficients

Calculates rolling ICs for features against forward returns, with sector-neutral ranking.

Input:
    - features_df: pd.DataFrame with MultiIndex columns (stock, feature)
    - Pxs_df: pd.DataFrame with stock prices (dates × stocks)
    - sectors_s: pd.Series mapping stocks to sectors (static)
    
Output:
    - ic_df: pd.DataFrame with same structure as features_df, showing rolling ICs
"""

import pandas as pd
import numpy as np
from typing import Optional


def sector_neutral_rank(values, sectors):
    """
    Rank values within each sector.
    
    Args:
        values: pd.Series with stock index
        sectors: pd.Series mapping stocks to sectors
    
    Returns:
        pd.Series of fractional ranks [0, 1] within sectors
    """
    # Align indices
    aligned = pd.DataFrame({
        'value': values,
        'sector': sectors
    }).dropna()
    
    # Rank within each sector
    ranked = aligned.groupby('sector')['value'].rank(pct=True)
    
    return ranked


def calculate_forward_returns(prices_df, horizon=21):
    """
    Calculate forward returns for each stock.
    
    Args:
        prices_df: pd.DataFrame (dates × stocks)
        horizon: number of days forward (default 21)
    
    Returns:
        pd.DataFrame of forward returns
    """
    return (prices_df.shift(-horizon) / prices_df) - 1


def calculate_ic_single_date(feature_values, forward_returns, sectors):
    """
    Calculate IC for a single date across all stocks.
    
    Args:
        feature_values: pd.Series (stock → feature value)
        forward_returns: pd.Series (stock → forward return)
        sectors: pd.Series (stock → sector)
    
    Returns:
        float: IC value (rank correlation)
    """
    # Align all three series
    aligned = pd.DataFrame({
        'feature': feature_values,
        'return': forward_returns,
        'sector': sectors
    }).dropna()
    
    if len(aligned) < 10:  # Need minimum stocks for meaningful IC
        return np.nan
    
    # Rank within sectors
    feature_rank = sector_neutral_rank(aligned['feature'], aligned['sector'])
    return_rank = sector_neutral_rank(aligned['return'], aligned['sector'])
    
    # IC = correlation of ranks
    return feature_rank.corr(return_rank)


def calculate_rolling_ic(features_df, prices_df, sectors_s, 
                        forward_horizon=21, rolling_window=252):
    """
    Calculate rolling ICs for all features across all stocks.
    
    Args:
        features_df: pd.DataFrame with MultiIndex columns (stock, feature)
                     Index = dates
        prices_df: pd.DataFrame with prices (dates × stocks)
        sectors_s: pd.Series mapping stocks to sectors (static)
        forward_horizon: days forward for return calculation (default 21)
        rolling_window: window size for rolling IC (default 252)
    
    Returns:
        pd.DataFrame with same structure as features_df, containing rolling ICs
    """
    print("="*80)
    print("ROLLING IC CALCULATION")
    print("="*80)
    print(f"Forward return horizon: {forward_horizon} days")
    print(f"Rolling window: {rolling_window} days")
    print(f"Features shape: {features_df.shape}")
    print(f"Prices shape: {prices_df.shape}")
    print(f"Sectors: {len(sectors_s)} stocks")
    
    # Calculate forward returns
    print("\nCalculating forward returns...")
    forward_returns_df = calculate_forward_returns(prices_df, forward_horizon)
    
    # Get list of stocks and features from MultiIndex columns
    stocks = features_df.columns.get_level_values(0).unique()
    features = features_df.columns.get_level_values(1).unique()
    
    print(f"Stocks: {len(stocks)}")
    print(f"Features: {list(features)}")
    
    # Initialize IC dataframe (same structure as features_df)
    ic_df = pd.DataFrame(
        index=features_df.index,
        columns=features_df.columns,
        dtype=float
    )
    
    # Get valid dates (need full window + forward horizon)
    valid_start_idx = rolling_window - 1
    valid_end_idx = len(features_df) - forward_horizon
    
    if valid_end_idx <= valid_start_idx:
        print(f"\n❌ Error: Not enough data points!")
        print(f"   Need at least {rolling_window + forward_horizon} dates")
        print(f"   Have {len(features_df)} dates")
        return ic_df
    
    valid_dates = features_df.index[valid_start_idx:valid_end_idx]
    print(f"\nValid date range for IC: {valid_dates[0]} to {valid_dates[-1]}")
    print(f"Total valid dates: {len(valid_dates)}")
    
    # Calculate rolling IC for each stock-feature pair
    total_pairs = len(stocks) * len(features)
    processed = 0
    
    print("\nCalculating rolling ICs...")
    
    for stock in stocks:
        for feature in features:
            processed += 1
            
            if processed % 10 == 0 or processed == total_pairs:
                print(f"  Progress: {processed}/{total_pairs} "
                      f"({processed/total_pairs*100:.1f}%)", end='\r')
            
            # Get feature time series for this stock
            feature_ts = features_df[(stock, feature)]
            
            # Get forward returns for this stock
            if stock not in forward_returns_df.columns:
                continue
            
            returns_ts = forward_returns_df[stock]
            
            # Calculate IC for each date in rolling window
            for i, date in enumerate(valid_dates):
                # Get window of data
                window_start_idx = features_df.index.get_loc(date) - rolling_window + 1
                window_end_idx = features_df.index.get_loc(date) + 1
                
                window_dates = features_df.index[window_start_idx:window_end_idx]
                
                # Extract window data (cross-sectional at each date)
                window_ics = []
                
                for window_date in window_dates:
                    # Cross-sectional feature values across all stocks
                    feature_cross_section = features_df.loc[window_date].xs(
                        feature, level=1, drop_level=True
                    )
                    
                    # Cross-sectional forward returns
                    if window_date not in forward_returns_df.index:
                        continue
                    
                    returns_cross_section = forward_returns_df.loc[window_date]
                    
                    # Calculate IC for this date
                    ic = calculate_ic_single_date(
                        feature_cross_section,
                        returns_cross_section,
                        sectors_s
                    )
                    
                    if not np.isnan(ic):
                        window_ics.append(ic)
                
                # Average IC over window
                if len(window_ics) > 0:
                    ic_df.loc[date, (stock, feature)] = np.mean(window_ics)
    
    print(f"\n  Progress: {total_pairs}/{total_pairs} (100.0%)")
    
    # Summary statistics
    print("\n" + "="*80)
    print("IC SUMMARY STATISTICS")
    print("="*80)
    
    for feature in features:
        feature_ics = ic_df.xs(feature, level=1, axis=1).values.flatten()
        feature_ics = feature_ics[~np.isnan(feature_ics)]
        
        if len(feature_ics) > 0:
            print(f"\n{feature}:")
            print(f"  Mean IC: {np.mean(feature_ics):.4f}")
            print(f"  Median IC: {np.median(feature_ics):.4f}")
            print(f"  Std IC: {np.std(feature_ics):.4f}")
            print(f"  Min IC: {np.min(feature_ics):.4f}")
            print(f"  Max IC: {np.max(feature_ics):.4f}")
            print(f"  % Positive: {(feature_ics > 0).sum() / len(feature_ics) * 100:.1f}%")
    
    return ic_df


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    features_list = ['Value', 'Growth', 'Quality']
    
    # Sample features_df with MultiIndex columns
    np.random.seed(42)
    columns = pd.MultiIndex.from_product([stocks, features_list], names=['Stock', 'Feature'])
    features_df = pd.DataFrame(
        np.random.randn(len(dates), len(stocks) * len(features_list)),
        index=dates,
        columns=columns
    )
    
    # Sample prices_df
    prices_df = pd.DataFrame(
        100 * (1 + np.random.randn(len(dates), len(stocks)).cumsum(axis=0) * 0.01),
        index=dates,
        columns=stocks
    )
    
    # Sample sectors
    sectors_s = pd.Series({
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'GOOGL': 'Technology',
        'AMZN': 'Consumer',
        'TSLA': 'Consumer'
    })
    
    # Calculate ICs
    ic_df = calculate_rolling_ic(
        features_df=features_df,
        prices_df=prices_df,
        sectors_s=sectors_s,
        forward_horizon=21,
        rolling_window=252
    )
    
    print("\n" + "="*80)
    print("SAMPLE OUTPUT")
    print("="*80)
    print("\nIC DataFrame (first 5 rows):")
    print(ic_df.head())
    print("\nIC DataFrame (last 5 rows):")
    print(ic_df.tail())
