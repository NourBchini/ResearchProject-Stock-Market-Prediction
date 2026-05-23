#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 11: Analyze Effect of AI on Stock Market Behavior
Compares market characteristics before and after AI dominance
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("AI EFFECT ON MARKET BEHAVIOR ANALYSIS")
print("="*60)

# Load data
import os
spy_path = '../../data/SPY.csv'
if not os.path.exists(spy_path):
    spy_path = '../data/SPY.csv'
spy = pd.read_csv(spy_path, parse_dates=['Date'], index_col='Date')

# Split at 2016-01-01
ai_split = '2016-01-01'
pre_ai = spy[spy.index < ai_split]
post_ai = spy[spy.index >= ai_split]

# Calculate market characteristics
def analyze_period(data, period_name):
    returns = data['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)  # Annualized
    avg_daily_return = returns.mean() * 252
    sharpe = avg_daily_return / volatility if volatility > 0 else 0
    
    # Price range
    price_range = data['High'].max() - data['Low'].min()
    avg_range = (data['High'] - data['Low']).mean()
    
    # Volume characteristics
    avg_volume = data['Volume'].mean()
    volume_volatility = data['Volume'].std() / data['Volume'].mean()
    
    # Trend characteristics
    price_change = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
    
    # Volatility clustering
    volatility_clustering = returns.rolling(20).std().std()
    
    return {
        'period': period_name,
        'days': len(data),
        'annualized_volatility': volatility * 100,
        'annualized_return': avg_daily_return * 100,
        'sharpe_ratio': sharpe,
        'price_range_pct': (price_range / data['Close'].mean()) * 100,
        'avg_daily_range_pct': (avg_range / data['Close'].mean()) * 100,
        'avg_volume': avg_volume,
        'volume_volatility': volume_volatility,
        'total_price_change_pct': price_change,
        'volatility_clustering': volatility_clustering
    }

# Analyze both periods
pre_analysis = analyze_period(pre_ai, "Pre-AI Era (Before 2016)")
post_analysis = analyze_period(post_ai, "Post-AI Era (After 2016)")

# Compare
print("\nMARKET CHARACTERISTICS COMPARISON:")
print("="*60)

comparison = pd.DataFrame([pre_analysis, post_analysis])

for col in comparison.columns:
    if col != 'period':
        pre_val = pre_analysis[col]
        post_val = post_analysis[col]
        change = ((post_val - pre_val) / pre_val * 100) if pre_val != 0 else 0
        
        print(f"\n{col.replace('_', ' ').title()}:")
        print(f"  Pre-AI:  {pre_val:.4f}")
        print(f"  Post-AI: {post_val:.4f}")
        print(f"  Change:  {change:+.2f}%")

# Save results
comparison.to_csv('results/ai_market_behavior_analysis.csv', index=False)

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print("\n1. Volatility Changes:")
vol_change = ((post_analysis['annualized_volatility'] - pre_analysis['annualized_volatility']) / 
              pre_analysis['annualized_volatility'] * 100)
print(f"   Volatility changed by {vol_change:+.2f}%")

print("\n2. Market Efficiency:")
print(f"   Sharpe Ratio - Pre-AI: {pre_analysis['sharpe_ratio']:.4f}")
print(f"   Sharpe Ratio - Post-AI: {post_analysis['sharpe_ratio']:.4f}")

print("\n3. Trading Activity:")
vol_vol_change = ((post_analysis['volume_volatility'] - pre_analysis['volume_volatility']) / 
                  pre_analysis['volume_volatility'] * 100)
print(f"   Volume volatility changed by {vol_vol_change:+.2f}%")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print("Results saved to: results/ai_market_behavior_analysis.csv")

