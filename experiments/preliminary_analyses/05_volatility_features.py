#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 5: Find Best Features - Test Volatility Indicators
Tests: RSI, MACD, Bollinger Bands, ATR, Volatility

TECHNICAL INDICATORS EXPLANATION:
================================================================================

1. RSI (Relative Strength Index)
   ------------------------------
   What it is: A momentum oscillator that measures the speed and magnitude of 
               price changes to identify overbought or oversold conditions.
   
   Calculation:
   - RSI = 100 - (100 / (1 + RS))
   - RS (Relative Strength) = Average Gain / Average Loss
   - Uses 14-period lookback (default)
   - Average Gain = average of price increases over 14 periods
   - Average Loss = average of price decreases over 14 periods
   
   Interpretation:
   - RSI ranges from 0 to 100
   - RSI > 70: Overbought (potential sell signal)
   - RSI < 30: Oversold (potential buy signal)
   - RSI = 50: Neutral
   
   Why it's useful:
   - Identifies momentum shifts before price reversals
   - Helps detect when prices are stretched too far
   - Provides early warning of trend exhaustion
   - Useful for timing entries and exits

2. MACD (Moving Average Convergence Divergence)
   ---------------------------------------------
   What it is: A trend-following momentum indicator that shows the relationship
               between two moving averages of a security's price.
   
   Calculation:
   - MACD Line = 12-period EMA - 26-period EMA
   - Signal Line = 9-period EMA of MACD Line
   - Histogram = MACD Line - Signal Line
   - EMA = Exponential Moving Average (gives more weight to recent prices)
   
   Interpretation:
   - MACD > Signal: Bullish momentum (upward trend)
   - MACD < Signal: Bearish momentum (downward trend)
   - MACD crosses above Signal: Buy signal
   - MACD crosses below Signal: Sell signal
   - Histogram size indicates momentum strength
   
   Why it's useful:
   - Identifies trend changes and momentum shifts
   - Works well in trending markets
   - Provides clear buy/sell signals
   - Combines trend and momentum information

3. Bollinger Bands
   ----------------
   What it is: A volatility indicator consisting of three lines: a middle band
               (SMA) and two outer bands (standard deviations from middle).
   
   Calculation:
   - Middle Band = 20-period Simple Moving Average (SMA)
   - Upper Band = Middle Band + (2 × Standard Deviation)
   - Lower Band = Middle Band - (2 × Standard Deviation)
   - Band Width = Upper Band - Lower Band
   
   Interpretation:
   - Price near Upper Band: Potentially overbought
   - Price near Lower Band: Potentially oversold
   - Narrow Bands: Low volatility (consolidation)
   - Wide Bands: High volatility (trending or volatile)
   - Band Width: Measures volatility level
   
   Why it's useful:
   - Identifies volatility regimes
   - Helps detect mean reversion opportunities
   - Band width indicates market uncertainty
   - Adapts to changing market conditions
   - Useful for position sizing based on volatility

4. ATR (Average True Range)
   -------------------------
   What it is: A volatility indicator that measures market volatility by
               calculating the average of true ranges over a period.
   
   Calculation:
   - True Range = Maximum of:
     * Current High - Current Low
     * |Current High - Previous Close|
     * |Current Low - Previous Close|
   - ATR = 14-period moving average of True Range
   
   Interpretation:
   - Higher ATR: Higher volatility (more price movement)
   - Lower ATR: Lower volatility (less price movement)
   - ATR increases: Market becoming more volatile
   - ATR decreases: Market becoming calmer
   
   Why it's useful:
   - Measures actual price volatility (not just returns)
   - Accounts for gaps between trading sessions
   - Useful for risk management and position sizing
   - Helps identify volatility breakouts
   - Adapts to changing market conditions
   - Used in stop-loss placement

5. Volatility (Rolling Standard Deviation)
   ---------------------------------------
   What it is: A statistical measure of price dispersion, calculated as the
               standard deviation of percentage returns over a rolling window.
   
   Calculation:
   - Returns = Percentage change in Close price
   - Volatility = Standard deviation of returns over 20-period window
   - Annualized Volatility = Daily Volatility × √252 (trading days)
   
   Interpretation:
   - Higher Volatility: More price uncertainty and risk
   - Lower Volatility: More stable, predictable prices
   - Volatility spikes: Market stress or major events
   - Volatility clusters: High volatility tends to follow high volatility
   
   Why it's useful:
   - Direct measure of price uncertainty
   - Essential for risk assessment
   - Helps predict future volatility (volatility clustering)
   - Useful for portfolio optimization
   - Indicates market regime (calm vs. turbulent)
   - Critical for option pricing models

================================================================================
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# NOTE: Preserved as part of the preliminary investigation (feature-engineering
# exploration). Not part of the revised paper. Paths assume execution from this
# script's directory: experiments/preliminary_analyses/.
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPY_CSV = REPO_ROOT / "data" / "SPY.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("VOLATILITY INDICATORS TESTING")
print("="*60)
print("\nTesting Technical Indicators:")
print("  1. RSI - Momentum oscillator (overbought/oversold)")
print("  2. MACD - Trend and momentum indicator")
print("  3. Bollinger Bands - Volatility bands")
print("  4. ATR - Average True Range (volatility measure)")
print("  5. Volatility - Rolling standard deviation of returns")
print("="*60)

# Load data
import os
spy = pd.read_csv(SPY_CSV, parse_dates=['Date'], index_col='Date')

# Calculate technical indicators
def calculate_indicators(df):
    """
    Calculate all technical indicators with detailed explanations.
    
    Returns DataFrame with added indicator columns.
    """
    print("\nCalculating Technical Indicators...")
    
    # ========== RSI (Relative Strength Index) ==========
    # Measures momentum and identifies overbought/oversold conditions
    # Range: 0-100, where >70 is overbought, <30 is oversold
    print("  → Calculating RSI (14-period)...")
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  # Average gains
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # Average losses
    rs = gain / loss  # Relative Strength
    df['RSI'] = 100 - (100 / (1 + rs))  # Normalize to 0-100 range
    
    # ========== MACD (Moving Average Convergence Divergence) ==========
    # Trend-following momentum indicator showing relationship between EMAs
    # MACD Line = 12 EMA - 26 EMA, Signal = 9 EMA of MACD
    print("  → Calculating MACD (12, 26, 9)...")
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()  # Fast EMA (12 periods)
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()  # Slow EMA (26 periods)
    df['MACD'] = exp1 - exp2  # MACD Line
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()  # Signal Line
    
    # ========== Bollinger Bands ==========
    # Volatility bands around moving average (SMA ± 2 standard deviations)
    # Band width indicates volatility level
    print("  → Calculating Bollinger Bands (20-period, 2 std)...")
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()  # 20-period SMA
    bb_std = df['Close'].rolling(window=20).std()  # Standard deviation
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)  # Upper band
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)  # Lower band
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']  # Volatility measure
    
    # ========== ATR (Average True Range) ==========
    # Measures volatility by averaging true price ranges
    # Accounts for gaps and provides actual volatility measure
    print("  → Calculating ATR (14-period)...")
    high_low = df['High'] - df['Low']  # Current period range
    high_close = np.abs(df['High'] - df['Close'].shift())  # Gap up
    low_close = np.abs(df['Low'] - df['Close'].shift())  # Gap down
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)  # True Range (accounts for gaps)
    df['ATR'] = true_range.rolling(window=14).mean()  # 14-period average
    
    # ========== Volatility (Rolling Standard Deviation) ==========
    # Statistical measure of price dispersion (returns volatility)
    # Higher values indicate more uncertainty and risk
    print("  → Calculating Volatility (20-period rolling std of returns)...")
    returns = df['Close'].pct_change()  # Percentage returns
    df['Volatility'] = returns.rolling(window=20).std()  # 20-period rolling std
    
    print("  ✓ All indicators calculated successfully")
    
    return df.fillna(method='bfill').fillna(0)

spy = calculate_indicators(spy)

# Prepare feature sets for testing
# Each set adds different indicators to baseline OHLCV features
feature_sets = {
    'Baseline (OHLCV)': {
        'features': ['Open', 'High', 'Low', 'Close', 'Volume'],
        'description': 'Raw price and volume data only (baseline)'
    },
    'With RSI': {
        'features': ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI'],
        'description': 'Adds momentum indicator (overbought/oversold signals)'
    },
    'With MACD': {
        'features': ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'MACD_Signal'],
        'description': 'Adds trend and momentum indicators (trend changes)'
    },
    'With Bollinger': {
        'features': ['Open', 'High', 'Low', 'Close', 'Volume', 'BB_Width'],
        'description': 'Adds volatility bands (volatility regime indicator)'
    },
    'With ATR': {
        'features': ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR'],
        'description': 'Adds true range volatility (actual price movement measure)'
    },
    'With Volatility': {
        'features': ['Open', 'High', 'Low', 'Close', 'Volume', 'Volatility'],
        'description': 'Adds returns volatility (statistical dispersion measure)'
    },
    'All Indicators': {
        'features': ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'BB_Width', 'ATR', 'Volatility'],
        'description': 'Combines all indicators (comprehensive feature set)'
    }
}

results = []

for name, config in feature_sets.items():
    features = config['features']
    description = config['description']
    
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Description: {description}")
    print(f"Features: {', '.join(features)}")
    print(f"{'='*60}")
    
    # Prepare data
    data = spy[features].dropna()
    targets = spy[['Close']].shift(-1).loc[data.index]
    
    if len(data) < 100:
        continue
    
    # Scale
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    targets_scaled = scaler.fit_transform(targets)
    
    # Create sequences
    length = 60
    X, y = [], []
    for i in range(length, len(data_scaled)):
        X.append(data_scaled[i-length:i])
        y.append(targets_scaled[i])
    
    X, y = np.array(X), np.array(y)
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Simple LSTM
    class FeatureLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=64):
            super(FeatureLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.fc(out)
            return out
    
    # Train
    model = FeatureLSTM(len(features))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    
    for epoch in range(30):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        pred_test = model(X_test_t).numpy()
    
    # Inverse transform
    target_scaler = MinMaxScaler()
    target_scaler.fit(targets)
    pred_rescaled = target_scaler.inverse_transform(pred_test)
    true_rescaled = target_scaler.inverse_transform(y_test)
    
    mae = np.mean(np.abs(pred_rescaled - true_rescaled), axis=0)
    rmse = np.sqrt(np.mean((pred_rescaled - true_rescaled) ** 2, axis=0))
    
    print(f"  Results:")
    # Close is index 3 (Open=0, High=1, Low=2, Close=3, Volume=4)
    close_mae = mae[3] if len(mae) > 3 else mae[0]
    close_rmse = rmse[3] if len(rmse) > 3 else rmse[0]
    print(f"    Close MAE: ${close_mae:.4f}")
    print(f"    Close RMSE: ${close_rmse:.4f}")
    
    # Calculate improvement vs baseline
    if name != 'Baseline (OHLCV)':
        baseline_mae = next((r['mae'] for r in results if r['feature_set'] == 'Baseline (OHLCV)'), None)
        if baseline_mae:
            improvement = ((baseline_mae - close_mae) / baseline_mae) * 100
            print(f"    Improvement vs Baseline: {improvement:+.2f}%")
    
    results.append({
        'feature_set': name,
        'description': description,
        'num_features': len(features),
        'mae': close_mae,
        'rmse': close_rmse
    })

# Save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('mae')
results_df.to_csv(RESULTS_DIR / 'volatility_features_results.csv', index=False)

print(f"\n{'='*60}")
print("BEST FEATURE SETS (sorted by MAE):")
print(f"{'='*60}")

# Calculate improvements
baseline_mae = results_df[results_df['feature_set'] == 'Baseline (OHLCV)']['mae'].values[0]
results_df['improvement_pct'] = ((baseline_mae - results_df['mae']) / baseline_mae * 100).round(2)

print("\nSummary:")
print(results_df[['feature_set', 'num_features', 'mae', 'improvement_pct']].to_string(index=False))

print(f"\n{'='*60}")
print("KEY INSIGHTS:")
print(f"{'='*60}")

best = results_df.loc[results_df['mae'].idxmin()]
print(f"\n✓ Best Feature Set: {best['feature_set']}")
print(f"  MAE: ${best['mae']:.4f}")
print(f"  Improvement: {best['improvement_pct']:.2f}% vs baseline")
print(f"  Description: {best['description']}")

print(f"\n✓ Indicator Effectiveness:")
for idx, row in results_df.iterrows():
    if row['feature_set'] != 'Baseline (OHLCV)' and row['feature_set'] != 'All Indicators':
        indicator_name = row['feature_set'].replace('With ', '')
        print(f"  {indicator_name}: {row['improvement_pct']:+.2f}% improvement")

print(f"\n{'='*60}")
print("Results saved to: results/volatility_features_results.csv")
print(f"{'='*60}")

