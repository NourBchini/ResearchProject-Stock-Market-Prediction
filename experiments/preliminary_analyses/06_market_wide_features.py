#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 6: Add Market-Wide Data (Indices, Sector Performance) as Features

MARKET-WIDE INDICATORS EXPLANATION:
================================================================================

Why Market-Wide Data Matters for Stock Prediction:
--------------------------------------------------
Individual stocks don't move in isolation. They're influenced by:
1. Overall market sentiment (bull/bear markets)
2. Sector rotation (money moving between industries)
3. Macroeconomic factors (interest rates, dollar strength)
4. Risk-on/risk-off sentiment (investors seeking safety vs. growth)
5. Volatility regimes (calm vs. turbulent markets)

By incorporating market-wide data, we capture:
- Cross-asset correlations
- Market regime information
- Sector leadership signals
- Risk appetite indicators
- Macroeconomic context

================================================================================

1. VIX (CBOE Volatility Index) - "Fear Index"
   -------------------------------------------
   What it is: Measures market's expectation of 30-day volatility based on 
               S&P 500 option prices. Often called the "fear gauge."
   
   Why it's useful:
   - High VIX (>30): Market fear, uncertainty, potential selloffs
   - Low VIX (<15): Market complacency, stability, potential rallies
   - VIX spikes: Often precede market corrections
   - Negative correlation with SPY: When VIX rises, SPY typically falls
   
   Prediction value:
   - Predicts volatility regimes (calm vs. turbulent)
   - Indicates market sentiment shifts
   - Helps identify risk-on vs. risk-off periods
   - Early warning of market stress
   
   Example: VIX spike in March 2020 predicted COVID crash

2. DXY (US Dollar Index)
   ----------------------
   What it is: Measures the value of US dollar against a basket of major 
               currencies (EUR, JPY, GBP, CAD, SEK, CHF).
   
   Why it's useful:
   - Strong dollar: Hurts exports, multinational earnings, commodities
   - Weak dollar: Helps exports, boosts commodities, foreign earnings
   - Dollar trends: Reflect monetary policy and economic strength
   - Currency impact: Affects international company valuations
   
   Prediction value:
   - Indicates macroeconomic strength/weakness
   - Predicts sector performance (exporters vs. importers)
   - Correlates with commodity prices (inverse relationship)
   - Reflects Federal Reserve policy expectations
   
   Example: Strong dollar in 2014-2015 hurt multinational stocks

3. TLT (20+ Year Treasury Bond ETF)
   ---------------------------------
   What it is: Tracks long-term US Treasury bonds. Represents "safe haven" 
               asset and interest rate expectations.
   
   Why it's useful:
   - Bond prices rise: Economic uncertainty, flight to safety
   - Bond prices fall: Economic optimism, risk-on sentiment
   - Interest rates: Inverse relationship (rates up = bonds down)
   - Risk-off indicator: Money flows from stocks to bonds
   
   Prediction value:
   - Indicates risk appetite (bonds up = stocks down typically)
   - Predicts economic cycle (recession fears = bond rally)
   - Interest rate expectations (Fed policy signals)
   - Flight to quality indicator
   
   Example: Bond rally in 2019 predicted 2020 recession concerns

4. QQQ (Nasdaq-100 ETF)
   ---------------------
   What it is: Tracks Nasdaq-100 index, heavily weighted toward technology 
               and growth stocks.
   
   Why it's useful:
   - Tech leadership: Shows if growth stocks are outperforming
   - Risk-on indicator: QQQ outperforms in bull markets
   - Sector rotation: Tech vs. value performance
   - Market breadth: Tech strength indicates broad market health
   
   Prediction value:
   - Indicates growth vs. value preference
   - Predicts sector rotation
   - Shows market leadership (tech-led rallies)
   - Correlates with SPY but with higher beta
   
   Example: QQQ outperformance in 2020-2021 showed tech dominance

5. DIA (Dow Jones Industrial Average ETF)
   ---------------------------------------
   What it is: Tracks Dow Jones, representing large-cap blue-chip stocks 
               across various sectors.
   
   Why it's useful:
   - Value indicator: More value-oriented than QQQ
   - Market breadth: Shows if large caps are leading
   - Sector diversity: Less tech-heavy than QQQ
   - Stability measure: Blue-chip performance
   
   Prediction value:
   - Indicates value vs. growth rotation
   - Shows large-cap leadership
   - Different sector exposure than SPY
   - Market breadth indicator

6. GLD (Gold ETF)
   --------------
   What it is: Tracks gold prices, representing inflation hedge and safe haven.
   
   Why it's useful:
   - Inflation hedge: Rises with inflation expectations
   - Safe haven: Rises during market stress
   - Dollar inverse: Weak dollar = strong gold
   - Risk-off indicator: Gold up = stocks often down
   
   Prediction value:
   - Predicts inflation expectations
   - Indicates risk-off sentiment
   - Correlates with dollar strength (inverse)
   - Early warning of market stress

SECTOR INDICATORS:
================================================================================

7. XLK (Technology Sector ETF)
   ----------------------------
   Why useful: Tech is largest SPY sector. XLK performance indicates:
   - Growth stock sentiment
   - Innovation cycle strength
   - Risk-on appetite
   - Market leadership direction

8. XLF (Financial Sector ETF)
   ---------------------------
   Why useful: Financials are rate-sensitive. XLF performance indicates:
   - Interest rate expectations
   - Economic cycle position
   - Credit market health
   - Yield curve signals

9. XLV (Healthcare Sector ETF)
   ----------------------------
   Why useful: Defensive sector. XLV performance indicates:
   - Defensive rotation (recession fears)
   - Demographic trends
   - Regulatory environment
   - Risk-off sentiment

10. XLE (Energy Sector ETF)
    ------------------------
    Why useful: Commodity-driven. XLE performance indicates:
    - Oil price trends
    - Economic growth expectations
    - Inflation pressures
    - Geopolitical risks

11. XLY (Consumer Discretionary ETF)
    ---------------------------------
    Why useful: Consumer confidence proxy. XLY performance indicates:
    - Consumer spending strength
    - Economic optimism
    - Discretionary spending trends
    - Economic cycle position

DERIVED FEATURES:
================================================================================

- Rolling Correlations: How SPY moves relative to each indicator
- Relative Strength: SPY performance vs. each sector/index
- Divergences: When SPY and indicators move differently (signals)
- Momentum: Rate of change in each indicator

================================================================================
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance, provide instructions if not available
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠ yfinance not installed. Installing...")
    print("   Run: pip install yfinance")
    print("   Or: python -m pip install yfinance")
    print("\n   Attempting to install automatically...")
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "--quiet"])
        import yfinance as yf
        YFINANCE_AVAILABLE = True
        print("   ✓ yfinance installed successfully")
    except:
        print("   ✗ Could not install automatically")
        print("   Please install manually: pip install yfinance")

print("="*60)
print("MARKET-WIDE FEATURES DATA COLLECTION")
print("="*60)

# NOTE: Preserved as part of the preliminary investigation (multi-asset feature
# exploration). Not part of the revised paper.
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
spy_path = REPO_ROOT / "data" / "SPY.csv"
spy = pd.read_csv(spy_path, parse_dates=['Date'], index_col='Date')
start_date = spy.index.min()
end_date = spy.index.max()

print(f"\nSPY Date Range: {start_date.date()} to {end_date.date()}")
print("\nDownloading Market-Wide Data...")
print("="*60)

# Define tickers to download
market_data = {
    'Indices': {
        'VIX': '^VIX',  # Volatility Index
        'DXY': 'DX-Y.NYB',  # Dollar Index (alternative: UUP)
        'TLT': 'TLT',  # 20+ Year Treasury
        'QQQ': 'QQQ',  # Nasdaq-100
        'DIA': 'DIA',  # Dow Jones
        'GLD': 'GLD'   # Gold
    },
    'Sectors': {
        'XLK': 'XLK',  # Technology
        'XLF': 'XLF',  # Financials
        'XLV': 'XLV',  # Healthcare
        'XLE': 'XLE',  # Energy
        'XLY': 'XLY'   # Consumer Discretionary
    }
}

all_data = {}
failed_downloads = []

print("\n1. Downloading Market Indices...")
for name, ticker in market_data['Indices'].items():
    try:
        print(f"   → Downloading {name} ({ticker})...")
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(start=start_date, end=end_date)
        if not data.empty:
            # Use Close price
            all_data[name] = data['Close'].rename(name)
            print(f"     ✓ {name}: {len(data)} days")
        else:
            print(f"     ✗ {name}: No data")
            failed_downloads.append(name)
    except Exception as e:
        print(f"     ✗ {name}: Error - {str(e)}")
        failed_downloads.append(name)

print("\n2. Downloading Sector ETFs...")
for name, ticker in market_data['Sectors'].items():
    try:
        print(f"   → Downloading {name} ({ticker})...")
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(start=start_date, end=end_date)
        if not data.empty:
            all_data[name] = data['Close'].rename(name)
            print(f"     ✓ {name}: {len(data)} days")
        else:
            print(f"     ✗ {name}: No data")
            failed_downloads.append(name)
    except Exception as e:
        print(f"     ✗ {name}: Error - {str(e)}")
        failed_downloads.append(name)

# Combine all market data
if all_data:
    print("\n3. Combining Market Data...")
    # Combine all series into DataFrame
    market_df = pd.DataFrame()
    for name, series in all_data.items():
        # Remove timezone if present
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        # Normalize to date only
        series.index = pd.to_datetime(series.index).normalize()
        # Remove duplicates
        series = series[~series.index.duplicated(keep='last')]
        market_df[name] = series
    
    market_df.index.name = 'Date'
    
    # Align with SPY dates (merge on closest date, then forward fill)
    # First, reindex to SPY dates
    market_df = market_df.reindex(spy.index)
    # Forward fill missing values (carry last known value forward)
    market_df = market_df.fillna(method='ffill')
    # Backward fill any remaining NaNs at the start
    market_df = market_df.fillna(method='bfill')
    
    print(f"   ✓ Combined data: {market_df.shape}")
    print(f"   Date range: {market_df.index.min().date()} to {market_df.index.max().date()}")
    print(f"   Columns: {', '.join(market_df.columns)}")
    
    # Calculate derived features
    print("\n4. Calculating Derived Features...")
    
    # Rolling correlations with SPY Close
    spy_close = spy['Close']
    for col in market_df.columns:
        correlation = spy_close.rolling(window=60).corr(market_df[col])
        market_df[f'{col}_Correlation'] = correlation
    
    # Relative strength (SPY vs each indicator)
    for col in market_df.columns:
        if '_Correlation' not in col:
            # Calculate relative performance
            spy_returns = spy_close.pct_change()
            indicator_returns = market_df[col].pct_change()
            market_df[f'{col}_RelativeStrength'] = spy_returns - indicator_returns
    
    # Momentum (rate of change)
    for col in market_df.columns:
        if '_Correlation' not in col and '_RelativeStrength' not in col:
            market_df[f'{col}_Momentum'] = market_df[col].pct_change(periods=5)
    
    print(f"   ✓ Derived features calculated")
    print(f"   Total features: {len(market_df.columns)}")
    
    # Save market data
    output_path = 'data/market_wide_data.csv'
    market_df.to_csv(output_path)
    print(f"\n✓ Market-wide data saved to: {output_path}")
    
    # Create summary
    summary = {
        'download_date': datetime.now().isoformat(),
        'date_range': {
            'start': str(start_date.date()),
            'end': str(end_date.date())
        },
        'successful_downloads': list(all_data.keys()),
        'failed_downloads': failed_downloads,
        'total_features': len(market_df.columns),
        'base_indicators': list(market_df.columns[:len(all_data)]),
        'derived_features': [col for col in market_df.columns if '_' in col],
        'data_quality': {
            'total_days': len(market_df),
            'missing_values': market_df.isnull().sum().to_dict(),
            'completeness': (1 - market_df.isnull().sum() / len(market_df)).to_dict()
        }
    }
    
    import json
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'market_wide_data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"✓ Summary saved to: results/market_wide_data_summary.json")
    
    # Display sample
    print(f"\n{'='*60}")
    print("SAMPLE DATA (First 5 rows):")
    print(f"{'='*60}")
    print(market_df.head().to_string())
    
    print(f"\n{'='*60}")
    print("DATA STATISTICS:")
    print(f"{'='*60}")
    print(market_df.describe().to_string())
    
    print(f"\n{'='*60}")
    print("CORRELATION WITH SPY CLOSE (where data available):")
    print(f"{'='*60}")
    correlations = {}
    for col in market_df.columns:
        if '_Correlation' not in col and '_RelativeStrength' not in col and '_Momentum' not in col:
            # Only calculate correlation where both series have data
            combined = pd.concat([spy_close, market_df[col]], axis=1)
            combined.columns = ['SPY', col]
            valid_data = combined.dropna()
            if len(valid_data) > 100:  # Need sufficient data points
                corr = valid_data['SPY'].corr(valid_data[col])
                correlations[col] = corr
                data_points = len(valid_data)
                pct_available = (len(valid_data) / len(spy_close)) * 100
                print(f"  {col:20s}: {corr:7.4f} (n={data_points} days, {pct_available:.1f}% coverage)")
            else:
                available = market_df[col].notna().sum()
                print(f"  {col:20s}: Insufficient data (only {available} days available)")
    
    print(f"\n{'='*60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"\nNext Steps:")
    print("  1. Review data quality in summary file")
    print("  2. Integrate market data into model features")
    print("  3. Test performance improvement")
    print("  4. Use correlations to select most relevant indicators")
    
else:
    print("\n✗ No data downloaded. Please check internet connection and ticker symbols.")
    print("  Note: Some tickers may require different symbols or data sources.")
