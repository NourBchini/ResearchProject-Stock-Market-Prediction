#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 10: Test One-Week Testing Periods to See if Performance Improves
Tests multiple one-week periods throughout the dataset
"""

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import sys
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Import frozen model
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("freeze_model", "freeze_model.py")
freeze_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze_module)
Frozen_CNN_LSTM = freeze_module.Frozen_CNN_LSTM

print("="*60)
print("ONE-WEEK TESTING PERIODS ANALYSIS")
print("="*60)

# Load data
import os
spy_path = '../../data/SPY.csv'
if not os.path.exists(spy_path):
    spy_path = '../data/SPY.csv'
spy = pd.read_csv(spy_path, parse_dates=['Date'], index_col='Date')

# Prepare sequences
def prepare_sequences(data, sequence_length=60):
    # Use only OHLCV features (5 features)
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    targets = features.shift(-1)
    # Rename target columns to avoid duplicates
    targets.columns = ['Target_' + col for col in targets.columns]
    df = pd.concat([features, targets], axis=1).ffill().dropna()
    
    feature_scaler = MinMaxScaler(feature_range=(0.01, 0.99))
    target_scaler = MinMaxScaler(feature_range=(0.01, 0.99))
    
    # Use only the 5 feature columns (not targets)
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    target_cols = ['Target_Open', 'Target_High', 'Target_Low', 'Target_Close', 'Target_Volume']
    
    features_scaled = feature_scaler.fit_transform(df[feature_cols])
    targets_scaled = target_scaler.fit_transform(df[target_cols])
    
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(targets_scaled)):
        X_seq.append(features_scaled[i - sequence_length:i])
        y_seq.append(targets_scaled[i])
    
    return np.array(X_seq), np.array(y_seq), feature_scaler, target_scaler, df.index[sequence_length:]

X_seq, y_seq, feature_scaler, target_scaler, dates = prepare_sequences(spy)

# Load frozen model
try:
    model = Frozen_CNN_LSTM()
    model.load_state_dict(torch.load('models/frozen/cnn_lstm_frozen.pth', map_location='cpu'))
    model.eval()
except FileNotFoundError:
    print("⚠ Warning: Frozen model weights not found. Please run Step 1 first.")
    sys.exit(1)

# Test multiple one-week periods
results = []
test_periods = [
    ('2015-01-05', '2015-01-09'),
    ('2016-01-04', '2016-01-08'),
    ('2017-01-03', '2017-01-06'),
    ('2018-01-02', '2018-01-05'),
    ('2019-01-02', '2019-01-04'),
    ('2019-10-14', '2019-10-18'),
    ('2020-01-02', '2020-01-03'),
    ('2021-01-04', '2021-01-08'),
    ('2022-01-03', '2022-01-07'),
    ('2023-01-03', '2023-01-06')
]

print("\nTesting One-Week Periods:")
print("="*60)

for start_date, end_date in test_periods:
    # Find indices for this week
    mask = (dates >= start_date) & (dates <= end_date)
    week_indices = np.where(mask)[0]
    
    if len(week_indices) < 3:  # Need at least 3 days
        continue
    
    X_week = X_seq[week_indices]
    y_week = y_seq[week_indices]
    
    # Predict
    X_tensor = torch.tensor(X_week, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_tensor).numpy()
    
    # Inverse transform
    pred_rescaled = target_scaler.inverse_transform(predictions)
    true_rescaled = target_scaler.inverse_transform(y_week)
    
    # Calculate Close MAE
    close_mae = np.mean(np.abs(pred_rescaled[:, 3] - true_rescaled[:, 3]))
    close_rmse = np.sqrt(np.mean((pred_rescaled[:, 3] - true_rescaled[:, 3]) ** 2))
    
    print(f"\n{start_date} to {end_date}:")
    print(f"  Days: {len(week_indices)}")
    print(f"  Close MAE: ${close_mae:.4f}")
    print(f"  Close RMSE: ${close_rmse:.4f}")
    
    results.append({
        'start_date': start_date,
        'end_date': end_date,
        'days': len(week_indices),
        'close_mae': close_mae,
        'close_rmse': close_rmse
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('results/one_week_testing_results.csv', index=False)

print("\n" + "="*60)
print("ONE-WEEK TESTING COMPLETE")
print("="*60)
print(f"\nAverage Close MAE: ${results_df['close_mae'].mean():.4f}")
print(f"Best Week MAE: ${results_df['close_mae'].min():.4f}")
print(f"Worst Week MAE: ${results_df['close_mae'].max():.4f}")
print(f"\nResults saved to: results/one_week_testing_results.csv")

