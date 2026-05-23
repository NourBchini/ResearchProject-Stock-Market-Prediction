
"""
Splitting Data Pre-Pandemic vs Post-Pandemic and Test Performance
Pandemic start: March 2020 (COVID-19 market crash)
"""

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import sys
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Import frozen model architecture
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("freeze_model", "freeze_model.py")
freeze_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze_module)
Frozen_CNN_LSTM = freeze_module.Frozen_CNN_LSTM

print("="*60)
print("PANDEMIC SPLIT ANALYSIS")
print("="*60)

# Load data
spy_path = '../../data/SPY.csv'
if not os.path.exists(spy_path):
    spy_path = '../data/SPY.csv'
spy = pd.read_csv(spy_path, parse_dates=['Date'], index_col='Date')

# Define pandemic start (March 2020 market crash)
pandemic_start = '2020-03-01'

# Split data
pre_pandemic = spy[spy.index < pandemic_start]
post_pandemic = spy[spy.index >= pandemic_start]

print(f"\nData Split:")
print(f"  Pre-Pandemic: {pre_pandemic.index.min()} to {pre_pandemic.index.max()} ({len(pre_pandemic)} days)")
print(f"  Post-Pandemic: {post_pandemic.index.min()} to {post_pandemic.index.max()} ({len(post_pandemic)} days)")

# Prepare features and targets
def prepare_data(data, sequence_length=60):
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
    
    return np.array(X_seq), np.array(y_seq), feature_scaler, target_scaler

# CRITICAL: Fit scalers on FULL dataset first, then use for all splits
# This ensures model sees data in same distribution it was trained on
print("\nFitting scalers on full dataset...")
full_features = spy[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
full_targets = full_features.shift(-1)
full_targets.columns = ['Target_' + col for col in full_targets.columns]
full_df = pd.concat([full_features, full_targets], axis=1).ffill().dropna()

feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
target_cols = ['Target_Open', 'Target_High', 'Target_Low', 'Target_Close', 'Target_Volume']

# Fit scalers on FULL dataset
global_feature_scaler = MinMaxScaler(feature_range=(0.01, 0.99))
global_target_scaler = MinMaxScaler(feature_range=(0.01, 0.99))
global_feature_scaler.fit(full_df[feature_cols])
global_target_scaler.fit(full_df[target_cols])

# Now prepare data for each period using the GLOBAL scalers
def prepare_data_with_global_scaler(data, sequence_length=60):
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    targets = features.shift(-1)
    targets.columns = ['Target_' + col for col in targets.columns]
    df = pd.concat([features, targets], axis=1).ffill().dropna()
    
    # Use GLOBAL scalers (fitted on full dataset)
    features_scaled = global_feature_scaler.transform(df[feature_cols])
    targets_scaled = global_target_scaler.transform(df[target_cols])
    
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(targets_scaled)):
        X_seq.append(features_scaled[i - sequence_length:i])
        y_seq.append(targets_scaled[i])
    
    return np.array(X_seq), np.array(y_seq)

print("\nPreparing datasets with global scalers...")
X_pre, y_pre = prepare_data_with_global_scaler(pre_pandemic)
X_post, y_post = prepare_data_with_global_scaler(post_pandemic)

# Split into train/test for each period
def split_train_test(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

X_pre_train, X_pre_test, y_pre_train, y_pre_test = split_train_test(X_pre, y_pre)
X_post_train, X_post_test, y_post_train, y_post_test = split_train_test(X_post, y_post)

print(f"\nPre-Pandemic Split:")
print(f"  Train: {len(X_pre_train)} samples")
print(f"  Test: {len(X_pre_test)} samples")
print(f"\nPost-Pandemic Split:")
print(f"  Train: {len(X_post_train)} samples")
print(f"  Test: {len(X_post_test)} samples")

# Load frozen model
try:
    model = Frozen_CNN_LSTM()
    model.load_state_dict(torch.load('models/frozen/cnn_lstm_frozen.pth', map_location='cpu'))
    model.eval()
except FileNotFoundError:
    print("⚠ Warning: Frozen model weights not found. Please run Step 1 first.")
    sys.exit(1)

# Evaluate function
def evaluate_model(X_test, y_test, model, period_name):
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    with torch.no_grad():
        predictions = model(X_tensor).numpy()
    
    # Inverse transform using GLOBAL scaler
    pred_rescaled = global_target_scaler.inverse_transform(predictions)
    true_rescaled = global_target_scaler.inverse_transform(y_test)
    
    # Calculate metrics for Close price (index 3)
    close_mae = np.mean(np.abs(pred_rescaled[:, 3] - true_rescaled[:, 3]))
    close_rmse = np.sqrt(np.mean((pred_rescaled[:, 3] - true_rescaled[:, 3]) ** 2))
    close_mape = np.mean(np.abs((pred_rescaled[:, 3] - true_rescaled[:, 3]) / true_rescaled[:, 3])) * 100
    
    print(f"\n{period_name} Performance:")
    print(f"  Close MAE: ${close_mae:.4f}")
    print(f"  Close RMSE: ${close_rmse:.4f}")
    print(f"  Close MAPE: {close_mape:.2f}%")
    
    return {
        'period': period_name,
        'close_mae': close_mae,
        'close_rmse': close_rmse,
        'close_mape': close_mape
    }

# Evaluate on both periods
results = []
results.append(evaluate_model(X_pre_test, y_pre_test, model, "Pre-Pandemic"))
results.append(evaluate_model(X_post_test, y_post_test, model, "Post-Pandemic"))

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('results/pandemic_split_results.csv', index=False)

print("\n" + "="*60)
print("PANDEMIC SPLIT ANALYSIS COMPLETE")
print("="*60)
print(f"\nResults saved to: results/pandemic_split_results.csv")

