#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4 (legacy): Open-price prediction with 70/30, 80/20, 90/10 splits.

Uses a global MinMax fit (preliminary protocol). For §6.5 Close-price results with
train-only scaling and the log-return fix, run:
  python analysis/05_split_anomaly_log_returns.py
"""

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("OPEN PRICE PREDICTION WITH DIFFERENT SPLITS")
print("="*60)

# Load data
import os
spy_path = '../../data/SPY.csv'
if not os.path.exists(spy_path):
    spy_path = '../data/SPY.csv'
spy = pd.read_csv(spy_path, parse_dates=['Date'], index_col='Date')

# Use only Open price
features = spy[['Open']]
targets = spy[['Open']].shift(-1)
df = pd.concat([features, targets], axis=1).ffill().dropna()

# Prepare sequences
length = 60
X_seq, y_seq = [], []
for i in range(length, len(df)):
    X_seq.append(df['Open'].iloc[i-length:i].values)
    y_seq.append(df['Open'].iloc[i])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# Scale
scaler = MinMaxScaler(feature_range=(0.01, 0.99))
X_scaled = scaler.fit_transform(X_seq.reshape(-1, 1)).reshape(X_seq.shape)
y_scaled = scaler.transform(y_seq.reshape(-1, 1)).flatten()

# Simple LSTM for Open prediction
class OpenPredictor(nn.Module):
    def __init__(self, hidden_size=64):
        super(OpenPredictor, self).__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Test different splits
splits = [0.7, 0.8, 0.9]
results = []

for split in splits:
    print(f"\n{'='*60}")
    print(f"Testing {int(split*100)}/{int((1-split)*100)} Split")
    print(f"{'='*60}")
    
    split_idx = int(len(X_scaled) * split)
    X_train = X_scaled[:split_idx]
    X_test = X_scaled[split_idx:]
    y_train = y_scaled[:split_idx]
    y_test = y_scaled[split_idx:]
    
    # Reshape for LSTM (batch, seq_len, features)
    X_train = X_train.reshape(-1, length, 1)
    X_test = X_test.reshape(-1, length, 1)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    # Ensure same length
    min_len = min(len(X_train), len(y_train))
    X_train = X_train[:min_len]
    y_train = y_train[:min_len]
    
    # Train model
    model = OpenPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    
    # Training
    for epoch in range(50):
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
    
    # Inverse transform - ensure same shape
    pred_rescaled = scaler.inverse_transform(pred_test.reshape(-1, 1)).flatten()
    true_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Ensure same length
    min_len = min(len(pred_rescaled), len(true_rescaled))
    pred_rescaled = pred_rescaled[:min_len]
    true_rescaled = true_rescaled[:min_len]
    
    mae = np.mean(np.abs(pred_rescaled - true_rescaled))
    rmse = np.sqrt(np.mean((pred_rescaled - true_rescaled) ** 2))
    mape = np.mean(np.abs((pred_rescaled - true_rescaled) / (true_rescaled + 1e-8))) * 100
    
    print(f"  MAE: ${mae:.4f}")
    print(f"  RMSE: ${rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    results.append({
        'split': f"{int(split*100)}/{int((1-split)*100)}",
        'train_size': len(X_train),
        'test_size': len(X_test),
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('results/open_prediction_splits.csv', index=False)

print(f"\n{'='*60}")
print("RESULTS SAVED")
print(f"{'='*60}")
print(results_df.to_string(index=False))

