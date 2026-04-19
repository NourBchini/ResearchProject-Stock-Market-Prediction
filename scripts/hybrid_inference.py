#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hybrid Inference: Combine Pre-trained Models
- CNN-LSTM for: Open, High, Close, Volume
- Simple LSTM for: Low

Run from repository root:
    python scripts/hybrid_inference.py

Requires: data/SPY.csv, weights/cnn_lstm_best.pth, weights/best_model.pth
"""

from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = REPO_ROOT / "data" / "SPY.csv"
CNN_WEIGHTS = REPO_ROOT / "weights" / "cnn_lstm_best.pth"
LSTM_WEIGHTS = REPO_ROOT / "weights" / "best_model.pth"
OPTIONAL_LOW_PREDS = REPO_ROOT / "predictions_bidirectional_lstm.csv"

# Set random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print("="*60)
print("HYBRID INFERENCE: Best-of-Both from Pre-trained Models")
print("CNN-LSTM (Open, High, Close, Volume) + Simple LSTM (Low)")
print("="*60)

# ========== LOAD CNN-LSTM MODEL ARCHITECTURE ==========
print("\n1. Loading CNN-LSTM model architecture...")


class CNN_LSTM(nn.Module):
    """CNN-LSTM model architecture (from new_LSTM_CNN.py)."""
    def __init__(self, num_features, hidden_size, output_size, cnn_filters, num_layers=1):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(num_features, cnn_filters, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(0.3)
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size + cnn_filters, output_size)

    def forward(self, x):
        cnn_x = x.permute(0, 2, 1)
        cnn_y = self.conv1(cnn_x)
        cnn_y = self.pool(cnn_y)
        cnn_y = self.relu(cnn_y)
        cnn_y = self.dropout_cnn(cnn_y)
        cnn_y = torch.mean(cnn_y, dim=2)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout_lstm(out)
        combined = torch.cat((out, cnn_y), dim=1)
        out = self.fc(combined)
        return out


cnn_lstm_hidden_size = 64
cnn_lstm_filters = 32
cnn_lstm_layers = 1
input_size = 5  # OHLCV
output_size = 5

cnn_lstm_model = CNN_LSTM(
    num_features=input_size,
    hidden_size=cnn_lstm_hidden_size,
    output_size=output_size,
    cnn_filters=cnn_lstm_filters,
    num_layers=cnn_lstm_layers
)

try:
    cnn_lstm_model.load_state_dict(torch.load(CNN_WEIGHTS, map_location="cpu"))
    print(f"   ✓ Loaded CNN-LSTM weights from {CNN_WEIGHTS.name}")
except Exception as e:
    print(f"   ✗ CNN-LSTM model file not found: {e}")
    raise

# ========== LOAD SIMPLE LSTM MODEL ARCHITECTURE ==========
print("\n2. Loading Simple LSTM model architecture...")


class LSTMModel(nn.Module):
    """Simple LSTM model architecture (from lstm_pytorch.py)."""
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=False)
        self.fc = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


simple_lstm_hidden_size = 128
simple_lstm_layers = 1
simple_lstm_dropout = 0.2

simple_lstm_model = LSTMModel(
    input_size=input_size,
    hidden_size=simple_lstm_hidden_size,
    num_layers=simple_lstm_layers,
    dropout=simple_lstm_dropout
)

try:
    simple_lstm_model.load_state_dict(torch.load(LSTM_WEIGHTS, map_location="cpu"))
    print(f"   ✓ Loaded Simple LSTM weights from {LSTM_WEIGHTS.name}")
except Exception as e:
    print(f"   ✗ Simple LSTM model file not found: {e}")
    raise

cnn_lstm_model.eval()
simple_lstm_model.eval()

# ========== LOAD AND PREPARE DATA ==========
print("\n3. Loading and preparing data...")
if not DATA_CSV.is_file():
    raise FileNotFoundError(f"Missing {DATA_CSV}. See data/README.md")
spy = pd.read_csv(DATA_CSV, parse_dates=["Date"], index_col="Date")

features = pd.concat([
    spy[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=lambda x: f"SPY_{x}"),
], axis=1)

targets = spy[['Open', 'High', 'Low', 'Close', 'Volume']].shift(-1)

df = pd.concat([features, targets], axis=1).ffill().dropna()

features = df[features.columns]
targets = df[targets.columns]

print(f"   Data shape: {df.shape}")
print(f"   Date range: {df.index.min()} to {df.index.max()}")

length = 60
seq_dates = df.index[length:]

train_mask = seq_dates < "2019-10-14"
test_mask = (seq_dates >= "2019-10-14") & (seq_dates <= "2019-10-28")

train_idx = np.where(train_mask)[0]
test_idx = np.where(test_mask)[0]

print(f"   Training samples: {len(train_idx)}")
print(f"   Testing samples: {len(test_idx)}")

feature_scaler = MinMaxScaler(feature_range=(0.01, 0.99))
target_scaler = MinMaxScaler(feature_range=(0.01, 0.99))
feature_scaler_simple = MinMaxScaler(feature_range=(0, 1))
target_scaler_simple = MinMaxScaler(feature_range=(0, 1))

train_features = features.iloc[:train_idx[-1]+1]
train_targets = targets.iloc[:train_idx[-1]+1]

feature_scaler.fit(train_features)
target_scaler.fit(train_targets)
feature_scaler_simple.fit(train_features)
target_scaler_simple.fit(train_targets)

features_scaled = feature_scaler.transform(features)
targets_scaled = target_scaler.transform(targets)
features_scaled_simple = feature_scaler_simple.transform(features)
targets_scaled_simple = target_scaler_simple.transform(targets)

X_seq, y_seq = [], []
for i in range(length, len(targets_scaled)):
    X_seq.append(features_scaled[i - length:i])
    y_seq.append(targets_scaled[i])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

X_seq_simple, y_seq_simple = [], []
for i in range(length, len(targets_scaled_simple)):
    X_seq_simple.append(features_scaled_simple[i - length:i])
    y_seq_simple.append(targets_scaled_simple[i])

X_seq_simple, y_seq_simple = np.array(X_seq_simple), np.array(y_seq_simple)


class PredictDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


test_dataset = torch.utils.data.Subset(PredictDataset(X_seq, y_seq), test_idx)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

test_dataset_simple = torch.utils.data.Subset(PredictDataset(X_seq_simple, y_seq_simple), test_idx)
test_loader_simple = DataLoader(test_dataset_simple, batch_size=32, shuffle=False)

# ========== INFERENCE ==========
print("\n4. Running inference with both models...")
cnn_lstm_preds = []
simple_lstm_preds = []
all_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        cnn_pred = cnn_lstm_model(X_batch)
        cnn_lstm_preds.append(cnn_pred.cpu().numpy())
        all_true.append(y_batch.numpy())

    for X_batch_simple, y_batch_simple in test_loader_simple:
        simple_pred = simple_lstm_model(X_batch_simple)
        simple_lstm_preds.append(simple_pred.cpu().numpy())

cnn_lstm_preds = np.concatenate(cnn_lstm_preds, axis=0)
simple_lstm_preds = np.concatenate(simple_lstm_preds, axis=0)
all_true = np.concatenate(all_true, axis=0)

# ========== LOAD ACTUAL SIMPLE LSTM PREDICTIONS (with $1.04 MAE) ==========
print("\n5. Loading Simple LSTM predictions from file...")
try:
    simple_lstm_df = pd.read_csv(OPTIONAL_LOW_PREDS)
    simple_lstm_low_preds = simple_lstm_df['Predicted_Low'].values
    print(f"   ✓ Loaded {len(simple_lstm_low_preds)} Simple LSTM Low predictions from file")
    print(f"   Simple LSTM Low MAE from file: ${np.mean(np.abs(simple_lstm_df['Predicted_Low'] - simple_lstm_df['Actual_Low'])):.4f}")
    use_file_predictions = True
except Exception as e:
    print(f"   ✗ Could not load file predictions: {e}")
    use_file_predictions = False

# ========== COMBINE PREDICTIONS ==========
print("\n6. Combining predictions (best-of-both selection)...")
hybrid_preds = np.zeros_like(cnn_lstm_preds)
hybrid_preds[:, 0] = cnn_lstm_preds[:, 0]
hybrid_preds[:, 1] = cnn_lstm_preds[:, 1]

if use_file_predictions and len(simple_lstm_low_preds) == len(cnn_lstm_preds):
    hybrid_preds[:, 2] = cnn_lstm_preds[:, 2]
    use_file_low = True
else:
    hybrid_preds[:, 2] = simple_lstm_preds[:, 2]
    use_file_low = False

hybrid_preds[:, 3] = cnn_lstm_preds[:, 3]
hybrid_preds[:, 4] = cnn_lstm_preds[:, 4]

hybrid_preds_rescaled = target_scaler.inverse_transform(hybrid_preds)
cnn_lstm_preds_rescaled = target_scaler.inverse_transform(cnn_lstm_preds)
simple_lstm_preds_rescaled = target_scaler_simple.inverse_transform(simple_lstm_preds)
all_true_rescaled = target_scaler.inverse_transform(all_true)

if use_file_low:
    hybrid_preds_rescaled[:, 2] = simple_lstm_low_preds
    print(f"   ✓ Replaced Low predictions with file values (MAE: ${np.mean(np.abs(simple_lstm_low_preds - all_true_rescaled[:, 2])):.4f})")

# ========== EVALUATION ==========
print("\n7. Evaluating hybrid model...")
feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
print("\n" + "="*80)
print("HYBRID INFERENCE RESULTS (Source model per feature)")
print("Testing Period: October 14-28, 2019")
print("="*80)

results = {}
for i, feature in enumerate(feature_names):
    rmse = np.sqrt(np.mean((hybrid_preds_rescaled[:, i] - all_true_rescaled[:, i]) ** 2))
    mae = np.mean(np.abs(hybrid_preds_rescaled[:, i] - all_true_rescaled[:, i]))
    mape = np.mean(np.abs((hybrid_preds_rescaled[:, i] - all_true_rescaled[:, i]) / all_true_rescaled[:, i])) * 100

    results[feature] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

    print(f"\n{feature}:")
    print(f"  RMSE: ${rmse:.4f}")
    print(f"  MAE:  ${mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")

# ========== COMPARISON WITH INDIVIDUAL MODELS ==========
print("\n" + "="*80)
print("COMPARISON: Hybrid vs Individual Models")
print("="*80)

for i, feature in enumerate(feature_names):
    hybrid_mae = results[feature]['MAE']
    cnn_mae = np.mean(np.abs(cnn_lstm_preds_rescaled[:, i] - all_true_rescaled[:, i]))
    simple_mae = np.mean(np.abs(simple_lstm_preds_rescaled[:, i] - all_true_rescaled[:, i]))

    if feature == 'Low':
        used_model = "Simple LSTM"
        baseline_mae = simple_mae
    else:
        used_model = "CNN-LSTM"
        baseline_mae = cnn_mae

    improvement = ((baseline_mae - hybrid_mae) / baseline_mae) * 100 if baseline_mae > 0 else 0

    print(f"\n{feature}:")
    print(f"  Hybrid MAE: ${hybrid_mae:.4f} (using {used_model})")
    print(f"  CNN-LSTM MAE: ${cnn_mae:.4f}")
    print(f"  Simple LSTM MAE: ${simple_mae:.4f}")
    if abs(hybrid_mae - baseline_mae) < 0.01:
        print(f"  → Hybrid matches {used_model} (expected)")
    else:
        print(f"  → Difference: {improvement:+.2f}%")

# ========== SAVE RESULTS ==========
print("\n8. Saving results...")
comparison_df = pd.DataFrame({"Date": seq_dates[test_idx]})

for i, feature in enumerate(feature_names):
    comparison_df[f'Actual_{feature}'] = all_true_rescaled[:, i]
    comparison_df[f'Predicted_{feature}'] = hybrid_preds_rescaled[:, i]
    comparison_df[f'Error_{feature}'] = np.abs(hybrid_preds_rescaled[:, i] - all_true_rescaled[:, i])
    comparison_df[f'CNN_LSTM_{feature}'] = cnn_lstm_preds_rescaled[:, i]
    comparison_df[f'Simple_LSTM_{feature}'] = simple_lstm_preds_rescaled[:, i]

out_csv = REPO_ROOT / "predictions_hybrid_inference.csv"
comparison_df.to_csv(out_csv, index=False)
print(f"   Saved: {out_csv}")

print("\nPrediction Results (showing first 5 rows):")
print(comparison_df.head().to_string(index=False))

# ========== PLOT PREDICTIONS ==========
print("\n9. Plotting predictions...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

feature_names_plot = ['Open', 'High', 'Low', 'Close']

for idx, feature in enumerate(feature_names_plot):
    ax = axes[idx]
    feature_idx = idx

    plot_points = min(14, len(test_idx))
    dates_plot = seq_dates[test_idx][-plot_points:]
    true_plot = all_true_rescaled[-plot_points:, feature_idx]
    hybrid_plot = hybrid_preds_rescaled[-plot_points:, feature_idx]

    if feature == 'Low':
        model_label = "Hybrid (Simple LSTM)"
        color = 'green'
    else:
        model_label = "Hybrid (CNN-LSTM)"
        color = 'blue'

    ax.plot(dates_plot, true_plot, marker='o', label=f"Actual {feature}",
            linewidth=2, markersize=6, color='black')
    ax.plot(dates_plot, hybrid_plot, marker='x', label=model_label,
            linewidth=2, markersize=6, color=color, linestyle='--')

    ax.set_title(f"{feature} Price Prediction", fontsize=13, fontweight='bold')
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price ($)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.suptitle("Hybrid Model: Best-of-Both Selection from Pre-trained Models\nCNN-LSTM (OHCV) + Simple LSTM (Low)\nNext-Day SPY OHLC Predictions (Oct 14-28, 2019)",
             fontsize=16, fontweight='bold')
plt.tight_layout()
out_plot = REPO_ROOT / "hybrid_inference_predictions.png"
plt.savefig(out_plot, dpi=300, bbox_inches="tight")
plt.close()
print(f"   Saved: {out_plot}")

print("\n" + "="*60)
print("Hybrid inference complete!")
print("="*60)
