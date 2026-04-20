
"""
Hybrid Model: CNN-LSTM for Open/High/Close, Simple LSTM for Low

This model combines:
- CNN-LSTM: Predicts Open, High, Close (3 features)
- Simple LSTM: Predicts Low (1 feature)
- Volume: Not predicted (can be set to 0 or use CNN-LSTM prediction)

Training period: Before 2019-10-14
Testing period: 2019-10-14 to 2019-10-28
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("="*60)
print("HYBRID MODEL: Best-of-Both Selection")
print("CNN-LSTM (Open, High, Close, Volume) + Simple LSTM (Low)")
print("="*60)

# ========== LOAD DATA ==========
print("\n1. Loading data...")
spy = pd.read_csv('../data/SPY.csv', parse_dates=['Date'], index_col='Date')

features = pd.concat([
    spy[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=lambda x: f"SPY_{x}"),
], axis=1)

targets = spy[['Open', 'High', 'Low', 'Close', 'Volume']].shift(-1)

df = pd.concat([features, targets], axis=1).ffill().dropna()

features = df[features.columns]
targets = df[targets.columns]

print(f"   Data shape: {df.shape}")
print(f"   Date range: {df.index.min()} to {df.index.max()}")

# ========== PREPARE SEQUENCES ==========
print("\n2. Creating sequences...")
length = 60
seq_dates = df.index[length:]

train_mask = seq_dates < "2019-10-14"
test_mask = (seq_dates >= "2019-10-14") & (seq_dates <= "2019-10-28")

train_idx = np.where(train_mask)[0]
test_idx = np.where(test_mask)[0]

print(f"   Training samples: {len(train_idx)}")
print(f"   Testing samples: {len(test_idx)}")

# ========== SCALE DATA ==========
feature_scaler = MinMaxScaler(feature_range=(0.01, 0.99))
target_scaler = MinMaxScaler(feature_range=(0.01, 0.99))

train_features = features.iloc[:train_idx[-1]+1]
train_targets = targets.iloc[:train_idx[-1]+1]

feature_scaler.fit(train_features)
target_scaler.fit(train_targets)

features_scaled = feature_scaler.transform(features)
targets_scaled = target_scaler.transform(targets)

# ========== CREATE SEQUENCES ==========
X_seq, y_seq = [], []
for i in range(length, len(targets_scaled)):
    X_seq.append(features_scaled[i - length:i])
    y_seq.append(targets_scaled[i])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)
print(f"   Sequence shape: {X_seq.shape}")

# ========== CREATE DATASET ==========
class PredictDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def __len__(self):
        return len(self.X)

dataset = PredictDataset(X_seq, y_seq)
train_dataset = torch.utils.data.Subset(dataset, train_idx)
test_dataset = torch.utils.data.Subset(dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ========== MODEL ARCHITECTURE ==========
input_size = X_seq.shape[2]  # 5 features (OHLCV)
hidden_size_cnn_lstm = 64    # CNN-LSTM hidden size
hidden_size_simple = 128     # Simple LSTM hidden size
num_layers = 1
cnn_filters = 32

print("\n3. Building Hybrid model...")
print(f"   Input size: {input_size}")
print(f"   CNN-LSTM hidden size: {hidden_size_cnn_lstm}")
print(f"   Simple LSTM hidden size: {hidden_size_simple}")
print(f"   CNN filters: {cnn_filters}")
print(f"   Strategy: Both learn ALL features, then select best for each")
print(f"     - Open, High, Close, Volume: CNN-LSTM")
print(f"     - Low: Simple LSTM")

class CNN_LSTM_Branch(nn.Module):
    """CNN-LSTM branch - learns ALL features (Open, High, Low, Close, Volume)"""
    def __init__(self, num_features, hidden_size, cnn_filters, num_layers=1):
        super(CNN_LSTM_Branch, self).__init__()
        
        # CNN branch
        self.conv1 = nn.Conv1d(num_features, cnn_filters, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(0.3)
        
        # LSTM branch
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.3)
        
        # Output: ALL 5 features (Open, High, Low, Close, Volume)
        self.fc = nn.Linear(hidden_size + cnn_filters, 5)
    
    def forward(self, x):
        # CNN branch
        cnn_x = x.permute(0, 2, 1)
        cnn_y = self.conv1(cnn_x)
        cnn_y = self.pool(cnn_y)
        cnn_y = self.relu(cnn_y)
        cnn_y = self.dropout_cnn(cnn_y)
        cnn_y = torch.mean(cnn_y, dim=2)  # Shape: (batch, cnn_filters)
        
        # LSTM branch
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Shape: (batch, hidden_size)
        out = self.dropout_lstm(out)
        
        # Fusion
        combined = torch.cat((out, cnn_y), dim=1)
        out = self.fc(combined)  # Shape: (batch, 5) - ALL features
        
        return out

class Simple_LSTM_Branch(nn.Module):
    """Simple LSTM branch - learns ALL features (Open, High, Low, Close, Volume)"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        super(Simple_LSTM_Branch, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=False)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 5)  # ALL 5 features
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)  # Shape: (batch, 5) - ALL features
        
        return out

class HybridModel(nn.Module):
    """
    Hybrid Model that selects best predictions:
    - CNN-LSTM for: Open, High, Close, Volume
    - Simple LSTM for: Low
    """
    def __init__(self, num_features, hidden_size_cnn_lstm, hidden_size_simple, cnn_filters, num_layers=1):
        super(HybridModel, self).__init__()
        
        # Both branches learn ALL features
        self.cnn_lstm_branch = CNN_LSTM_Branch(num_features, hidden_size_cnn_lstm, cnn_filters, num_layers)
        self.simple_lstm_branch = Simple_LSTM_Branch(num_features, hidden_size_simple, num_layers)
    
    def forward(self, x, select_best=True):
        # Both branches predict ALL 5 features
        cnn_lstm_pred = self.cnn_lstm_branch(x)  # Shape: (batch, 5) - [O, H, L, C, V]
        simple_lstm_pred = self.simple_lstm_branch(x)  # Shape: (batch, 5) - [O, H, L, C, V]
        
        if select_best:
            # Select best predictions based on performance:
            # Open: CNN-LSTM (index 0)
            # High: CNN-LSTM (index 1)
            # Low: Simple LSTM (index 2)
            # Close: CNN-LSTM (index 3)
            # Volume: CNN-LSTM (index 4)
            combined = torch.zeros_like(cnn_lstm_pred)
            combined[:, 0] = cnn_lstm_pred[:, 0]  # Open from CNN-LSTM
            combined[:, 1] = cnn_lstm_pred[:, 1]  # High from CNN-LSTM
            combined[:, 2] = simple_lstm_pred[:, 2]  # Low from Simple LSTM
            combined[:, 3] = cnn_lstm_pred[:, 3]  # Close from CNN-LSTM
            combined[:, 4] = cnn_lstm_pred[:, 4]  # Volume from CNN-LSTM
            return combined
        else:
            # Return both predictions for analysis
            return cnn_lstm_pred, simple_lstm_pred

# Create model
model = HybridModel(num_features=input_size, 
                   hidden_size_cnn_lstm=hidden_size_cnn_lstm,
                   hidden_size_simple=hidden_size_simple,
                   cnn_filters=cnn_filters, 
                   num_layers=num_layers)

print(model)
print(f"   Total parameters: {sum(p.numel() for p in model.parameters())}")

# ========== TRAINING SETUP ==========
loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.002)

patience = 25
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False
EPOCHS = 60
train_losses, val_losses = [], []

# ========== TRAINING LOOP ==========
print("\n4. Training model...")
for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        # During training, both branches predict all features
        # We compute loss for both and add them together (or average)
        cnn_lstm_pred, simple_lstm_pred = model(X_batch, select_best=False)
        
        # Compute losses for both branches
        loss_cnn_lstm = loss_fn(cnn_lstm_pred, y_batch)
        loss_simple_lstm = loss_fn(simple_lstm_pred, y_batch)
        
        # Combined loss - both branches learn all features
        loss = loss_cnn_lstm + loss_simple_lstm
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_train_loss += loss.item() * X_batch.size(0)
    
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # During validation, use best-of-both selection
            preds = model(X_batch, select_best=True)
            loss = loss_fn(preds, y_batch)
            epoch_val_loss += loss.item() * X_batch.size(0)
    epoch_val_loss /= len(test_loader.dataset)
    val_losses.append(epoch_val_loss)
    
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()
        torch.save(model.state_dict(), "../weights/hybrid_model_best.pth")
        print(f"Epoch {epoch+1}: New best validation loss - Model saved!")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            early_stop = True
            break
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

if early_stop:
    model.load_state_dict(best_model_state)
    print("Restored best model from early stopping.")

# ========== PLOT TRAINING CURVES ==========
print("\n5. Plotting training curves...")
plt.figure(figsize=(10,6))
plt.plot(range(1,len(train_losses)+1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1,len(val_losses)+1), val_losses, marker='x', label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Hybrid Model Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig('../plots/hybrid_model_training_curves.png')
plt.close()
print("   Saved: ../plots/hybrid_model_training_curves.png")

# ========== EVALUATION ==========
print("\n6. Evaluating on test set...")
model.eval()
all_preds, all_true = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        # Use best-of-both selection during evaluation
        preds = model(X_batch, select_best=True)
        all_preds.append(preds.cpu().numpy())
        all_true.append(y_batch.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_true = np.concatenate(all_true, axis=0)

all_preds_rescaled = target_scaler.inverse_transform(all_preds)
all_true_rescaled = target_scaler.inverse_transform(all_true)

feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
print("\n" + "="*80)
print("HYBRID MODEL EVALUATION RESULTS")
print("Architecture: Best-of-Both Selection")
print("  - Open, High, Close, Volume: CNN-LSTM predictions")
print("  - Low: Simple LSTM prediction")
print("Testing Period: October 14-28, 2019")
print("="*80)

for i, feature in enumerate(feature_names):
    rmse = np.sqrt(np.mean((all_preds_rescaled[:, i] - all_true_rescaled[:, i]) ** 2))
    mae = np.mean(np.abs(all_preds_rescaled[:, i] - all_true_rescaled[:, i]))
    mape = np.mean(np.abs((all_preds_rescaled[:, i] - all_true_rescaled[:, i]) / all_true_rescaled[:, i])) * 100
    
    print(f"\n{feature}:")
    print(f"  RMSE: ${rmse:.4f}")
    print(f"  MAE:  ${mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")

# ========== SAVE RESULTS ==========
print("\n7. Saving results...")
comparison_df = pd.DataFrame({"Date": seq_dates[test_idx]})

for i, feature in enumerate(feature_names):
    comparison_df[f'Actual_{feature}'] = all_true_rescaled[:, i]
    comparison_df[f'Predicted_{feature}'] = all_preds_rescaled[:, i]
    comparison_df[f'Error_{feature}'] = np.abs(all_preds_rescaled[:, i] - all_true_rescaled[:, i])

comparison_df.to_csv('predictions_hybrid_model.csv', index=False)
print(f"   Saved: predictions_hybrid_model.csv")

print("\nPrediction Results (showing first 5 rows):")
print(comparison_df.head().to_string(index=False))

# ========== PLOT PREDICTIONS ==========
print("\n8. Plotting predictions...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

feature_names_plot = ['Open', 'High', 'Low', 'Close']

for idx, feature in enumerate(feature_names_plot):
    ax = axes[idx]
    feature_idx = idx  # Open=0, High=1, Low=2, Close=3
    
    plot_points = min(14, len(test_idx))
    dates_plot = seq_dates[test_idx][-plot_points:]
    true_plot = all_true_rescaled[-plot_points:, feature_idx]
    pred_plot = all_preds_rescaled[-plot_points:, feature_idx]
    
    ax.plot(dates_plot, true_plot, marker='o', label=f"Actual {feature}", linewidth=2, markersize=6)
    ax.plot(dates_plot, pred_plot, marker='x', label=f"Predicted {feature}", linewidth=2, markersize=6)
    
    ax.set_title(f"{feature} Price Prediction", fontsize=13, fontweight='bold')
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price ($)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.suptitle("Hybrid Model: Best-of-Both Selection\nCNN-LSTM (OHCV) + Simple LSTM (Low)\nNext-Day SPY OHLC Predictions (Oct 14-28, 2019)", 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../plots/hybrid_model_predictions.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: ../plots/hybrid_model_predictions.png")

print("\n" + "="*60)
print("Training and evaluation complete!")
print("="*60)

