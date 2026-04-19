import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Load data
spy = pd.read_csv('../data/SPY.csv', parse_dates=['Date'], index_col='Date')

# Prepare features - only SPY since we're not using TLT, GLD, DXY
features = spy[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=lambda x: f"SPY_{x}")
# CHANGE: Predict all OHLCV instead of just Close
# OLD: targets = spy[['Close']].shift(-1)
# NEW: Predict next day's Open, High, Low, Close, Volume
targets = spy[['Open', 'High', 'Low', 'Close', 'Volume']].shift(-1)

df = pd.concat([features, targets], axis=1).ffill().dropna()

features_df = df[features.columns]
targets_df = df[targets.columns]

# CHANGE: Changed from 90 to 60 days to match other models for fair comparison
# OLD: length = 90
# NEW: 60-day windows to match lstm_pytorch.py and new_LSTM_CNN.py
length = 60
seq_dates = df.index[length:]

# Split data
train_mask = seq_dates < "2019-10-14"
test_mask = (seq_dates >= "2019-10-14") & (seq_dates <= "2019-10-28")

train_idx = np.where(train_mask)[0]
test_idx = np.where(test_mask)[0]

# Scale data
feature_scaler = MinMaxScaler(feature_range=(0.01, 0.99))
target_scaler = MinMaxScaler(feature_range=(0.01, 0.99))

train_features = features_df.iloc[:train_idx[-1]+1]
train_targets = targets_df.iloc[:train_idx[-1]+1]

feature_scaler.fit(train_features)
target_scaler.fit(train_targets)

features_scaled = feature_scaler.transform(features_df)
targets_scaled = target_scaler.transform(targets_df)

# Create sequences
X_seq, y_seq = [], []
for i in range(length, len(targets_scaled)):
    X_seq.append(features_scaled[i - length:i])
    y_seq.append(targets_scaled[i])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)

class Predict(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def __len__(self):
        return len(self.X)

dataset = Predict(X_seq, y_seq)
train_dataset = torch.utils.data.Subset(dataset, train_idx)
test_dataset = torch.utils.data.Subset(dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model parameters
input_size = X_seq.shape[2]
hidden_size = 64
num_layers = 1
# CHANGE: Output size changed from 1 to 5 to predict all OHLCV
# OLD: output_size = 1
# NEW: Predict 5 values (Open, High, Low, Close, Volume)
output_size = 5
cnn_filters = 32

class CNN_LSTM(nn.Module):
    def __init__(self, num_features, hidden_size, output_size, cnn_filters, num_layers=1):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=cnn_filters, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to (batch, features, seq_len)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # Change back to (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

model = CNN_LSTM(num_features=input_size, hidden_size=hidden_size, output_size=output_size, 
                 cnn_filters=cnn_filters, num_layers=num_layers)

loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.002)

# Training
patience = 25
min_delta = 1e-6
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

EPOCHS = 60
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    epoch_train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        batch_loss = loss_fn(preds, y_batch)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_train_loss += batch_loss.item() * X_batch.size(0)
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            batch_loss = loss_fn(preds, y_batch)
            epoch_val_loss += batch_loss.item()       * X_batch.size(0)
    epoch_val_loss /= len(test_loader.dataset)
    val_losses.append(epoch_val_loss)

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            early_stop = True
            break

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

if early_stop:
    model.load_state_dict(best_model_state)
    print("Restored best model from early stopping.")

# Evaluation
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch)
        all_preds.append(preds.cpu().numpy())
        all_true.append(y_batch.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_true = np.concatenate(all_true, axis=0)

all_preds_rescaled = target_scaler.inverse_transform(all_preds)
all_true_rescaled = target_scaler.inverse_transform(all_true)

# CHANGE: Calculate metrics for each feature separately
# OLD: Single RMSE/MAE for all values
# NEW: Print metrics for each OHLCV feature
feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']

print("\n" + "="*80)
print("CNN-LSTM MODEL RESULTS (90-day windows)")
print("Test Period: October 14-28, 2019")
print("="*80)

for i, feature in enumerate(feature_names):
    rmse = np.sqrt(np.mean((all_preds_rescaled[:, i] - all_true_rescaled[:, i]) ** 2))
    mae = np.mean(np.abs(all_preds_rescaled[:, i] - all_true_rescaled[:, i]))
    mape = np.mean(np.abs((all_preds_rescaled[:, i] - all_true_rescaled[:, i]) / all_true_rescaled[:, i])) * 100
    
    print(f"\n{feature}:")
    print(f"  RMSE: ${rmse:.4f}")
    print(f"  MAE:  ${mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")

# Save predictions to CSV
# CHANGE: Save all OHLCV features instead of just Close
# OLD: Only saved Close price
# NEW: Save all 5 features
dates = df.index[length:]
feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']

comparison_df = pd.DataFrame({"Date": dates[test_idx]})

for i, feature in enumerate(feature_names):
    comparison_df[f'Actual_{feature}'] = all_true_rescaled[:, i]
    comparison_df[f'Predicted_{feature}'] = all_preds_rescaled[:, i]
    comparison_df[f'Error_{feature}'] = np.abs(all_preds_rescaled[:, i] - all_true_rescaled[:, i])

comparison_df.to_csv('predictions_cnn_lstm_90day.csv', index=False)
print(f"\nPredictions saved to predictions_cnn_lstm_90day.csv")

