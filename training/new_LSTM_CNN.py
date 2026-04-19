"""
Please note: this code was used to explain how the LSTM model works 
to a group of students who are participating in similar researsh therefore 
there are lots of comments and explanations in the code.
"""




"""
CNN-LSTM Model for Stock Price Prediction

This model combines Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) 
for predicting next-day OHLCV (Open, High, Low, Close, Volume).

Training period: Before 2019-10-14
Testing period: 2019-10-14 to 2019-10-28

Architecture:
- CNN branch: Extracts local patterns from time series
- LSTM branch: Captures temporal dependencies
- Both outputs are concatenated and fed to a fully connected layer
- Outputs 5 values: Open, High, Low, Close, Volume
"""

# Import necessary libraries
import torch  # PyTorch deep learning framework
import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Plotting
import torch.nn as nn  # Neural network modules
from torch.utils.data import Dataset, DataLoader  # Dataset handling
from sklearn.preprocessing import MinMaxScaler  # Data scaling
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

# ========== SET RANDOM SEEDS FOR REPRODUCIBILITY ==========
# Ensures same results every time
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("="*60)
print("CNN-LSTM Model for SPY Price Prediction")
print("="*60)

# ========== LOAD DATA ==========
print("\n1. Loading data...")
# Load SPY stock data
spy = pd.read_csv('../data/SPY.csv', parse_dates=['Date'], index_col='Date')

# Prepare features (input data) - only SPY data
# Features: Open, High, Low, Close, Volume
features = pd.concat([
    spy[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=lambda x: f"SPY_{x}"),
], axis=1)

# Prepare targets (what we want to predict)
# CHANGE: Now predicting all OHLCV instead of just Close price ================================================================
# OLD: targets = spy[['Close']].shift(-1)  # Only Close price
# NEW: Predict next day's Open, High, Low, Close, and Volume (5 values total)
targets = spy[['Open', 'High', 'Low', 'Close', 'Volume']].shift(-1)

# Combine features and targets
df = pd.concat([features, targets], axis=1).ffill().dropna()

features = df[features.columns]
targets = df[targets.columns]

print(f"   Data shape: {df.shape}")
print(f"   Date range: {df.index.min()} to {df.index.max()}")

# ========== PREPARE SEQUENCES ==========
print("\n2. Creating sequences...")
# CHANGE: Changed from 90 to 60 days to match Simple LSTM for fair comparison
# OLD: length = 90  # 90 days of history
# NEW: Using 60-day windows to match lstm_pytorch.py
length = 60
seq_dates = df.index[length:]

# Split into training and testing periods
# Training: All data before Oct 14, 2019
# Testing: Oct 14-28, 2019
train_mask = seq_dates < "2019-10-14"
test_mask = (seq_dates >= "2019-10-14") & (seq_dates <= "2019-10-28")

train_idx = np.where(train_mask)[0]
test_idx = np.where(test_mask)[0]

print(f"   Training samples: {len(train_idx)}")
print(f"   Testing samples: {len(test_idx)}")

# ========== SCALE DATA ==========
# Scale features and targets to [0.01, 0.99] range
feature_scaler = MinMaxScaler(feature_range=(0.01, 0.99))
target_scaler = MinMaxScaler(feature_range=(0.01, 0.99))

# Get training data for scaling
train_features = features.iloc[:train_idx[-1]+1]
train_targets = targets.iloc[:train_idx[-1]+1]

# Fit scalers on training data ONLY (prevents data leakage)
feature_scaler.fit(train_features)
target_scaler.fit(train_targets)

# Transform all data
features_scaled = feature_scaler.transform(features)
targets_scaled = target_scaler.transform(targets)

# ========== CREATE SEQUENCES ==========
# Create sliding window sequences
X_seq, y_seq = [], []
for i in range(length, len(targets_scaled)):
    # Input: past 90 days of features
    X_seq.append(features_scaled[i - length:i])
    # Target: next day's closing price
    y_seq.append(targets_scaled[i])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)
print(f"   Sequence shape: {X_seq.shape}")

# ========== CREATE DATASET ==========
class PredictDataset(Dataset):
    """
    Custom Dataset class
    Wraps data for PyTorch DataLoader
    """
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def __len__(self):
        return len(self.X)

# Create datasets
dataset = PredictDataset(X_seq, y_seq)
train_dataset = torch.utils.data.Subset(dataset, train_idx)
test_dataset = torch.utils.data.Subset(dataset, test_idx)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ========== MODEL ARCHITECTURE ==========
# Model hyperparameters
input_size = X_seq.shape[2]  # Number of features (5: OHLCV)
hidden_size = 64              # LSTM hidden size
num_layers = 1                # LSTM layers
# CHANGE: Output size increased from 1 to 5 to predict all OHLCV values==========================================================
# OLD: output_size = 1  # Only Close price prediction
# NEW: Predicting 5 values - Open, High, Low, Close, and Volume
output_size = 5               # Predicting 5 values (Open, High, Low, Close, Volume)
cnn_filters = 32              # Number of CNN filters

print("\n3. Building CNN-LSTM model...")
print(f"   Input size: {input_size}")
print(f"   Hidden size: {hidden_size}")
print(f"   CNN filters: {cnn_filters}")

class CNN_LSTM(nn.Module):
    """
    Hybrid CNN-LSTM Model
    
    Architecture:
    1. CNN branch: Extracts local patterns from time series
       - Conv1D: Detects patterns in temporal data
       - MaxPool: Reduces dimensionality
       - ReLU: Non-linearity
       - Average pooling: Summarizes CNN features
    
    2. LSTM branch: Captures temporal dependencies
       - LSTM: Processes sequences over time
       - Takes last time step output
    
    3. Fusion: Concatenate CNN and LSTM outputs
    
    4. Final layer: Maps combined features to prediction
    """
    def __init__(self, num_features, hidden_size, output_size, cnn_filters, num_layers=1):
        super(CNN_LSTM, self).__init__()
        
        # ========== CNN BRANCH ==========
        # Conv1D: processes time series data
        # Input: (batch, num_features, seq_len) after permute
        # Output: (batch, cnn_filters, reduced_seq_len)
        self.conv1 = nn.Conv1d(num_features, cnn_filters, kernel_size=3)
        
        # MaxPool: reduces sequence length by half
        self.pool = nn.MaxPool1d(2)
        
        # ReLU activation
        self.relu = nn.ReLU()
        
        # Dropout for CNN branch
        self.dropout_cnn = nn.Dropout(0.3)
        
        # ========== LSTM BRANCH ==========
        # LSTM: processes sequences to capture temporal patterns
        # Input: (batch, seq_len, num_features)
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers=num_layers, batch_first=True)
        
        # Dropout for LSTM branch
        self.dropout_lstm = nn.Dropout(0.3)
        
        # ========== FUSION LAYER ==========
        # Fully connected layer that combines CNN and LSTM outputs
        # Input size: hidden_size (from LSTM) + cnn_filters (from CNN)
        self.fc = nn.Linear(hidden_size + cnn_filters, output_size)
    
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
        
        Returns:
            Predicted value
        """
        # ========== CNN BRANCH ==========
        # Permute: change from (batch, seq_len, features) to (batch, features, seq_len)
        # Conv1D expects (batch, channels, seq_len)
        cnn_x = x.permute(0, 2, 1)
        
        # Apply CNN layers
        cnn_y = self.conv1(cnn_x)     # Convolution
        cnn_y = self.pool(cnn_y)      # Max pooling
        cnn_y = self.relu(cnn_y)      # ReLU activation
        cnn_y = self.dropout_cnn(cnn_y)  # Dropout
        
        # Average over time dimension to get single feature vector
        cnn_y = torch.mean(cnn_y, dim=2)  # Shape: (batch, cnn_filters)
        
        # ========== LSTM BRANCH ==========
        # Process through LSTM
        out, _ = self.lstm(x)  # Shape: (batch, seq_len, hidden_size)
        
        # Take only the last time step
        out = out[:, -1, :]  # Shape: (batch, hidden_size)
        
        # Apply dropout
        out = self.dropout_lstm(out)
        
        # ========== FUSION ==========
        # Concatenate CNN and LSTM outputs
        # Shape: (batch, hidden_size + cnn_filters)
        combined = torch.cat((out, cnn_y), dim=1)
        
        # Final prediction
        out = self.fc(combined)  # Shape: (batch, 1)
        
        return out

# Create model
model = CNN_LSTM(num_features=input_size, hidden_size=hidden_size,
                 output_size=output_size, cnn_filters=cnn_filters, num_layers=num_layers)

# Print model structure
print(model)
print(f"   Total parameters: {sum(p.numel() for p in model.parameters())}")

# ========== TRAINING SETUP ==========
# Loss function: Mean Squared Error
loss_fn = nn.MSELoss()

# Optimizer: AdamW with weight decay for regularization
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.002)

# Early stopping parameters
patience = 25  # Stop if no improvement for 25 epochs
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False
EPOCHS = 60
train_losses, val_losses = [], []

# ========== TRAINING LOOP ==========
print("\n4. Training model...")
for epoch in range(EPOCHS):
    # ========== TRAINING PHASE ==========
    model.train()
    epoch_train_loss = 0
    
    for X_batch, y_batch in train_loader:
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping: prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        epoch_train_loss += loss.item() * X_batch.size(0)
    
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    # ========== VALIDATION PHASE ==========
    model.eval()
    epoch_val_loss = 0
    
    with torch.no_grad():  # No gradient computation during validation
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            epoch_val_loss += loss.item() * X_batch.size(0)
    
    epoch_val_loss /= len(test_loader.dataset)
    val_losses.append(epoch_val_loss)
    
    # ========== EARLY STOPPING ==========
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()
        torch.save(model.state_dict(), "../weights/cnn_lstm_best.pth")
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
plt.title("CNN-LSTM Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig('../plots/cnn_lstm_training_curves.png')
plt.close()
print("   Saved: ../plots/cnn_lstm_training_curves.png")

# ========== EVALUATION ==========
print("\n6. Evaluating on test set...")
model.eval()
all_preds, all_true = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch)
        all_preds.append(preds.cpu().numpy())
        all_true.append(y_batch.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_true = np.concatenate(all_true, axis=0)

# Convert back to original scale
all_preds_rescaled = target_scaler.inverse_transform(all_preds)
all_true_rescaled = target_scaler.inverse_transform(all_true)

# CHANGE: Calculate metrics for EACH feature separately==========================================================================

# OLD: Single RMSE/MAE for just Close price
# NEW: Loop through all 5 features and calculate metrics for each
feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
print("\n" + "="*80)
print("EVALUATION RESULTS BY FEATURE")
print("Testing Period: October 14-28, 2019")
print("="*80)#=====================================================================================================================


for i, feature in enumerate(feature_names):
    # Calculate RMSE: how far off predictions are (in dollars)
    rmse = np.sqrt(np.mean((all_preds_rescaled[:, i] - all_true_rescaled[:, i]) ** 2))
    # Calculate MAE: average absolute error (in dollars)
    mae = np.mean(np.abs(all_preds_rescaled[:, i] - all_true_rescaled[:, i]))
    # Calculate MAPE: percentage error
    mape = np.mean(np.abs((all_preds_rescaled[:, i] - all_true_rescaled[:, i]) / all_true_rescaled[:, i])) * 100
    
    print(f"\n{feature}:")
    print(f"  RMSE: ${rmse:.4f}")
    print(f"  MAE:  ${mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")

# ========== SAVE RESULTS ==========
print("\n7. Saving results...")
feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']#=================================================================================

# CHANGE: Save predictions for ALL features instead of just Close
# OLD: Only saved Actual_Close and Predicted_Close
# NEW: Loop through all 5 features and save actual, predicted, and error for each
# Create comparison DataFrame for all features
comparison_df = pd.DataFrame({"Date": seq_dates[test_idx]})

for i, feature in enumerate(feature_names):
    comparison_df[f'Actual_{feature}'] = all_true_rescaled[:, i]
    comparison_df[f'Predicted_{feature}'] = all_preds_rescaled[:, i]
    comparison_df[f'Error_{feature}'] = np.abs(all_preds_rescaled[:, i] - all_true_rescaled[:, i])

comparison_df.to_csv('predictions_cnn_lstm.csv', index=False)
print(f"   Saved: predictions_cnn_lstm.csv")

# Display results
# CHANGE: Display only first 5 rows since we now have 15 columns (Date + 14 data columns)=================================================================================
# OLD: Showed all rows for Close price only
# NEW: Compact display with first 5 rows for all features
print("\nPrediction Results (showing first 5 rows):")
print(comparison_df.head().to_string(index=False))

# ========== PLOT PREDICTIONS ==========
print("\n8. Plotting predictions...")
#=================================================================================
# CHANGE: Create 4 subplots for OHLC features instead of single plot
# OLD: Single plot showing only Close price
# NEW: 2x2 grid showing Open, High, Low, and Close (Volume excluded from plots)
# Create subplots for all OHLC features (not Volume)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

feature_names = ['Open', 'High', 'Low', 'Close']

for idx, feature in enumerate(feature_names):
    ax = axes[idx]
    feature_idx = idx  # Open=0, High=1, Low=2, Close=3
    
    # Plot last 14 days
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

plt.suptitle("CNN-LSTM: Next-Day SPY OHLC Predictions\n(Oct 14-28, 2019)", 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../plots/cnn_lstm_predictions.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: ../plots/cnn_lstm_predictions.png")

print("\n" + "="*60)
print("Training and evaluation complete!")
print("="*60)

