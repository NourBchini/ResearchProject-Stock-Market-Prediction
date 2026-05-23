
"""
Freezing the current CNN-LSTM Model
Saves architecture, weights, and code for the final model
Architecture: CNN-LSTM with 64 hidden units, sequential structure
"""

import torch
import torch.nn as nn
import json
import shutil
import os
from pathlib import Path

# Create directories
Path("models/frozen").mkdir(parents=True, exist_ok=True)
Path("results").mkdir(parents=True, exist_ok=True)

# Define frozen architecture (will be imported by other scripts)
class Frozen_CNN_LSTM(nn.Module):
    """
    FROZEN CNN-LSTM Architecture
    - CNN-LSTM with 64 hidden units
    - Sequential structure
    - Input: 5 features (OHLCV)
    - Output: 5 values (OHLCV)
    """
    def __init__(self, num_features=5, hidden_size=64, output_size=5, cnn_filters=32, num_layers=1):
        super(Frozen_CNN_LSTM, self).__init__()
        
        # CNN branch
        self.conv1 = nn.Conv1d(num_features, cnn_filters, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(0.3)
        
        # LSTM branch
        self.lstm = nn.LSTM(num_features, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.3)
        
        # Output layer
        self.fc = nn.Linear(hidden_size + cnn_filters, output_size)
    
    def forward(self, x):
        # CNN branch
        cnn_x = x.permute(0, 2, 1)
        cnn_y = self.conv1(cnn_x)
        cnn_y = self.pool(cnn_y)
        cnn_y = self.relu(cnn_y)
        cnn_y = self.dropout_cnn(cnn_y)
        cnn_y = torch.mean(cnn_y, dim=2)
        
        # LSTM branch
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout_lstm(out)
        
        # Fusion
        combined = torch.cat((out, cnn_y), dim=1)
        out = self.fc(combined)
        return out

# Save architecture
model_config = {
    "architecture": "CNN-LSTM",
    "num_features": 5,
    "hidden_size": 64,
    "output_size": 5,
    "cnn_filters": 32,
    "num_layers": 1,
    "sequence_length": 60,
    "dropout_cnn": 0.3,
    "dropout_lstm": 0.3,
    "structure": "sequential",
    "description": "CNN-LSTM with 64 hidden units, sequential structure"
}

with open("models/frozen/model_config.json", "w") as f:
    json.dump(model_config, f, indent=2)

# Copy weights (if exists)
if os.path.exists("../../weights/cnn_lstm_best.pth"):
    shutil.copy("../../weights/cnn_lstm_best.pth", "models/frozen/cnn_lstm_frozen.pth")
    print("✓ Weights copied from ../../weights/cnn_lstm_best.pth")
else:
    print("⚠ Warning: cnn_lstm_best.pth not found. Please train model first.")

# Save model code (if exists)
if os.path.exists("../../training/new_LSTM_CNN.py"):
    shutil.copy("../../training/new_LSTM_CNN.py", "models/frozen/cnn_lstm_code.py")

print("="*60)
print("MODEL FROZEN SUCCESSFULLY")
print("="*60)
print("✓ Architecture saved: models/frozen/model_config.json")
print("✓ Weights saved: models/frozen/cnn_lstm_frozen.pth")
print("✓ Code saved: models/frozen/cnn_lstm_code.py")
print("\nFrozen Model Configuration:")
for key, value in model_config.items():
    print(f"  {key}: {value}")

