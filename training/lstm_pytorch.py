"""
Please note: this code was used to explain how the LSTM model works 
to a group of students who are participating in similar researsh therefore 
there are lots of comments and explanations in the code.
"""
# Import necessary libraries
from pathlib import Path

import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation (reading CSV files, etc.)
from sklearn.preprocessing import MinMaxScaler  # For scaling data to [0,1] range
from sklearn.metrics import mean_squared_error, mean_absolute_error  # For evaluation metrics
import torch  # PyTorch deep learning library
import torch.nn as nn  # PyTorch neural network modules
import torch.optim as optim  # PyTorch optimizers (Adam, SGD, etc.)
from torch.utils.data import Dataset, DataLoader  # For creating datasets and loading data in batches
import warnings
warnings.filterwarnings('ignore')  # Suppress warning messages

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = REPO_ROOT / "data" / "SPY.csv"
WEIGHTS_DIR = REPO_ROOT / "weights"

# Set random seeds for reproducibility - ensures we get the same results every time
# This is important for experiments to be reproducible
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():  # If GPU is available, also set its random seed
    torch.cuda.manual_seed(42)

class StockDataset(Dataset):
    """
    Custom Dataset class for stock data
    PyTorch requires data to be in a Dataset format for DataLoader
    This wraps our numpy arrays and converts them to PyTorch tensors
    """
    def __init__(self, X, y):
        # X: input features (sequences of past days)
        # y: target values (what we want to predict)
        # Convert numpy arrays to PyTorch tensors (floats)
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        # Return total number of samples in dataset
        return len(self.X)
    
    def __getitem__(self, idx):
        # Return single sample (input features and target) at given index
        # This is called by DataLoader to get a batch of data
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """
    Bidirectional LSTM model for stock prediction
    
    Architecture:
    1. Bidirectional LSTM - processes sequences in both directions
    2. Dropout - prevents overfitting by randomly disabling neurons
    3. Fully connected layer - maps LSTM output to predictions
    
    IMPROVEMENT: Bidirectional LSTM processes the SAME sequence in BOTH directions:
    - Forward: Day1 → Day2 → ... → Day60
    - Backward: Day60 → Day59 → ... → Day1
    - Combines both outputs for better context
    - NO future data needed! Just better feature extraction.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size  # Number of neurons in LSTM
        self.num_layers = num_layers    # Number of stacked LSTM layers
        
        # Unidirectional LSTM - processes sequence forward only
        # batch_first=True means input shape is (batch, seq_len, features)
        # bidirectional=False processes sequence in forward direction only
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=False)  # Unidirectional
        
        # Fully connected (dense) layer - converts LSTM output to predictions
        # Maps from hidden_size neurons to input_size outputs (OHLCV predictions)
        self.fc = nn.Linear(hidden_size, input_size)  # No * 2 for unidirectional
        
        # Dropout layer - randomly sets 20% of neurons to 0 during training
        # This prevents the model from overfitting (memorizing training data)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through the network
        x: input batch of shape (batch_size, sequence_length, num_features)
        """
        # Unidirectional LSTM initial states
        # Size: (num_layers, batch_size, hidden_size) for unidirectional
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate through unidirectional LSTM
        # Input: sequence of 60 days of data
        # Unidirectional processes sequence forward only
        # Output: sequence of hidden states (one for each day)
        out, _ = self.lstm(x, (h0, c0))
        
        # Get only the output from the last time step (most recent day)
        # Unidirectional output has shape (batch_size, hidden_size)
        out = out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Apply dropout to prevent overfitting
        # During training: randomly drops 20% of values
        # During inference: does nothing
        out = self.dropout(out)
        
        # Pass through fully connected layer to get final predictions
        # Maps from hidden_size to input_size (e.g., 64 -> 5 for OHLCV)
        out = self.fc(out)
        
        return out

def load_data(filepath):
    """
    Load and preprocess the SPY stock data from CSV file
    
    Steps:
    1. Read CSV file into pandas DataFrame
    2. Convert Date column to datetime format
    3. Sort by date (chronological order)
    4. Extract OHLCV features (Open, High, Low, Close, Volume)
    5. Convert to numpy array for processing
    
    Returns:
        df: DataFrame with dates
        data: numpy array of OHLCV values
        features: list of feature names
    """
    df = pd.read_csv(filepath)  # Read CSV file
    df['Date'] = pd.to_datetime(df['Date'])  # Convert date string to datetime
    df = df.sort_values('Date')  # Sort by date (oldest to newest)
    
    # Select relevant features for stock prediction
    # Open, High, Low, Close: price data
    # Volume: trading volume
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values  # Convert to numpy array (removes column names, keeps only values)
    
    return df, data, features

def create_sequences(data, n_steps=60):
    """
    Create sequences for LSTM model
    
    LSTM needs sequences of past data to predict future values
    We use 60 days of history to predict the next day
    
    Example with n_steps=3:
        X[0] = data[0:3]  (days 0, 1, 2)
        y[0] = data[3]    (day 3)
        
        X[1] = data[1:4]  (days 1, 2, 3)
        y[1] = data[4]    (day 4)
    
    Returns:
        X: sequences of n_steps days (input to model)
        y: next day's values (target for model to predict)
    """
    X, y = [], []
    for i in range(n_steps, len(data)):
        # Input: past n_steps days of OHLCV data
        X.append(data[i-n_steps:i])
        
        # Target: next day's OHLCV data (what we want to predict)
        y.append(data[i])
    
    return np.array(X), np.array(y)

def prepare_data(df, data, train_start_date, eval_start_date, eval_end_date, n_steps=60):
    """
    Prepare and split data for training and evaluation
    
    Steps:
    1. Find the date boundaries
    2. Split data into training (before test period) and evaluation sets
    3. Scale all data to [0,1] range (normalization)
    4. Create sequences for LSTM
    
    CRITICAL: Only fit scaler on training data to prevent data leakage!
    """
    dates = pd.to_datetime(df['Date'])
    
    # Get indices for train and evaluation periods
    # eval_start_idx: how many rows before the evaluation period starts
    eval_start_idx = dates[dates < eval_start_date].shape[0]
    
    # eval_end_idx: how many rows until the end of evaluation period
    eval_end_idx = dates[dates <= eval_end_date].shape[0]
    
    # Split data chronologically (time series - never shuffle!)
    # Train on all historical data before the evaluation period
    train_data = data[:eval_start_idx]
    
    # Evaluation data: need to include n_steps days before eval_start for sequences
    # Then extend to eval_end to get full evaluation period
    eval_data = data[eval_start_idx-n_steps:eval_end_idx]
    
    # Scale the data to [0, 1] range using MinMaxScaler
    # IMPORTANT: fit ONLY on training data to prevent data leakage
    # fit_transform: learns min/max from training data AND transforms it
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    
    # transform: applies same scaling learned from training data to evaluation data
    # This ensures we use the same scale for train and test
    eval_scaled = scaler.transform(eval_data)
    
    # Create sequences: convert daily data into sliding windows
    X_train, y_train = create_sequences(train_scaled, n_steps)
    X_eval, y_eval = create_sequences(eval_scaled, n_steps)
    
    # Return scaled sequences and scaler (need scaler to reverse transform predictions)
    return X_train, y_train, X_eval, y_eval, scaler

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, device='cpu'):
    """
    Train the model using training data and validate on validation data
    
    Training process:
    1. For each epoch (complete pass through training data):
       a. Train on training data
       b. Validate on validation data
       c. Save best model based on validation loss
    2. Stop early if validation loss stops improving (early stopping)
    
    Early stopping: Prevents overfitting by stopping when validation loss doesn't improve
    """
    best_val_loss = float('inf')  # Track best validation loss
    patience = 20  # How many epochs to wait for improvement
    patience_counter = 0  # Counter for epochs without improvement
    
    for epoch in range(epochs):
        # ========== TRAINING PHASE ==========
        # Set model to training mode (enables dropout, gradient computation, etc.)
        model.train()
        train_loss = 0.0
        
        # Process each batch in the training set
        for X_batch, y_batch in train_loader:
            # Move data to GPU if available, otherwise CPU
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass: get model predictions
            # zero_grad clears old gradients from previous iteration
            optimizer.zero_grad()
            outputs = model(X_batch)  # Model makes predictions
            
            # Calculate loss (difference between predictions and actual values)
            loss = criterion(outputs, y_batch)
            
            # Backward pass: compute gradients
            # backpropagation calculates how much each parameter affects the loss
            loss.backward()
            
            # Update parameters based on gradients
            optimizer.step()
            
            # Accumulate loss for this batch
            train_loss += loss.item()
        
        # ========== VALIDATION PHASE ==========
        # Set model to evaluation mode (disables dropout, more efficient)
        model.eval()
        val_loss = 0.0
        
        # Process validation batches (don't compute gradients - faster)
        with torch.no_grad():  # Disable gradient computation
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)  # Get predictions
                loss = criterion(outputs, y_batch)  # Calculate loss
                val_loss += loss.item()
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # ========== EARLY STOPPING ==========
        # If validation loss improved, save this as best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save model weights to file
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), WEIGHTS_DIR / "best_model.pth")
        else:
            # No improvement - increment counter
            patience_counter += 1
        
        # Print progress every 10 epochs or on first epoch
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # If no improvement for 'patience' epochs, stop training early
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load the best model (lowest validation loss)
    model.load_state_dict(torch.load(WEIGHTS_DIR / "best_model.pth", map_location="cpu"))
    return model

def evaluate_model(model, test_loader, scaler, device='cpu'):
    """
    Evaluate model on test/evaluation data
    
    Steps:
    1. Set model to evaluation mode (no dropout, no gradients)
    2. Get predictions for all test batches
    3. Combine predictions from all batches
    4. Convert back from normalized [0,1] to original price scale
    
    Returns predictions and actual values in original scale
    """
    # Set model to evaluation mode (disables dropout and other training features)
    model.eval()
    predictions = []  # Store model predictions
    actuals = []      # Store actual/true values
    
    # Disable gradient computation (not needed for evaluation, saves memory)
    with torch.no_grad():
        # Process each batch in test data
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)  # Move to GPU if available
            outputs = model(X_batch)  # Get model predictions
            
            # Convert PyTorch tensors to numpy arrays and store
            # cpu() moves data from GPU to CPU if needed
            # numpy() converts tensor to numpy array
            predictions.append(outputs.cpu().numpy())
            actuals.append(y_batch.numpy())
    
    # Combine all batches into single arrays
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    # Inverse transform: convert from normalized [0,1] back to original price scale
    # This undoes the MinMaxScaler transformation
    predictions_inv = scaler.inverse_transform(predictions)
    actuals_inv = scaler.inverse_transform(actuals)
    
    return predictions_inv, actuals_inv

def save_predictions(eval_dates, y_pred, y_actual, features, filename='predictions_pytorch.csv'):
    """
    Save model predictions to CSV file for analysis
    
    Creates a DataFrame with:
    - Date: when the prediction was made
    - Predicted values for each feature
    - Actual values for each feature
    - Error: absolute difference between predicted and actual
    
    Only saves Open, High, Low, Close (not Volume)
    """
    results_df = pd.DataFrame()
    results_df['Date'] = eval_dates  # Add date column
    
    # Save predicted, actual, and error for each price feature (Open, High, Low, Close)
    for i, feature in enumerate(features[:4]):  # Only save OHLC, not Volume
        # Predicted values from model
        results_df[f'Predicted_{feature}'] = y_pred[:, i]
        
        # Actual/true values
        results_df[f'Actual_{feature}'] = y_actual[:, i]
        
        # Error: absolute difference between predicted and actual
        results_df[f'Error_{feature}'] = np.abs(y_pred[:, i] - y_actual[:, i])
    
    # Save to CSV file
    results_df.to_csv(filename, index=False)
    print(f"\nPredictions saved to {filename}")
    return results_df

if __name__ == "__main__":
    # ========== CONFIGURATION ==========
    # Data file and preprocessing parameters
    filepath = str(DATA_CSV)  # Repository: data/SPY.csv (see data/README.md)
    n_steps = 60  # Number of past days to use for prediction (sequence length)
    
    # Date ranges for training and evaluation
    train_start_date = '2010-01-01'  # Start of training data (not used - we use all available data)
    eval_start_date = '2019-10-14'   # Start of evaluation/testing period
    eval_end_date = '2019-10-28'     # End of evaluation/testing period
    
    # ========== MODEL HYPERPARAMETERS ==========
    # Hidden size: Number of neurons in LSTM layer (determines model capacity)
    HIDDEN_SIZE = 128
    NUM_LAYERS = 1      # Number of stacked LSTM layers (Unidirectional, 1 layer, 128 units)
    DROPOUT = 0.2       # Dropout rate (20% of neurons randomly disabled during training)
    LEARNING_RATE = 0.001  # How fast model learns (step size for optimization)
    BATCH_SIZE = 32     # Number of samples processed before updating weights
    EPOCHS = 100        # Maximum number of training iterations
    
    # ========== DEVICE CONFIGURATION ==========
    # Use GPU if available (much faster), otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("="*60)
    print("PyTorch UNIDIRECTIONAL LSTM Model for SPY Price Prediction")
    print("="*60)
    print(f"Model Architecture:")
    print(f"  Type: Unidirectional LSTM (1 layer, 128 hidden units, forward direction only)")
    print(f"  Hidden Units: {HIDDEN_SIZE}")
    print(f"  Layers: {NUM_LAYERS}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Optimizer: Adam (lr={LEARNING_RATE})")
    
    # ========== STEP 1: LOAD DATA ==========
    print("\n1. Loading data...")
    df, data, features = load_data(filepath)
    print(f"   Data shape: {data.shape}")
    print(f"   Features: {features}")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # ========== STEP 2: PREPARE DATA ==========
    print("\n2. Preparing data...")
    # Split data chronologically and create sequences
    X_train, y_train, X_eval, y_eval, scaler = prepare_data(
        df, data, train_start_date, eval_start_date, eval_end_date, n_steps
    )
    print(f"   Training sequences: {X_train.shape}")
    print(f"   Evaluation sequences: {X_eval.shape}")
    
    # ========== STEP 3: SPLIT TRAINING DATA FOR VALIDATION ==========
    # Use 90% of training data for training, 10% for validation
    # Validation set is used to monitor training and prevent overfitting
    split_idx = int(0.9 * len(X_train))
    X_train_split = X_train[:split_idx]
    y_train_split = y_train[:split_idx]
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    
    # ========== STEP 4: CREATE DATASETS AND DATA LOADERS ==========
    # Convert numpy arrays to PyTorch Dataset objects
    train_dataset = StockDataset(X_train_split, y_train_split)
    val_dataset = StockDataset(X_val, y_val)
    eval_dataset = StockDataset(X_eval, y_eval)
    
    # Create DataLoaders that handle batching
    # Training: shuffle=True randomizes order each epoch (helps learning)
    # Validation/Test: shuffle=False keeps order (not needed, saves time)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ========== STEP 5: BUILD MODEL ==========
    print("\n3. Building LSTM model...")
    # Create model instance with specified architecture
    model = LSTMModel(
        input_size=len(features),  # 5 features (OHLCV)
        hidden_size=HIDDEN_SIZE,  # 128 neurons
        num_layers=NUM_LAYERS,     # 1 layer (best performing configuration)
        dropout=DROPOUT           # 0.2 dropout
    ).to(device)  # Move model to GPU if available
    
    # Print model information
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(model)
    
    # ========== STEP 6: DEFINE LOSS FUNCTION AND OPTIMIZER ==========
    # MSE (Mean Squared Error): penalizes large errors more than small ones
    # Good for regression tasks (predicting continuous values like prices)
    criterion = nn.MSELoss()
    
    # Adam optimizer: adaptive learning rate algorithm
    # Automatically adjusts learning rate for each parameter
    # lr=LEARNING_RATE sets initial learning rate
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ========== STEP 7: TRAIN MODEL ==========
    print("\n4. Training model...")
    # Train model on training data, validate on validation data
    # Returns best model (lowest validation loss)
    model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS, device=device)
    
    # ========== STEP 8: EVALUATE MODEL ==========
    print("\n5. Evaluating model on evaluation period...")
    # Get predictions on evaluation period (future dates, never seen during training)
    # Returns predictions and actual values in original price scale (not normalized)
    y_pred, y_actual = evaluate_model(model, eval_loader, scaler, device=device)
    
    # ========== STEP 9: CALCULATE METRICS ==========
    # Calculate performance metrics for each price feature
    print("\nEvaluation Results:")
    print("-" * 60)
    for i, feature in enumerate(features):
        # MSE (Mean Squared Error): average of squared errors (penalizes large errors more)
        mse = mean_squared_error(y_actual[:, i], y_pred[:, i])
        
        # MAE (Mean Absolute Error): average of absolute errors (in dollars)
        mae = mean_absolute_error(y_actual[:, i], y_pred[:, i])
        
        # RMSE (Root Mean Squared Error): sqrt of MSE (in dollars, comparable to MAE)
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error): percentage error
        mape = np.mean(np.abs((y_actual[:, i] - y_pred[:, i]) / y_actual[:, i])) * 100
        
        print(f"\n{feature}:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")
    
    # ========== STEP 10: SAVE PREDICTIONS ==========
    print("\n6. Saving predictions...")
    # Get dates for the evaluation period
    eval_dates = df[df['Date'].between(eval_start_date, eval_end_date, inclusive='both')]['Date'].values
    
    # Adjust dates to match sequence length (we lose first n_steps days for sequences)
    eval_dates = eval_dates[n_steps:]
    
    # Save predictions to CSV file for analysis
    predictions_df = save_predictions(eval_dates, y_pred, y_actual, features, filename='predictions_bidirectional_lstm.csv')
    
    print("\n" + "="*60)
    print("Training and evaluation complete!")
    print("="*60)

