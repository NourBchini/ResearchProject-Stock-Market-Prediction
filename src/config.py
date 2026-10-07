


from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_CSV = REPO_ROOT / "Data" / "SPY.csv"

# Output folders

RESULTS_DIR = REPO_ROOT / "results"           # tables and summaries
PREDS_DIR = RESULTS_DIR / "preds"             # one CSV per (model, seed)
WEIGHTS_DIR = REPO_ROOT / "weights"           # saved model files (.pth)
FIGURES_DIR = REPO_ROOT / "figures"           # plots for the paper



# THE DATA

# once the data becomes a plain NumPy array the column names are gone so we name them here

FEATURES = ["Open", "High", "Low", "Close", "Volume"]

# for result tables later, making sure it is at culumn 3
CLOSE_INDEX = FEATURES.index("Close")

# Gu–Kelly–Xiu style signals you can compute from SPY alone.
# Leave False so Tables 2–4 stay the OHLCV-only paper run.
# Turn True, then retrain from scratch (old weights/*.pth will not match).
USE_FIRM_SIGNALS = False
SIGNAL_FEATURES = [
    "reversal_21",   # last ~1 month of close log-returns (short-term reversal)
    "mom_63",        # last ~1 quarter
    "mom_252",       # last ~1 year (momentum)
    "vol_21",        # 21-day realized volatility
    "liquidity_21",  # today's volume / 21-day average volume
    "amihud_21",     # 21-day Amihud illiquidity: |return| / dollar volume
]


DATA_END_DATE = "2026-05-22"


#SPLITTING THE DATA

# How many past trading days the model reads before making one prediction.
SEQ_LEN = 60

# The train/test split date
TEST_START_DATE = "2019-10-14"

# short evaluation window: 2019-10-14 to 2019-10-28 (11 days). for reporting results and continuity

SHORT_WINDOW_END = "2019-10-28"

# What fraction of the pre-test data is held back for early stopping. (chronologically the last 10 percent)

VAL_FRACTION = 0.10


# # MinMaxScaler squeezes every column into [0.01, 0.99].
# inputs are small and have a similar scale --> better for Neural nets performance.

SCALER_RANGE = (0.01, 0.99)


# WHAT THE MODEL PREDICTS


TARGET = "log_return"






# TRAINING SETTINGS 


SEEDS = [42, 1337, 2024, 7, 12345]

# our parameters for training


BATCH_SIZE = 64        # samples processed before each weight update
MAX_EPOCHS = 30        # hard ceiling on training rounds
PATIENCE = 8           # stop after this many rounds with no validation gain

LEARNING_RATE_LSTM = 1e-3      # for the standalone LSTM-128
LEARNING_RATE_HYBRID = 3e-4    # for Fusion and Cascade CNN-LSTM


# L2 regularization
# In PyTorch, Adam's weight_decay argument IS classic L2 regularization.
WEIGHT_DECAY_LSTM = 1e-4

# list dropout for the hybrids but no weight decay.
WEIGHT_DECAY_HYBRID = 0.0
