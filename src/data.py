


import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import config  


#This is used to store the data in a structured way. (a pachage passed to training and evaluation)

@dataclass
class Splits:
    #dataclass is a decorator that automatically build the constructor
    #defining three piles of data: train, validation, test

    #training data
    X_train: np.ndarray # field name : numpy array
    y_train: np.ndarray
    dates_train: np.ndarray


    #validation data
    X_val: np.ndarray
    y_val: np.ndarray
    dates_val: np.ndarray

    #test data
    X_test: np.ndarray
    y_test: np.ndarray
    dates_test: np.ndarray

    scaler: MinMaxScaler        # needed later to turn predictions back into dollars
    n_scaler_rows: int          # how many raw rows the scaler was allowed to see

    # "price" or "log_return" 
    target: str = "log_return"

    
    # We need this to turn a predicted return back into a dollar price:
    #     P_hat = P_(t-1) * exp(predicted_log_return)
    
    base_train: np.ndarray = None #yesterday’s OHLCV for each training example
    base_val: np.ndarray = None #yesterday’s OHLCV for each validation example
    base_test: np.ndarray = None #yesterday’s OHLCV for each test example


#Loading the raw price history

def load_spy(csv_path=config.DATA_CSV, end_date=config.DATA_END_DATE):
    
    #Read the CSV and hand back two plain NumPy arrays: the dates and the prices.

    
    # Read the file into a pandas table.
    df = pd.read_csv(csv_path)

    # The Date column arrives as text like "2019-10-14". Convert it to real
    # datetime objects so we can compare and sort dates properly.
    df["Date"] = pd.to_datetime(df["Date"])

    # Force oldest-to-newest order. 
    df = df.sort_values("Date").reset_index(drop=True)

    # Cut the history at data end_date.
    df = df[df["Date"] <= pd.Timestamp(end_date)].reset_index(drop=True)

    # Refuse to continue if any price is missing. 
    if df[config.FEATURES].isna().any().any():
        raise ValueError("Missing values in SPY.csv")

    # .values strips away the column names and gives a raw NumPy array.
    #the column names in config.FEATURES 
    dates = df["Date"].values                                # shape (n,)
    values = df[config.FEATURES].values.astype("float64")    # shape (n, 5)

    return dates, values



#convert prices into daily log-returns


def to_log_returns(values, dates):

    # values[1:]  is every day from day 2 onward   -> P_t
    # values[:-1] is every day from day 1 to n-1   -> P_(t-1)
    # Dividing them element-wise pairs each day with the day before it.
    log_rets = np.log(values[1:] / values[:-1])

    # Drop day 1's date so dates[k] still describes log_rets[k].
    dates_out = dates[1:]

    # Remember yesterday's price for every row. 
    # Without this we could never turn a predicted return back 
    # into a dollar figure for the MAE tables.
    base_prices = values[:-1]

    return log_rets, dates_out, base_prices


#MAKING 60-DAYS CHUNKS TO FEED THE LSTMs
#60 past days → next day 

def make_sequences(values, dates, n_steps=config.SEQ_LEN):
    
    X_list, y_list, date_list = [], [], []

    # Start at n_steps, because example number i needs the n_steps rows that
    # come before row i. Before row 60 there is not enough history.
    for i in range(n_steps, len(values)):
        X_list.append(values[i - n_steps:i])   # the 60 days of input
        #reminder a[start:end] not inclusing end
        y_list.append(values[i])               # the day we are predicting
        date_list.append(dates[i])             # WHEN that predicted day is

    # Stack the Python lists into single NumPy arrays.
    X = np.array(X_list, dtype="float32")      # (N, 60, 5) 
    y = np.array(y_list, dtype="float32")      # (N, 5)
    forecast_dates = np.array(date_list)       # (N,)

    return X, y, forecast_dates


#MAIN FUNCTION TO PREPARE THE DATA

def prepare_data(
    csv_path=config.DATA_CSV,
    end_date=config.DATA_END_DATE,
    test_start=config.TEST_START_DATE,
    n_steps=config.SEQ_LEN,
    val_fraction=config.VAL_FRACTION,
    scaler_range=config.SCALER_RANGE,
    target=config.TARGET,
    verbose=True,
):
    
    # Loading data 
    dates, values = load_spy(csv_path, end_date)

   #LOG RETURNS
    
    if target == "log_return":
        # values becomes returns; dates loses its first day; base_prices keeps
        # yesterday's dollar price so we can undo this later.
        values, dates, base_prices = to_log_returns(values, dates)
        #values becomes the log-returns table (n-1, 5)
        #dates becomes the trimmed dates (n-1,)
        #base_prices becomes yesterday’s prices (n-1, 5)

    elif target == "price":
        base_prices = None         
    else:
        raise ValueError(f"target must be 'price' or 'log_return', got {target!r}")

# BOUNDARIES FOR THE DATA SPLIT
    test_start_ts = pd.Timestamp(test_start)

    # How many raw rows come before the test period begins.
    # For 2019-10-14 this is 6725.
    n_pre_test_rows = int((dates < np.datetime64(test_start_ts)).sum())

    # Every window costs us n_steps rows of runway at the start, so the number
    # of usable pre-test examples is 6725 - 60 = 6665.
    n_pre_test_seqs = n_pre_test_rows - n_steps

    # Split those chronologically: first 90% train, last 10% validation.
    
    n_train = int((1.0 - val_fraction) * n_pre_test_seqs)
    n_val = n_pre_test_seqs - n_train

    # FITTING THE SCALLER ON TRAINING RAWS
    
    n_scaler_rows = n_steps + n_train

    scaler = MinMaxScaler(feature_range=scaler_range)
    scaler.fit(values[:n_scaler_rows])       # LEARN min/max: training rows only
    values_scaled = scaler.transform(values)  # APPLY to everything, including test


    #n_steps: how many past days in one input window
    #n_train: number of training examples
   

    # make_sequences creates the input windows for the LSTM 60x + 1y
    X_all, y_all, dates_all = make_sequences(values_scaled, dates, n_steps)

    # make_sequences throws away the first n_steps rows (no history before
    # them).
    # base_prices are yesterday prices 
    base_all = None if base_prices is None else base_prices[n_steps:]

    # The windows 
    # first 5,997 examples → train
    # next 667 → validation
    # everything after 2019-10-14 → test
    # Positions 0 .. n_train-1                 -> train
    # Positions n_train .. n_pre_test_seqs-1   -> validation
    # Positions n_pre_test_seqs .. end         -> test
    train_slice = slice(0, n_train)

    
    val_slice = slice(n_train , n_pre_test_seqs)

    test_slice = slice(n_pre_test_seqs, len(X_all))

    splits = Splits(
        X_train=X_all[train_slice],
        y_train=y_all[train_slice],
        dates_train=dates_all[train_slice],
        X_val=X_all[val_slice],
        y_val=y_all[val_slice],
        dates_val=dates_all[val_slice],
        X_test=X_all[test_slice],
        y_test=y_all[test_slice],
        dates_test=dates_all[test_slice],
        scaler=scaler,
        n_scaler_rows=n_scaler_rows,
        target=target,
        base_train=None if base_all is None else base_all[train_slice],
        base_val=None if base_all is None else base_all[val_slice],
        base_test=None if base_all is None else base_all[test_slice],
    )

    

    return splits



def to_dollars(scaler, scaled_values, base_prices=None):
    #converting back to dollars
    raw = scaler.inverse_transform(np.asarray(scaled_values, dtype="float64"))

    if base_prices is None:
        return raw                          # already dollars

    return base_prices * np.exp(raw)        # returns -> dollars


def filter_by_date(dates, start=None, end=None):
   
    # train once, then slice the saved test predictions by date.
    mask = np.ones(len(dates), dtype=bool)
    if start is not None:
        mask &= dates >= np.datetime64(pd.Timestamp(start))
    if end is not None:
        mask &= dates <= np.datetime64(pd.Timestamp(end))
    return mask





if __name__ == "__main__":
    prepare_data()

