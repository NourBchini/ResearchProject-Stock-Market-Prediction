# Deep Learning for Stock Price Prediction

**A Hybrid CNN–LSTM Approach for SPY Forecasting**

Nour Bchini · Skidmore College  
Faculty Supervisor: Professor Wenlu Du  
Undergraduate Independent Research Project | 2025–2026


## What is in this repo

```
Data/SPY.csv      # daily OHLCV for SPY (Yahoo Finance, 1993–2026)
src/config.py     # dates, splits, hyperparameters (required by the other scripts)
src/data.py       # load, log-returns, 60-day windows, train/val/test split
src/models.py     # LSTM-128, Fusion CNN–LSTM, Cascade CNN–LSTM
src/train.py      # 3 models × 5 seeds, validation early stopping, MAE vs persistence
```

`config.py` is required even if you only care about the other files. `data.py`, `models.py`, and `train.py` all read settings from it.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the paper training protocol

```bash
python src/train.py
```

This trains LSTM-128, Fusion CNN–LSTM, and Cascade CNN–LSTM on five seeds `[42, 1337, 2024, 7, 12345]`, then prints dollar MAE against a persistence baseline (`tomorrow = today`).

### Protocol (matches the paper)

- **Input:** 60-day windows of Open, High, Low, Close, Volume
- **Target:** next-day log-return, converted back to dollars with \(P_t = P_{t-1}\,e^{r_t}\)
- **Split:** train on sequences with forecast dates before 2019-10-14; last 10% of that block is validation (early stopping only); test is 2019-10-14 through 2026-05-22
- **Scaling:** `MinMaxScaler` fit on training rows only, range `[0.01, 0.99]`
- **Training:** batch size 64, max 30 epochs, patience 8 on validation MSE
- **Optimizers:** LSTM-128 uses Adam (`lr=1e-3`, L2 `1e-4`); hybrids use Adam (`lr=3e-4`, dropout 0.3, no weight decay)

Early stopping uses the **validation set**, never the test set.

## Models

| Name | Architecture |
|---|---|
| **LSTM-128** | 128 hidden units, 1 layer, dropout 0.2 |
| **Fusion CNN–LSTM** | CNN (32 filters, kernel 3) and LSTM (64 units) see the same window in parallel, then concatenate |
| **Cascade CNN–LSTM** | CNN first (length 29 after pooling), then LSTM (64 units) on the CNN maps |
| **Persistence** | \(P_{t+1}=P_t\); no training |

## Main result

On the 1,661-day test window, none of the deep models beats persistence on Close. Fusion CNN–LSTM Close MAE is **$3.63 ± 0.01** versus **$3.63** for persistence (Diebold–Mariano \(p \approx 0.72\)). That is consistent with a weak-form efficient daily close: the lowest-MAE forecast of tomorrow’s price is today’s price.

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `scikit-learn`, `torch`
