#!/usr/bin/env python
"""
§6.5 — 80/20 split anomaly: price-level MinMax extrapolation vs log-return targets.

Reproduces the preliminary split experiment (70/30, 80/20, 90/10) on SPY Close with
a train-only MinMax scaler (no leakage). Compares:

  1. Price-level targets  — scaler extrapolation when test prices exceed train range.
  2. Log-return targets   — stationary targets; dollar MAE via P_{t+1} = P_t * exp(r_hat).

Writes analysis/results/split_anomaly_comparison.csv (cited in README §6.5).

Run from repo root:
  python analysis/05_split_anomaly_log_returns.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = REPO_ROOT / "data" / "SPY.csv"
RESULTS_DIR = REPO_ROOT / "analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 60
EPOCHS = 50
LR = 1e-3
HIDDEN = 64
SPLITS = (0.7, 0.8, 0.9)
# Paper sample window (preliminary + revised §6.5)
DATA_END = "2020-12-31"


class SeqLSTM(nn.Module):
    def __init__(self, hidden_size: int = HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def load_close() -> pd.Series:
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"Missing {DATA_CSV}. Run: python scripts/download_spy.py"
        )
    spy = pd.read_csv(DATA_CSV, parse_dates=["Date"], index_col="Date")
    close = spy["Close"].astype(float).sort_index()
    return close.loc[:DATA_END]


def make_price_sequences(close: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows of raw Close; target = next-day Close. dates[i] is date of target."""
    values = close.values
    dates = close.index.to_numpy()
    X, y, y_dates = [], [], []
    for i in range(SEQ_LEN, len(values)):
        X.append(values[i - SEQ_LEN : i])
        y.append(values[i])
        y_dates.append(dates[i])
    return np.array(X, dtype=float), np.array(y, dtype=float), np.array(y_dates)


def make_return_sequences(
    close: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Windows of log-returns; target = next log-return; anchor = Close at window end."""
    log_ret = np.log(close / close.shift(1)).dropna()
    anchors = close.loc[log_ret.index]
    r = log_ret.values
    dates = log_ret.index.to_numpy()
    anchor_px = anchors.values
    X, y, y_dates, p_anchor = [], [], [], []
    for i in range(SEQ_LEN, len(r)):
        X.append(r[i - SEQ_LEN : i])
        y.append(r[i])
        y_dates.append(dates[i])
        p_anchor.append(anchor_px[i])
    return (
        np.array(X, dtype=float),
        np.array(y, dtype=float),
        np.array(y_dates),
        np.array(p_anchor, dtype=float),
    )


def train_lstm(X_train: np.ndarray, y_train: np.ndarray) -> SeqLSTM:
    model = SeqLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(X_train.reshape(-1, SEQ_LEN, 1), dtype=torch.float32)
    yt = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
    return model


def mae_dollars_price(
    model: SeqLSTM,
    scaler_x: MinMaxScaler,
    scaler_y: MinMaxScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    Xs = scaler_x.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    Xs = Xs.reshape(-1, SEQ_LEN, 1)
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.tensor(Xs, dtype=torch.float32)).numpy()
    pred = scaler_y.inverse_transform(pred_s).flatten()
    return float(np.mean(np.abs(pred - y_test)))


def mae_dollars_returns(
    model: SeqLSTM,
    scaler_x: MinMaxScaler,
    scaler_y: MinMaxScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    p_anchor: np.ndarray,
) -> float:
    Xs = scaler_x.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    Xs = Xs.reshape(-1, SEQ_LEN, 1)
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.tensor(Xs, dtype=torch.float32)).numpy()
    pred_r = scaler_y.inverse_transform(pred_s).flatten()
    pred_px = p_anchor * np.exp(pred_r)
    true_px = p_anchor * np.exp(y_test)
    return float(np.mean(np.abs(pred_px - true_px)))


def split_label(split: float) -> str:
    train_pct = int(round(split * 100))
    test_pct = 100 - train_pct
    return f"{train_pct}/{test_pct}"


def run_split(
    split: float,
    target_mode: str,
    close: pd.Series,
) -> dict:
    if target_mode in ("price", "price_global_scaler"):
        X, y, _ = make_price_sequences(close)
        p_anchor = None
    elif target_mode == "log_return":
        X, y, _, p_anchor = make_return_sequences(close)
    else:
        raise ValueError(target_mode)

    n = len(X)
    split_idx = int(n * split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler_x = MinMaxScaler(feature_range=(0.01, 0.99))
    scaler_y = MinMaxScaler(feature_range=(0.01, 0.99))
    if target_mode == "price_global_scaler":
        # Preliminary protocol (04_open_prediction_splits.py): scaler fit on full series.
        scaler_x.fit(X.reshape(-1, 1))
        scaler_y.fit(y.reshape(-1, 1))
    else:
        scaler_x.fit(X_train.reshape(-1, 1))
        scaler_y.fit(y_train.reshape(-1, 1))

    X_tr = scaler_x.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    y_tr = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
    model = train_lstm(X_tr, y_tr)

    if target_mode.startswith("price"):
        mae = mae_dollars_price(model, scaler_x, scaler_y, X_test, y_test)
        train_max = float(y_train.max())
        test_max = float(y_test.max())
    else:
        p_test = p_anchor[split_idx:]
        mae = mae_dollars_returns(model, scaler_x, scaler_y, X_test, y_test, p_test)
        train_max = float(y_train.max())
        test_max = float(y_test.max())

    return {
        "split": split_label(split),
        "target_mode": target_mode,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "close_mae_usd": round(mae, 4),
        "train_target_max": round(train_max, 2),
        "test_target_max": round(test_max, 2),
    }


def main() -> None:
    print("=" * 60)
    print("§6.5 Split anomaly: price level vs log-return targets (Close)")
    print("=" * 60)

    close = load_close()
    rows = []
    for split in SPLITS:
        for mode in ("price", "price_global_scaler", "log_return"):
            row = run_split(split, mode, close)
            rows.append(row)
            print(
                f"  {row['split']:6} {row['target_mode']:11} "
                f"Close MAE ${row['close_mae_usd']:.2f}  "
                f"(train max {row['train_target_max']:.1f}, test max {row['test_target_max']:.1f})"
            )

    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "split_anomaly_comparison.csv"
    out.to_csv(out_path, index=False)

    def mae_80(mode: str) -> float | None:
        sub = out[(out["split"] == "80/20") & (out["target_mode"] == mode)]
        return float(sub["close_mae_usd"].iloc[0]) if len(sub) else None

    p_train = mae_80("price")
    p_global = mae_80("price_global_scaler")
    r80 = mae_80("log_return")
    if p_train is not None and r80 is not None:
        print("\n80/20 summary (paper §6.5, Close, through 2020):")
        print(f"  Price + train-only scaler:  ${p_train:.2f}")
        if p_global is not None:
            print(f"  Price + global scaler:      ${p_global:.2f}  (legacy extrapolation)")
        print(f"  Log-return + train scaler:  ${r80:.2f}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
