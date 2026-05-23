"""
Multi-seed comparison and statistical tests for the paper.

What this script does (paper-grade revision of the original ad-hoc table):

1. Trains three architectures with 5 random seeds each, on a FIXED train/val/test
   split (no test-set peeking via early stopping):
     - Fusion CNN-LSTM (parallel: 32 CNN filters, 64 LSTM units, dropout 0.3, Adam lr=3e-4)
     - Cascade CNN-LSTM (CNN -> LSTM stack; from cnn_lstm_model.py)
     - Unidirectional LSTM-128 (lstm_pytorch.py baseline)
2. Adds a Persistence (random-walk) baseline: y_hat_{t+1} = y_t .
3. Evaluates on two windows:
     - SHORT: 2019-10-14 .. 2019-10-28 (original paper window, n ~ 11 days)
     - LONG : 2019-10-14 .. end of data (n ~ 1600 days), for proper stats power
4. Statistical tests on per-day Close absolute error (LONG window):
     - Diebold-Mariano (HLN-corrected) for forecast accuracy
     - Wilcoxon signed-rank on paired |error|
5. Saves results to analysis/results/multi_seed_compare.{json,md}

No test-set peeking: validation = last 10% of training data (chronological),
matching lstm_pytorch.py's design. Scalers are fit on TRAIN only.

Run from the analysis directory or repo root; paths are resolved automatically.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = REPO_ROOT / "data" / "SPY.csv"
RESULTS_DIR = REPO_ROOT / "analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SHORT_TEST_START = "2019-10-14"
SHORT_TEST_END = "2019-10-28"
LONG_TEST_START = "2019-10-14"

SEQ_LEN = 60
BATCH = 64
EPOCHS = 30
PATIENCE = 8
SEEDS = [42, 1337, 2024, 7, 12345]
FEATURE_NAMES = ["Open", "High", "Low", "Close", "Volume"]


def get_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------- Data pipeline (no leakage) -------------------------

def load_spy() -> pd.DataFrame:
    spy = pd.read_csv(DATA_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    feat = spy[["Open", "High", "Low", "Close", "Volume"]]
    targ = spy[["Open", "High", "Low", "Close", "Volume"]].shift(-1)
    df = pd.concat([feat.add_prefix("SPY_"), targ], axis=1).ffill().dropna()
    return df


def build_arrays(df: pd.DataFrame, test_start: str, test_end: str | None):
    feat_cols = [c for c in df.columns if c.startswith("SPY_")]
    targ_cols = ["Open", "High", "Low", "Close", "Volume"]

    seq_dates = df.index[SEQ_LEN:]
    test_mask = seq_dates >= test_start
    if test_end is not None:
        test_mask &= seq_dates <= test_end
    train_mask = seq_dates < test_start

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    val_size = max(1, int(0.1 * len(train_idx)))
    val_idx = train_idx[-val_size:]
    train_idx = train_idx[:-val_size]

    feat_scaler = MinMaxScaler(feature_range=(0.01, 0.99))
    targ_scaler = MinMaxScaler(feature_range=(0.01, 0.99))

    train_last_row = train_idx[-1] + 1
    feat_scaler.fit(df[feat_cols].iloc[: SEQ_LEN + train_last_row])
    targ_scaler.fit(df[targ_cols].iloc[: SEQ_LEN + train_last_row])

    feats_scaled = feat_scaler.transform(df[feat_cols].values)
    targs_scaled = targ_scaler.transform(df[targ_cols].values)

    X_all, y_all = [], []
    for i in range(SEQ_LEN, len(targs_scaled)):
        X_all.append(feats_scaled[i - SEQ_LEN : i])
        y_all.append(targs_scaled[i])
    X_all = np.asarray(X_all, dtype=np.float32)
    y_all = np.asarray(y_all, dtype=np.float32)

    test_dates = seq_dates[test_idx]
    return {
        "X": X_all,
        "y": y_all,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "test_dates": test_dates,
        "targ_scaler": targ_scaler,
    }


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ------------------------- Models -------------------------

class FusionCNN_LSTM(nn.Module):
    """Two-stream: CNN branch and LSTM branch over the same window, concat -> FC."""

    def __init__(self, num_features=5, hidden=64, cnn_filters=32, out=5):
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, cnn_filters, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(0.3)
        self.lstm = nn.LSTM(num_features, hidden, num_layers=1, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden + cnn_filters, out)

    def forward(self, x):
        c = self.dropout_cnn(self.relu(self.pool(self.conv1(x.permute(0, 2, 1)))))
        c = c.mean(dim=2)
        out, _ = self.lstm(x)
        h = self.dropout_lstm(out[:, -1, :])
        return self.fc(torch.cat([h, c], dim=1))


class CascadeCNN_LSTM(nn.Module):
    """Cascade: CNN -> LSTM consumes CNN time-step features."""

    def __init__(self, num_features=5, hidden=64, cnn_filters=32, out=5):
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, cnn_filters, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_filters, hidden, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden, out)

    def forward(self, x):
        z = self.relu(self.pool(self.conv1(x.permute(0, 2, 1))))
        z = z.permute(0, 2, 1)
        out, _ = self.lstm(z)
        return self.fc(self.dropout(out[:, -1, :]))


class Plain_LSTM(nn.Module):
    """Unidirectional LSTM-128 baseline (matches lstm_pytorch.py architecture)."""

    def __init__(self, num_features=5, hidden=128, out=5):
        super().__init__()
        self.lstm = nn.LSTM(num_features, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, out)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))


# ------------------------- Train / eval -------------------------

@dataclass
class RunResult:
    model: str
    seed: int
    window: str
    mae: dict
    rmse: dict
    pred: np.ndarray
    true: np.ndarray
    dates: pd.DatetimeIndex


def train_model(model: nn.Module, train_loader, val_loader, opt, loss_fn) -> nn.Module:
    model.to(DEVICE)
    best_val = float("inf")
    best_state = None
    bad = 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        model.eval()
        v = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                v += loss_fn(model(xb), yb).item() * xb.size(0)
                n += xb.size(0)
        v /= max(n, 1)
        if v < best_val - 1e-7:
            best_val = v
            best_state = {k: t.detach().clone().cpu() for k, t in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    return model


def evaluate(model: nn.Module, loader, targ_scaler) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    P, T = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            P.append(model(xb).cpu().numpy())
            T.append(yb.numpy())
    P = targ_scaler.inverse_transform(np.concatenate(P, axis=0))
    T = targ_scaler.inverse_transform(np.concatenate(T, axis=0))
    return P, T


def metrics(P, T) -> tuple[dict, dict]:
    mae, rmse = {}, {}
    for i, name in enumerate(FEATURE_NAMES):
        e = P[:, i] - T[:, i]
        mae[name] = float(np.mean(np.abs(e)))
        rmse[name] = float(np.sqrt(np.mean(e ** 2)))
    return mae, rmse


def run_seed_dual(model_name: str, seed: int, arr_long, short_mask) -> tuple[RunResult, RunResult]:
    """Train once on the long-window pipeline; eval on both LONG and SHORT (subset)."""
    set_seed(seed)
    X, y = arr_long["X"], arr_long["y"]
    train_ds = Subset(SeqDataset(X, y), arr_long["train_idx"])
    val_ds = Subset(SeqDataset(X, y), arr_long["val_idx"])
    test_ds = Subset(SeqDataset(X, y), arr_long["test_idx"])
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

    if model_name == "Fusion CNN-LSTM":
        model = FusionCNN_LSTM()
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    elif model_name == "Cascade CNN-LSTM":
        model = CascadeCNN_LSTM()
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    elif model_name == "LSTM-128":
        model = Plain_LSTM()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        raise ValueError(model_name)

    loss_fn = nn.MSELoss()
    train_model(model, train_loader, val_loader, opt, loss_fn)
    P, T = evaluate(model, test_loader, arr_long["targ_scaler"])

    long_mae, long_rmse = metrics(P, T)
    long_run = RunResult(model_name, seed, "long", long_mae, long_rmse, P, T, arr_long["test_dates"])

    P_s = P[short_mask]
    T_s = T[short_mask]
    short_dates = arr_long["test_dates"][short_mask]
    short_mae, short_rmse = metrics(P_s, T_s)
    short_run = RunResult(model_name, seed, "short", short_mae, short_rmse, P_s, T_s, short_dates)
    return short_run, long_run


def persistence(arr) -> RunResult:
    feat_scaler_dummy = None  # not needed: we use raw, not scaled
    spy = pd.read_csv(DATA_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    targ = spy[["Open", "High", "Low", "Close", "Volume"]]
    test_dates = arr["test_dates"]
    P = []
    T = []
    for d in test_dates:
        if d not in targ.index:
            continue
        prev_dates = targ.index[targ.index < d]
        if len(prev_dates) == 0:
            continue
        prev = targ.loc[prev_dates[-1]].values
        actual = targ.loc[d].values
        P.append(prev)
        T.append(actual)
    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)
    mae, rmse = metrics(P, T)
    return RunResult("Persistence", -1, "", mae, rmse, P, T, test_dates[: len(P)])


# ------------------------- Stats tests -------------------------

def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict:
    """HLN-corrected DM test on squared errors. e1, e2 are forecast errors (P - T)."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    mean_d = float(np.mean(d))
    gamma0 = float(np.var(d, ddof=0))
    var_d = gamma0
    for k in range(1, h):
        gk = float(np.mean((d[:-k] - mean_d) * (d[k:] - mean_d)))
        var_d += 2 * gk
    se = math.sqrt(var_d / n) if var_d > 0 else float("nan")
    dm = mean_d / se if se > 0 else float("nan")
    correction = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    hln = dm * correction
    df = max(n - 1, 1)
    p_two = 2 * (1 - stats.t.cdf(abs(hln), df=df))
    return {"DM_stat": dm, "HLN_stat": hln, "p_value_two_sided": p_two, "n": n}


def wilcoxon_paired(abs_e1: np.ndarray, abs_e2: np.ndarray) -> dict:
    try:
        res = stats.wilcoxon(abs_e1, abs_e2, zero_method="wilcox", alternative="two-sided")
        return {"stat": float(res.statistic), "p_value": float(res.pvalue), "n": int(len(abs_e1))}
    except Exception as ex:
        return {"error": str(ex)}


def chow_test(returns: np.ndarray, break_idx: int) -> dict:
    """Chow test on AR(1) of returns at break_idx."""
    y = returns[1:]
    x = returns[:-1]
    bk = break_idx - 1
    if bk < 5 or bk > len(y) - 5:
        return {"error": "break index out of range"}
    X = np.column_stack([np.ones_like(x), x])
    bhat, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(np.sum((y - X @ bhat) ** 2))
    X1, y1 = X[:bk], y[:bk]
    X2, y2 = X[bk:], y[bk:]
    b1, *_ = np.linalg.lstsq(X1, y1, rcond=None)
    b2, *_ = np.linalg.lstsq(X2, y2, rcond=None)
    rss1 = float(np.sum((y1 - X1 @ b1) ** 2))
    rss2 = float(np.sum((y2 - X2 @ b2) ** 2))
    k = 2
    n = len(y)
    f = ((rss - (rss1 + rss2)) / k) / ((rss1 + rss2) / (n - 2 * k))
    p = 1 - stats.f.cdf(f, k, n - 2 * k)
    return {"F": float(f), "p_value": float(p), "n": int(n), "break_idx": int(break_idx)}


# ------------------------- Driver -------------------------

def main():
    print("=" * 78)
    print("MULTI-SEED PAPER-GRADE COMPARISON")
    print("=" * 78)
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"Epochs: {EPOCHS}, patience: {PATIENCE}, seq_len: {SEQ_LEN}, batch: {BATCH}")
    print()

    df = load_spy()
    arr_long = build_arrays(df, LONG_TEST_START, None)
    test_dates_long = arr_long["test_dates"]
    short_mask = (test_dates_long >= SHORT_TEST_START) & (test_dates_long <= SHORT_TEST_END)
    print(f"LONG  window test rows: {len(arr_long['test_idx'])}")
    print(f"SHORT window test rows (subset of long): {int(short_mask.sum())}")
    print()

    models = ["Fusion CNN-LSTM", "Cascade CNN-LSTM", "LSTM-128"]
    runs: dict[tuple[str, str], list[RunResult]] = {(m, w): [] for m in models for w in ("short", "long")}

    t0 = time.time()
    for m in models:
        for s in SEEDS:
            t1 = time.time()
            short_r, long_r = run_seed_dual(m, s, arr_long, short_mask)
            runs[(m, "short")].append(short_r)
            runs[(m, "long")].append(long_r)
            print(
                f"  {m} seed={s}  "
                f"Close MAE short=${short_r.mae['Close']:.4f}  long=${long_r.mae['Close']:.4f}  "
                f"({time.time() - t1:.1f}s)"
            )
        print()

    arr_short_for_persistence = {"test_dates": test_dates_long[short_mask]}
    pers_short = persistence(arr_short_for_persistence)
    pers_long = persistence(arr_long)

    summary = {"models": {}}
    for window_label, persistence_run in [("short", pers_short), ("long", pers_long)]:
        for m in models:
            rs = runs[(m, window_label)]
            ma = {f: [r.mae[f] for r in rs] for f in FEATURE_NAMES}
            rm = {f: [r.rmse[f] for r in rs] for f in FEATURE_NAMES}
            summary["models"].setdefault(m, {})[window_label] = {
                "mae_mean": {f: float(np.mean(ma[f])) for f in FEATURE_NAMES},
                "mae_std": {f: float(np.std(ma[f], ddof=1)) for f in FEATURE_NAMES},
                "rmse_mean": {f: float(np.mean(rm[f])) for f in FEATURE_NAMES},
                "rmse_std": {f: float(np.std(rm[f], ddof=1)) for f in FEATURE_NAMES},
                "per_seed_close_mae": ma["Close"],
            }
        summary["models"].setdefault("Persistence", {})[window_label] = {
            "mae_mean": {f: persistence_run.mae[f] for f in FEATURE_NAMES},
            "mae_std": {f: 0.0 for f in FEATURE_NAMES},
            "rmse_mean": {f: persistence_run.rmse[f] for f in FEATURE_NAMES},
            "rmse_std": {f: 0.0 for f in FEATURE_NAMES},
            "per_seed_close_mae": [persistence_run.mae["Close"]],
        }

    def stack_preds(rs: list[RunResult]) -> np.ndarray:
        Ps = np.stack([r.pred for r in rs], axis=0)
        return Ps.mean(axis=0)

    long_true = runs[("Fusion CNN-LSTM", "long")][0].true
    fusion_pred = stack_preds(runs[("Fusion CNN-LSTM", "long")])
    cascade_pred = stack_preds(runs[("Cascade CNN-LSTM", "long")])
    lstm_pred = stack_preds(runs[("LSTM-128", "long")])
    pers_pred = pers_long.pred
    pers_true = pers_long.true

    n_min = min(fusion_pred.shape[0], pers_pred.shape[0])
    fusion_pred_c = fusion_pred[-n_min:, 3]
    cascade_pred_c = cascade_pred[-n_min:, 3]
    lstm_pred_c = lstm_pred[-n_min:, 3]
    pers_pred_c = pers_pred[-n_min:, 3]
    true_c = long_true[-n_min:, 3]
    pers_true_c = pers_true[-n_min:, 3]
    if not np.allclose(true_c, pers_true_c, atol=1e-4):
        print("WARNING: persistence and model true vectors differ; check alignment.")

    def err(p): return p - true_c

    tests = {}
    pairs = [
        ("Fusion vs LSTM-128", err(fusion_pred_c), err(lstm_pred_c)),
        ("Fusion vs Cascade", err(fusion_pred_c), err(cascade_pred_c)),
        ("Fusion vs Persistence", err(fusion_pred_c), err(pers_pred_c)),
        ("LSTM-128 vs Persistence", err(lstm_pred_c), err(pers_pred_c)),
    ]
    for label, e1, e2 in pairs:
        dm = diebold_mariano(e1, e2)
        w = wilcoxon_paired(np.abs(e1), np.abs(e2))
        tests[label] = {"diebold_mariano": dm, "wilcoxon": w,
                        "mean_abs_err_model_A": float(np.mean(np.abs(e1))),
                        "mean_abs_err_model_B": float(np.mean(np.abs(e2)))}

    print("\nStructural-break (Chow) tests on SPY daily log-returns")
    spy_close = df["Close"].values if "Close" in df.columns else None
    if spy_close is None:
        spy = pd.read_csv(DATA_CSV, parse_dates=["Date"], index_col="Date").sort_index()
        rets = np.diff(np.log(spy["Close"].values))
        dates = spy.index[1:]
    else:
        spy = pd.read_csv(DATA_CSV, parse_dates=["Date"], index_col="Date").sort_index()
        rets = np.diff(np.log(spy["Close"].values))
        dates = spy.index[1:]
    chow = {}
    for year in [2012, 2013, 2014, 2015, 2016, 2017, 2018]:
        cand = pd.Timestamp(f"{year}-01-01")
        bk = int(np.searchsorted(dates, cand))
        res = chow_test(rets, bk)
        chow[str(year)] = res
        if "F" in res:
            print(f"  break {year}-01-01 idx={bk}  F={res['F']:.3f}  p={res['p_value']:.4g}")

    out = {
        "config": {
            "seeds": SEEDS, "epochs": EPOCHS, "patience": PATIENCE,
            "seq_len": SEQ_LEN, "batch": BATCH,
            "short_test": [SHORT_TEST_START, SHORT_TEST_END],
            "long_test_start": LONG_TEST_START,
            "long_test_days": int(len(arr_long["test_idx"])),
        },
        "summary": summary,
        "stats_tests_long_window_close": tests,
        "chow_tests_on_log_returns": chow,
        "elapsed_seconds": time.time() - t0,
    }
    (RESULTS_DIR / "multi_seed_compare.json").write_text(json.dumps(out, indent=2, default=str))

    lines = []
    lines.append("# Multi-seed model comparison\n")
    lines.append(f"Seeds: {SEEDS}  |  epochs={EPOCHS}, patience={PATIENCE}, seq_len={SEQ_LEN}\n")
    lines.append(f"Short test: {SHORT_TEST_START}..{SHORT_TEST_END}  (n={int(short_mask.sum())})\n")
    lines.append(f"Long test : {LONG_TEST_START}..end  (n={len(arr_long['test_idx'])})\n\n")
    for window_label in ("short", "long"):
        lines.append(f"## Close MAE (mean +/- std across seeds) — {window_label} window\n")
        lines.append("| Model | Open | High | Low | Close | Volume |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for m in models + ["Persistence"]:
            d = summary["models"][m][window_label]
            row = [m]
            for f in FEATURE_NAMES:
                row.append(f"{d['mae_mean'][f]:.3f} ± {d['mae_std'][f]:.3f}")
            lines.append("| " + " | ".join(row) + " |\n")
        lines.append("\n")
    lines.append("## Stats tests on long-window Close errors (seed-averaged forecasts)\n")
    for label, t in tests.items():
        lines.append(
            f"- **{label}**: mean|e_A|={t['mean_abs_err_model_A']:.3f}, "
            f"mean|e_B|={t['mean_abs_err_model_B']:.3f}, "
            f"DM(HLN)={t['diebold_mariano']['HLN_stat']:.3f} (p={t['diebold_mariano']['p_value_two_sided']:.3g}), "
            f"Wilcoxon p={t['wilcoxon'].get('p_value', float('nan')):.3g}\n"
        )
    lines.append("\n## Chow tests on SPY log-returns (AR(1))\n")
    lines.append("| Break year | F | p-value |\n|---|---|---|\n")
    for y, r in chow.items():
        if "F" in r:
            lines.append(f"| {y}-01-01 | {r['F']:.3f} | {r['p_value']:.4g} |\n")
    (RESULTS_DIR / "multi_seed_compare.md").write_text("".join(lines))
    print(f"\nSaved: {RESULTS_DIR / 'multi_seed_compare.json'}")
    print(f"Saved: {RESULTS_DIR / 'multi_seed_compare.md'}")


if __name__ == "__main__":
    main()
