# Main experiment in the paper (Section 5): train 3 models x 5 seeds.
# Early stopping uses the validation set only (not the test set).
#
# Run:  python src/train.py

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import data as data_module
from models import MODEL_NAMES, build_model, build_optimizer

# We report Open/High/Low/Close in dollars. Volume is trained but not in the tables.
OHLC = ["Open", "High", "Low", "Close"]


def set_seed(seed):
    # Same seed = same weight start and same shuffle.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def loader(X, y, shuffle, seed=None):
    # Batches of 64. Shuffle train only.
    data = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    g = torch.Generator().manual_seed(seed) if shuffle and seed is not None else None
    return DataLoader(data, batch_size=config.BATCH_SIZE, shuffle=shuffle, generator=g)


def run_epoch(model, batches, loss_fn, dev, optimizer=None):
    # One pass over the data. If optimizer is given, we learn; if not, we only measure val MSE.
    train = optimizer is not None
    model.train() if train else model.eval()
    total = 0.0
    with torch.set_grad_enabled(train):
        for X, y in batches:
            X, y = X.to(dev), y.to(dev)
            loss = loss_fn(model(X), y)          # MSE on scaled log-returns
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * X.size(0)
    return total / len(batches.dataset)


def mae_ohlc(pred, true):
    # Dollar MAE used in the paper tables (not the training loss).
    return {col: float(np.abs(pred[:, i] - true[:, i]).mean())
            for i, col in enumerate(OHLC)}


def train_one(name, seed, splits, dev):
    # Train one model, one seed. Stop if val MSE does not improve for 8 epochs.
    set_seed(seed)
    n_in = splits.X_train.shape[-1]
    n_out = splits.y_train.shape[-1]
    model, spec = build_model(name, n_features=n_in, output_size=n_out)
    model = model.to(dev)
    optimizer = build_optimizer(model, spec)
    loss_fn = nn.MSELoss()

    train_loader = loader(splits.X_train, splits.y_train, shuffle=True, seed=seed)
    val_loader = loader(splits.X_val, splits.y_val, shuffle=False)

    best_val = float("inf")
    best_state = None
    wait = 0

    print(f"\n{name}  seed {seed}")
    for epoch in range(1, config.MAX_EPOCHS + 1):
        tr = run_epoch(model, train_loader, loss_fn, dev, optimizer)
        va = run_epoch(model, val_loader, loss_fn, dev)
        if va < best_val:
            best_val = va
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = "  <- best"
        else:
            wait += 1
            mark = ""
        print(f"  epoch {epoch:>2}/{config.MAX_EPOCHS}  train {tr:.6f}  val {va:.6f}{mark}")
        if wait >= config.PATIENCE:
            print("  early stop")
            break

    # Reload the best val weights (not the last epoch).
    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(splits.X_test).float().to(dev)
        pred_scaled = model(X).cpu().numpy()

    # Convert log-returns back to dollars: P_hat = P_yesterday * exp(r_hat)
    pred = data_module.to_dollars(splits.scaler, pred_scaled, splits.base_test)
    true = data_module.to_dollars(splits.scaler, splits.y_test, splits.base_test)
    print(f"  Close MAE ${mae_ohlc(pred, true)['Close']:.2f}")
    return pred, true


def print_tables(runs, persistence, true, dates):
    # Mean +/- std across 5 seeds, vs persistence (tomorrow = today).
    first = pd.Timestamp(dates[0]).date()
    last = pd.Timestamp(dates[-1]).date()
    print("\n" + "=" * 64)
    print(f"RESULTS  {first} to {last}  ({len(dates)} days, {len(config.SEEDS)} seeds)")
    print("=" * 64)

    rows = [{"model": "persistence"}]
    rows[0].update({col: f"${v:.2f}" for col, v in mae_ohlc(persistence, true).items()})

    for name in MODEL_NAMES:
        per_seed = [mae_ohlc(pred, true) for pred, _ in runs[name]]
        row = {"model": name}
        for col in OHLC:
            vals = [m[col] for m in per_seed]
            row[col] = f"${np.mean(vals):.2f} ± ${np.std(vals, ddof=1):.2f}"
        rows.append(row)

    print(pd.DataFrame(rows).to_string(index=False))

    print("\nClose MAE by seed")
    seed_rows = []
    for name in MODEL_NAMES:
        for seed, (pred, _) in zip(config.SEEDS, runs[name]):
            seed_rows.append({
                "model": name,
                "seed": seed,
                "Close": f"${mae_ohlc(pred, true)['Close']:.2f}",
            })
    print(pd.DataFrame(seed_rows).to_string(index=False))
    print("=" * 64)


def main():
    splits = data_module.prepare_data(verbose=False)
    dev = device()
    print(f"device {dev}")

    # Persistence: tomorrow's Open/High/Low/Close/Volume = today's values.
    true = data_module.to_dollars(splits.scaler, splits.y_test, splits.base_test)
    persistence = np.asarray(splits.base_test, dtype="float64")

    runs = {name: [] for name in MODEL_NAMES}
    for name in MODEL_NAMES:
        for seed in config.SEEDS:
            pred, true = train_one(name, seed, splits, dev)
            runs[name].append((pred, true))

    print_tables(runs, persistence, true, splits.dates_test)


if __name__ == "__main__":
    main()
