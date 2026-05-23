#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 2: Create Comprehensive Hyperparameter Table
Documents all final hyperparameters for all models tested in the
preliminary paper (including the Hybrid model not used in the revised paper).

NOTE: Preserved as part of the preliminary investigation. The revised paper
uses Tables 1 (LSTM tuning) and 2-4 (multi-seed comparison) in
analysis/multi_seed_compare.py instead.
"""

from pathlib import Path
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Comprehensive hyperparameter table
hyperparameters = {
    "Model": [
        "CNN-LSTM (Frozen)",
        "Simple LSTM (128 units)",
        "Simple LSTM (64 units)",
        "Simple LSTM (256 units)",
        "Bidirectional LSTM (1 layer)",
        "Bidirectional LSTM (2 layers)",
        "Unidirectional LSTM (2 layers)",
        "Hybrid Model"
    ],
    "Architecture": [
        "CNN-LSTM Sequential",
        "LSTM Unidirectional",
        "LSTM Unidirectional",
        "LSTM Unidirectional",
        "LSTM Bidirectional",
        "LSTM Bidirectional",
        "LSTM Unidirectional",
        "CNN-LSTM + Simple LSTM"
    ],
    "Hidden_Size": [64, 128, 64, 256, 128, 128, 128, "64/128"],
    "Num_Layers": [1, 1, 1, 1, 1, 2, 2, "1/1"],
    "CNN_Filters": [32, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", 32],
    "Sequence_Length": [60, 60, 60, 60, 60, 60, 60, 60],
    "Batch_Size": [32, 32, 32, 32, 32, 32, 32, 32],
    "Learning_Rate": [0.0003, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0003],
    "Dropout_CNN": [0.3, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", 0.3],
    "Dropout_LSTM": [0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, "0.3/0.2"],
    "Optimizer": ["AdamW", "Adam", "Adam", "Adam", "Adam", "Adam", "Adam", "AdamW"],
    "Weight_Decay": [0.002, 0, 0, 0, 0, 0, 0, 0.002],
    "Bidirectional": [False, False, False, False, True, True, False, "Mixed"],
    "Close_MAE": ["$1.04", "$1.10", "$2.31", "$1.06", "$1.22", "$1.35", "$3.32", "$1.07"],
    "Status": [
        "BEST - Frozen",
        "Good",
        "Underfit",
        "Overfit",
        "Good",
        "Worse",
        "Worst",
        "Ensemble"
    ]
}

df = pd.DataFrame(hyperparameters)
df.to_csv(OUT_DIR / "hyperparameter_table.csv", index=False)
df.to_markdown(OUT_DIR / "hyperparameter_table.md", index=False)

print("="*60)
print("HYPERPARAMETER TABLE CREATED")
print("="*60)
print("\nSaved to:")
print("  - results/hyperparameter_table.csv")
print("  - results/hyperparameter_table.md")
print("\nTable Preview:")
print(df.to_string(index=False))

