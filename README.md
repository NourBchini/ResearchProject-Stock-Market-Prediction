# SPY Price Forecasting — Hybrid CNN–LSTM Research

**A Hybrid CNN–LSTM Approach for Stock Market Price Prediction Across Market Regimes**  
Nour Bchini · Skidmore College · Independent Research Project 2024–2025  
Faculty Supervisor: Professor Wenlu Du

---

## Overview

This repository contains the full research codebase for a deep learning study on next-day SPY (S&P 500 ETF) price forecasting. The project systematically benchmarks standalone LSTM models against two hybrid CNN–LSTM architectures — sequential and parallel — across 28 years of daily market data (1993–2020), with a focus on how model performance changes across different market regimes.

**Key finding:** The parallel CNN–LSTM architecture achieved the strongest overall performance (Close MAE: $1.04), but model accuracy degraded 3.86× in the post-2016 era — a result attributed to the structural transformation of markets by AI-driven and algorithmic trading.

---

## Results Summary

| Architecture | Open MAE | High MAE | Low MAE | Close MAE |
|---|---|---|---|---|
| Simple LSTM (128 units) | $1.09 | $1.21 | **$1.04** | $1.10 |
| CNN–LSTM Sequential | $5.79 | $5.66 | $6.46 | $6.04 |
| **CNN–LSTM Parallel** | **$0.92** | **$1.01** | $1.18 | **$1.04** |

| Era | MAE |
|---|---|
| Pre-AI (1993–2015) | $2.31 |
| Post-AI (2016–2020) | $8.91 |

---

## Repository Structure

```
ResearchProject-Stock-Market-Prediction/
│
├── data/                   # Raw and processed SPY OHLCV data
│
├── scripts/                # Model definitions and architecture files
│   ├── lstm_model.py           # Standalone LSTM baseline
│   ├── new_LSTM_CNN.py         # Parallel CNN–LSTM (CNN and LSTM branches run independently, outputs concatenated)
│   └── cnn_lstm_model.py       # Sequential CNN→LSTM (CNN output feeds into LSTM)
│
├── training/               # Training loops, hyperparameter experiments
│
├── weights/                # Saved model weights (.pt files)
│
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

## Model Architectures

### Simple LSTM (Baseline)
- Hidden Units: 128 · Layers: 1 · Optimizer: Adam (lr=0.001)
- Dropout: 0.2 · Sequence Length: 60 days · Batch Size: 32
- Loss: MSE · Regularization: L2 + Early Stopping (patience=20)

### CNN–LSTM Sequential (`cnn_lstm_model.py`)
The convolutional layer processes the input sequence first, extracting localized temporal features that are then fed directly into the LSTM. Emphasizes hierarchical feature extraction.

### CNN–LSTM Parallel (`new_LSTM_CNN.py`) ← Best performing
The same input sequence is fed simultaneously into two independent branches — one convolutional, one recurrent. Their outputs are concatenated and passed into a fully connected prediction layer. This preserves the original temporal information in both branches before fusion.

**Final architecture:**
- CNN Layer: 32 filters · LSTM Layer: 64 hidden units
- Dropout: 0.3 (both branches) · Sequence Length: 60 trading days
- Optimizer: Adam · Learning Rate: 0.0003

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `yfinance`

### Data

SPY daily OHLCV data (1993–2020) is included in the `data/` folder. Data was sourced from Yahoo Finance via `yfinance`.

### Training

```bash
# Train the parallel CNN–LSTM model
python training/train_parallel.py

# Train the standalone LSTM baseline
python training/train_lstm.py

# Run the sequential CNN–LSTM
python training/train_sequential.py
```

### Evaluation

```bash
# Run full regime analysis
python analysis/regime_analysis.py

# Evaluate on a specific date window
python analysis/evaluate.py --start 2019-10-14 --end 2019-10-28
```

---

## Experiments

### Hyperparameter Tuning (LSTM)
Systematic comparison of 64, 128, and 256 hidden units with cost-benefit and ROI analysis. 128 units selected as optimal (50% MAE improvement over 64 units, 14% ROI; 256 units showed only 4% further gain at 4× the compute cost).

### Optimizer Comparison
Adam vs AdamW on the 128-unit LSTM. Adam outperformed AdamW — AdamW's extra regularization hindered convergence on this large dataset with dropout already applied.

### Depth Experiment
1-layer vs 2-layer LSTM. Adding a second layer worsened most metrics (Close MAE: $1.22 → $1.35), likely due to overfitting under current dropout settings.

### Regime Analysis
| Split | MAE | Notes |
|---|---|---|
| Pre-AI era (1993–2015) | $2.31 | Stable, predictable |
| Post-AI era (2016–2020) | $8.91 | 3.86× degradation |
| Pre-pandemic | $3.26 | — |
| Post-pandemic | $3.35 | MAPE improved despite higher absolute error |

### Train/Test Split
| Split | Train | Test | MAE | Result |
|---|---|---|---|---|
| 70/30 | 4,846 | 4,156 | $44.98 | Baseline |
| 80/20 | 5,539 | 2,770 | $172.67 | Overfit |
| **90/10** | **6,231** | **1,386** | **$7.55** | **Best** |

---

## Key Findings

1. **Parallel > Sequential.** The parallel CNN–LSTM outperforms the sequential variant because each branch specializes independently — CNN captures short-term local patterns, LSTM captures long-range dependencies — before fusion. The sequential model loses temporal resolution by filtering input before LSTM sees it.

2. **Hyperparameter transfer doesn't work across architectures.** Tuning gains from standalone LSTM do not directly carry over to CNN–LSTM hybrids, because the LSTM's input changes fundamentally (from raw prices to CNN feature maps).

3. **The 2016 structural break is real.** Post-2016, MAE rose 3.86×. Market structure metrics confirm the cause: annual returns improved 52.5%, Sharpe ratio rose 51.1%, but volume volatility dropped 55.6% and volatility clustering increased 28.2% — consistent with HFT liquidity flooding masking momentum signals and AI systems eliminating exploitable patterns faster than models can learn them.

4. **Single-week evaluations are misleading.** MAE across six random weekly windows ranged from $1.74 to $10.11 (5.8× spread), underscoring that cherry-picked demo windows can dramatically misrepresent true model behavior.

---

## Paper

The full research paper is available here:  
**[A Hybrid CNN–LSTM Approach for SPY Price Forecasting Across Market Regimes](https://nourbchini.domains.skidmore.edu)**

---

## Citation

```
Bchini, N. (2025). A Hybrid CNN–LSTM Approach for Stock Market Price Prediction
Across Market Regimes. Independent Research Project, Skidmore College.
Supervisor: Professor Wenlu Du.
```

---

## References

- Mehtab, S., Sen, J., & Dasgupta, S. (2020). Analysis and Forecasting of Financial Time Series Using CNN and LSTM-Based Deep Learning Models. *arXiv:2011.08011*
- Lu, W., Li, J., Li, Y., Sun, A., & Wang, J. (2020). A CNN–LSTM-based model to forecast stock prices. *Complexity*, 2020, Article 6622927.

---

## License

MIT License — see `LICENSE` for details.

---

*Not investment advice. This is an academic research project.*
