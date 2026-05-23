# SPY Price Forecasting: Hybrid CNN–LSTM Research

**Deep Learning for Stock Price Prediction**
A Hybrid CNN–LSTM Approach for SPY Forecasting: Methodology, Evaluation, and Regime Analysis

Nour Bchini · Skidmore College · Faculty Supervisor: Prof. Wenlu Du
Undergraduate Independent Research Project · 2024–2026

This repo contains **two generations** of the same project, kept side by side:

| Paper | Code lives in | What it represents |
| ----- | -------------- | ------------------ |
| **Preliminary** (2024–2025) | [`experiments/`](experiments/README.md) | Original single-seed results, hybrid model, feature-engineering studies. Headline: parallel CNN–LSTM Close MAE **$1.04**. |
| **Revised** (2025–2026) | [`analysis/`](analysis/README.md) + [`training/`](training/) | Corrects a test-set peeking bug, adds a persistence baseline, runs 5 seeds, Diebold–Mariano, Wilcoxon, Chow, and a §6.5 log-return fix. Headline: **persistence beats all deep models on the long window**. |

Both pipelines run from the same data (`data/SPY.csv`) and the same env (`requirements.txt`).

---

## Quick Start (works for both pipelines)

```bash
git clone https://github.com/NourBchini/ResearchProject-Stock-Market-Prediction.git
cd ResearchProject-Stock-Market-Prediction

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_spy.py     # writes data/SPY.csv
```

After this you can run any script from either pipeline below.

---

## Revised Paper — `analysis/`

Headline (from [`analysis/results/multi_seed_compare.md`](analysis/results/multi_seed_compare.md)):

| Model | Close MAE — short window (11 days) | Close MAE — long window (1,622 days) |
| ----- | ---------------------------------- | ------------------------------------ |
| **Persistence (baseline)** | **$1.20** | **$3.60** |
| Fusion CNN–LSTM | $2.46 ± 2.62 | $22.11 ± 12.81 |
| LSTM-128 | $2.30 ± 0.98 | $23.63 ± 7.82 |
| Cascade CNN–LSTM | $11.13 ± 2.47 | $78.99 ± 20.02 |

5 seeds (`[42, 1337, 2024, 7, 12345]`), validation-based early stopping, no test-set peeking.
Diebold–Mariano (HLN-corrected) and Wilcoxon both reject equal accuracy with *p* < 0.001.

### How to reproduce each result (revised)

| Paper section | Run | Output file (committed) |
| ------------- | --- | ------------------------ |
| §6.2–6.3 + §7.1, Tables 2–4 + Chow | `python analysis/multi_seed_compare.py` | [`analysis/results/multi_seed_compare.json`](analysis/results/multi_seed_compare.json) + [`.md`](analysis/results/multi_seed_compare.md) |
| §6.5 legacy 80/20 anomaly (**$172.67**) | `python analysis/open_prediction_splits.py` | [`analysis/results/open_prediction_splits.csv`](analysis/results/open_prediction_splits.csv) |
| §6.5 log-return fix (Close, 1993–2020) | `python analysis/split_anomaly_log_returns.py` | [`analysis/results/split_anomaly_comparison.csv`](analysis/results/split_anomaly_comparison.csv) |
| §7.2 Pre-AI vs Post-AI | `python analysis/ai_era_split.py` | [`analysis/results/ai_era_split_results.csv`](analysis/results/ai_era_split_results.csv) |
| §7.2 Table 6 market structure | `python analysis/ai_market_analysis.py` | [`analysis/results/ai_market_behavior_analysis.csv`](analysis/results/ai_market_behavior_analysis.csv) |
| §7.3 Pandemic split | `python analysis/pandemic_split.py` | [`analysis/results/pandemic_split_results.csv`](analysis/results/pandemic_split_results.csv) |
| §7.4 Weekly windows | `python analysis/one_week_testing.py` | [`analysis/results/one_week_testing_results.csv`](analysis/results/one_week_testing_results.csv) |

`analysis/freeze_model.py` is a shared helper imported by `pandemic_split.py`, `ai_era_split.py`, and `one_week_testing.py`.
`training/lstm_pytorch.py`, `new_LSTM_CNN.py`, and `cnn_lstm_model.py` are the three architectures `multi_seed_compare.py` benchmarks.

---

## Preliminary Paper — `experiments/`

Headline (single-seed, original protocol): parallel CNN–LSTM Close MAE **$1.04**, 70/30 vs 80/20 vs 90/10 split study, hybrid CNN–LSTM + dedicated Low-price LSTM.

### How to reproduce each result (preliminary)

```bash
cd experiments/preliminary_analyses
```

| Preliminary topic | Run | Output |
| ----------------- | --- | ------ |
| Hyperparameter table (Tables 1–2 of preliminary paper) | `python 02_hyperparameter_table.py` | `results/hyperparameter_table.{csv,md}` |
| Volatility / technical-indicator features | `python 05_volatility_features.py` | `results/volatility_features_results.csv` |
| Multi-asset features (yfinance pulls) | `python 06_market_wide_features.py` | `results/market_wide_data_summary.json` |
| AI-trading narrative timeline | `python 07_08_ai_trading_research.py` | `research/ai_trading_history.json` |
| Aggregate preliminary report | `python 12_final_comparison.py` | `results/final_comparison_{report.txt,summary.json}` |
| LinkedIn infographic | `python generate_linkedin_attachment.py` | `results/linkedin_research_infographic.png` |

Hybrid model from the preliminary paper:

```bash
cd experiments/hybrid_model
python hybrid_model.py         # train both branches
python hybrid_inference.py     # combine predictions
```

Outputs: `predictions_*.csv` and the three figures in `figures/`.

See [`experiments/README.md`](experiments/README.md) for the full file map and why each piece was dropped in the revised paper.

---

## Repository Layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/                       SPY OHLCV (gitignored; regenerate with scripts/download_spy.py)
├── scripts/
│   └── download_spy.py         Daily SPY via yfinance
├── training/                   Architectures used by the revised benchmark
│   ├── lstm_pytorch.py         LSTM-128 baseline
│   ├── new_LSTM_CNN.py         Fusion (parallel) CNN–LSTM
│   └── cnn_lstm_model.py       Cascade (sequential) CNN–LSTM
├── analysis/                   Revised-paper pipeline + committed results
│   ├── README.md
│   ├── multi_seed_compare.py        Tables 2–4 + Chow test
│   ├── freeze_model.py              Shared CNN–LSTM helper
│   ├── pandemic_split.py            §7.3
│   ├── open_prediction_splits.py    §6.5 legacy
│   ├── split_anomaly_log_returns.py §6.5 log-return fix
│   ├── ai_era_split.py              §7.2
│   ├── one_week_testing.py          §7.4
│   ├── ai_market_analysis.py        Table 6
│   └── results/                CSV / JSON cited by the revised paper
├── experiments/                Preliminary-paper pipeline (preserved)
│   ├── README.md
│   ├── hybrid_model/
│   └── preliminary_analyses/
└── weights/                    Trained checkpoints (gitignored)
```

---

## Reproducibility Notes (revised pipeline)

- Five seeds share identical hyperparameters and chronological train/val/test boundaries.
- Train ends **2019-10-14**; validation = last 10% of pre-test sequences; test is never used for early stopping or scaler fitting.
- `MinMaxScaler` on `[0.01, 0.99]`, fit on training data only.
- Stochasticity: `numpy` / `torch` seeds per run in `multi_seed_compare.py`.

---

## Citation

```bibtex
@misc{bchini2026spy,
  author  = {Bchini, Nour},
  title   = {Deep Learning for Stock Price Prediction: A Hybrid CNN--LSTM Approach for SPY Forecasting},
  year    = {2026},
  note    = {Undergraduate Independent Research Project, Skidmore College. Supervisor: Wenlu Du.}
}
```

## References

- Mehtab, S., Sen, J., & Dasgupta, S. (2020). *Analysis and Forecasting of Financial Time Series Using CNN and LSTM-Based Deep Learning Models.* arXiv:2011.08011
- Lu, W., Li, J., Li, Y., Sun, A., & Wang, J. (2020). *A CNN–LSTM-based model to forecast stock prices.* Complexity, 2020, Article 6622927.

## License

MIT — see [`LICENSE`](LICENSE).

*Not investment advice. Academic research only.*
