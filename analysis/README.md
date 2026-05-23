# analysis/ — Revised Paper Pipeline

Every script in this directory is referenced by the revised paper. Outputs
written to `analysis/results/` are the exact numbers cited in Tables 2–6 and
Sections 6.5 / 7.

## Files

| Script | Paper reference | What it does |
| ------ | --------------- | ------------ |
| `multi_seed_compare.py` | Tables 2–4, §6.3, §7.1 | Trains Fusion / Cascade / LSTM-128 over 5 seeds with strict train/val/test splits, evaluates against the persistence baseline on the short (11-day) and long (1,622-day) windows, then runs Diebold–Mariano (HLN-corrected) and Wilcoxon signed-rank tests. Also runs the Chow test on AR(1) log-returns (Table 5). Writes `results/multi_seed_compare.{json,md}`. |
| `01_freeze_model.py` | (helper) | Defines `Frozen_CNN_LSTM` — the parallel CNN+LSTM architecture imported by the regime analyses (03, 04, 09, 10) so that all of them share a single, frozen model definition. |
| `03_pandemic_split.py` | §7.3 | Splits the post-test window at March 2020 and reports pre- vs post-pandemic MAE and MAPE. Writes `results/pandemic_split_results.csv`. |
| `04_open_prediction_splits.py` | §6.5 (legacy) | Open-price splits with **global** MinMax scaler (preliminary protocol). Produces the cited **$172.67** on 80/20. Writes `results/open_prediction_splits.csv`. |
| `05_split_anomaly_log_returns.py` | §6.5 (revised) | Close-price splits through 2020: price-level (train-only vs global scaler) vs **log-return** targets. Writes `results/split_anomaly_comparison.csv`. |
| `09_ai_era_split.py` | §7.2 | Compares Close MAE before and after 2016 (pre-AI: $2.31 vs post-AI: $8.91). Writes `results/ai_era_split_results.csv`. |
| `10_one_week_testing.py` | §7.4 | Evaluates six randomly selected one-week windows to expose week-to-week variability. Writes `results/one_week_testing_results.csv`. |
| `11_ai_market_analysis.py` | Table 6 | Computes the market structure metrics (volatility, Sharpe, volume volatility, GARCH-β) for the pre-AI and post-AI eras. Writes `results/ai_market_behavior_analysis.csv`. |

## Outputs (`analysis/results/`)

These files are tracked in git because they are the numbers cited in the paper.
Regenerate them by running the scripts above; the JSON files are deterministic
given the same seeds.

```
results/
├── multi_seed_compare.json   Tables 2–4 + Chow test (§7.1) numbers
├── multi_seed_compare.md     Human-readable summary of the above
├── ai_era_split_results.csv  §7.2 pre/post-AI MAE
├── ai_market_behavior_analysis.csv  Table 6 (market structure)
├── one_week_testing_results.csv  §7.4 weekly windows
├── open_prediction_splits.csv  §6.5 legacy Open splits ($172.67 on 80/20)
├── split_anomaly_comparison.csv  §6.5 Close: price vs log-return targets
└── pandemic_split_results.csv  §7.3 pre/post COVID-19
```

## Conventions

- All scripts assume the working directory is the repo root *or* this
  directory; `multi_seed_compare.py` resolves paths via `Path(__file__).resolve()`
  so it works from anywhere.
- The Chow test (§7.1) is implemented at the top of `multi_seed_compare.py`
  using SciPy's F-distribution, restricted to candidate breakpoints 2012–2018
  to avoid contamination by the 2020 COVID shock.
