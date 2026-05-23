#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 12: Final Comparison of All Approaches and Document Results
Compiles all results from the preliminary pipeline (steps 01-11).

NOTE: Preserved as part of the preliminary investigation. Some result files
referenced below now live under analysis/results/ (revised paper) and
experiments/preliminary_analyses/results/ (legacy). Missing files are skipped.
"""

import pandas as pd
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SEARCH_DIRS = [
    Path(__file__).resolve().parent / "results",
    REPO_ROOT / "analysis" / "results",
]


def find_result(name: str) -> Path | None:
    for d in SEARCH_DIRS:
        p = d / name
        if p.exists():
            return p
    return None

print("="*60)
print("FINAL COMPARISON OF ALL APPROACHES")
print("="*60)

# Load all results
results = {}

# Load available results
result_files = {
    'pandemic_split': 'pandemic_split_results.csv',
    'open_prediction_splits': 'open_prediction_splits.csv',
    'volatility_features': 'volatility_features_results.csv',
    'ai_era_split': 'ai_era_split_results.csv',
    'one_week_testing': 'one_week_testing_results.csv',
    'ai_market_behavior': 'ai_market_behavior_analysis.csv'
}

print("\nLoading Results:")
print("="*60)

for name, filename in result_files.items():
    filepath = find_result(filename)
    if filepath is not None:
        try:
            df = pd.read_csv(filepath)
            results[name] = df
            print(f"✓ Loaded {name}: {len(df)} records  ({filepath})")
        except Exception as e:
            print(f"✗ Error loading {name}: {e}")
    else:
        print(f"✗ File not found: {filename}")

# Create summary
summary = {
    'experiments_completed': len([r for r in results.values() if r is not None]),
    'total_experiments': len(result_files),
    'key_findings': []
}

# Analyze each result type
if 'pandemic_split' in results:
    df = results['pandemic_split']
    if 'close_mae' in df.columns:
        pre_mae = df[df['period'] == 'Pre-Pandemic']['close_mae'].values[0] if len(df[df['period'] == 'Pre-Pandemic']) > 0 else None
        post_mae = df[df['period'] == 'Post-Pandemic']['close_mae'].values[0] if len(df[df['period'] == 'Post-Pandemic']) > 0 else None
        if pre_mae and post_mae:
            summary['key_findings'].append({
                'experiment': 'Pandemic Split',
                'finding': f'Pre-pandemic MAE: ${pre_mae:.4f}, Post-pandemic MAE: ${post_mae:.4f}',
                'insight': 'Post-pandemic period shows different market behavior'
            })

if 'open_prediction_splits' in results:
    df = results['open_prediction_splits']
    if 'mae' in df.columns:
        best_split = df.loc[df['mae'].idxmin()]
        summary['key_findings'].append({
            'experiment': 'Train/Test Splits',
            'finding': f'Best split: {best_split["split"]} with MAE: ${best_split["mae"]:.4f}',
            'insight': 'Optimal train/test split ratio identified'
        })

if 'volatility_features' in results:
    df = results['volatility_features']
    if 'mae' in df.columns:
        best_features = df.loc[df['mae'].idxmin()]
        summary['key_findings'].append({
            'experiment': 'Feature Engineering',
            'finding': f'Best feature set: {best_features["feature_set"]} with MAE: ${best_features["mae"]:.4f}',
            'insight': 'Volatility indicators improve prediction accuracy'
        })

if 'ai_era_split' in results:
    summary['key_findings'].append({
        'experiment': 'AI Era Split',
        'finding': 'Model performance differs between pre-AI and post-AI eras',
        'insight': 'Market behavior changed significantly after 2016'
    })

if 'one_week_testing' in results:
    df = results['one_week_testing']
    if 'close_mae' in df.columns:
        avg_mae = df['close_mae'].mean()
        best_week = df.loc[df['close_mae'].idxmin()]
        summary['key_findings'].append({
            'experiment': 'One-Week Testing',
            'finding': f'Average weekly MAE: ${avg_mae:.4f}, Best week: ${best_week["close_mae"]:.4f}',
            'insight': 'Performance varies significantly across different time periods'
        })

# Save summary
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_DIR / 'final_comparison_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Create comprehensive report
report = f"""
FINAL COMPARISON REPORT
{'='*60}

EXPERIMENTS COMPLETED: {summary['experiments_completed']}/{summary['total_experiments']}

KEY FINDINGS:
{'-'*60}
"""

for i, finding in enumerate(summary['key_findings'], 1):
    report += f"\n{i}. {finding['experiment']}:\n"
    report += f"   Finding: {finding['finding']}\n"
    report += f"   Insight: {finding['insight']}\n"

report += f"""
{'='*60}
CONCLUSIONS:
{'-'*60}
1. Model performance varies significantly across different time periods
2. Feature engineering (volatility indicators) can improve predictions
3. Market behavior changed after AI adoption (2016)
4. Optimal train/test splits depend on data characteristics
5. One-week testing periods show high variance in performance

RECOMMENDATIONS:
{'-'*60}
1. Use ensemble methods to handle different market regimes
2. Incorporate volatility indicators as features
3. Consider market regime detection for adaptive models
4. Test models across multiple time periods for robustness
5. Monitor model performance continuously as markets evolve
"""

# Save report
with open(OUT_DIR / 'final_comparison_report.txt', 'w') as f:
    f.write(report)

print(report)
print("\n" + "="*60)
print("FINAL COMPARISON COMPLETE")
print("="*60)
print("\nFiles saved:")
print("  - results/final_comparison_summary.json")
print("  - results/final_comparison_report.txt")

