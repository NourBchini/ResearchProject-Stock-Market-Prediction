#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Master Script: Run All Final Analysis Steps
Executes all 12 steps in sequence.

NOTE: Preserved from the preliminary pipeline. Several step files were moved
during cleanup; this orchestrator is kept for transparency and reproducibility
of the preliminary investigation. The revised-paper pipeline is driven by
analysis/multi_seed_compare.py.
"""

import subprocess
import sys
import os

print("="*60)
print("RUNNING ALL FINAL ANALYSIS STEPS")
print("="*60)

steps = [
    ("Step 1: Freeze Model", "01_freeze_model.py"),
    ("Step 2: Hyperparameter Table", "02_hyperparameter_table.py"),
    ("Step 3: Pandemic Split", "03_pandemic_split.py"),
    ("Step 4: Open Prediction Splits", "04_open_prediction_splits.py"),
    ("Step 5: Volatility Features", "05_volatility_features.py"),
    ("Step 6: Market-Wide Features", "06_market_wide_features.py"),
    ("Step 7-8: AI Trading Research", "07_08_ai_trading_research.py"),
    ("Step 9: AI Era Split", "09_ai_era_split.py"),
    ("Step 10: One-Week Testing", "10_one_week_testing.py"),
    ("Step 11: AI Market Analysis", "11_ai_market_analysis.py"),
    ("Step 12: Final Comparison", "12_final_comparison.py"),
]

completed = []
failed = []

for step_name, script_name in steps:
    print(f"\n{'='*60}")
    print(f"Running {step_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per step
        )
        
        if result.returncode == 0:
            print(f"✓ {step_name} completed successfully")
            completed.append(step_name)
            if result.stdout:
                print(result.stdout[-500:])  # Last 500 chars
        else:
            print(f"✗ {step_name} failed")
            print(result.stderr)
            failed.append(step_name)
    except subprocess.TimeoutExpired:
        print(f"✗ {step_name} timed out")
        failed.append(step_name)
    except Exception as e:
        print(f"✗ {step_name} error: {e}")
        failed.append(step_name)

print(f"\n{'='*60}")
print("ANALYSIS SUMMARY")
print(f"{'='*60}")
print(f"Completed: {len(completed)}/{len(steps)}")
print(f"Failed: {len(failed)}/{len(steps)}")

if completed:
    print("\n✓ Completed Steps:")
    for step in completed:
        print(f"  - {step}")

if failed:
    print("\n✗ Failed Steps:")
    for step in failed:
        print(f"  - {step}")

print(f"\n{'='*60}")
print("All results saved to 'results/' directory")
print(f"{'='*60}")

