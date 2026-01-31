#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
# SPDX-License-Identifier: MIT
"""
DUMSTO RIGOROUS RELIABILITY TEST (20 RUNS)
==========================================
Executes the Maximum Accuracy Benchmark 20 times to establish 
statistical confidence intervals and cross-platform consistency.

Settings:
- Runs: 20
- Validation: Maximum Accuracy (No Early Stopping, 1000 estimators)
- Comparison: vs Pop!_OS Reference
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import time
import json
import statistics
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Configuration
N_RUNS = 20
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using Device: {DEVICE}")

# Pop!_OS Reference Values (MAE)
REF_POPOS = {
    'D1': 3.13,
    'D2': 6.97,
    'D3': 6.29,
    'D4': 14.61
}

# Calibration Data
DATASET_CALIBRATIONS = {
    'D1': {'s_intrinsic': 80.0, 'k_slag': 1.0, 'k_fly_ash': 1.0, 'k_ref': 0.55, 'early_boost': 1.2},
    'D2': {'s_intrinsic': 60.0, 'k_slag': 0.2, 'k_fly_ash': 0.22, 'k_ref': 0.5, 'early_boost': 1.4},
    'D3': {'s_intrinsic': 60.0, 'k_slag': 0.2, 'k_fly_ash': 0.2, 'k_ref': 0.5, 'early_boost': 1.6},
    'D4': {'s_intrinsic': 81.0, 'k_slag': 0.2, 'k_fly_ash': 0.2, 'k_ref': 0.7, 'early_boost': 1.1},
}

FEATURE_COLS = ['cement', 'slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_agg', 'fine_agg', 'age']
TARGET_COL = 'strength'

# ============================================================================
# PHYSICS KERNEL
# ============================================================================

def compute_hydration_degree(age, temp_c, scm_ratio, k_ref):
    alpha_max = 0.95 - scm_ratio * 0.15
    t_ref_k = 293.15
    t_k = temp_c + 273.15
    e_over_r = 5000.0
    
    temp_factor = np.exp(e_over_r * (1.0/t_ref_k - 1.0/t_k))
    scm_factor = 1.0 - scm_ratio * 0.4
    k = k_ref * temp_factor * scm_factor
    
    alpha = alpha_max * (1.0 - np.exp(-k * np.sqrt(age)))
    return np.clip(alpha, 0.0, 1.0)

def compute_physics_strength(row, cal):
    cement = row['cement']
    slag = row.get('slag', 0.0)
    fly_ash = row.get('fly_ash', 0.0)
    water = row['water']
    age = row['age']
    
    binder = cement + slag + fly_ash
    if binder <= 0: return 0.0
    
    effective_cement = cement + cal['k_slag'] * slag + cal['k_fly_ash'] * fly_ash
    if effective_cement <= 0: return 0.0
    
    w_c = np.clip(water / effective_cement, 0.25, 1.0)
    scm_ratio = (slag + fly_ash) / binder
    
    alpha = compute_hydration_degree(age, 20.0, scm_ratio, cal['k_ref'])
    
    vg = 0.68 * alpha
    vc = w_c - 0.36 * alpha
    space = vg + max(0, vc) + 0.02
    
    if space <= 0.001: return 0.0
    
    x = vg / space
    fc = cal['s_intrinsic'] * (x ** 3)
    
    if age < 7.0:
        fc *= cal['early_boost']
    
    return np.clip(fc, 0.0, 150.0)

def check_admissibility(predictions):
    neg_violations = np.sum(predictions < 0)
    high_violations = np.sum(predictions > 120)
    total_violations = neg_violations + high_violations
    return 100.0 * (1.0 - total_violations / max(len(predictions), 1))

# ============================================================================
# HYBRID MODEL TRAINER (Max Accuracy)
# ============================================================================

def train_hybrid_max_acc(X_train, y_train, X_test, y_test, cal):
    # 1. Physics Baseline
    physics_train = np.array([compute_physics_strength(row, cal) for _, row in X_train.iterrows()])
    residuals_train = y_train - physics_train
    
    # 2. Residual Learner (Max Accuracy: 1000 estimators, no early stop)
    # Using random_state=None to test stability across runs if preferred,
    # but strictly random_state varies per run in the outer loop for true validation?
    # Actually, usually 20 runs implies different seeds.
    
    model = GradientBoostingRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        # random_state is handled by the seed passed to the function or global stability
    )
    
    model.fit(X_train, residuals_train)
    
    # 3. Predict
    physics_test = np.array([compute_physics_strength(row, cal) for _, row in X_test.iterrows()])
    residual_pred = model.predict(X_test)
    
    # 4. Bind and Combine
    max_correction = 0.5 * np.abs(physics_test)
    residual_pred = np.clip(residual_pred, -max_correction, max_correction)
    
    final_pred = np.clip(physics_test + residual_pred, 0, 150)
    
    mae = mean_absolute_error(y_test, final_pred)
    adm = check_admissibility(final_pred)
    
    return mae, adm

# ============================================================================
# MAIN LOOP
# ============================================================================

def run_reliability_test():
    print("=" * 60)
    print(f"STARTING 20-RUN DEEP RELIABILITY TEST")
    print("=" * 60)
    
    results = {'D1': [], 'D2': [], 'D3': [], 'D4': []}
    
    # Load Datasets
    base_path = Path(__file__).parent.parent / 'data'
    dfs = {}
    for did in results.keys():
        dfs[did] = pd.read_csv(base_path / f'dataset_{did}.csv')
        for col in FEATURE_COLS + [TARGET_COL]:
            if col not in dfs[did].columns: dfs[did][col] = 0.0

    # Execute Runs
    for run_i in range(1, N_RUNS + 1):
        print(f"\nRun {run_i}/{N_RUNS}...")
        
        for did in results.keys():
            df = dfs[did]
            cal = DATASET_CALIBRATIONS[did]
            
            # Use different random seed per run to test stability
            X_train, X_test, y_train, y_test = train_test_split(
                df[FEATURE_COLS], df[TARGET_COL].values, 
                test_size=0.2, 
                random_state=42 + run_i # Different split/seed each time
            )
            
            mae, adm = train_hybrid_max_acc(X_train, y_train, X_test, y_test, cal)
            results[did].append({'mae': mae, 'adm': adm})
            
            print(f"  {did}: MAE={mae:.2f} | Adm={adm:.1f}%")

    # Aggregate
    final_stats = {}
    print("\n" + "="*80)
    print("FINAL RELIABILITY STATISTICS (20 RUNS)")
    print("="*80)
    print(f"{'Dataset':<10} | {'MAE (Mean ± Std)':<20} | {'Min / Max':<15} | {'Ref (Pop!_OS)':<15} | {'Status'}")
    print("-" * 80)
    
    for did in results.keys():
        maes = [r['mae'] for r in results[did]]
        adms = [r['adm'] for r in results[did]]
        
        mean_mae = statistics.mean(maes)
        stdev_mae = statistics.stdev(maes)
        min_mae = min(maes)
        max_mae = max(maes)
        mean_adm = statistics.mean(adms)
        
        ref = REF_POPOS[did]
        status = "✅ PASS" if abs(mean_mae - ref) / ref < 0.1 else "⚠️ CHECK"
        
        print(f"{did:<10} | {mean_mae:.2f} ± {stdev_mae:.2f}      | {min_mae:.2f} / {max_mae:.2f} | {ref:<15} | {status}")
        
        final_stats[did] = {
            'mae_mean': mean_mae,
            'mae_std': stdev_mae,
            'mae_min': min_mae,
            'mae_max': max_mae,
            'adm_mean': mean_adm,
            'runs': results[did]
        }

    # Save artifact
    output_path = Path(__file__).parent.parent / 'results' / 'macos' / 'ssot' / 'SSOT_v6_20run_reliability_macos.json'
    with open(output_path, 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    print("-" * 80)
    print(f"Artifact saved to: {output_path}")
    print("Deep Reliability Test Complete.")

if __name__ == "__main__":
    run_reliability_test()
