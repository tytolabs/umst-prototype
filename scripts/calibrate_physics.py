#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
# SPDX-License-Identifier: MIT
"""
PHYSICS CALIBRATION - Proper Parameter Fitting
==============================================

Fits DUMSTO physics parameters to each dataset using optimization.
Ensures physics models are properly calibrated before benchmarking.

Calibration Parameters:
- s_intrinsic: Intrinsic gel strength (MPa)
- k_slag: Slag reactivity factor
- k_fly_ash: Fly ash reactivity factor
- k_ref: Hydration rate constant
- early_boost: Early age strength boost factor
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
from dataclasses import dataclass
import time
from pathlib import Path

# ============================================================================
# PHYSICS KERNEL (Same as benchmark)
# ============================================================================

def compute_hydration_tensor(age: np.ndarray, scm_ratio: np.ndarray, k_ref: float) -> np.ndarray:
    """Avrami kinetics for hydration degree"""
    alpha_max = 0.95 - scm_ratio * 0.15
    scm_factor = 1.0 - scm_ratio * 0.4
    k = k_ref * scm_factor
    alpha = alpha_max * (1.0 - np.exp(-k * np.sqrt(age)))
    return np.clip(alpha, 0.0, 1.0)

def compute_physics_tensor(X: pd.DataFrame, params: np.ndarray) -> np.ndarray:
    """
    Compute physics predictions with parameters to optimize

    params: [s_intrinsic, k_slag, k_fly_ash, k_ref, early_boost]
    """
    s_intrinsic, k_slag, k_fly_ash, k_ref, early_boost = params

    cement = X['cement'].values.astype(np.float64)
    slag = X['slag'].values.astype(np.float64) if 'slag' in X.columns else np.zeros(len(X))
    fly_ash = X['fly_ash'].values.astype(np.float64) if 'fly_ash' in X.columns else np.zeros(len(X))
    water = X['water'].values.astype(np.float64)
    age = X['age'].values.astype(np.float64)

    binder = cement + slag + fly_ash
    binder = np.maximum(binder, 1e-6)

    effective_cement = cement + k_slag * slag + k_fly_ash * fly_ash
    effective_cement = np.maximum(effective_cement, 1e-6)

    w_c = np.clip(water / effective_cement, 0.25, 1.0)
    scm_ratio = (slag + fly_ash) / binder

    alpha = compute_hydration_tensor(age, scm_ratio, k_ref)

    vg = 0.68 * alpha
    vc = w_c - 0.36 * alpha
    space = vg + np.maximum(vc, 0) + 0.02

    x = vg / np.maximum(space, 1e-6)
    fc = s_intrinsic * (x ** 3)
    fc = np.where(age < 7.0, fc * early_boost, fc)

    return np.clip(fc, 0.0, 150.0)

# ============================================================================
# CALIBRATION OBJECTIVE FUNCTION
# ============================================================================

def calibration_objective(params: np.ndarray, X: pd.DataFrame, y: np.ndarray) -> float:
    """
    Objective function to minimize: MAE between physics predictions and actual data

    params: [s_intrinsic, k_slag, k_fly_ash, k_ref, early_boost]
    """
    try:
        pred = compute_physics_tensor(X, params)
        return mean_absolute_error(y, pred)
    except:
        return 1e6  # High penalty for invalid parameters

# ============================================================================
# PARAMETER BOUNDS AND CONSTRAINTS
# ============================================================================

# Parameter bounds for optimization
PARAM_BOUNDS = [
    (30.0, 120.0),   # s_intrinsic: 30-120 MPa (reasonable gel strength range)
    (0.1, 2.0),      # k_slag: 0.1-2.0 (reactivity factor)
    (0.1, 2.0),      # k_fly_ash: 0.1-2.0 (reactivity factor)
    (0.3, 1.0),      # k_ref: 0.3-1.0 (hydration rate constant)
    (1.0, 2.0),      # early_boost: 1.0-2.0 (early age boost factor)
]

# Initial guesses for different datasets
INITIAL_GUESSES = {
    'D1': [80.0, 1.0, 1.0, 0.55, 1.2],      # UCI Concrete (well-studied)
    'D2': [60.0, 0.2, 0.22, 0.5, 1.4],     # General concrete mix
    'D3': [60.0, 0.2, 0.2, 0.5, 1.6],      # General concrete mix
    'D4': [81.0, 0.2, 0.2, 0.7, 1.1],      # High performance mix
    'full': [70.0, 0.4, 0.4, 0.55, 1.3],   # Combined dataset
}

# ============================================================================
# CALIBRATION FUNCTION
# ============================================================================

def calibrate_physics_parameters(X: pd.DataFrame, y: np.ndarray, dataset_id: str,
                               max_iter: int = 100) -> dict:
    """
    Calibrate physics parameters for a specific dataset

    Args:
        X: Feature DataFrame
        y: Target values
        dataset_id: Dataset identifier
        max_iter: Maximum optimization iterations

    Returns:
        dict: Calibration results with fitted parameters and metrics
    """

    print(f"  Calibrating {dataset_id} with {len(X)} samples...")

    # Use dataset-specific initial guess
    initial_guess = INITIAL_GUESSES.get(dataset_id, [70.0, 0.4, 0.4, 0.55, 1.3])

    start_time = time.time()

    # Optimization using L-BFGS-B (bounded optimization)
    result = minimize(
        calibration_objective,
        initial_guess,
        args=(X, y),
        method='L-BFGS-B',
        bounds=PARAM_BOUNDS,
        options={
            'maxiter': max_iter,
            'ftol': 1e-6,
            'gtol': 1e-6,
            'disp': False
        }
    )

    calibration_time = time.time() - start_time

    # Extract fitted parameters
    fitted_params = result.x
    s_intrinsic, k_slag, k_fly_ash, k_ref, early_boost = fitted_params

    # Compute final metrics
    final_pred = compute_physics_tensor(X, fitted_params)
    final_mae = mean_absolute_error(y, final_pred)
    final_rmse = np.sqrt(np.mean((final_pred - y) ** 2))
    final_r2 = 1 - np.sum((final_pred - y) ** 2) / np.sum((y - np.mean(y)) ** 2)

    # Initial guess metrics for comparison
    initial_pred = compute_physics_tensor(X, initial_guess)
    initial_mae = mean_absolute_error(y, initial_pred)

    improvement = initial_mae - final_mae

    return {
        'dataset': dataset_id,
        'fitted_parameters': {
            's_intrinsic': float(s_intrinsic),
            'k_slag': float(k_slag),
            'k_fly_ash': float(k_fly_ash),
            'k_ref': float(k_ref),
            'early_boost': float(early_boost),
        },
        'initial_guess': initial_guess,
        'metrics': {
            'final_mae': float(final_mae),
            'final_rmse': float(final_rmse),
            'final_r2': float(final_r2),
            'initial_mae': float(initial_mae),
            'improvement': float(improvement),
        },
        'optimization': {
            'success': result.success,
            'message': result.message,
            'nfev': result.nfev,
            'njev': result.njev,
            'calibration_time': float(calibration_time),
        }
    }

# ============================================================================
# BATCH CALIBRATION
# ============================================================================

def calibrate_all_datasets(output_path: str = None) -> dict:
    """
    Calibrate physics parameters for all datasets

    Args:
        output_path: Where to save calibration results

    Returns:
        dict: Calibration results for all datasets
    """

    print("=" * 70)
    print("PHYSICS CALIBRATION - Fitting Parameters to Data")
    print("=" * 70)

    datasets = ['D1', 'D2', 'D3', 'D4', 'full']
    results = {}

    for dataset_id in datasets:
        print(f"\nDATASET: {dataset_id}")
        print("-" * 50)

        # Load dataset
        base_path = Path(__file__).parent.parent / 'data'
        csv_path = base_path / f'dataset_{dataset_id}.csv'

        if not csv_path.exists():
            print(f"  ERROR: Dataset {dataset_id} not found at {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # Ensure required columns exist
        required_cols = ['cement', 'slag', 'fly_ash', 'water', 'superplasticizer',
                        'coarse_agg', 'fine_agg', 'age', 'strength']
        for col in required_cols:
            if col not in df.columns:
                if col == 'strength':
                    continue  # Target column
                df[col] = 0.0

        # Split data (use training portion for calibration)
        X = df[['cement', 'slag', 'fly_ash', 'water', 'superplasticizer',
                'coarse_agg', 'fine_agg', 'age']]
        y = df['strength'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        print(f"  Using {len(X_train)} samples for calibration, {len(X_test)} for validation")

        # Calibrate parameters
        cal_result = calibrate_physics_parameters(X_train, y_train, dataset_id)

        # Validate on test set
        test_pred = compute_physics_tensor(X_test, list(cal_result['fitted_parameters'].values()))
        test_mae = mean_absolute_error(y_test, test_pred)

        cal_result['validation'] = {
            'test_mae': float(test_mae),
            'test_samples': len(X_test)
        }

        results[dataset_id] = cal_result

        # Print summary
        params = cal_result['fitted_parameters']
        metrics = cal_result['metrics']
        opt = cal_result['optimization']

        print(f"  FITTED PARAMETERS:")
        print(f"    s_intrinsic: {params['s_intrinsic']:.1f} MPa")
        print(f"    k_slag: {params['k_slag']:.3f}")
        print(f"    k_fly_ash: {params['k_fly_ash']:.3f}")
        print(f"    k_ref: {params['k_ref']:.3f}")
        print(f"    early_boost: {params['early_boost']:.3f}")
        print(f"  PERFORMANCE:")
        print(f"    Train MAE: {metrics['final_mae']:.2f} MPa (was {metrics['initial_mae']:.2f})")
        print(f"    Test MAE: {test_mae:.2f} MPa")
        print(f"    Improvement: +{metrics['improvement']:.2f} MPa")
        print(f"    R²: {metrics['final_r2']:.3f}")
        print(f"  OPTIMIZATION: {opt['success']} ({opt['nfev']} evaluations, {opt['calibration_time']:.1f}s)")

    # Save results
    if output_path is None:
        output_path = Path(__file__).parent.parent / 'results' / 'macos' / 'ssot' / 'physics_calibration_2026-01-22.json'

    output_data = {
        'metadata': {
            'title': 'Physics Parameter Calibration',
            'date': '2026-01-22',
            'method': 'L-BFGS-B optimization',
            'objective': 'Minimize MAE between physics predictions and data',
            'parameters_fitted': ['s_intrinsic', 'k_slag', 'k_fly_ash', 'k_ref', 'early_boost'],
            'bounds': PARAM_BOUNDS,
        },
        'calibrations': results
    }

    with open(output_path, 'w') as f:
        import json
        json.dump(output_data, f, indent=2)

    print(f"\n Calibration results saved to: {output_path}")

    return output_data

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_calibrated_parameters(dataset_id: str, calibration_file: str = None) -> dict:
    """
    Load calibrated parameters for a specific dataset

    Args:
        dataset_id: Dataset identifier
        calibration_file: Path to calibration file

    Returns:
        dict: Fitted parameters
    """
    if calibration_file is None:
        base_path = Path(__file__).parent.parent / 'results' / 'macos' / 'ssot'
        calibration_file = base_path / 'physics_calibration_2026-01-22.json'

    try:
        with open(calibration_file) as f:
            data = json.load(f)
        return data['calibrations'][dataset_id]['fitted_parameters']
    except:
        # Fallback to initial guesses if calibration failed
        print(f"Warning: Could not load calibrated parameters for {dataset_id}, using defaults")
        return {
            's_intrinsic': INITIAL_GUESSES[dataset_id][0],
            'k_slag': INITIAL_GUESSES[dataset_id][1],
            'k_fly_ash': INITIAL_GUESSES[dataset_id][2],
            'k_ref': INITIAL_GUESSES[dataset_id][3],
            'early_boost': INITIAL_GUESSES[dataset_id][4],
        }

if __name__ == '__main__':
    calibrate_all_datasets()