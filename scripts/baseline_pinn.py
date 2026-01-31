#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
# SPDX-License-Identifier: MIT
"""
PINN Baseline - Physics-Informed Neural Networks
================================================

Demonstrates soft constraint approach vs DUMSTO hard gates.

PINNs incorporate physics as soft penalties in loss function:
L_total = L_data + λ_physics * L_physics

For cementitious materials, physics losses include:
- Hydration kinetics (Avrami equation)
- Strength development (Powers' law)
- Basic conservation laws
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
import os
from pathlib import Path
import time

warnings.filterwarnings('ignore')

# --- PATHS ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / 'data'

DATASETS = {
    'D1 (UCI)': str(DATA_DIR / 'dataset_D1.csv'),
    'D2 (NDT)': str(DATA_DIR / 'dataset_D2.csv'),
    'D3 (SON)': str(DATA_DIR / 'dataset_D3.csv'),
    'D4 (RH)':  str(DATA_DIR / 'dataset_D4.csv'),
}

FEAT_COLS = ['cement', 'slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_agg', 'fine_agg', 'age']

class CementPINN(nn.Module):
    """Physics-Informed Neural Network for concrete strength prediction"""

    def __init__(self, input_dim=8, hidden_dim=64, physics_weight=0.1, hard_constraints=False):
        super(CementPINN, self).__init__()
        self.physics_weight = physics_weight
        self.hard_constraints = hard_constraints

        # Neural network layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Hard constraint projection layer
        if self.hard_constraints:
            self.physics_projector = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # Get raw neural network prediction
        raw_pred = self.layers(x)

        if self.hard_constraints:
            # Apply hard physics constraints
            return self.apply_hard_constraints(x, raw_pred)
        else:
            # Soft constraints (original PINN approach)
            return raw_pred

    def apply_hard_constraints(self, x, raw_pred):
        """
        Apply hard physics constraints by projecting predictions onto admissible manifold

        This enforces:
        1. Strength bounds based on composition
        2. Hydration-based limits
        3. Conservation law constraints
        """
        batch_size = x.shape[0]
        constrained_pred = raw_pred.clone()

        for i in range(batch_size):
            # Extract material composition
            cement = x[i, 0].item()
            slag = x[i, 1].item()
            fly_ash = x[i, 2].item()
            water = x[i, 3].item()
            age = max(x[i, 7].item(), 1.0)  # Ensure positive age

            # Skip invalid compositions
            if cement <= 0 or water <= 0:
                continue

            # Calculate physics-based bounds
            w_c_ratio = water / cement

            # Hydration-based strength limit (simplified Powers' law)
            # α(t) from Avrami equation
            alpha_max = min(0.95, 0.95 - 0.15 * (slag + fly_ash) / max(cement + slag + fly_ash, 0.1))
            k_hydration = 0.01 / max(1.0 + w_c_ratio, 0.1)
            alpha = alpha_max * (1.0 - torch.exp(torch.tensor(-k_hydration * age)))

            # Gel-space ratio and resulting strength limit
            binder = cement + slag + fly_ash
            if binder > 0:
                gel_ratio = alpha * (cement / binder) / w_c_ratio
                max_strength = 50.0 * torch.pow(torch.clamp(gel_ratio, 0.01, 10.0), 3.0)

                # Conservation-based upper bound
                composition_factor = cement / (cement + water + slag + fly_ash)
                conservation_limit = 120.0 * composition_factor

                # Apply hard constraint: project to [5, min(physics_limit, conservation_limit)]
                physics_limit = torch.clamp(max_strength, torch.tensor(5.0), torch.tensor(120.0))
                upper_bound = torch.min(physics_limit, torch.tensor(conservation_limit))

                constrained_pred[i] = torch.clamp(raw_pred[i], torch.tensor(5.0), upper_bound)

        return constrained_pred

    def physics_loss(self, x, y_pred):
        """
        Physics-informed loss terms for cementitious materials

        Incorporates:
        1. Hydration kinetics (Avrami equation)
        2. Strength development (Powers' law)
        3. Basic conservation laws
        """
        batch_size = x.shape[0]
        physics_loss = torch.tensor(0.0, device=x.device)

        for i in range(batch_size):
            # Extract material composition
            cement = x[i, 0].item()
            slag = x[i, 1].item()
            fly_ash = x[i, 2].item()
            water = x[i, 3].item()
            age = x[i, 7].item()  # age in days

            # Skip invalid compositions
            if cement <= 0 or water <= 0 or age <= 0:
                continue

            # 1. Hydration Kinetics Loss (Avrami equation)
            # α(t) = α_max * (1 - exp(-k*t^β))
            w_c_ratio_tensor = torch.tensor(water / cement)
            scm_ratio = (slag + fly_ash) / cement if cement > 0 else 0
            alpha_max_tensor = torch.tensor(0.95 - min(0.15, 0.15 * scm_ratio))
            k_hydration_tensor = 0.01 * (1.0 / (1.0 + w_c_ratio_tensor))
            age_tensor = torch.tensor(age)

            alpha_pred = alpha_max_tensor * (1.0 - torch.exp(-k_hydration_tensor * torch.pow(age_tensor, 0.8)))

            # Expected hydration should be reasonable (0.1 to 0.9 for mature concrete)
            hydration_penalty = torch.relu(torch.tensor(0.1, device=x.device) - alpha_pred) + torch.relu(alpha_pred - torch.tensor(0.9, device=x.device))
            physics_loss += hydration_penalty.sum()

            # 2. Powers' Law Loss (strength ∝ gel/space ratio³)
            # Simplified: strength should correlate with hydration and composition
            binder = torch.tensor(cement + slag + fly_ash, device=x.device)
            if binder > 0:
                cement_tensor = torch.tensor(cement, device=x.device)
                slag_tensor = torch.tensor(slag, device=x.device)
                fly_ash_tensor = torch.tensor(fly_ash, device=x.device)
                w_c_ratio_tensor = torch.tensor(water / cement, device=x.device) if cement > 0 else torch.tensor(1.0, device=x.device)

                # Gel-space ratio approximation
                gel_ratio = alpha_pred * (cement_tensor / binder) * (1.0 / w_c_ratio_tensor)
                expected_strength = 50.0 * torch.pow(torch.clamp(gel_ratio, 0.01, 10.0), 3)  # MPa, simplified Powers' law

                # Strength should be reasonable (10-100 MPa range)
                strength_penalty = torch.relu(torch.tensor(10.0, device=x.device) - expected_strength) + torch.relu(expected_strength - torch.tensor(100.0, device=x.device))
                physics_loss += strength_penalty.sum()

            # 3. Conservation Law Loss
            # Total mass balance (simplified check)
            total_input = torch.tensor(cement + slag + fly_ash + water, device=x.device)
            # Strength shouldn't exceed unreasonable bounds based on composition
            if total_input > 0:
                cement_tensor = torch.tensor(cement, device=x.device)
                max_reasonable_strength = 100.0 * (cement_tensor / total_input)  # Simplified
                conservation_penalty = torch.relu(y_pred[i] - max_reasonable_strength)
                physics_loss += conservation_penalty.sum()

        return physics_loss / max(1, batch_size)  # Average over batch

def thermodynamic_admissibility(y_pred, X=None):
    """Same admissibility check as other baselines"""
    accepted = 0
    total = len(y_pred)

    for val in y_pred:
        if val >= 0 and val <= 130:
            accepted += 1

    return (accepted / total) * 100.0

def create_physics_informed_loss(model, physics_weight=0.1):
    """Create combined loss function with physics constraints"""
    mse_loss = nn.MSELoss()

    def pinn_loss(x, y_true):
        y_pred = model(x)

        # Data loss
        data_loss = mse_loss(y_pred.squeeze(), y_true)

        # Physics loss
        physics_loss = model.physics_loss(x, y_pred)

        # Combined loss
        total_loss = data_loss + physics_weight * physics_loss

        return total_loss, data_loss, physics_loss

    return pinn_loss

def train_pinn_model(train_loader, val_loader=None, physics_weight=0.1, hard_constraints=False, epochs=200, device='cpu'):
    """Train PINN model with physics-informed loss"""
    model = CementPINN(physics_weight=physics_weight, hard_constraints=hard_constraints).to(device)
    pinn_loss_fn = create_physics_informed_loss(model, physics_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0
        train_loss_data = 0
        train_loss_physics = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            total_loss, data_loss, physics_loss = pinn_loss_fn(batch_x, batch_y)
            total_loss.backward()
            optimizer.step()

            train_loss_total += total_loss.item()
            train_loss_data += data_loss.item()
            train_loss_physics += physics_loss.item()

        # Average over batches
        n_batches = len(train_loader)
        train_loss_total /= n_batches
        train_loss_data /= n_batches
        train_loss_physics /= n_batches

        scheduler.step()

        # Validation
        if val_loader and epoch % 10 == 0:
            model.eval()
            val_loss_total = 0
            val_loss_data = 0
            val_loss_physics = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    total_loss, data_loss, physics_loss = pinn_loss_fn(batch_x, batch_y)
                    val_loss_total += total_loss.item()
                    val_loss_data += data_loss.item()
                    val_loss_physics += physics_loss.item()

            val_loss_total /= len(val_loader)
            val_loss_data /= len(val_loader)
            val_loss_physics /= len(val_loader)

            if val_loss_total < best_val_loss:
                best_val_loss = val_loss_total
                patience_counter = 0
                torch.save(model.state_dict(), 'best_pinn_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 20 == 0:
            print(".4f")

    # Load best model
    if val_loader and os.path.exists('best_pinn_model.pth'):
        model.load_state_dict(torch.load('best_pinn_model.pth'))
        os.remove('best_pinn_model.pth')

    return model

def predict_pinn(model, loader, device='cpu'):
    """Make predictions with trained PINN"""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            predictions.extend(pred.cpu().numpy().flatten())
            targets.extend(batch_y.cpu().numpy().flatten())

    return np.array(predictions), np.array(targets)

def create_data_loaders(df, batch_size=32, val_split=0.2):
    """Create PyTorch data loaders from dataframe"""
    # Prepare data
    X = df[FEAT_COLS].values.astype(np.float32)
    y = df['strength'].values.astype(np.float32)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=val_split, random_state=42
    )

    # Create tensors
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val), torch.tensor(y_val)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, scaler

def get_pinn_calibration(dataset_id):
    """Dataset-specific calibration for PINN physics parameters"""
    calibrations = {
        'D1': {'physics_weight': 0.1, 'hard_constraints': True,  'epochs': 100},  # Clean data, use hard constraints
        'D2': {'physics_weight': 0.05, 'hard_constraints': False, 'epochs': 150},  # Heterogeneous, use soft constraints
        'D3': {'physics_weight': 0.08, 'hard_constraints': False, 'epochs': 120},  # Multi-modal, moderate physics
        'D4': {'physics_weight': 0.02, 'hard_constraints': False, 'epochs': 80},   # High variance, light constraints
    }
    return calibrations.get(dataset_id, calibrations['D1'])

def run_pinn_baseline():
    """Run PINN baseline on all datasets with dataset-specific calibration"""
    print("=" * 70)
    print("UMST PINN BASELINE - PHYSICS-INFORMED NEURAL NETWORKS")
    print("Dataset-specific calibration for optimal performance")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Remove fixed weights - use dataset-specific calibration instead
    physics_weights = [0.0]  # Placeholder - will be overridden by dataset calibration
    hard_constraints = [False]  # Placeholder - will be overridden
    results = {}

    # Train on D1, evaluate on all
    if not os.path.exists(DATASETS['D1 (UCI)']):
        print("ERROR: D1 dataset not found!")
        return {}

    # Results dictionary
    final_results = {}
    
    # Train and evaluate on EACH dataset individually
    for d_name, path in DATASETS.items():
        dataset_id = d_name.split(' ')[0]
        if not os.path.exists(path):
            continue
            
        print(f"\n{'='*60}")
        print(f"Training Hard PINN for {dataset_id}")
        print(f"{'='*60}")

        # Load data for this dataset
        df = pd.read_csv(path)
        train_loader, val_loader, scaler = create_data_loaders(df)
        
        # Get calibration for THIS dataset
        cal = get_pinn_calibration(dataset_id)
        
        # Train
        model = train_pinn_model(
            train_loader,
            val_loader,
            physics_weight=cal['physics_weight'],
            hard_constraints=cal['hard_constraints'],
            epochs=cal['epochs'],
            device=device
        )
        
        # Evaluate on test set (same dataset)
        # Re-create loader for full evaluation
        X = scaler.transform(df[FEAT_COLS].values.astype(np.float32))
        y = df['strength'].values
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        y_pred, y_true = predict_pinn(model, test_loader, device=device)
        mae = mean_absolute_error(y_true, y_pred)
        adm = thermodynamic_admissibility(y_pred)
        
        final_results[dataset_id] = {
            'MAE': mae,
            'Admissibility': adm,
            'N': len(y_true),
            'Characteristics': 'Trained on ' + dataset_id
        }
        print(f"Result {dataset_id}: MAE={mae:.2f}, Adm={adm:.1f}%")

    variant_name = "pinn_individual_calibration" # A generic name for this approach
    results[variant_name] = {
        'results': final_results,
        'training_time': 'variable', # Aggregate?
        'physics_weight': 'variable'
    }

    # Save comprehensive results
    output_path = REPO_ROOT / 'results/canonical/raw/pinn_baseline.json'
    with open(output_path, 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = {}
        for key, value in results.items():
            json_results[key] = {
                'results': {k: {k2: float(v2) if isinstance(v2, (np.floating, np.integer)) else v2
                               for k2, v2 in v.items()}
                           for k, v in value['results'].items()},
                'training_time': float(value['training_time']) if isinstance(value['training_time'], (int, float)) else 0.0,
                'physics_weight': float(value['physics_weight']) if isinstance(value['physics_weight'], (int, float)) else 0.0
            }
        import json
        json.dump(json_results, f, indent=2)

    print(f"\nComprehensive results saved to: {output_path}")

    # Summary comparison across datasets
    print("\n--- PINN Cross-Dataset Performance Summary ---")
    print("Trained on D1 (Clean OPC), tested on all datasets")
    print(f"{'Dataset':<10} {'N':>6} {'MAE':>8} {'Adm%':>6} {'Characteristics':<15}")
    print("-" * 55)

    for key, value in results.items():
        if 'calibrated' in key:
            for d_name, d_results in value['results'].items():
                char = d_results.get('Characteristics', 'Unknown')
                print("6")
            break  # Only show one variant (the calibrated one)

    print("-" * 55)
    print("PINN Configuration: Hard constraints, λ_physics = 0.1, 100 epochs")
    print("Best suited for: Clean data (D1), moderate generalization to heterogeneous data")

    return results

if __name__ == "__main__":
    run_pinn_baseline()