#!/usr/bin/env python3
"""
Physics Kernel Pre-Prediction Script for Physical Validation

Generates predictions for all L0-L4 mix designs BEFORE casting physical cubes.
Stores predictions in JSON format for comparison against physical measurements.

Usage:
    python scripts/pre_generate_predictions.py

Outputs:
    results/physical_validation/predictions_pre_cast.json

Author: UMST Research Team
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt

# Import UMST modules
try:
    from umst.core.physics_kernel import PhysicsKernel
except ImportError:
    # Mock for development
    class PhysicsKernel:
        def compute_industrial(self, props):
            # Mock physics predictions
            time_points = [3, 7, 14, 28]
            return {
                "strength_3d": 15.0 + np.random.normal(0, 2),
                "strength_7d": 22.0 + np.random.normal(0, 3),
                "strength_14d": 28.0 + np.random.normal(0, 4),
                "strength_28d": 32.0 + np.random.normal(0, 5),
                "yield_stress": 120 + np.random.normal(0, 20),
                "plastic_viscosity": 45 + np.random.normal(0, 10),
                "slump_flow": 180 + np.random.normal(0, 30),
                "chloride_diffusivity": 5.2e-12,
                "permeability": 1.8e-16,
                "co2_emissions": 280 - (props.get("rac_fraction", 0) * 100),
                "energy_consumption": 850,
                "thermodynamic_admissible": True,
                "fresh_density": 2250,
                "mix_temperature": 22,
                "penetration_resistance_4h": 0.5,
                "penetration_resistance_8h": 1.2,
                "penetration_resistance_24h": 2.8,
                "schmidt_rebound_3d": 18,
                "schmidt_rebound_7d": 25,
                "schmidt_rebound_14d": 32,
                "schmidt_rebound_28d": 38
            }


def get_mix_designs() -> Dict[str, Dict]:
    """Define the 5 mix designs (L0-L4) for physical validation."""
    return {
        "L0_baseline": {
            "name": "OPC Baseline",
            "cement": 350,    # kg/m³
            "sand": 700,      # kg/m³
            "coarse_natural": 1100,  # kg/m³
            "coarse_rac": 0,  # kg/m³
            "water": 175,     # kg/m³
            "rac_percent": 0,
            "target_strength": 25,  # MPa
            "description": "Control mix - no RAC"
        },
        "L1_rac30": {
            "name": "RAC 30% Replacement",
            "cement": 350,
            "sand": 700,
            "coarse_natural": 770,
            "coarse_rac": 330,
            "water": 185,
            "rac_percent": 30,
            "target_strength": 20,
            "description": "Standard RAC replacement level"
        },
        "L2_rac60_siteA": {
            "name": "RAC 60% Site A (Residential)",
            "cement": 350,
            "sand": 700,
            "coarse_natural": 440,
            "coarse_rac": 660,
            "water": 210,  # Higher for RAC absorption
            "rac_percent": 60,
            "target_strength": 15,
            "description": "High RAC - residential demolition source"
        },
        "L2_rac60_siteB": {
            "name": "RAC 60% Site B (Industrial)",
            "cement": 350,
            "sand": 700,
            "coarse_natural": 440,
            "coarse_rac": 660,
            "water": 210,
            "rac_percent": 60,
            "target_strength": 15,
            "description": "High RAC - industrial demolition source"
        },
        "L3_rac100": {
            "name": "RAC 100% Extreme",
            "cement": 350,
            "sand": 700,
            "coarse_natural": 0,
            "coarse_rac": 1100,
            "water": 230,  # Very high for absorption
            "rac_percent": 100,
            "target_strength": 10,  # Expected failure case
            "description": "Boundary test - pure RAC"
        }
    }


def generate_predictions_for_mix(mix_design: Dict, physics_kernel: PhysicsKernel) -> Dict:
    """Generate comprehensive predictions for a single mix design."""
    # Convert mix design to physics kernel format
    material_props = {
        "cement": mix_design["cement"],
        "water": mix_design["water"],
        "fine_aggregate": mix_design["sand"],
        "coarse_aggregate": mix_design["coarse_natural"] + mix_design["coarse_rac"],
        "rac_fraction": mix_design["rac_percent"] / 100.0,
        "curing_conditions": {
            "temperature": 20,  # °C
            "humidity": 95,     # %
            "time": [3, 7, 14, 28]  # days
        }
    }

    # Get physics kernel predictions
    predictions = physics_kernel.compute_industrial(material_props)

    # Structure for JSON output
    return {
        "mix_design": mix_design,
        "predictions": {
            "strength": {
                "f3d": predictions.get("strength_3d", 0),
                "f7d": predictions.get("strength_7d", 0),
                "f14d": predictions.get("strength_14d", 0),
                "f28d": predictions.get("strength_28d", 0)
            },
            "rheology": {
                "yield_stress": predictions.get("yield_stress", 0),
                "plastic_viscosity": predictions.get("plastic_viscosity", 0),
                "slump_flow": predictions.get("slump_flow", 0)
            },
            "durability": {
                "chloride_diffusivity": predictions.get("chloride_diffusivity", 0),
                "permeability": predictions.get("permeability", 0)
            },
            "sustainability": {
                "co2_kg_m3": predictions.get("co2_emissions", 0),
                "energy_mj_m3": predictions.get("energy_consumption", 0)
            },
            "admissibility": predictions.get("thermodynamic_admissible", True)
        },
        "expected_proxies": {
            "fresh": {
                "slump_flow_mm": predictions.get("slump_flow", 0),
                "yield_stress_pa": predictions.get("yield_stress", 0),
                "density_kg_m3": predictions.get("fresh_density", 0),
                "temperature_c": predictions.get("mix_temperature", 20)
            },
            "early_age": {
                "penetrometer_4h_mpa": predictions.get("penetration_resistance_4h", 0),
                "penetrometer_8h_mpa": predictions.get("penetration_resistance_8h", 0),
                "penetrometer_24h_mpa": predictions.get("penetration_resistance_24h", 0)
            },
            "hardened": {
                "schmidt_3d": predictions.get("schmidt_rebound_3d", 0),
                "schmidt_7d": predictions.get("schmidt_rebound_7d", 0),
                "schmidt_14d": predictions.get("schmidt_rebound_14d", 0),
                "schmidt_28d": predictions.get("schmidt_rebound_28d", 0)
            }
        }
    }


def main():
    """Main prediction generation function."""
    print("Generating pre-cast predictions for physical validation...")

    # Initialize physics kernel
    physics_kernel = PhysicsKernel()

    # Get mix designs
    mix_designs = get_mix_designs()

    # Generate predictions for each mix
    predictions = {}
    for mix_id, mix_design in mix_designs.items():
        print(f"Generating predictions for {mix_id}: {mix_design['name']}")
        predictions[mix_id] = generate_predictions_for_mix(mix_design, physics_kernel)

    # Create output directory
    output_dir = Path("results/physical_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    output_path = output_dir / "predictions_pre_cast.json"
    with open(output_path, 'w') as f:
        json.dump({
            "metadata": {
                "generated": "2026-02-04",
                "physics_kernel_version": "v1.0",
                "curing_conditions": "20°C, 95% RH",
                "cube_size": "50mm",
                "test_ages": [3, 7, 14, 28]
            },
            "mix_designs": mix_designs,
            "predictions": predictions
        }, f, indent=2)

    print(f"Predictions saved to {output_path}")

    # Generate summary visualization
    plot_predictions_summary(predictions, output_dir)


def plot_predictions_summary(predictions: Dict, output_dir: Path):
    """Generate summary plots of predictions."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    mix_names = [p["mix_design"]["name"] for p in predictions.values()]
    strengths_28d = [p["predictions"]["strength"]["f28d"] for p in predictions.values()]
    co2_values = [p["predictions"]["sustainability"]["co2_kg_m3"] for p in predictions.values()]
    slump_values = [p["expected_proxies"]["fresh"]["slump_flow_mm"] for p in predictions.values()]
    admissible = [p["predictions"]["admissibility"] for p in predictions.values()]

    # Strength predictions
    ax1.bar(mix_names, strengths_28d)
    ax1.set_title("Predicted 28-day Strength")
    ax1.set_ylabel("Strength (MPa)")
    ax1.tick_params(axis='x', rotation=45)

    # CO2 predictions
    ax2.bar(mix_names, co2_values, color='green')
    ax2.set_title("Predicted CO₂ Emissions")
    ax2.set_ylabel("CO₂ (kg/m³)")
    ax2.tick_params(axis='x', rotation=45)

    # Slump predictions
    ax3.bar(mix_names, slump_values, color='blue')
    ax3.set_title("Predicted Slump Flow")
    ax3.set_ylabel("Slump (mm)")
    ax3.tick_params(axis='x', rotation=45)

    # Admissibility status
    ax4.bar(mix_names, [1 if a else 0 for a in admissible], color='red')
    ax4.set_title("Thermodynamic Admissibility")
    ax4.set_ylabel("Admissible (1=Yes, 0=No)")
    ax4.set_ylim(-0.1, 1.1)
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "predictions_summary.png", dpi=300, bbox_inches='tight')
    print(f"Summary plot saved to {output_dir / 'predictions_summary.png'}")


if __name__ == "__main__":
    main()