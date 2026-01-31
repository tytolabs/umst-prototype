// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT
//
// UMST — Material Agnostic Operating System
// ConcreteRewardProvider: Recycled Aggregate Concrete Implementation
//
// This file is part of UMST.
// For licensing terms, see the LICENSE file in the project root.

//! Concrete Science Cartridge - Reward Provider Implementation
//!
//! Implements IRewardProvider for Recycled Aggregate Concrete (RAC).
//! Wraps existing science engines (rheology, strength, sustainability, etc.)
//! into the cartridge interface.

use super::reward::{RewardComponents, RewardConfig, RewardType};
use super::state::RLAction;
use super::traits::{
    CartridgeInfo, IDataProvider, IRewardProvider, IScienceCartridge, MaterialData,
};
use crate::science::{
    colloidal::ColloidalEngine, cost::CostEngine, fracture::FractureEngine, itz::ITZEngine,
    rheology::RheologyEngine, strength::StrengthEngine, sustainability::SustainabilityEngine,
    thermo::ThermoEngine, transport::TransportEngine,
};
use crate::tensors::MixTensor;

/// ConcreteRewardProvider: Recycled Aggregate Concrete reward computation
pub struct ConcreteRewardProvider {
    is_ready: bool,
}

impl ConcreteRewardProvider {
    pub fn new() -> Self {
        ConcreteRewardProvider { is_ready: false }
    }
}

impl Default for ConcreteRewardProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl IRewardProvider for ConcreteRewardProvider {
    fn compute_components(&self, mix: &MixTensor, action: &RLAction) -> RewardComponents {
        // 1. Apply action to create simulated mix
        let mut sim_mix = mix.clone();
        sim_mix.apply_action(
            action.delta_wc as f32,
            action.delta_scms as f32,
            action.delta_sp as f32,
        );

        // 2. Run Science Kernels (WASM) - "God Grade" Simulation
        let packing = 0.74; // Assume optimal packing
        let rheology = RheologyEngine::compute(&sim_mix, packing);

        let w_c = sim_mix.water_cement_ratio();
        let strength = StrengthEngine::compute_powers(w_c, 0.7, 0.02, 240.0);

        let sustainability = SustainabilityEngine::compute_impact(&sim_mix);

        let fracture = FractureEngine::compute_lefm(strength.compressive_strength * 0.1, 2.0, 1.0);

        let cost = CostEngine::compute(&sim_mix);

        // [New Science Engines]
        let itz = ITZEngine::compute_properties(10.0, w_c);
        let colloidal = ColloidalEngine::compute_potential(2.0, 25.0, 0.5);
        let thermo = ThermoEngine::compute_heat_rate(20.0, 0.5, 40_000.0);
        let transport = TransportEngine::compute_sorptivity(50.0, 0.072, 0.001);

        // 3. Assemble Reward Components
        RewardComponents {
            strength_fc: strength.compressive_strength as f64,
            cost: cost.cost_per_m3 as f64,
            co2: sustainability.gwp_total as f64,
            fracture_kic: fracture.k_ic as f64,
            diffusivity: transport.diffusion_coeff as f64,
            damage: 0.0,
            bond: 2.5 + (action.delta_sp * 0.5),
            yield_stress: rheology.yield_stress as f64,
            viscosity: rheology.viscosity as f64,
            slump_flow: rheology.slump_flow as f64,

            // [NEW] 35-Dim Upgrade Metrics
            itz_thickness: itz.thickness_microns as f64,
            itz_porosity: itz.porosity_itz as f64,
            colloidal_potential: colloidal.potential_energy as f64,
            heat_rate: thermo.heat_rate as f64,
            temp_rise: thermo.adiabatic_temp_rise as f64,
            permeability: 1e-12, // Placeholder
            suction: 50.0,       // Placeholder
        }
    }

    fn get_supported_reward_types(&self) -> Vec<RewardType> {
        vec![
            RewardType::Balanced,
            RewardType::StrengthFirst,
            RewardType::Sustainability,
            RewardType::CostOptimal,
            RewardType::DurabilityFirst,
            RewardType::Printability, // 3DCP support
        ]
    }

    fn cartridge_id(&self) -> &'static str {
        "concrete"
    }

    fn default_config(&self) -> RewardConfig {
        RewardConfig::balanced()
    }
}

impl IDataProvider for ConcreteRewardProvider {
    fn get_default_materials(&self) -> Vec<MaterialData> {
        vec![
            MaterialData {
                id: "opc_42_5n".to_string(),
                name: "OPC 42.5N".to_string(),
                material_type: "cement".to_string(),
                density: 3150.0,
                rheology: crate::science::materials::RheologyProfile {
                    yield_stress: 0.0,
                    viscosity: 0.0,
                    thixotropy: 0.0,
                    slump_flow: None,
                },
                ecology: crate::science::materials::EcologyProfile {
                    embodied_carbon: 0.9,
                    certification: "EN 197-1".to_string(),
                },
                economy: crate::science::materials::EconomyProfile {
                    cost_per_kg: 0.1,
                    supplier: "Generic".to_string(),
                    lead_time_days: Some(2),
                },
                provenance: crate::science::materials::ProvenanceProfile {
                    origin: "Local".to_string(),
                    batch_id: "Batch-001".to_string(),
                    production_date: None,
                },
            },
            MaterialData {
                id: "potable_water".to_string(),
                name: "Potable Water".to_string(),
                material_type: "water".to_string(),
                density: 1000.0,
                rheology: crate::science::materials::RheologyProfile {
                    yield_stress: 0.0,
                    viscosity: 0.001,
                    thixotropy: 0.0,
                    slump_flow: None,
                },
                ecology: crate::science::materials::EcologyProfile {
                    embodied_carbon: 0.0,
                    certification: "WHO".to_string(),
                },
                economy: crate::science::materials::EconomyProfile {
                    cost_per_kg: 0.001,
                    supplier: "Municipal".to_string(),
                    lead_time_days: Some(0),
                },
                provenance: crate::science::materials::ProvenanceProfile {
                    origin: "Local".to_string(),
                    batch_id: "H2O-001".to_string(),
                    production_date: None,
                },
            },
            MaterialData {
                id: "river_sand".to_string(),
                name: "River Sand".to_string(),
                material_type: "aggregate".to_string(),
                density: 2650.0,
                rheology: crate::science::materials::RheologyProfile {
                    yield_stress: 0.0,
                    viscosity: 0.0,
                    thixotropy: 0.0,
                    slump_flow: None,
                },
                ecology: crate::science::materials::EcologyProfile {
                    embodied_carbon: 0.005,
                    certification: "None".to_string(),
                },
                economy: crate::science::materials::EconomyProfile {
                    cost_per_kg: 0.02,
                    supplier: "Quarry A".to_string(),
                    lead_time_days: Some(1),
                },
                provenance: crate::science::materials::ProvenanceProfile {
                    origin: "River Bed".to_string(),
                    batch_id: "SAND-001".to_string(),
                    production_date: None,
                },
            },
            MaterialData {
                id: "recycled_coarse_10mm".to_string(),
                name: "Recycled Coarse Aggregate (10mm)".to_string(),
                material_type: "aggregate".to_string(),
                density: 2400.0,
                rheology: crate::science::materials::RheologyProfile {
                    yield_stress: 0.0,
                    viscosity: 0.0,
                    thixotropy: 0.0,
                    slump_flow: None,
                },
                ecology: crate::science::materials::EcologyProfile {
                    embodied_carbon: 0.002,
                    certification: "GreenCert".to_string(),
                },
                economy: crate::science::materials::EconomyProfile {
                    cost_per_kg: 0.01,
                    supplier: "Demo Site B".to_string(),
                    lead_time_days: Some(3),
                },
                provenance: crate::science::materials::ProvenanceProfile {
                    origin: "Building 4".to_string(),
                    batch_id: "RCA-001".to_string(),
                    production_date: None,
                },
            },
        ]
    }

    fn get_training_data_path(&self) -> Option<&'static str> {
        Some("cartridges/concrete/data/training/")
    }

    fn get_onnx_models(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "rheology_predictor",
                "cartridges/concrete/models/rheology.onnx",
            ),
            (
                "strength_predictor",
                "cartridges/concrete/models/strength.onnx",
            ),
        ]
    }

    fn validate_material(&self, material: &MaterialData) -> bool {
        let valid_types = ["cement", "water", "aggregate", "scm", "admixture", "fiber"];
        valid_types.contains(&material.material_type.as_str())
    }
}

impl IScienceCartridge for ConcreteRewardProvider {
    fn info(&self) -> CartridgeInfo {
        CartridgeInfo {
            id: "concrete",
            name: "Recycled Aggregate Concrete",
            version: "1.0.0",
            supported_materials: vec!["cement", "water", "aggregate", "scm", "admixture", "fiber"],
        }
    }

    fn init(&mut self) -> Result<(), String> {
        // Initialize WASM engines, load models, etc.
        self.is_ready = true;
        Ok(())
    }

    fn dispose(&mut self) {
        self.is_ready = false;
    }

    fn is_ready(&self) -> bool {
        self.is_ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concrete_provider_info() {
        let provider = ConcreteRewardProvider::new();
        let info = provider.info();
        assert_eq!(info.id, "concrete");
        assert_eq!(info.version, "1.0.0");
    }

    #[test]
    fn test_supported_reward_types() {
        let provider = ConcreteRewardProvider::new();
        let types = provider.get_supported_reward_types();
        assert!(types.contains(&RewardType::Printability));
        assert!(types.contains(&RewardType::Sustainability));
    }

    #[test]
    fn test_default_materials() {
        let provider = ConcreteRewardProvider::new();
        let materials = provider.get_default_materials();
        assert!(!materials.is_empty());
        assert!(materials.iter().any(|m| m.id == "opc_42_5n"));
    }
}
