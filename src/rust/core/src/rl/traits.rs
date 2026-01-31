// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT
//
// UMST — Material Agnostic Operating System
// IRewardProvider Trait: Material-Agnostic Reward Interface
//
// This file is part of UMST.
// For licensing terms, see the LICENSE file in the project root.

//! Reward Provider Traits for Cartridge Modularization
//!
//! The IRewardProvider trait abstracts reward computation away from PPOAgent,
//! allowing different material science cartridges to provide their own
//! physics-based reward functions.
//!
//! Architecture:
//! ```text
//! PPOAgent (Core) --uses--> IRewardProvider (Trait)
//!                                    ^
//!                                    |
//! //!         +--------------------------+--------------------------+
//!         |                          |                          |
//! ConcreteRewardProvider    PolymerRewardProvider    RegolithRewardProvider
//! ```

use super::reward::{RewardComponents, RewardConfig, RewardFunction, RewardType};
use super::state::RLAction;
use crate::tensors::MixTensor;

/// IRewardProvider: Material-agnostic reward computation interface
///
/// Implement this trait for each science cartridge to provide
/// domain-specific reward signals to the PPO agent.
pub trait IRewardProvider: Send + Sync {
    /// Compute reward components from mix state and action
    ///
    /// This method runs the cartridge's physics engines to produce
    /// the metrics needed for reward calculation.
    fn compute_components(&self, mix: &MixTensor, action: &RLAction) -> RewardComponents;

    /// Get supported reward types for this cartridge
    ///
    /// Different materials may support different optimization objectives.
    /// e.g., Concrete supports Printability, Regolith supports RadiationShielding
    fn get_supported_reward_types(&self) -> Vec<RewardType>;

    /// Get the cartridge identifier
    fn cartridge_id(&self) -> &'static str;

    /// Get the default reward configuration
    fn default_config(&self) -> RewardConfig;

    /// Create a reward function with the given type
    fn create_reward_function(&self, reward_type: RewardType) -> RewardFunction {
        let mut config = self.default_config();
        config.reward_type = reward_type;
        RewardFunction::new(config)
    }
}

/// IDataProvider: Material-agnostic data interface
///
/// Provides access to cartridge-specific data: materials, training datasets,
/// ONNX models, and calibration parameters.
pub trait IDataProvider: Send + Sync {
    /// Get default materials for this cartridge
    fn get_default_materials(&self) -> Vec<MaterialData>;

    /// Get training data path (relative to cartridge root)
    fn get_training_data_path(&self) -> Option<&'static str>;

    /// Get ONNX model paths
    fn get_onnx_models(&self) -> Vec<(&'static str, &'static str)>; // (name, path)

    /// Validate a material against cartridge constraints
    fn validate_material(&self, material: &MaterialData) -> bool;
}

/// Simple material data structure (serializable)
use crate::science::materials::{
    EcologyProfile, EconomyProfile, ProvenanceProfile, RheologyProfile,
};

/// Simple material data structure (serializable)
/// [God-Grade] Parity with TypeScript Material2Schema
#[derive(Clone, Debug)]
pub struct MaterialData {
    pub id: String,
    pub name: String,
    pub material_type: String,
    pub density: f32, // Density persists as a core physical property

    // [New] Nested Profiles
    pub rheology: RheologyProfile,
    pub ecology: EcologyProfile,
    pub economy: EconomyProfile,
    pub provenance: ProvenanceProfile,
}

/// CartridgeInfo: Metadata about a science cartridge
#[derive(Clone, Debug)]
pub struct CartridgeInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub version: &'static str,
    pub supported_materials: Vec<&'static str>,
}

/// IScienceCartridge: Combined interface for full cartridge
///
/// A complete science cartridge implements both IRewardProvider and IDataProvider
pub trait IScienceCartridge: IRewardProvider + IDataProvider {
    /// Get cartridge metadata
    fn info(&self) -> CartridgeInfo;

    /// Initialize the cartridge (load WASM, models, etc.)
    fn init(&mut self) -> Result<(), String>;

    /// Cleanup resources
    fn dispose(&mut self);

    /// Check if cartridge is ready
    fn is_ready(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test that traits are object-safe
    fn _assert_object_safe(_: &dyn IRewardProvider) {}
    fn _assert_data_object_safe(_: &dyn IDataProvider) {}
}
