// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT
//
// UMST — Material Agnostic Operating System
// Minimal Core Profile for Reproducibility Package
//

pub mod formulas;
// pub mod geometry;
// pub mod ibe;
// pub mod ml;
// pub mod neural;
pub mod optimization;
// pub mod oracle;
// pub mod physics;
pub mod physics_kernel;
// pub mod profiler;
pub mod rl;
// pub mod robotics;
pub mod safety; // Guardrails needed
pub mod science;
// pub mod search;
pub mod tensors;
// pub mod tests;
#[cfg(test)]
pub mod tests_physics;
// pub mod trust;
// pub mod validation;

// Re-export core types
pub use science::rheology::CartridgeRegistry;
// pub use tensors::hyper_graph_tensor::HyperGraphTensor; // Disabled

pub use physics_kernel::PhysicsKernel;
pub use rl::{PPOAgent, RLAction, RLState, RewardFunction, RewardType};
// pub use tensors::SparseTensor; // Disabled
pub use tensors::MixTensor;

// pub use science::maturity::MaturityEngine;
