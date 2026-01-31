// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto

//! Monte Carlo Optimizer for DUMSTO
//!
//! Provides stochastic optimization for concrete mix design
//! using the DUMSTO physics kernel constraints.

use crate::physics_kernel::{IndustrialResult, PhysicsKernel};
use crate::tensors::MixTensor;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct MonteCarloOptimizer;

#[wasm_bindgen]
impl MonteCarloOptimizer {
    /// Run Monte Carlo Optimization in Rust (WASM)
    ///
    /// Args:
    /// - base_components: JSON string of MixComponentInput[]
    /// - materials: JSON string of MaterialInput[]
    /// - n: Number of iterations
    ///
    /// Returns:
    /// - JSON string of Array<IndustrialResult & { id: string, components: MixComponentInput[] }>
    pub fn optimize(base_components: &str, materials: &str, n: usize) -> String {
        // 1. Hydrate Base Tensor
        let base_tensor = match MixTensor::from_json(base_components, materials) {
            Ok(t) => t,
            Err(_) => return "[]".to_string(), // Fail gracefully
        };

        let mut results = Vec::with_capacity(n);

        for i in 0..n {
            // 2. Clone & Perturb
            let mut trial = base_tensor.clone();
            let data = trial.buffer_mut();

            // Perturb Mass (Stride 8, Offset 0)
            let variance = 0.3; // +/- 30%

            // Use rand for native compatibility
            for j in (0..data.len()).step_by(8) {
                let r = rand::random::<f32>(); // 0.0 to 1.0
                let factor = 1.0 + (r - 0.5) * variance; // 0.85 to 1.15

                let current_mass = data[j];
                // Apply
                data[j] = (current_mass * factor).max(0.0);
            }

            // 3. Compute Physics (Shared Logic)
            let result =
                PhysicsKernel::compute(&trial, None, &crate::physics_kernel::PhysicsConfig::default());

            // 4. Wrap for Output
            // We need to reconstruct the components structure for the frontend
            // This is slightly expensive but necessary for the UI to know what the mix IS.
            // Alternatively, we return the tensor data and let TS map it back?
            // TS expects `components: [{materialId, mass}, ...]`.
            // The MixTensor doesn't store materialIDs internally in the vector (it's lossy).
            // BUT: The structure (order) corresponds to the input `base_components` order
            // IF we assume MixTensor inserts in order.
            // MixTensor::from_json iterates components and adds them. So order is preserved.

            // We can re-map masses back to a component list if we parse the input once.
            // For speed, let's just return the `IndustrialResult` and let TS handle the mix mapping?
            // No, TS needs to know the *ingredients* of the result.

            // Let's create a lightweight result struct
            let optimization_result = OptimizationResult {
                id: format!("mc-rust-{}", i),
                physics: result,
                tensor_data: trial.data().clone(), // Return the raw data, TS can map it back using reference
            };
            results.push(optimization_result);
        }

        serde_json::to_string(&results).unwrap_or("[]".to_string())
    }
}

#[derive(Serialize)]
struct OptimizationResult {
    id: String,
    #[serde(flatten)]
    physics: IndustrialResult,
    tensor_data: Vec<f32>, // [mass, sg, type, ... ] stride 8
}
