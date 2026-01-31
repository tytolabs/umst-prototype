// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT
//
// UMST — Material Agnostic Operating System
// SparseTensor: Core Unified Tensor Data Structure

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use std::collections::HashMap;

#[wasm_bindgen]
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SparseTensor {
    indices: Vec<u32>, // Flat indices (row * cols + col)
    values: Vec<f32>,
    shape: Vec<u32>,
}

#[wasm_bindgen]
impl SparseTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(shape: Vec<u32>) -> SparseTensor {
        SparseTensor {
            indices: Vec::new(),
            values: Vec::new(),
            shape,
        }
    }

    pub fn set(&mut self, index: u32, value: f32) {
        self.indices.push(index);
        self.values.push(value);
    }

    /**
     * [Phase 16] Sparse Matrix Multiplication (A x B)
     * Implementation: Robust Hash-Map based accumulation.
     * C[i, j] = sum(A[i, k] * B[k, j])
     */
    pub fn matmul(&self, other: &SparseTensor) -> SparseTensor {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            // Error handling: Return empty 0x0 tensor indicating failure
            return SparseTensor::new(vec![0, 0]);
        }

        let rows_a = self.shape[0];
        let cols_a = self.shape[1];
        let rows_b = other.shape[0];
        let cols_b = other.shape[1];

        if cols_a != rows_b {
             return SparseTensor::new(vec![0, 0]);
        }

        // 1. Build Row-Lookup for A (Iterate non-zeros)
        // Map: Row -> Vec<(Col, Value)>
        let mut a_rows: HashMap<u32, Vec<(u32, f32)>> = HashMap::new();
        for (i, &flat_idx) in self.indices.iter().enumerate() {
            let r = flat_idx / cols_a;
            let c = flat_idx % cols_a;
            a_rows.entry(r).or_default().push((c, self.values[i]));
        }

        // 2. Build Row-Lookup for B (Used to find B[k, :])
        let mut b_rows: HashMap<u32, Vec<(u32, f32)>> = HashMap::new();
        for (i, &flat_idx) in other.indices.iter().enumerate() {
            let r = flat_idx / cols_b;
            let c = flat_idx % cols_b;
            b_rows.entry(r).or_default().push((c, other.values[i]));
        }

        // 3. Compute Product
        // For each row r in A:
        //   For each (k, val_a) in A[r, :]:
        //     For each (c, val_b) in B[k, :]:
        //       C[r, c] += val_a * val_b
        let mut c_map: HashMap<u32, f32> = HashMap::new();

        for (r, a_cols) in &a_rows {
            for &(k, val_a) in a_cols {
                if let Some(b_cols) = b_rows.get(&k) {
                    for &(c, val_b) in b_cols {
                        let flat_idx_c = r * cols_b + c;
                        *c_map.entry(flat_idx_c).or_insert(0.0) += val_a * val_b;
                    }
                }
            }
        }

        // 4. Construct Result
        let mut result = SparseTensor::new(vec![rows_a, cols_b]);
        
        // Sort keys for deterministic output (optional but good for testing)
        let mut sorted_indices: Vec<u32> = c_map.keys().cloned().collect();
        sorted_indices.sort();

        for idx in sorted_indices {
            let val = c_map[&idx];
            if val.abs() > 1e-9 { // Filter near-zero
                result.indices.push(idx);
                result.values.push(val);
            }
        }

        result
    }

    // --- Inspection ---

    pub fn density(&self) -> f32 {
        let total = self.shape.iter().product::<u32>() as f32;
        if total == 0.0 { return 0.0; }
        self.indices.len() as f32 / total
    }

    // Getters for JS
    #[wasm_bindgen(getter)]
    pub fn indices(&self) -> Vec<u32> {
        self.indices.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f32> {
        self.values.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<u32> {
        self.shape.clone()
    }
}
