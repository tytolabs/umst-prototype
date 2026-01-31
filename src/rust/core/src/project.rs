// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT
use crate::oracle::Oracle;
use crate::tensors::{MixTensor, SparseTensor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TensorVersion {
    pub hash: String,
    pub timestamp: u64,
    // Vector Clock: UserID -> Counter
    pub vector: HashMap<String, u64>,
}

// Placeholder for ToolpathTensor (Vector Path)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolpathTensor {
    pub points: Vec<f32>, // Flat [x,y,z, x,y,z...]
    pub speed: f32,
}

#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct ProjectTensor {
    id: String,
    mix: Option<MixTensor>,
    geometry: Option<SparseTensor>, // Voxel/G-Code sparse representation
    toolpath: Option<ToolpathTensor>,
    version: TensorVersion,

    // Last known good state hash (Merkle Root)
    merkle_root: String,
}

#[wasm_bindgen]
impl ProjectTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(id: String) -> ProjectTensor {
        let mut vector = HashMap::new();
        vector.insert("system".to_string(), 0);

        ProjectTensor {
            id,
            mix: None,
            geometry: None,
            toolpath: None,
            version: TensorVersion {
                hash: "genesis".to_string(),
                timestamp: instant::now() as u64,
                vector,
            },
            merkle_root: "genesis".to_string(),
        }
    }

    /// Commit a Mix change.
    /// Runs Oracle validation FIRST.
    /// Returns:
    /// - Ok(true) if committed
    /// - Ok(false) if rejected by Oracle
    /// - Err(String) if critical failure
    pub fn commit_mix(&mut self, _mix_json: String, _author: String) -> Result<JsValue, JsValue> {
        Err(JsValue::from_str("Use commit_mix_tensor instead"))
    }

    pub fn commit_mix_tensor(
        &mut self,
        mix: &MixTensor,
        author: String,
    ) -> Result<JsValue, JsValue> {
        // 1. Oracle Validation (The Judge)
        let violations = Oracle::validate_mix(mix);
        if !violations.is_empty() {
            return Ok(serde_wasm_bindgen::to_value(&violations)?);
        }

        // 2. Update State
        self.mix = Some(mix.clone());

        // 3. Update Vector Clock
        let counter = self.version.vector.entry(author.clone()).or_insert(0);
        *counter += 1;

        // 4. Calculate Merkle Hash (SHA256 of State)
        self.merkle_root = self.calculate_hash();

        self.version.timestamp = instant::now() as u64;
        self.version.hash = self.merkle_root.clone();

        // Return null to indicate success (no violations)
        Ok(JsValue::NULL)
    }

    pub fn commit_geometry(
        &mut self,
        geometry: &SparseTensor,
        author: String,
    ) -> Result<JsValue, JsValue> {
        self.geometry = Some(geometry.clone());
        self.update_version(author);
        Ok(JsValue::NULL)
    }

    fn update_version(&mut self, author: String) {
        let counter = self.version.vector.entry(author).or_insert(0);
        *counter += 1;
        self.merkle_root = self.calculate_hash();
        self.version.timestamp = instant::now() as u64;
    }

    pub fn get_merkle_root(&self) -> String {
        self.merkle_root.clone()
    }

    pub fn get_mix(&self) -> JsValue {
        match &self.mix {
            Some(m) => serde_wasm_bindgen::to_value(m).unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }

    fn calculate_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.id.as_bytes());

        if let Some(m) = &self.mix {
            for val in m.data() {
                hasher.update(val.to_le_bytes());
            }
        }

        if let Some(g) = &self.geometry {
            for idx in g.indices() {
                hasher.update(idx.to_le_bytes());
            }
            for val in g.values() {
                hasher.update(val.to_le_bytes());
            }
        }

        if let Some(t) = &self.toolpath {
            for val in &t.points {
                hasher.update(val.to_le_bytes());
            }
        }

        // Hash vector clock
        let mut sorted_keys: Vec<&String> = self.version.vector.keys().collect();
        sorted_keys.sort();
        for k in sorted_keys {
            hasher.update(k.as_bytes());
            hasher.update(self.version.vector[k].to_le_bytes());
        }

        let result = hasher.finalize();
        hex::encode(result)
    }
}
