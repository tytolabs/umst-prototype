// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialType {
    Cement = 0,
    Aggregate = 1,
    Water = 2,
    Admixture = 3,
    Air = 4,
    SCM = 5,
}

impl MaterialType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "cement" => MaterialType::Cement,
            "scm" | "flyash" | "slag" | "silica_fume" => MaterialType::SCM,
            "aggregate" | "sand" | "gravel" => MaterialType::Aggregate,
            "water" => MaterialType::Water,
            "admixture" | "superplasticizer" | "pigment" => MaterialType::Admixture,
            _ => MaterialType::Air,
        }
    }
}

/// Incoming material from TypeScript (JSON)
/// This struct is designed to be FLEXIBLE and handle the complex TS Material schema
/// using `#[serde(default)]` for all optional fields.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaterialInput {
    pub id: String,
    #[serde(rename = "type", default)]
    pub material_type: String,
    #[serde(default = "default_density")]
    pub density: f32,
    #[serde(default)]
    pub ecology: Option<EcologyInput>,
    #[serde(default)]
    pub economy: Option<EconomyInput>,
    #[serde(default)]
    pub properties: Option<PhysicsPropertiesInput>,
    // Ignore all other fields from TS schema
    #[serde(flatten)]
    #[allow(dead_code)]
    _extra: std::collections::HashMap<String, serde_json::Value>,
}

fn default_density() -> f32 {
    2400.0
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct EcologyInput {
    #[serde(rename = "embodiedCarbon", default)]
    pub embodied_carbon: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct EconomyInput {
    #[serde(rename = "costPerKg", default)]
    pub cost_per_kg: f32,
}

/// Physics properties from TypeScript Material.properties
/// These drive science engine calculations for accurate physics
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PhysicsPropertiesInput {
    /// Blaine fineness (m²/kg) - affects hydration rate and rheology
    #[serde(default)]
    pub blaine: f32,
    /// CO2 per kg material - for sustainability calculation
    #[serde(rename = "co2_kg", default)]
    pub co2_kg: f32,
    /// Reactivity coefficient (0-1+) - hydraulic activity index
    #[serde(default)]
    pub reactivity: f32,
    /// Fineness modulus - aggregate grading (0-8 scale)
    #[serde(default)]
    pub fm: f32,
    /// Water absorption (%) - affects effective W/C
    #[serde(default)]
    pub absorption: f32,
    /// Dosage (% by cement weight) - for admixtures
    #[serde(default)]
    pub dosage: f32,
    /// Shape factor (0-1, 1=spherical) - affects packing
    #[serde(default)]
    pub shape: f32,
    /// Moisture content (%) - current moisture in aggregates
    #[serde(default)]
    pub moisture: f32,
}

/// Incoming mix component from TypeScript (JSON)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MixComponentInput {
    #[serde(rename = "materialId")]
    pub material_id: String,
    #[serde(default)]
    pub mass: f32,
}

#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MixTensor {
    // Stores material data: [mass, sg, type, co2, cost, blaine, fm, shape]
    // Stride = 8
    data: Vec<f32>,
}

#[wasm_bindgen]
impl MixTensor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> MixTensor {
        MixTensor { data: Vec::new() }
    }

    /// Hydrate tensor directly from JSON (TypeScript sends raw JSON strings)
    /// This moves ALL marshalling logic into Rust.
    #[wasm_bindgen]
    pub fn from_json(components_json: &str, materials_json: &str) -> Result<MixTensor, JsValue> {
        let components: Vec<MixComponentInput> = serde_json::from_str(components_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse components: {}", e)))?;

        let materials: Vec<MaterialInput> = serde_json::from_str(materials_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse materials: {}", e)))?;

        let mut tensor = MixTensor::new();

        for comp in components {
            // Find material by ID
            if let Some(mat) = materials.iter().find(|m| m.id == comp.material_id) {
                let sg = mat.density / 1000.0; // Convert density to specific gravity
                let type_id = MaterialType::from_str(&mat.material_type) as u8;
                let co2 = mat
                    .ecology
                    .as_ref()
                    .map(|e| e.embodied_carbon)
                    .unwrap_or(0.0);
                let cost = mat.economy.as_ref().map(|e| e.cost_per_kg).unwrap_or(0.0);

                // Extract Physics Properties
                let (blaine, fm, shape) = if let Some(props) = &mat.properties {
                    (props.blaine, props.fm, props.shape)
                } else {
                    (0.0, 0.0, 0.5) // Defaults
                };

                tensor.add_material(comp.mass, sg, type_id, co2, cost, blaine, fm, shape);
            }
        }

        Ok(tensor)
    }

    pub fn add_material(
        &mut self,
        mass: f32,
        sg: f32,
        type_id: u8,
        co2: f32,
        cost: f32,
        blaine: f32,
        fm: f32,
        shape: f32,
    ) {
        self.data.push(mass);
        self.data.push(sg);
        self.data.push(type_id as f32);
        self.data.push(co2);
        self.data.push(cost);
        self.data.push(blaine);
        self.data.push(fm);
        self.data.push(shape);
    }

    pub fn total_mass(&self) -> f32 {
        let mut total = 0.0;
        for i in (0..self.data.len()).step_by(8) {
            total += self.data[i];
        }
        total
    }

    /// Legacy w/c ratio with hardcoded k_scm=0.5 (kept for backward compatibility)
    pub fn water_cement_ratio(&self) -> f32 {
        self.water_cement_ratio_calibrated(0.5)
    }

    /// Calibrated w/c ratio using dataset-specific SCM efficiency factor.
    ///
    /// k_scm: blended SCM efficiency factor from calibration.
    /// - D1 (UCI Concrete): ~1.0 (L-BFGS-B fitted k_slag=1.18, k_fly_ash=1.15)
    /// - D2-D4: ~0.2 (fitted)
    /// - Generic fallback: 0.5
    pub fn water_cement_ratio_calibrated(&self, k_scm: f32) -> f32 {
        let mut water = 0.0;
        let mut cement = 0.0;

        for i in (0..self.data.len()).step_by(8) {
            let mass = self.data[i];
            let type_id = self.data[i + 2] as u8;

            if type_id == MaterialType::Water as u8 {
                water += mass;
            } else if type_id == MaterialType::Cement as u8 {
                cement += mass;
            } else if type_id == MaterialType::SCM as u8 {
                cement += mass * k_scm;
            }
        }

        if cement == 0.0 {
            f32::INFINITY
        } else {
            water / cement
        }
    }

    pub fn total_co2(&self) -> f32 {
        let mut total = 0.0;
        for i in (0..self.data.len()).step_by(8) {
            let mass = self.data[i];
            let co2_factor = self.data[i + 3];
            total += mass * co2_factor;
        }
        total
    }

    pub fn scm_ratio(&self) -> f32 {
        let mut total_cement = 0.0;
        let mut total_scm = 0.0;

        for i in (0..self.data.len()).step_by(8) {
            let mass = self.data[i];
            let type_id = self.data[i + 2] as u8;

            if type_id == MaterialType::Cement as u8 {
                total_cement += mass;
            } else if type_id == MaterialType::SCM as u8 {
                total_scm += mass;
            }
        }

        let total_binder = total_cement + total_scm;
        if total_binder > 0.0 {
            total_scm / total_binder
        } else {
            0.0
        }
    }
}

impl MixTensor {
    // Accessor for Rust internal use (not wasm-bindgen)
    pub fn data(&self) -> &Vec<f32> {
        &self.data
    }

    pub fn buffer_mut(&mut self) -> &mut Vec<f32> {
        &mut self.data
    }

    /// [ACTION] Apply RL Action Deltas directly to the mix tensor
    /// This allows the PPO agent to "mutate" reality in the optimization loop.
    pub fn apply_action(&mut self, delta_wc: f32, delta_scms: f32, delta_sp: f32) {
        // 1. Identification Pass
        let mut cement_indices = Vec::new();
        let mut water_indices = Vec::new();
        let mut scm_indices: Vec<usize> = Vec::new();
        let mut sp_indices = Vec::new(); // Superplasticizer

        for i in (0..self.data.len()).step_by(8) {
            let type_id = self.data[i + 2] as u8;
            if type_id == MaterialType::Cement as u8 {
                cement_indices.push(i);
            } else if type_id == MaterialType::Water as u8 {
                water_indices.push(i);
            } else if type_id == MaterialType::Admixture as u8 {
                sp_indices.push(i);
            } else if type_id == MaterialType::SCM as u8 {
                scm_indices.push(i);
            }
        }

        // Safety: If no cement or water, we can't adjust w/c effectively
        if cement_indices.is_empty() {
            return;
        }

        // 2. Apply W/C Delta (Adjust Water Mass)
        let total_cement: f32 = cement_indices.iter().map(|&i| self.data[i]).sum();
        let total_scm: f32 = scm_indices.iter().map(|&i| self.data[i]).sum();
        let total_binder = total_cement + total_scm;

        if !water_indices.is_empty() && total_binder > 0.0 {
            // Adjust water based on TOTAL binder (Cement + SCM) for W/B ratio
            let water_change = delta_wc * total_binder;
            let num_water = water_indices.len() as f32;
            for &i in &water_indices {
                let new_mass = (self.data[i] + water_change / num_water).max(0.0);
                self.data[i] = new_mass;
            }
        }

        // 3. Apply SCM Delta (Replace Cement with SCM)
        // delta_scms is percentage shift (e.g. +0.10 means move 10% of total binder from cement to SCM)
        if delta_scms.abs() > 0.001 {
            if !scm_indices.is_empty() {
                // We have SCMs, so we perform actual replacement
                // Target SCM Ratio += delta
                // But simpler: just move mass.
                // Mass to move = delta_scms * total_binder

                // Ensure we don't try to move more mass than available cement
                let mass_to_move = (delta_scms * total_binder).min(total_cement);

                // Reduce Cement
                let num_cement = cement_indices.len() as f32;
                if num_cement > 0.0 {
                    for &i in &cement_indices {
                        self.data[i] = (self.data[i] - (mass_to_move / num_cement)).max(1.0);
                        // Safety floor
                    }
                }

                // Increase SCM
                let num_scm = scm_indices.len() as f32;
                if num_scm > 0.0 {
                    for &i in &scm_indices {
                        self.data[i] = (self.data[i] + (mass_to_move / num_scm)).max(0.0);
                    }
                }
            } else {
                // No SCM existing. Just reduce cement to simulate "leaner" mix?
                // Or ignore? Let's reduce cement to penalize "Low Strength" if goal is checking robustness.
                // If delta_scms is positive, it means we want to increase SCM, so reduce cement.
                // If delta_scms is negative, it means we want to decrease SCM, but there are none, so do nothing.
                if delta_scms > 0.0 {
                    let mass_reduce = delta_scms * total_cement; // Reduce cement by a fraction of its current mass
                    for &i in &cement_indices {
                        self.data[i] = (self.data[i] - mass_reduce).max(10.0); // Safety floor
                    }
                }
            }
        }

        // 4. Apply SP Delta (Adjust Admixture Mass)
        if !sp_indices.is_empty() {
            for &i in &sp_indices {
                let change = delta_sp * total_binder * 0.01; // 1% of binder
                self.data[i] = (self.data[i] + change).max(0.0);
            }
        }
    }
}
