// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT
//! State and Action definitions for RL agent
//!
//! State Space: 35 dimensions (27 proxies + 6 simulation outputs + 2 weather)
//! Action Space: 9 dimensions (mix adjustments)

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// 33-dimensional state vector for RL agent
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RLState {
    /// 27 Sensing Proxy values (Jar Settle, Tilt Flow, etc.)
    proxies: Vec<f64>,

    /// Heat generation (Q) from hydration
    pub heat_q: f64,
    /// Accumulated damage (D)
    pub damage_d: f64,
    /// Fracture toughness (K_IC)
    pub fracture_kic: f64,
    /// Diffusivity coefficient
    pub diffusivity: f64,
    /// Shrinkage strain
    pub shrinkage: f64,
    /// Bond strength (MPa)
    /// Bond strength (MPa)
    pub bond_strength: f64,

    // Weather Data (New Dimensions 34 & 35)
    /// Ambient Temperature (°C)
    pub temperature: f64,
    /// Relative Humidity (0-1)
    pub humidity: f64,
}

#[wasm_bindgen]
impl RLState {
    #[wasm_bindgen(constructor)]
    pub fn new() -> RLState {
        RLState {
            proxies: vec![0.0; 27],
            heat_q: 0.0,
            damage_d: 0.0,
            fracture_kic: 1.5,
            diffusivity: 0.001,
            shrinkage: 0.0003,
            bond_strength: 2.5,
            temperature: 20.0, // Standard Lab Temp
            humidity: 0.5,     // Standard Lab Humidity
        }
    }

    pub fn set_proxy(&mut self, index: usize, value: f64) {
        if index < 27 {
            self.proxies[index] = value;
        }
    }

    pub fn to_vector(&self) -> Vec<f64> {
        let mut vec = self.proxies.clone();
        vec.push(self.heat_q);
        vec.push(self.damage_d);
        vec.push(self.fracture_kic);
        vec.push(self.diffusivity);
        vec.push(self.shrinkage);
        vec.push(self.bond_strength);
        // Add new dimensions
        vec.push(self.temperature);
        vec.push(self.humidity);
        vec
    }
}

/// 9-dimensional action space for mix optimization
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RLAction {
    /// Δ(w/c) - water-cement ratio change
    pub delta_wc: f64,
    /// Δ(SCMs %) - supplementary cementitious materials change
    pub delta_scms: f64,
    /// Δ(Superplasticizer %)
    pub delta_sp: f64,
    /// Pretreatment flag (0 or 1)
    pub pretreatment: f64,
    /// Earth content percentage
    pub earth_content: f64,
    /// Accelerator dosage
    pub accelerator_dose: f64,
    /// Extrusion speed (mm/s)
    pub extrusion_speed: f64,
    /// Path strategy (0=LINEAR, 1=ZIGZAG, 2=SPIRAL)
    pub path_strategy: f64,
    /// Cure profile (0=STANDARD, 1=ACCELERATED, 2=EXTENDED)
    pub cure_profile: f64,
}

#[wasm_bindgen]
impl RLAction {
    #[wasm_bindgen(constructor)]
    pub fn new() -> RLAction {
        RLAction {
            delta_wc: 0.0,
            delta_scms: 0.0,
            delta_sp: 0.0,
            pretreatment: 0.0,
            earth_content: 0.0,
            accelerator_dose: 0.0,
            extrusion_speed: 50.0,
            path_strategy: 0.0,
            cure_profile: 0.0,
        }
    }

    pub fn to_vector(&self) -> Vec<f64> {
        vec![
            self.delta_wc,
            self.delta_scms,
            self.delta_sp,
            self.pretreatment,
            self.earth_content,
            self.accelerator_dose,
            self.extrusion_speed,
            self.path_strategy,
            self.cure_profile,
        ]
    }

    pub fn from_vector(v: &[f64]) -> RLAction {
        RLAction {
            delta_wc: v.get(0).copied().unwrap_or(0.0),
            delta_scms: v.get(1).copied().unwrap_or(0.0),
            delta_sp: v.get(2).copied().unwrap_or(0.0),
            pretreatment: v.get(3).copied().unwrap_or(0.0),
            earth_content: v.get(4).copied().unwrap_or(0.0),
            accelerator_dose: v.get(5).copied().unwrap_or(0.0),
            extrusion_speed: v.get(6).copied().unwrap_or(50.0),
            path_strategy: v.get(7).copied().unwrap_or(0.0),
            cure_profile: v.get(8).copied().unwrap_or(0.0),
        }
    }
}
