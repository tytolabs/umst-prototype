// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT
//
// UMST — Material Agnostic Operating System
// RheologyEngine: Herschel-Bulkley & Krieger-Dougherty Models
//
// This file is part of UMST.
// For licensing terms, see the LICENSE file in the project root.

use crate::tensors::{MaterialType, MixTensor};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct RheologyResult {
    pub yield_stress: f32, // Pa
    pub viscosity: f32,    // Pa.s
    pub slump_flow: f32,   // mm
}

#[wasm_bindgen]
pub struct RheologyEngine;

#[wasm_bindgen]
impl RheologyEngine {
    /// Calculate rheological properties using Herschel-Bulkley & Krieger-Dougherty models
    pub fn compute(mix: &MixTensor, packing_density: f32) -> RheologyResult {
        // Extract raw data view from tensor
        // In a real generic tensor, accessors would be safer
        // Here we know stride=5 layout: [mass, sg, type, co2, cost]

        let data = mix.data();
        let stride = 8;
        let count = data.len() / stride;

        let mut water_vol = 0.0;
        let mut solid_vol = 0.0;
        let mut sp_dosage = 0.0; // Superplasticizer (kg)

        for i in 0..count {
            let offset = i * stride;
            let mass = data[offset];
            let sg = data[offset + 1];
            let type_id = data[offset + 2] as u8;

            // Volume = mass / (sg * 1000) -> assuming sg is specific gravity relative to water?
            // TS Code: mass / mat.density. Let's assume SG is standard (e.g. 2.4 for agg).
            // Density = SG * 1000 kg/m^3.
            // Density = SG * 1000 kg/m^3.
            // Robust Fallback: If SG is 0 (missing), assume 2.4 (Aggregate/Cement avg)
            let density = if sg > 0.1 { sg * 1000.0 } else { 2400.0 };
            let vol = mass / density;

            if type_id == MaterialType::Water as u8 {
                water_vol += vol;
            } else if type_id == MaterialType::Admixture as u8 {
                // Heuristic: Admixture mass counts as SP dosage if type 3
                sp_dosage += mass;
            } else if type_id != MaterialType::Air as u8 {
                solid_vol += vol;
            }
        }

        let total_vol = water_vol + solid_vol;
        if total_vol <= 0.0001 {
            return RheologyResult {
                yield_stress: 0.0,
                viscosity: 0.0,
                slump_flow: 0.0,
            };
        }

        let phi = solid_vol / total_vol;
        // Clamp packing density to avoid division by zero
        let phi_max = if packing_density > 0.0 {
            packing_density
        } else {
            0.64
        }; // Random close packing default

        let closeness = (phi / phi_max).min(0.99);

        // 1. Yield Stress (Pa) - Yudelovitch-like exponential
        // Tuned: 1.0 * exp(7.5 * closeness)
        let mut yield_stress = 1.0 * (7.5 * closeness).exp();

        // SP Reduction
        // TS: exp(-spDosage * 0.15) - Gentler reduction to keep it measurable (10-500 Pa)
        let sp_factor = (-sp_dosage * 0.15).exp();
        yield_stress *= sp_factor;

        // 2. Plastic Viscosity (Pa.s) - Krieger-Dougherty
        // eta = 0.1 * (1 - closeness)^-2
        let mut viscosity = 0.1 * (1.0 - closeness).powf(-2.0);

        // Clamp Viscosity for pumping realism
        if viscosity > 150.0 {
            viscosity = 150.0;
        }

        // 3. Slump Flow (mm) - Murata's Model linear approximation
        // 900 - 0.45 * YS (Matching formulas.rs)
        let mut slump_flow = 0.0;
        if yield_stress < 2000.0 {
            slump_flow = 900.0 - 0.45 * yield_stress;
        }
        slump_flow = slump_flow.max(0.0).min(850.0);

        RheologyResult {
            yield_stress,
            viscosity,
            slump_flow,
        }
    }
}

// --- Standard Material Cartridge System ---

#[derive(Clone, Debug)]
#[wasm_bindgen]
pub struct MaterialCartridge {
    #[wasm_bindgen(skip)]
    pub id: String,
    pub yield_stress: f32, // Pa
    pub viscosity: f32,    // Pa.s
    pub density: f32,      // kg/m3
}

#[wasm_bindgen]
impl MaterialCartridge {
    #[wasm_bindgen(constructor)]
    pub fn new(id: String, yield_stress: f32, viscosity: f32, density: f32) -> MaterialCartridge {
        MaterialCartridge {
            id,
            yield_stress,
            viscosity,
            density,
        }
    }
}

#[wasm_bindgen]
pub struct CartridgeRegistry;

#[wasm_bindgen]
impl CartridgeRegistry {
    /// Retrieve a standard material cartridge by ID
    /// Supports: "StandardConcrete", "HighPerformanceConcrete", "Clay", "BioHydrogel"
    pub fn get_standard(type_id: &str) -> Option<MaterialCartridge> {
        match type_id {
            "StandardConcrete" => Some(MaterialCartridge {
                id: "StandardConcrete".to_string(),
                yield_stress: 2000.0,
                viscosity: 50.0,
                density: 2300.0,
            }),
            "HighPerformanceConcrete" => Some(MaterialCartridge {
                id: "HighPerformanceConcrete".to_string(),
                yield_stress: 5000.0,
                viscosity: 150.0,
                density: 2400.0,
            }),
            "RAC" => Some(MaterialCartridge {
                // Recycled Aggregate Concrete (Higher Viscosity usually due to absorption)
                id: "RAC".to_string(),
                yield_stress: 3000.0,
                viscosity: 80.0,
                density: 2250.0,
            }),
            "Clay" => Some(MaterialCartridge {
                // High thixotropy, very high yield stress, high viscosity
                id: "Clay".to_string(),
                yield_stress: 8000.0,
                viscosity: 600.0,
                density: 1800.0,
            }),
            "BioHydrogel" => Some(MaterialCartridge {
                // Extremely low viscosity compared to concrete
                id: "BioHydrogel".to_string(),
                yield_stress: 200.0,
                viscosity: 5.0,
                density: 1050.0,
            }),
            _ => None,
        }
    }
}
