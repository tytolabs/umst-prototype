// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT
//
// UMST — Unified Material-State Tensor
// Geometry Integration Module
//
// This file is part of UMST.
// For licensing terms, see the LICENSE file in the project root.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Geometry data extracted from hypergraph for physics calculations
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeometryData {
    /// Surface area (m²) - affects hydration rates, reaction kinetics
    pub surface_area: f32,
    /// Volume (m³) - actual volume from SDF
    pub volume: f32,
    /// Bounding box volume (m³) - for packing calculations
    pub bounding_volume: f32,
    /// Surface area to volume ratio - affects transport properties
    pub sav_ratio: f32,
    /// Shape complexity factor (0-1) - how irregular the geometry is
    pub complexity: f32,
    /// Principal curvatures for stress concentration analysis
    pub mean_curvature: f32,
    pub gaussian_curvature: f32,
}

impl Default for GeometryData {
    fn default() -> Self {
        Self {
            surface_area: 0.0,
            volume: 0.0,
            bounding_volume: 0.0,
            sav_ratio: 0.0,
            complexity: 0.5, // Default spherical
            mean_curvature: 0.0,
            gaussian_curvature: 0.0,
        }
    }
}

#[wasm_bindgen]
impl GeometryData {
    #[wasm_bindgen(constructor)]
    pub fn new() -> GeometryData {
        Self::default()
    }

    /// Create from SDF data (simplified implementation)
    /// In practice, this would analyze the actual SDF binary data
    #[wasm_bindgen]
    pub fn from_sdf_data(_sdf_bytes: &[u8], bounds: &[f32]) -> GeometryData {
        // Simplified: assume spherical geometry for now
        // Real implementation would parse SDF and compute actual properties

        let width = bounds[3] - bounds[0];
        let height = bounds[4] - bounds[1];
        let depth = bounds[5] - bounds[2];

        let bounding_volume = width * height * depth;
        let radius = (width.min(height).min(depth)) / 2.0;
        let volume = (4.0/3.0) * std::f32::consts::PI * radius * radius * radius;
        let surface_area = 4.0 * std::f32::consts::PI * radius * radius;
        let sav_ratio = surface_area / volume;

        GeometryData {
            surface_area,
            volume,
            bounding_volume,
            sav_ratio,
            complexity: 0.1, // Low complexity for sphere
            mean_curvature: 1.0 / radius, // For sphere
            gaussian_curvature: 1.0 / (radius * radius), // For sphere
        }
    }

    /// Calculate packing efficiency based on geometry
    pub fn packing_efficiency(&self) -> f32 {
        if self.bounding_volume > 0.0 {
            self.volume / self.bounding_volume
        } else {
            0.5 // Default
        }
    }

    /// Calculate transport resistance factor (higher surface area = higher resistance)
    pub fn transport_resistance(&self) -> f32 {
        1.0 / (1.0 + self.sav_ratio * 0.1) // Simplified model
    }
}