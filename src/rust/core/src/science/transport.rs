// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct TransportResult {
    pub sorptivity: f32,      // mm/min^0.5
    pub diffusion_coeff: f32, // m2/s
}

#[wasm_bindgen]
pub struct TransportEngine;

#[wasm_bindgen]
impl TransportEngine {
    /// Computes Sorptivity (S) based on capillary scaling.
    /// S approx sqrt(C * sigma * r * cos(theta) / 2eta)
    pub fn compute_sorptivity(
        pore_radius_nm: f32,
        surface_tension: f32, // N/m typ 0.072
        viscosity: f32,       // Pa.s typ 0.001
    ) -> TransportResult {
        let r = pore_radius_nm * 1e-9;

        // Washburn relation for penetration depth L = S * sqrt(t)
        // S = sqrt( (r * sigma * cos(theta)) / (2 * eta) )
        // Assuming contact angle 0

        if r <= 0.0 {
            return TransportResult {
                sorptivity: 0.0,
                diffusion_coeff: 0.0,
            };
        }

        let s_squared = (r * surface_tension) / (2.0 * viscosity);
        let s_si = s_squared.sqrt(); // m / s^0.5

        // Convert to mm / min^0.5
        // S_mm = S_si * 1000
        // t_min = t_sec / 60 -> sqrt(t_sec) = sqrt(t_min) * sqrt(60)
        // x = S_si * sqrt(t_sec) = S_si * sqrt(60) * sqrt(t_min)
        // x_mm = 1000 * S_si * sqrt(60) * sqrt(t_min)
        // So S_metric = S_si * 1000 * 7.746

        let sorptivity_metric = s_si * 1000.0 * 7.746;

        TransportResult {
            sorptivity: sorptivity_metric, // mm/min^0.5
            diffusion_coeff: s_squared,    // Rough proxy
        }
    }
}
