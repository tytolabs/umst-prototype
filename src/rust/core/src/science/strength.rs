// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct StrengthResult {
    pub compressive_strength: f32, // MPa
    pub gel_space_ratio: f32,      // 0.0 - 1.0 (ξ)
    #[wasm_bindgen(getter_with_clone)]
    pub predicted_class: String, // e.g., "C30/37"
}

#[wasm_bindgen]
pub struct StrengthEngine;

#[wasm_bindgen]
impl StrengthEngine {
    /// Computes compressive strength using Powers' Gel-Space Ratio model.
    ///
    /// # Arguments
    /// * `wc_ratio`: Water-to-cement ratio (by mass)
    /// * `degree_hydration`: Alpha (0.0 - 1.0), typical 28-day is ~0.85
    /// * `air_content`: Entrapped/entrained air fraction (0.0 - 0.1)
    /// * `intrinsic_strength`: Strength of the gel itself (approx 240 MPa)
    pub fn compute_powers(
        wc_ratio: f32,
        degree_hydration: f32,
        air_content: f32,
        intrinsic_strength: f32,
    ) -> StrengthResult {
        // Volume of cement (approx density 3.15) vs water (1.0)
        // Vc = 1/3.15, Vw = wc_ratio
        // Gel-Space Ratio (x) = Volume of Gel / (Volume of Gel + Capillary Pores)

        // Simplified Powers model:
        // x = (0.68 * alpha) / (0.32 * alpha + wc_ratio)
        // But let's use the explicit volume approach for accuracy.

        // [GUARDRAIL] Infinite W/C (Zero Cement) Protection
        if wc_ratio > 100.0 {
            return StrengthResult {
                compressive_strength: 0.0,
                gel_space_ratio: 0.0,
                predicted_class: "N/A".to_string(),
            };
        }

        let vg_volume_gel = 0.68 * degree_hydration;
        let vc_volume_capillary = wc_ratio - 0.36 * degree_hydration;

        // Total space available for gel = Volume of Gel + Capillary Pores + Air
        // Note: Powers usually ignores air in the base equation, but for concrete we strictly include it.
        // x = Vgel / (Vgel + Vcap + Vair)

        let space = vg_volume_gel + vc_volume_capillary + air_content;

        // Handle physical impossibility (wc < 0.36 alpha)
        if space <= 0.001 {
            return StrengthResult {
                compressive_strength: 0.0,
                gel_space_ratio: 0.0,
                predicted_class: "INVALID".to_string(),
            };
        }

        let x = vg_volume_gel / space; // Gel-Space Ratio

        // Strength = S * x^3
        let fc = intrinsic_strength * x.powi(3);

        StrengthResult {
            compressive_strength: fc,
            gel_space_ratio: x,
            predicted_class: Self::classify_strength(fc),
        }
    }

    /// Computes strength using Bolomey's empirical equation (Standard Industrial).
    /// fc = K * (1/WC - 0.5)
    ///
    /// This is the CALIBRATED version that accounts for SCM k-values.
    pub fn compute_bolomey(wc_ratio: f32, k_factor: f32) -> f32 {
        if wc_ratio <= 0.01 {
            return 0.0;
        }
        // Clamp w/c to prevent extreme predictions
        let wc_clamped = wc_ratio.clamp(0.25, 1.0);
        let fc = k_factor * ((1.0 / wc_clamped) - 0.5);
        if fc < 0.0 {
            0.0
        } else {
            fc.min(120.0) // Clamp to realistic max
        }
    }

    /// Compute strength with effective w/c (accounting for SCM k-values)
    pub fn compute_calibrated(
        cement: f32,
        slag: f32,
        fly_ash: f32,
        water: f32,
        k_factor: f32,
        k_slag: f32,
        k_fly_ash: f32,
    ) -> f32 {
        let effective_cement = cement + k_slag * slag + k_fly_ash * fly_ash;
        if effective_cement <= 0.0 {
            return 0.0;
        }
        let effective_wc = water / effective_cement;
        Self::compute_bolomey(effective_wc, k_factor)
    }

    fn classify_strength(fc: f32) -> String {
        if fc < 12.0 {
            return "C8/10".to_string();
        }
        if fc < 16.0 {
            return "C12/15".to_string();
        }
        if fc < 20.0 {
            return "C16/20".to_string();
        }
        if fc < 25.0 {
            return "C20/25".to_string();
        }
        if fc < 30.0 {
            return "C25/30".to_string();
        }
        if fc < 37.0 {
            return "C30/37".to_string();
        }
        if fc < 45.0 {
            return "C35/45".to_string();
        }
        if fc < 50.0 {
            return "C40/50".to_string();
        }
        if fc < 60.0 {
            return "C50/60".to_string();
        }
        "C60+".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_cement_safety() {
        // CASE: Zero Cement means W/C is Infinite.
        // The engine must handle f32::INFINITY gracefully and return 0.0 strength.
        let result = StrengthEngine::compute_powers(f32::INFINITY, 0.85, 0.02, 150.0);
        assert_eq!(result.compressive_strength, 0.0);
        assert_eq!(result.predicted_class, "N/A");
    }

    #[test]
    fn test_wc_trend_abrams_law() {
        // CASE: Lower W/C should yield Higher Strength
        // W/C = 0.3 (Strong)
        let strong = StrengthEngine::compute_powers(0.3, 0.85, 0.02, 150.0);

        // W/C = 0.6 (Weak)
        let weak = StrengthEngine::compute_powers(0.6, 0.85, 0.02, 150.0);

        println!("Strong (0.3): {} MPa", strong.compressive_strength);
        println!("Weak (0.6): {} MPa", weak.compressive_strength);

        assert!(strong.compressive_strength > weak.compressive_strength);
        assert!(strong.compressive_strength > 50.0); // Expect high strength for 0.3
        assert!(weak.compressive_strength < 50.0); // Expect lower strength for 0.6
    }
}
