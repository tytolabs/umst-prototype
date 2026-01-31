// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT
//! Universal Physics Formulas for Concrete Science
//!
//! These are pure mathematical functions operating on scalar values.
//! They form the foundation of UMST physics calculations.
//!
//! References:
//! - Bolomey (1927): Strength prediction
//! - Krieger-Dougherty (1959): Suspension viscosity
//! - Powers (1958): Gel-space ratio
//! - ACI 318: Elastic modulus

use wasm_bindgen::prelude::*;

// ============================================================================
// STRENGTH MODELS
// ============================================================================

/// Bolomey Equation for 28-day compressive strength
///
/// f_c = K * (1/wc - 0.5)
///
/// # Arguments
/// * `wc` - Water/cement ratio (typically 0.3-0.7)
/// * `k` - Cement class constant (25 for CEM I 42.5N, 30 for 52.5R)
///
/// # Returns
/// Predicted 28-day compressive strength in MPa
#[wasm_bindgen]
pub fn bolomey_strength(wc: f32, k: f32) -> f32 {
    if wc <= 0.0 {
        return 0.0;
    }
    let result = k * ((1.0 / wc) - 0.5);
    result.max(0.0)
}

/// Powers' Gel-Space Ratio for strength prediction
///
/// f_c = 234 * x^3 where x = 0.647α / (0.319α + w/c)
///
/// # Arguments
/// * `wc` - Water/cement ratio
/// * `alpha` - Degree of hydration (0-1, typically 0.7-0.9 at 28 days)
#[wasm_bindgen]
pub fn powers_gel_space_strength(wc: f32, alpha: f32) -> f32 {
    if wc <= 0.0 || alpha <= 0.0 {
        return 0.0;
    }
    // Powers (1947): V_gel = 0.68·α, water consumed = 0.36·α
    // x = V_gel / (V_gel + V_cap) = 0.68α / (0.32α + w/c)
    let x = (0.68 * alpha) / (0.32 * alpha + wc);
    234.0 * x.powi(3)
}

/// ACI 318 Elastic Modulus
///
/// E = 4700 * sqrt(f_c) MPa, or 4.7 * sqrt(f_c) GPa
///
/// # Arguments
/// * `fc` - Compressive strength in MPa
///
/// # Returns
/// Elastic modulus in GPa
#[wasm_bindgen]
pub fn aci_elastic_modulus(fc: f32) -> f32 {
    if fc <= 0.0 {
        return 0.0;
    }
    4.7 * fc.sqrt()
}

// ============================================================================
// RHEOLOGY MODELS
// ============================================================================

/// Krieger-Dougherty equation for suspension viscosity
///
/// η = η₀ * (1 - φ/φₘ)^(-[η]φₘ)
///
/// # Arguments
/// * `phi` - Solid volume fraction (0-1)
/// * `phi_max` - Maximum packing fraction (typically 0.64-0.74)
/// * `intrinsic_viscosity` - Intrinsic viscosity [η], typically 2.5 for spheres
///
/// # Returns
/// Relative viscosity multiplier
#[wasm_bindgen]
pub fn krieger_dougherty_viscosity(phi: f32, phi_max: f32, intrinsic_viscosity: f32) -> f32 {
    if phi >= phi_max || phi_max <= 0.0 {
        return f32::MAX; // Solid state
    }
    if phi <= 0.0 {
        return 1.0; // Pure fluid
    }

    let ratio = 1.0 - (phi / phi_max);
    let exponent = -intrinsic_viscosity * phi_max;
    ratio.powf(exponent)
}

/// Simplified Krieger-Dougherty with default intrinsic viscosity
#[wasm_bindgen]
pub fn krieger_dougherty_simple(phi: f32, phi_max: f32) -> f32 {
    krieger_dougherty_viscosity(phi, phi_max, 2.5)
}

/// Yield stress estimation based on packing closeness
///
/// τ₀ = A * exp(B * (φ/φₘ))
///
/// Tuned for concrete: A=1.0, B=7.5 gives realistic range
#[wasm_bindgen]
pub fn yield_stress_exponential(phi: f32, phi_max: f32) -> f32 {
    if phi_max <= 0.0 {
        return 0.0;
    }
    let closeness = (phi / phi_max).min(0.99);
    1.0 * (7.5 * closeness).exp()
}

/// Murata's slump flow correlation
///
/// Slump (mm) ≈ 900 - 0.45 * τ₀ for τ₀ < 2000 Pa
///
/// # Arguments
/// * `yield_stress` - Yield stress in Pa
///
/// # Returns
/// Slump flow in mm (0-850 range)
#[wasm_bindgen]
pub fn murata_slump_flow(yield_stress: f32) -> f32 {
    if yield_stress >= 2000.0 {
        return 0.0;
    }
    let slump = 900.0 - 0.45 * yield_stress;
    slump.max(0.0).min(850.0)
}

// ============================================================================
// DURABILITY MODELS
// ============================================================================

/// Chloride diffusivity estimation
///
/// Based on W/C and packing. Higher W/C = more permeable.
///
/// # Returns
/// Chloride diffusivity in 10^-12 m²/s (typical range 1-20)
#[wasm_bindgen]
pub fn chloride_diffusivity(wc: f32, packing_density: f32) -> f32 {
    let permeability_factor = (wc - 0.3).max(0.0) + (1.0 - packing_density);
    2.0 + 30.0 * permeability_factor
}

/// ASR (Alkali-Silica Reaction) risk based on SCM content
///
/// SCM > 20% significantly reduces ASR risk
#[wasm_bindgen]
pub fn asr_risk(scm_ratio: f32) -> f32 {
    if scm_ratio > 0.2 {
        0.05
    } else {
        0.8
    }
}

// ============================================================================
// SUSTAINABILITY MODELS
// ============================================================================

/// LCA Score calculation
///
/// Benchmarked: 300 kg CO2/m³ = 50 score
/// Range: 0-100 (100 is best)
#[wasm_bindgen]
pub fn lca_score(co2_kg_m3: f32) -> f32 {
    (100.0 - (co2_kg_m3 / 500.0 * 100.0)).max(0.0)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- STRENGTH TESTS ----

    #[test]
    fn test_bolomey_standard_c30() {
        // W/C = 0.5, K = 25 → f_c = 25 * (2 - 0.5) = 37.5 MPa
        let result = bolomey_strength(0.5, 25.0);
        assert!(
            (result - 37.5).abs() < 0.1,
            "Expected ~37.5, got {}",
            result
        );
    }

    #[test]
    fn test_bolomey_high_strength() {
        // W/C = 0.35, K = 30 → f_c = 30 * (2.857 - 0.5) = 70.7 MPa
        let result = bolomey_strength(0.35, 30.0);
        assert!(
            result > 65.0 && result < 75.0,
            "Expected ~70.7, got {}",
            result
        );
    }

    #[test]
    fn test_bolomey_zero_wc_returns_zero() {
        assert_eq!(bolomey_strength(0.0, 25.0), 0.0);
    }

    #[test]
    fn test_bolomey_negative_wc_returns_zero() {
        assert_eq!(bolomey_strength(-0.5, 25.0), 0.0);
    }

    #[test]
    fn test_powers_gel_space() {
        // Typical hydrated concrete: α = 0.85, w/c = 0.5
        // Powers' formula: x = 0.647*0.85 / (0.319*0.85 + 0.5) = 0.55/0.77 = 0.714
        // f_c = 234 * 0.714^3 = 85 MPa (theoretical max, real-world ~60-70)
        let result = powers_gel_space_strength(0.5, 0.85);
        assert!(
            result > 70.0 && result < 100.0,
            "Expected 70-100 MPa, got {}",
            result
        );
    }

    #[test]
    fn test_aci_elastic_modulus() {
        // f_c = 30 MPa → E = 4.7 * sqrt(30) = 25.7 GPa
        let result = aci_elastic_modulus(30.0);
        assert!(
            (result - 25.7).abs() < 0.5,
            "Expected ~25.7, got {}",
            result
        );
    }

    // ---- RHEOLOGY TESTS ----

    #[test]
    fn test_krieger_dougherty_low_solids() {
        // Low solid fraction → viscosity near 1
        let result = krieger_dougherty_simple(0.2, 0.64);
        assert!(result < 5.0, "Expected low viscosity, got {}", result);
    }

    #[test]
    fn test_krieger_dougherty_high_solids() {
        // φ approaching φ_max → viscosity explodes
        let result = krieger_dougherty_simple(0.62, 0.64);
        assert!(result > 50.0, "Expected high viscosity, got {}", result);
    }

    #[test]
    fn test_krieger_dougherty_at_max_returns_max() {
        let result = krieger_dougherty_simple(0.64, 0.64);
        assert_eq!(result, f32::MAX);
    }

    #[test]
    fn test_yield_stress_realistic_range() {
        // For φ/φ_max ~ 0.8 (typical concrete), yield stress ~50-500 Pa
        let result = yield_stress_exponential(0.56, 0.70);
        assert!(
            result > 50.0 && result < 1000.0,
            "Expected 50-1000 Pa, got {}",
            result
        );
    }

    #[test]
    fn test_murata_slump_flowing() {
        // Low yield stress → high slump
        let result = murata_slump_flow(100.0);
        assert!(result > 800.0, "Expected >800mm, got {}", result);
    }

    #[test]
    fn test_murata_slump_stiff() {
        // High yield stress → low/no slump
        let result = murata_slump_flow(2500.0);
        assert_eq!(result, 0.0);
    }

    // ---- DURABILITY TESTS ----

    #[test]
    fn test_chloride_low_wc() {
        // Low W/C + high packing = moderate diffusivity
        // Formula: 2 + 30 * ((wc - 0.3) + (1 - packing))
        //        = 2 + 30 * (0.05 + 0.25) = 2 + 9 = 11
        let result = chloride_diffusivity(0.35, 0.75);
        assert!(result < 15.0, "Expected <15, got {}", result);
    }

    #[test]
    fn test_asr_with_scm() {
        assert_eq!(asr_risk(0.25), 0.05);
    }

    #[test]
    fn test_asr_without_scm() {
        assert_eq!(asr_risk(0.10), 0.8);
    }

    // ---- SUSTAINABILITY TESTS ----

    #[test]
    fn test_lca_score_benchmark() {
        // 300 kg CO2 → score ~40
        let result = lca_score(300.0);
        assert!(
            result > 35.0 && result < 45.0,
            "Expected ~40, got {}",
            result
        );
    }

    #[test]
    fn test_lca_score_bounds() {
        assert!(lca_score(0.0) <= 100.0);
        assert!(lca_score(1000.0) >= 0.0);
    }
}
