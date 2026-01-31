// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT

//! Thermodynamic Admissibility Filter
//!
//! Enforces the Clausius-Duhem inequality as a hard constraint (Algorithm 1).
//! Reference: "Towards Unified Material-State Tensors for Physics-Gated AI"
//!
//! ## Physics
//!
//! The Clausius-Duhem inequality for isothermal, closed-system hydration at rest
//! (σ:ε̇ = 0, ∇T = 0, Ṫ = 0) reduces to:
//!
//!   D_int = −ρ · ψ̇ ≥ 0
//!
//! where ψ is the Helmholtz free energy per unit mass. For cement hydration,
//! ψ(α) = ψ₀ − Q_hyd · α, where Q_hyd is the specific heat of hydration
//! (~450 J/kg for OPC). Since hydration is exothermic, ψ DECREASES with α,
//! giving positive dissipation for forward hydration (α̇ > 0).

use wasm_bindgen::prelude::*;

/// Result of thermodynamic admissibility check
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct AdmissibilityResult {
    pub accepted: bool,
    pub dissipation: f64, // D_int value (W/m³)
    pub mass_conserved: bool,
    pub energy_positive: bool,
}

#[wasm_bindgen]
impl AdmissibilityResult {
    #[wasm_bindgen(getter)]
    pub fn is_admissible(&self) -> bool {
        self.accepted
    }

    #[wasm_bindgen(getter)]
    pub fn get_rejection_reason(&self) -> String {
        if self.accepted {
            "ACCEPTED".to_string()
        } else if !self.mass_conserved {
            "MASS_VIOLATION".to_string()
        } else if !self.energy_positive {
            "NEGATIVE_DISSIPATION".to_string()
        } else {
            "UNKNOWN".to_string()
        }
    }
}

/// Thermodynamic state for admissibility checking
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct ThermodynamicState {
    pub density: f64,          // kg/m³
    pub temperature: f64,      // K
    pub free_energy: f64,      // Helmholtz ψ (J/kg)
    pub entropy: f64,          // η (J/kg·K)
    pub hydration_degree: f64, // α (0-1)
    pub strength: f64,         // f_c (MPa)
}

/// Default intrinsic gel strength for the Powers model (MPa).
/// Used by `from_mix()` for backward compatibility.
const S_INT_DEFAULT: f64 = 240.0;

/// Specific heat of hydration for OPC (J/kg).
/// Cement hydration releases ~250-500 J/g of cement; we use a representative
/// value normalised per kg of paste for the Helmholtz free energy model.
const Q_HYDRATION: f64 = 450.0;

#[wasm_bindgen]
impl ThermodynamicState {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ThermodynamicState {
        ThermodynamicState {
            density: 2400.0,
            temperature: 293.0, // 20°C
            free_energy: 0.0,
            entropy: 0.0,
            hydration_degree: 0.0,
            strength: 0.0,
        }
    }

    /// Create state from mix parameters using default intrinsic strength (240 MPa).
    pub fn from_mix(w_c: f64, alpha: f64, temp: f64) -> ThermodynamicState {
        Self::from_mix_calibrated(w_c, alpha, temp, S_INT_DEFAULT)
    }

    /// Create state from mix parameters with explicit intrinsic strength.
    ///
    /// # Arguments
    /// * `w_c` — water-cement ratio
    /// * `alpha` — hydration degree (0–1)
    /// * `temp` — temperature (K)
    /// * `s_intrinsic` — intrinsic gel strength (MPa), e.g. 80 for D1 calibration
    pub fn from_mix_calibrated(
        w_c: f64,
        alpha: f64,
        temp: f64,
        s_intrinsic: f64,
    ) -> ThermodynamicState {
        // Gel-space ratio (Powers model)
        let x = 0.68 * alpha / (0.32 * alpha + w_c + 1e-6);

        // Compressive strength from Powers model
        let fc = s_intrinsic * x.powi(3);

        // Helmholtz free energy: DECREASES with hydration (exothermic reaction)
        // ψ(α) = ψ₀ − Q_hyd · α
        // At α = 0: ψ = 0 (reference state)
        // At α > 0: ψ < 0 (energy released)
        let psi = -Q_HYDRATION * alpha;

        ThermodynamicState {
            density: 2400.0 - 400.0 * w_c,
            temperature: temp,
            free_energy: psi,
            entropy: alpha * 0.1, // Simplified entropy (increases with hydration)
            hydration_degree: alpha,
            strength: fc,
        }
    }
}

/// Thermodynamic Admissibility Filter
///
/// Enforces the Clausius-Duhem inequality (Algorithm 1):
///
///   D_int = −ρ · ψ̇ ≥ 0   (isothermal, no work, no heat flux)
///
/// with numerical tolerance ε = 10⁻⁶ and mass conservation check.
#[wasm_bindgen]
pub struct ThermodynamicFilter {
    tolerance: f64,
    rejections: u64,
    acceptances: u64,
}

#[wasm_bindgen]
impl ThermodynamicFilter {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ThermodynamicFilter {
        ThermodynamicFilter {
            tolerance: 1e-6,
            rejections: 0,
            acceptances: 0,
        }
    }

    /// Check if a state transition is thermodynamically admissible.
    ///
    /// Implements Algorithm 1:
    ///   1. Compute ψ̇ = (ψ_new − ψ_old) / Δt
    ///   2. Compute D_int = −ρ · ψ̇  (isothermal Clausius-Duhem)
    ///   3. Accept if D_int ≥ −ε AND mass conserved AND strength non-regressing
    ///
    /// # Arguments
    /// * `old_state` — Previous thermodynamic state
    /// * `new_state` — Proposed new state
    /// * `dt` — Time step (seconds)
    pub fn check_transition(
        &mut self,
        old_state: &ThermodynamicState,
        new_state: &ThermodynamicState,
        dt: f64,
    ) -> AdmissibilityResult {
        // 1. Mass conservation: V = M/ρ  (density change bounded)
        let mass_conserved = (new_state.density - old_state.density).abs() < 100.0;

        // 2. Clausius-Duhem dissipation (isothermal, no work):
        //    D_int = −ρ · ψ̇ ≥ 0
        //
        //    With ψ(α) = −Q_hyd · α:
        //      ψ̇ = −Q_hyd · α̇  (negative for forward hydration)
        //      D_int = −ρ · (−Q_hyd · α̇) = ρ · Q_hyd · α̇
        //
        //    Positive when hydration progresses (α̇ > 0).
        //    Negative when hydration reverses (α̇ < 0) → REJECTED.
        let rho = (old_state.density + new_state.density) / 2.0;
        let psi_dot = (new_state.free_energy - old_state.free_energy) / (dt + 1e-10);
        let d_int = -rho * psi_dot;

        // 3. Strength monotonicity (consequence of 2nd law for non-damage evolution)
        //    Under the Powers model without damage, strength is monotonically
        //    non-decreasing with hydration. A strength decrease without an
        //    explicit damage mechanism violates the constitutive model.
        let strength_valid = new_state.strength >= old_state.strength - self.tolerance;

        let energy_positive = d_int >= -self.tolerance && strength_valid;
        let accepted = mass_conserved && energy_positive;

        if accepted {
            self.acceptances += 1;
        } else {
            self.rejections += 1;
        }

        AdmissibilityResult {
            accepted,
            dissipation: d_int,
            mass_conserved,
            energy_positive,
        }
    }

    /// Get filter statistics
    pub fn get_stats(&self) -> String {
        let total = self.acceptances + self.rejections;
        if total == 0 {
            return "No transitions checked".to_string();
        }
        let rate = self.acceptances as f64 / total as f64 * 100.0;
        format!(
            "Accepted: {}, Rejected: {}, Rate: {:.1}%",
            self.acceptances, self.rejections, rate
        )
    }

    /// Get acceptance count
    pub fn get_acceptances(&self) -> u64 {
        self.acceptances
    }

    /// Get rejection count
    pub fn get_rejections(&self) -> u64 {
        self.rejections
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.acceptances = 0;
        self.rejections = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_admissible_hydration() {
        let mut filter = ThermodynamicFilter::new();

        // Forward hydration: α increases from 0.3 to 0.5 at fixed w/c
        let old = ThermodynamicState::from_mix(0.5, 0.3, 293.0);
        let new = ThermodynamicState::from_mix(0.5, 0.5, 293.0);

        // Verify free energy decreases (exothermic)
        assert!(
            new.free_energy < old.free_energy,
            "Free energy must decrease during hydration: ψ_old={}, ψ_new={}",
            old.free_energy,
            new.free_energy
        );

        let result = filter.check_transition(&old, &new, 3600.0);

        assert!(result.accepted, "Forward hydration should be admissible");
        assert!(
            result.dissipation > 0.0,
            "Dissipation D_int = −ρ·ψ̇ should be positive for forward hydration: {}",
            result.dissipation
        );
    }

    #[test]
    fn test_inadmissible_reverse_hydration() {
        let mut filter = ThermodynamicFilter::new();

        // Reverse hydration: α decreases from 0.7 to 0.3 (forbidden by 2nd law)
        let old = ThermodynamicState::from_mix(0.5, 0.7, 293.0);
        let new = ThermodynamicState::from_mix(0.5, 0.3, 293.0);

        // Verify free energy increases (anti-thermodynamic)
        assert!(
            new.free_energy > old.free_energy,
            "Free energy must increase for reverse hydration (violation)"
        );

        let result = filter.check_transition(&old, &new, 3600.0);

        assert!(!result.accepted, "Reverse hydration should be rejected");
        assert!(
            result.dissipation < 0.0,
            "Dissipation should be negative for reverse hydration: {}",
            result.dissipation
        );
    }

    #[test]
    fn test_strength_monotonicity() {
        let mut filter = ThermodynamicFilter::new();

        // Strength decrease without damage mechanism (inadmissible)
        let mut old = ThermodynamicState::new();
        old.strength = 30.0;
        old.hydration_degree = 0.5;

        let mut new = ThermodynamicState::new();
        new.strength = 25.0; // Decreased!
        new.hydration_degree = 0.5;

        let result = filter.check_transition(&old, &new, 1.0);

        assert!(!result.accepted, "Strength decrease should be rejected");
    }

    #[test]
    fn test_filter_statistics() {
        let mut filter = ThermodynamicFilter::new();

        // Run multiple forward hydration transitions
        for i in 0..10 {
            let old = ThermodynamicState::from_mix(0.5, i as f64 * 0.1, 293.0);
            let new = ThermodynamicState::from_mix(0.5, (i + 1) as f64 * 0.1, 293.0);
            filter.check_transition(&old, &new, 3600.0);
        }

        let stats = filter.get_stats();
        assert!(stats.contains("Accepted: 10"));
        assert_eq!(filter.get_acceptances(), 10);
        assert_eq!(filter.get_rejections(), 0);
    }

    #[test]
    fn test_dissipation_is_rho_times_q_hyd_times_alpha_dot() {
        let mut filter = ThermodynamicFilter::new();

        // Verify D_int = ρ · Q_hyd · α̇ quantitatively
        let w_c = 0.45;
        let alpha_old = 0.4;
        let alpha_new = 0.6;
        let dt = 7.0 * 86400.0; // 7 days in seconds

        let old = ThermodynamicState::from_mix(w_c, alpha_old, 293.0);
        let new = ThermodynamicState::from_mix(w_c, alpha_new, 293.0);
        let result = filter.check_transition(&old, &new, dt);

        let rho = (old.density + new.density) / 2.0;
        let alpha_dot = (alpha_new - alpha_old) / dt;
        let expected_d_int = rho * Q_HYDRATION * alpha_dot;

        let relative_error = ((result.dissipation - expected_d_int) / expected_d_int).abs();
        assert!(
            relative_error < 1e-10,
            "D_int should equal ρ·Q_hyd·α̇: got {}, expected {}",
            result.dissipation,
            expected_d_int
        );
    }

    #[test]
    fn test_from_mix_calibrated() {
        // Verify that from_mix_calibrated with different s_intrinsic
        // changes strength but not free energy
        let state_240 = ThermodynamicState::from_mix_calibrated(0.5, 0.5, 293.0, 240.0);
        let state_80 = ThermodynamicState::from_mix_calibrated(0.5, 0.5, 293.0, 80.0);

        // Same free energy (depends on Q_hyd and alpha, not s_intrinsic)
        assert_eq!(state_240.free_energy, state_80.free_energy);

        // Different strength (proportional to s_intrinsic)
        let ratio = state_240.strength / state_80.strength;
        assert!(
            (ratio - 3.0).abs() < 1e-10,
            "Strength ratio should be 240/80 = 3.0, got {}",
            ratio
        );
    }
}
