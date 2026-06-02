// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT

//! **Deprecation shim** — constitutional transition gate delegates to `umst_manifold::gate::mix_proposal`.
//!
//! SSOT for Algorithm 1 math: `umst-manifold/src/gate/mix_proposal.rs` (parity: `gate_dual_run_parity` 8/8).
//! Keep this module for `wasm-bindgen` types and legacy imports until PPO/WASM consumers migrate.

use wasm_bindgen::prelude::*;

use umst_manifold::gate::mix_proposal::{
    ThermodynamicMixFilter, ThermodynamicStateSnapshot, ThermodynamicTransitionOutcome,
    DEFAULT_S_INTRINSIC_MPA,
};

/// Result of thermodynamic admissibility check (WASM-facing; logic from manifold).
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct AdmissibilityResult {
    pub accepted: bool,
    pub dissipation: f64,
    pub mass_conserved: bool,
    pub energy_positive: bool,
}

impl From<ThermodynamicTransitionOutcome> for AdmissibilityResult {
    fn from(r: ThermodynamicTransitionOutcome) -> Self {
        AdmissibilityResult {
            accepted: r.accepted,
            dissipation: r.dissipation,
            mass_conserved: r.mass_conserved,
            energy_positive: r.energy_positive,
        }
    }
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

/// Thermodynamic state for admissibility checking (WASM-facing snapshot).
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct ThermodynamicState {
    pub density: f64,
    pub temperature: f64,
    pub free_energy: f64,
    pub entropy: f64,
    pub hydration_degree: f64,
    pub strength: f64,
}

impl From<ThermodynamicStateSnapshot> for ThermodynamicState {
    fn from(s: ThermodynamicStateSnapshot) -> Self {
        ThermodynamicState {
            density: s.density,
            temperature: s.temperature,
            free_energy: s.free_energy,
            entropy: s.entropy,
            hydration_degree: s.hydration_degree,
            strength: s.strength,
        }
    }
}

impl From<&ThermodynamicState> for ThermodynamicStateSnapshot {
    fn from(s: &ThermodynamicState) -> Self {
        ThermodynamicStateSnapshot {
            density: s.density,
            temperature: s.temperature,
            free_energy: s.free_energy,
            entropy: s.entropy,
            hydration_degree: s.hydration_degree,
            strength: s.strength,
        }
    }
}

#[wasm_bindgen]
impl ThermodynamicState {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ThermodynamicState {
        ThermodynamicStateSnapshot::new_idle().into()
    }

    pub fn from_mix(w_c: f64, alpha: f64, temp: f64) -> ThermodynamicState {
        ThermodynamicStateSnapshot::from_mix(w_c, alpha, temp).into()
    }

    pub fn from_mix_calibrated(
        w_c: f64,
        alpha: f64,
        temp: f64,
        s_intrinsic: f64,
    ) -> ThermodynamicState {
        ThermodynamicStateSnapshot::from_mix_calibrated(w_c, alpha, temp, s_intrinsic).into()
    }
}

/// Deprecated alias — use manifold `ThermodynamicMixFilter` via HTTP gate or `manifold-gate` shim.
#[wasm_bindgen]
pub struct ThermodynamicFilter {
    inner: ThermodynamicMixFilter,
}

#[wasm_bindgen]
impl ThermodynamicFilter {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ThermodynamicFilter {
        ThermodynamicFilter {
            inner: ThermodynamicMixFilter::new(),
        }
    }

    pub fn check_transition(
        &mut self,
        old_state: &ThermodynamicState,
        new_state: &ThermodynamicState,
        dt: f64,
    ) -> AdmissibilityResult {
        let old_s: ThermodynamicStateSnapshot = old_state.into();
        let new_s: ThermodynamicStateSnapshot = new_state.into();
        self.inner
            .check_transition(&old_s, &new_s, dt)
            .into()
    }

    pub fn get_stats(&self) -> String {
        let total = self.inner.acceptances() + self.inner.rejections();
        if total == 0 {
            return "No transitions checked".to_string();
        }
        let rate = self.inner.acceptances() as f64 / total as f64 * 100.0;
        format!(
            "Accepted: {}, Rejected: {}, Rate: {:.1}%",
            self.inner.acceptances(),
            self.inner.rejections(),
            rate
        )
    }

    pub fn get_acceptances(&self) -> u64 {
        self.inner.acceptances()
    }

    pub fn get_rejections(&self) -> u64 {
        self.inner.rejections()
    }

    pub fn reset_stats(&mut self) {
        self.inner.reset_stats();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_admissible_hydration() {
        let mut filter = ThermodynamicFilter::new();
        let old = ThermodynamicState::from_mix(0.5, 0.3, 293.0);
        let new = ThermodynamicState::from_mix(0.5, 0.5, 293.0);
        assert!(new.free_energy < old.free_energy);
        let result = filter.check_transition(&old, &new, 3600.0);
        assert!(result.accepted);
        assert!(result.dissipation > 0.0);
    }

    #[test]
    fn test_inadmissible_reverse_hydration() {
        let mut filter = ThermodynamicFilter::new();
        let old = ThermodynamicState::from_mix(0.5, 0.7, 293.0);
        let new = ThermodynamicState::from_mix(0.5, 0.3, 293.0);
        let result = filter.check_transition(&old, &new, 3600.0);
        assert!(!result.accepted);
        assert!(result.dissipation < 0.0);
    }

    #[test]
    fn test_strength_monotonicity() {
        let mut filter = ThermodynamicFilter::new();
        let mut old = ThermodynamicState::new();
        old.strength = 30.0;
        old.hydration_degree = 0.5;
        let mut new = ThermodynamicState::new();
        new.strength = 25.0;
        new.hydration_degree = 0.5;
        let result = filter.check_transition(&old, &new, 1.0);
        assert!(!result.accepted);
    }

    #[test]
    fn test_from_mix_calibrated_strength_ratio() {
        let state_240 = ThermodynamicState::from_mix_calibrated(0.5, 0.5, 293.0, 240.0);
        let state_80 = ThermodynamicState::from_mix_calibrated(0.5, 0.5, 293.0, 80.0);
        assert_eq!(state_240.free_energy, state_80.free_energy);
        let ratio = state_240.strength / state_80.strength;
        assert!((ratio - 3.0).abs() < 1e-10);
    }

    #[test]
    fn delegates_default_s_intrinsic_to_manifold() {
        let s = ThermodynamicState::from_mix(0.5, 0.5, 293.0);
        let snap = ThermodynamicStateSnapshot::from_mix(0.5, 0.5, 293.0);
        assert_eq!(s.strength, snap.strength);
        assert_eq!(DEFAULT_S_INTRINSIC_MPA, 240.0);
    }
}
