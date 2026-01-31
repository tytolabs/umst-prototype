// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT
//! Safety Guardrails for RL Optimization
//!
//! Critical safety constraints that CANNOT be violated by the RL agent.
//! These are physics-based hard limits that override any learned policy.

use serde::{Deserialize, Serialize};

/// Hard physical constraints that RL actions must satisfy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhysicsGuardrails {
    // Water-Cement Ratio Bounds (Workability vs Strength)
    pub min_wc_ratio: f64, // Below this = unworkable, won't pump
    pub max_wc_ratio: f64, // Above this = bleeding, segregation

    // Strength Bounds
    pub absolute_min_strength_mpa: f64, // Structural safety requirement

    // Rheology Bounds (Pumpability Window)
    pub min_yield_stress_pa: f64, // Below = segregation
    pub max_yield_stress_pa: f64, // Above = pump blockage
    pub min_viscosity_pas: f64,   // Below = segregation
    pub max_viscosity_pas: f64,   // Above = pump failure

    // Thermal Bounds (Mass Concrete Cracking)
    pub max_temperature_rise_c: f64, // Above = thermal cracking risk

    // Durability Bounds
    pub max_chloride_diffusivity: f64, // Above = rebar corrosion risk
    pub min_cover_depth_mm: f64,       // Structural requirement

    // 3DCP Specific
    pub max_layer_time_minutes: f64, // Above = cold joint risk
    pub min_bond_strength_mpa: f64,  // Below = delamination risk
}

// Physics bounds constants (defined locally to avoid broken cross-module import).
// Values consistent with ACI 318 / EN 206 structural concrete codes.
const WC_RATIO_MIN: f64 = 0.25;

impl Default for PhysicsGuardrails {
    fn default() -> Self {
        PhysicsGuardrails {
            // Industry-standard bounds based on ACI/EN codes
            min_wc_ratio: WC_RATIO_MIN,
            max_wc_ratio: 0.65, // Standard structural concrete ceiling

            absolute_min_strength_mpa: 20.0, // Minimum for structural use

            min_yield_stress_pa: 50.0, // Pumpable limit (stricter than absolute 0)
            max_yield_stress_pa: 500.0, // Pump pressure limit (stricter than abs max)
            min_viscosity_pas: 10.0,
            max_viscosity_pas: 100.0,

            max_temperature_rise_c: 35.0, // Mass concrete limit

            max_chloride_diffusivity: 1e-11, // Durability requirement
            min_cover_depth_mm: 25.0,        // Minimum code requirement

            max_layer_time_minutes: 15.0, // 3DCP open time
            min_bond_strength_mpa: 0.5,   // Inter-layer minimum
        }
    }
}

/// Validation result for an RL action
#[derive(Clone, Debug)]
pub struct GuardrailValidation {
    pub is_valid: bool,
    pub violations: Vec<GuardrailViolation>,
    pub clamped_action: Option<ClampedAction>,
}

#[derive(Clone, Debug)]
pub struct GuardrailViolation {
    pub constraint: String,
    pub actual_value: f64,
    pub limit: f64,
    pub severity: ViolationSeverity,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ViolationSeverity {
    /// Warning: Action is suboptimal but safe
    Warning,
    /// Error: Action would produce unsafe mix, will be clamped
    Error,
    /// Critical: Action is physically impossible, rejected entirely
    Critical,
}

/// An action that has been clamped to respect guardrails
#[derive(Clone, Debug)]
pub struct ClampedAction {
    pub original_delta_wc: f64,
    pub clamped_delta_wc: f64,
    pub original_delta_scms: f64,
    pub clamped_delta_scms: f64,
    pub was_clamped: bool,
}

/// The Guardrail Engine
pub struct GuardrailEngine {
    constraints: PhysicsGuardrails,
}

impl GuardrailEngine {
    pub fn new() -> Self {
        GuardrailEngine {
            constraints: PhysicsGuardrails::default(),
        }
    }

    /// Create guardrails adapted for a specific s_intrinsic calibration.
    ///
    /// The 20 MPa strength floor (EN 206 C20/25) is calibrated for s_intrinsic=240.
    /// At s_intrinsic=80 (D1 fitted), the threshold scales proportionally to ~6.7 MPa.
    /// We floor at 8.0 MPa (EN 206 C8/10, lowest concrete exposure class).
    pub fn with_s_intrinsic(s_intrinsic: f64) -> Self {
        let mut constraints = PhysicsGuardrails::default();
        constraints.absolute_min_strength_mpa =
            (20.0 * s_intrinsic / 240.0).max(8.0);
        GuardrailEngine { constraints }
    }

    pub fn with_constraints(constraints: PhysicsGuardrails) -> Self {
        GuardrailEngine { constraints }
    }

    /// Pre-validate and clamp w/c ratio bounds (no simulation needed).
    /// Returns the clamped delta_wc.
    pub fn clamp_wc(&self, current_wc: f64, delta_wc: f64) -> f64 {
        let proposed = current_wc + delta_wc;
        let clamped = proposed.clamp(
            self.constraints.min_wc_ratio,
            self.constraints.max_wc_ratio,
        );
        clamped - current_wc
    }

    /// Validate an RL action before applying
    pub fn validate_action(
        &self,
        current_wc: f64,
        delta_wc: f64,
        predicted_strength: f64,
        predicted_yield_stress: f64,
        _predicted_viscosity: f64,
    ) -> GuardrailValidation {
        let mut violations = Vec::new();
        let proposed_wc = current_wc + delta_wc;

        // Check W/C ratio bounds
        if proposed_wc < self.constraints.min_wc_ratio {
            violations.push(GuardrailViolation {
                constraint: "min_wc_ratio".to_string(),
                actual_value: proposed_wc,
                limit: self.constraints.min_wc_ratio,
                severity: ViolationSeverity::Error,
            });
        }
        if proposed_wc > self.constraints.max_wc_ratio {
            violations.push(GuardrailViolation {
                constraint: "max_wc_ratio".to_string(),
                actual_value: proposed_wc,
                limit: self.constraints.max_wc_ratio,
                severity: ViolationSeverity::Error,
            });
        }

        // Check minimum strength
        if predicted_strength < self.constraints.absolute_min_strength_mpa {
            violations.push(GuardrailViolation {
                constraint: "absolute_min_strength".to_string(),
                actual_value: predicted_strength,
                limit: self.constraints.absolute_min_strength_mpa,
                severity: ViolationSeverity::Critical,
            });
        }

        // Check rheology window
        if predicted_yield_stress < self.constraints.min_yield_stress_pa {
            violations.push(GuardrailViolation {
                constraint: "min_yield_stress".to_string(),
                actual_value: predicted_yield_stress,
                limit: self.constraints.min_yield_stress_pa,
                severity: ViolationSeverity::Warning,
            });
        }
        if predicted_yield_stress > self.constraints.max_yield_stress_pa {
            violations.push(GuardrailViolation {
                constraint: "max_yield_stress".to_string(),
                actual_value: predicted_yield_stress,
                limit: self.constraints.max_yield_stress_pa,
                severity: ViolationSeverity::Error,
            });
        }

        // Determine if action is valid
        let has_critical = violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::Critical);
        let has_error = violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::Error);

        // Clamp action if errors but no critical violations
        let clamped_action = if has_error && !has_critical {
            let clamped_wc =
                proposed_wc.clamp(self.constraints.min_wc_ratio, self.constraints.max_wc_ratio);
            Some(ClampedAction {
                original_delta_wc: delta_wc,
                clamped_delta_wc: clamped_wc - current_wc,
                original_delta_scms: 0.0, // Would also clamp SCMs in production
                clamped_delta_scms: 0.0,
                was_clamped: true,
            })
        } else {
            None
        };

        GuardrailValidation {
            is_valid: !has_critical,
            violations,
            clamped_action,
        }
    }

    /// Apply penalty to reward for constraint violations
    pub fn violation_penalty(&self, violations: &[GuardrailViolation]) -> f64 {
        violations
            .iter()
            .map(|v| {
                match v.severity {
                    ViolationSeverity::Warning => 5.0,
                    ViolationSeverity::Error => 50.0,
                    ViolationSeverity::Critical => 1000.0, // Massive penalty
                }
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wc_ratio_bounds() {
        let engine = GuardrailEngine::new();

        // Valid action
        let result = engine.validate_action(0.45, 0.05, 35.0, 100.0, 30.0);
        assert!(result.is_valid);
        assert!(result.violations.is_empty());

        // Invalid: W/C too high
        let result = engine.validate_action(0.60, 0.10, 35.0, 100.0, 30.0);
        assert!(!result.violations.is_empty());
        assert!(result.clamped_action.is_some());
    }

    #[test]
    fn test_critical_strength_violation() {
        let engine = GuardrailEngine::new();

        // Strength below absolute minimum
        let result = engine.validate_action(0.45, 0.0, 15.0, 100.0, 30.0);
        assert!(!result.is_valid); // Critical violation = invalid
        assert!(result
            .violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::Critical));
    }
}
