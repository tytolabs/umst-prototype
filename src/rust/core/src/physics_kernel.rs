// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT
//
// UMST — Material Agnostic Operating System
// PhysicsKernel: Unified Rust/WASM Physics Orchestrator
//
// This file is part of UMST, developed by Santhosh Shyamsundar, Prabhu S., and Studio Tyto.
// For licensing terms, see the LICENSE file in the project root.

// ============================================================================
// PhysicsKernel: Unified Rust Physics Orchestrator
// ============================================================================
// This module centralizes ALL physics/science computation.
// TypeScript should call ONLY this module, not individual engines.
// ============================================================================

use crate::science::{
    colloidal::ColloidalEngine, fracture::FractureEngine, itz::ITZEngine, porosity::PorosityEngine,
    rheology::RheologyEngine, strength::StrengthEngine, sustainability::SustainabilityEngine,
    thermo::ThermoEngine, transport::TransportEngine,
};
use crate::tensors::{GeometryData, MaterialType, MixTensor};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhysicsConfig {
    pub enable_rheology: bool,
    pub enable_strength: bool,
    pub enable_thermo: bool,
    pub enable_durability: bool,
    pub enable_sustainability: bool,
    pub enable_mechanics: bool,
    pub enable_transport: bool,
    pub enable_colloidal: bool,
    pub enable_itz: bool,
    pub enable_cost: bool,
    pub enable_maturity: bool,
    /// Calibrated intrinsic gel strength (MPa). Default 80.0 for D1 (UCI Concrete).
    pub s_intrinsic: f32,
    /// Calibrated SCM efficiency factor for w/c computation.
    /// Since MixTensor doesn't distinguish slag from fly_ash (both SCM),
    /// this is a blended k-factor. For D1: ~1.0 (L-BFGS-B fitted k_slag=1.18, k_fly_ash=1.15).
    /// For D2-D4: ~0.2 (fitted). Default 1.0 for D1.
    pub k_scm: f32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            enable_rheology: true,
            enable_strength: true,
            enable_thermo: true,
            enable_durability: true,
            enable_sustainability: true,
            enable_mechanics: true,
            enable_transport: true,
            enable_colloidal: true,
            enable_itz: true,
            enable_cost: true,
            enable_maturity: true,
            // Calibrated for D1 (UCI Concrete): s_intrinsic=80.0, k_scm≈1.0
            // L-BFGS-B fitted: k_slag=1.184, k_fly_ash=1.153, blended ≈ 1.0
            s_intrinsic: 80.0,
            k_scm: 1.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationEvent {
    pub topic: String,
    pub message: String,
    pub severity: String, // 'INFO', 'WARNING', 'CRITICAL'
}

#[derive(Serialize, Deserialize)]
pub struct PhysicsResponse {
    pub result: IndustrialResult,
    pub events: Vec<ValidationEvent>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndustrialResult {
    pub fresh: FreshProperties,
    pub hardened: HardenedProperties,
    pub durability: DurabilityProperties,
    pub sustainability: SustainabilityProperties,
    pub packing: PackingProperties,
    pub mechanics: MechanicsProperties,
    pub thermal: ThermalProperties,
    pub transport: TransportProperties,
    pub chemical: ChemicalProperties,
    pub economics: EconomicsProperties,
    pub colloidal: ColloidalProperties,
    pub itz: ITZProperties,
    pub compute_time_ms: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FreshProperties {
    pub slump_flow: f32,
    pub yield_stress: f32,
    pub plastic_viscosity: f32,
    pub thixotropy_index: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HardenedProperties {
    pub f28_compressive: f32,
    pub maturity_index: f32,
    pub e_modulus: f32,
    pub creep_coefficient: f32,
    pub hydration_degree: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DurabilityProperties {
    pub chloride_diffusivity: f32,
    pub sulfate_resistance: f32,
    pub asr_risk: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SustainabilityProperties {
    pub co2_kg_m3: f32,
    pub embodied_energy: f32,
    pub lca_score: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackingProperties {
    pub density: f32,
    pub voids: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MechanicsProperties {
    pub fracture_toughness: f32,
    pub split_tensile: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThermalProperties {
    pub adiabatic_rise: f32,
    pub heat_of_hydration: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransportProperties {
    pub sorptivity: f32,
    pub permeability: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChemicalProperties {
    pub ph_pore_solution: f32,
    pub mineralogy: Vec<String>,
    pub diffusivity: f32, // from chemo_water
    pub suction: f32,     // from chemo_water
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EconomicsProperties {
    pub total_cost: f32,
    pub cost_per_m3: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColloidalProperties {
    pub zeta_potential: f32,
    pub interparticle_distance: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ITZProperties {
    pub thickness: f32,
    pub porosity: f32,
}

#[wasm_bindgen]
pub struct PhysicsKernel;

#[wasm_bindgen]
impl PhysicsKernel {
    /// Main entry point: Compute ALL industrial physics from JSON inputs.
    /// TypeScript calls this ONCE with raw JSON, Rust handles everything.
    /// Computes comprehensive industrial concrete properties from component masses.
    ///
    /// This is the primary API function for industrial applications. It takes concrete
    /// component masses and material properties, then returns a complete analysis
    /// including fresh properties, hardened strength, durability, sustainability, and cost.
    ///
    /// # Arguments
    /// * `components_json` - JSON array of concrete components with masses and types
    /// * `materials_json` - JSON array defining material properties and densities
    ///
    /// # Returns
    /// JSON string containing complete analysis results or error message
    ///
    /// # Example
    /// ```json
    /// components_json: [{"materialId": "cement", "mass": 300.0, "type": 0}, ...]
    /// materials_json: [{"id": "cement", "density": 3150.0, "type": "cement"}, ...]
    /// ```
    ///
    /// # Errors
    /// Returns JSON error object if component parsing fails or computation errors occur
    ///
    /// # Notes
    /// This function is deterministic and reproducible across platforms.
    /// All computations follow first-principles physics with calibrated parameters.
    #[wasm_bindgen]
    pub fn compute_industrial(components_json: &str, materials_json: &str) -> String {
        // 1. Hydrate MixTensor from JSON (ALL marshalling in Rust)
        let tensor = match MixTensor::from_json(components_json, materials_json) {
            Ok(t) => t,
            Err(e) => {
                return serde_json::to_string(&serde_json::json!({
                    "error": format!("Failed to hydrate tensor: {:?}", e)
                }))
                .unwrap_or_default();
            }
        };

        // 2. Run Pure Rust Computation (Shared with Monte Carlo)
        // Use default config (ALL engines enabled) for industrial endpoint
        let result = Self::compute(&tensor, None, &PhysicsConfig::default());

        // 3. Generate Events (The Bridge)
        let mut events = Vec::new();

        // Example: Check Yield Stress
        if result.fresh.yield_stress < 10.0 {
            events.push(ValidationEvent {
                topic: "PHYSICS.RHEOLOGY".to_string(),
                message: format!(
                    "Low Yield Stress: {:.1} Pa (Risk of collapse)",
                    result.fresh.yield_stress
                ),
                severity: "WARNING".to_string(),
            });
        }

        // Example: Check Strength
        if result.hardened.f28_compressive < 25.0 {
            events.push(ValidationEvent {
                topic: "PHYSICS.STRENGTH".to_string(),
                message: format!(
                    "Low 28d Strength: {:.1} MPa",
                    result.hardened.f28_compressive
                ),
                severity: "CRITICAL".to_string(),
            });
        }

        // 4. Serialize output
        let response = PhysicsResponse { result, events };

        serde_json::to_string(&response).unwrap_or_default()
    }
}

/// Pure Rust Implementation (Non-WASM-Bindgen)
impl PhysicsKernel {
    /// [CORE] Pure Rust Physics Pipeline
    /// Zero-overhead access for internal agents (RL, Monte Carlo).
    pub fn compute(tensor: &MixTensor, geometry: Option<&GeometryData>, config: &PhysicsConfig) -> IndustrialResult {
        let start = instant::Instant::now();

        // 1. Calculate W/C ratio using calibrated k_scm
        let w_c = tensor.water_cement_ratio_calibrated(config.k_scm);

        // 2. Packing (Calculated Dynamic with Geometry)
        let mut packing_fraction = Self::compute_packing_density(tensor);

        // Apply geometry corrections if available
        if let Some(geo) = geometry {
            // Geometry affects packing efficiency
            let geo_packing_factor = geo.packing_efficiency();
            packing_fraction *= geo_packing_factor.clamp(0.8, 1.2); // Reasonable bounds
        }

        let total_mass = tensor.total_mass();

        // 3. Rheology (Rust engine)
        let rheology = if config.enable_rheology {
            RheologyEngine::compute(tensor, packing_fraction)
        } else {
            crate::science::rheology::RheologyResult {
                yield_stress: 0.0,
                viscosity: 0.0,
                slump_flow: 0.0,
            }
        };

        // 4. Strength (Rust engine) - Now with proper maturity/age handling
        // Use compute_strength_with_maturity for age-dependent strength
        let scm_ratio = tensor.scm_ratio();
        let age_days = 28.0; // Default to 28 days - MixTensor doesn't store age
        let temp_c = 20.0;   // Standard curing temperature
        
        // Compute age-dependent hydration degree using Parrot's equation
        let alpha = Self::compute_hydration_degree(age_days, temp_c, scm_ratio);
        let air = Self::estimate_air_content(tensor);
        
        // CALIBRATION: Use config.s_intrinsic (default 80.0 for D1, dataset-specific otherwise)
        let strength = if config.enable_strength {
            StrengthEngine::compute_powers(w_c, alpha, air, config.s_intrinsic)
        } else {
            crate::science::strength::StrengthResult {
                compressive_strength: 0.0,
                gel_space_ratio: 0.0,
                predicted_class: "None".to_string(),
            }
        };

        // 5. Sustainability (Rust engine)
        let sustainability = if config.enable_sustainability {
            SustainabilityEngine::compute_impact(tensor)
        } else {
            crate::science::sustainability::SustainabilityResult {
                score: 0.0,
                gwp_total: 0.0,
                energy_total: 0.0,
            }
        };

        // 6. Porosity (Rust engine)
        let porosity = PorosityEngine::compute_kozeny_carman(1.0 - packing_fraction, 300.0, 2.0);

        // 7. Derived calculations
        let f28 = strength.compressive_strength;
        let ft = 0.3 * f28.powf(0.67); // Split tensile (EC2)

        // 8. Fracture (Rust engine)
        let k_ic_est = Self::estimate_fracture_toughness(f28);
        let fracture = if config.enable_mechanics {
            FractureEngine::compute_lefm(f28 * 0.1, 2.0, k_ic_est)
        } else {
            crate::science::fracture::FractureResult {
                k_ic: 0.0,
                critical_crack_len: 0.0,
                failure_mode: "Unknown".to_string(),
            }
        };
        let k_ic = fracture.k_ic;

        // 9. Thermal (Rust engine)
        let scm_ratio = tensor.scm_ratio();
        let ea = Self::estimate_activation_energy(scm_ratio);
        let thermo = if config.enable_thermo {
            ThermoEngine::compute_heat_rate(20.0, alpha, ea)
        } else {
            crate::science::thermo::ThermoResult {
                heat_rate: 0.0,
                adiabatic_temp_rise: 0.0,
            }
        };
        let adiabatic_rise = thermo.adiabatic_temp_rise;

        // 10. Transport (Rust engine)
        let pore_radius = Self::estimate_mean_pore_radius(w_c);
        let transport_result = if config.enable_transport {
            TransportEngine::compute_sorptivity(pore_radius, 0.072, 0.001)
        } else {
            crate::science::transport::TransportResult {
                sorptivity: 0.0,
                diffusion_coeff: 0.0,
            }
        };

        // 11. Colloidal (Rust engine)
        let zeta = Self::estimate_zeta_potential(scm_ratio);
        let separation = Self::estimate_interparticle_distance(packing_fraction);
        let colloidal = if config.enable_colloidal {
            ColloidalEngine::compute_potential(separation, zeta, 0.5)
        } else {
            crate::science::colloidal::DLVOResult {
                potential_energy: 0.0,
                force: 0.0,
                stability: "Unknown".to_string(),
            }
        };

        // 12. ITZ (Rust engine)
        let agg_size_mm = Self::estimate_mean_aggregate_size(tensor);
        let itz = if config.enable_itz {
            ITZEngine::compute_properties(agg_size_mm, w_c)
        } else {
            crate::science::itz::ITZResult {
                thickness_microns: 0.0,
                porosity_itz: 0.0,
            }
        };

        // 13. Cost (Rust engine)
        let cost = if config.enable_cost {
            crate::science::cost::CostEngine::compute(tensor)
        } else {
            crate::science::cost::CostResult {
                total_cost: 0.0,
                cost_per_m3: 0.0,
            }
        };

        // 14. Chemo-Water (Rust engine)
        let calculated_porosity = 1.0 - packing_fraction;
        let chemo_water = crate::science::chemo_water::ChemoWaterEngine::compute_diffusivity(
            w_c,
            calculated_porosity,
        );

        // 15. Maturity (Full Engine Integration)
        let maturity_index = if config.enable_maturity {
            // Simulate standard curing: 20C for 28 days (672 hours)
            let scm_ratio = tensor.scm_ratio();
            let ea = Self::estimate_activation_energy(scm_ratio);
            let maturity_engine = crate::science::maturity::MaturityEngine::new(Some(ea as f64));

            // Generate a 28-day temperature profile (flat 20C)
            // 24 points (1 per hour roughly? let's do 28 points, 1 per day)
            // Interval = 24.0 hours
            let temp_history = vec![20.0; 28];
            let mat_result = maturity_engine.calculate(&temp_history, 24.0);
            mat_result.equivalent_age
        } else {
            Self::estimate_maturity_index(20.0, 28.0, ea).into()
        };

        // 15. Build Result
        IndustrialResult {
            fresh: FreshProperties {
                slump_flow: rheology.slump_flow,
                yield_stress: rheology.yield_stress,
                plastic_viscosity: rheology.viscosity,
                thixotropy_index: rheology.yield_stress * 0.1,
            },
            hardened: HardenedProperties {
                f28_compressive: f28,
                maturity_index: maturity_index as f32,
                e_modulus: 22.0 * (f28 / 10.0).powf(0.3),
                creep_coefficient: Self::estimate_creep_coefficient(f28),
                hydration_degree: alpha,
            },
            durability: DurabilityProperties {
                chloride_diffusivity: 10.0 * (w_c).powf(3.0),
                sulfate_resistance: Self::estimate_sulfate_resistance(w_c),
                asr_risk: 0.1,
            },
            sustainability: SustainabilityProperties {
                co2_kg_m3: sustainability.gwp_total,
                embodied_energy: sustainability.gwp_total * 5.0,
                lca_score: sustainability.score,
            },
            packing: PackingProperties {
                density: total_mass,
                voids: 1.0 - packing_fraction,
            },
            mechanics: MechanicsProperties {
                fracture_toughness: k_ic,
                split_tensile: ft,
            },
            thermal: ThermalProperties {
                adiabatic_rise,
                heat_of_hydration: thermo.heat_rate / 1000.0,
            },
            transport: TransportProperties {
                sorptivity: transport_result.sorptivity,
                permeability: porosity.permeability,
            },
            chemical: ChemicalProperties {
                ph_pore_solution: Self::estimate_ph(scm_ratio),
                mineralogy: vec![
                    "CSH".to_string(),
                    "CH".to_string(),
                    "Ettringite".to_string(),
                ],
                diffusivity: chemo_water.diffusivity,
                suction: chemo_water.suction,
            },
            economics: EconomicsProperties {
                total_cost: cost.total_cost,
                cost_per_m3: cost.cost_per_m3,
            },
            colloidal: ColloidalProperties {
                zeta_potential: -15.0,
                interparticle_distance: colloidal.force.abs(),
            },
            itz: ITZProperties {
                thickness: itz.thickness_microns,
                porosity: itz.porosity_itz,
            },
            compute_time_ms: start.elapsed().as_secs_f32() * 1000.0,
        }
    }
}

/// Helpers (WASM-Compatible)
#[wasm_bindgen]
impl PhysicsKernel {
    /// Calculate Packing Density (phi_max) based on material properties
    /// Uses a simplified model considering Shape Factor and Polydispersity
    pub fn compute_packing_density(mix: &MixTensor) -> f32 {
        let data = mix.data();
        let stride = 8; // MUST match MixTensor definition
        let mut total_solid_vol = 0.0;

        // Weighted accumulators
        let mut weighted_shape = 0.0;

        // Check for polydispersity (Cement + Aggregates?)
        let mut has_cement = false;
        let mut has_aggregates = false;

        // Calculate Solid Volume and properties
        for i in (0..data.len()).step_by(stride) {
            let mass = data[i];
            let sg = data[i + 1];
            let type_id = data[i + 2] as u8;
            // blaine = data[i+5], fm = data[i+6], shape = data[i+7]

            // Ignore Water (Type 2), Admixture (3), Air (Type 4) for SOLID skeleton packing
            if type_id != MaterialType::Water as u8
                && type_id != MaterialType::Air as u8
                && type_id != MaterialType::Admixture as u8
                && sg > 0.0
            {
                let vol = mass / (sg * 1000.0);
                total_solid_vol += vol;

                let shape = data[i + 7]; // 0.0-1.0 (1.0 = sphere)
                                         // Default shape if 0?
                let effective_shape = if shape > 0.01 { shape } else { 0.5 };

                weighted_shape += effective_shape * vol;

                if type_id == MaterialType::Cement as u8 || type_id == MaterialType::SCM as u8 {
                    has_cement = true;
                }
                if type_id == MaterialType::Aggregate as u8 {
                    has_aggregates = true;
                }
            }
        }

        if total_solid_vol <= 0.0001 {
            return 0.64; // Default Random Close Packing if no solids
        }

        let avg_shape = weighted_shape / total_solid_vol;

        // Base Packing Model (Linear fit between Angular and Spherical)
        // Shape 0.5 (Angular) -> ~0.55
        // Shape 1.0 (Sphere) -> ~0.64 (RCP Monodisperse)
        let mut phi_max = 0.46 + 0.18 * avg_shape;

        // Polydispersity Bonus (Gap grading)
        // If we have both Fines (Cement) and Coarse (Aggregates), packing improves significantly.
        // Heuristic: +0.12 for binary mix
        if has_cement && has_aggregates {
            phi_max += 0.12;
        }

        // Clamp to realistic values for Concrete (0.50 - 0.85)
        phi_max.max(0.50).min(0.85)
    }

    /// [HELPER] Estimate Air Content from Tensor
    pub fn estimate_air_content(mix: &MixTensor) -> f32 {
        let data = mix.data();
        let stride = 8;
        let mut air_vol = 0.0;
        let mut total_vol = 0.0;

        for i in (0..data.len()).step_by(stride) {
            let mass = data[i];
            let sg = data[i + 1];
            let type_id = data[i + 2] as u8;

            if sg > 0.0 {
                let vol = mass / (sg * 1000.0);
                total_vol += vol;
                if type_id == MaterialType::Air as u8 {
                    air_vol += vol;
                }
            }
        }

        if total_vol > 0.0 {
            let content = air_vol / total_vol;
            // If implicit 0, assume entrapped air 1.5% - 2.0%
            if content < 0.005 {
                0.02
            } else {
                content
            }
        } else {
            0.02
        }
    }

    /// [HELPER] Estimate Fracture Toughness (K_Ic) from Compressive Strength
    /// Empirical correlation: K_Ic ~ 0.05 * f_c^0.75 (approx)
    pub fn estimate_fracture_toughness(fc_mpa: f32) -> f32 {
        // Range: 0.5 (Weak) to 2.0 (HPC)
        let k = 0.15 * fc_mpa.powf(0.5);
        k.max(0.4).min(2.5)
    }

    /// [HELPER] Estimate Activation Energy based on SCMs
    /// OPC ~ 40-45 kJ/mol. Slag/FlyAsh ~ 50-60 kJ/mol (Slower reaction)
    pub fn estimate_activation_energy(scm_ratio: f32) -> f32 {
        // Linear interp
        // 0% SCM -> 40,000
        // 50% SCM -> 55,000
        40_000.0 + (scm_ratio * 30_000.0)
    }

    /// [HELPER] Estimate Pore Radius based on W/C
    /// Higher W/C -> Larger Capillary Pores
    pub fn estimate_mean_pore_radius(wc: f32) -> f32 {
        // 0.3 wc -> 20 nm
        // 0.6 wc -> 100 nm
        // Exponential growth
        // r = 10 * exp(3.5 * wc)
        let r = 10.0 * (3.5 * wc).exp();
        r.max(5.0).min(500.0)
    }

    /// [HELPER] Estimate Mean Aggregate Size (D50) from FM
    /// FM 2.0 (Sand) -> 0.5mm
    /// FM 7.0 (Coarse) -> 20mm
    /// Simple exponential mapping
    pub fn estimate_mean_aggregate_size(mix: &MixTensor) -> f32 {
        let data = mix.data();
        let stride = 8;
        let mut weighted_fm = 0.0;
        let mut total_agg_vol = 0.0;

        for i in (0..data.len()).step_by(stride) {
            let mass = data[i];
            let sg = data[i + 1];
            let type_id = data[i + 2] as u8;
            let fm = data[i + 6];

            if type_id == MaterialType::Aggregate as u8 && sg > 0.0 {
                let vol = mass / (sg * 1000.0);
                total_agg_vol += vol;
                weighted_fm += fm * vol;
            }
        }

        if total_agg_vol > 0.0 {
            let avg_fm = weighted_fm / total_agg_vol;
            // Mapping FM to mm (Empirical)
            // FM 2 => 0.5mm
            // FM 3 => 1.5mm
            // FM 7 => 20mm
            // d = 0.1 * e^(0.75 * FM)
            0.1 * (0.75 * avg_fm).exp()
        } else {
            5.0 // Default 5mm if no aggregates
        }
    }

    /// [HELPER] Estimate Zeta Potential from SCM Ratio
    /// OPC = -15mV (Typ), SCMs make it less negative (closer to 0) or more negative depending on type.
    /// Assuming Silica Fume/Fly Ash makes it slightly more stable (more negative) usually?
    /// Actually, let's assume SCMs dilute the surface charge density or bridge.
    /// Model: OPC (-20) -> High SCM (-10)
    pub fn estimate_zeta_potential(scm_ratio: f32) -> f32 {
        -20.0 + (scm_ratio * 10.0)
    }

    /// [HELPER] Estimate Interparticle Distance from Packing
    /// Denser packing = particles closer
    /// h = D * ((phi_max/phi)^(1/3) - 1)
    /// Simplified: h ~ 50nm * (1-phi)/phi
    pub fn estimate_interparticle_distance(packing: f32) -> f32 {
        let phi = packing.max(0.01);
        100.0 * (1.0 - phi) / phi
    }

    /// [HELPER] Estimate Maturity Index (deg-hours)
    /// Simple Nurse-Saul: Sum(T - T0) * dt
    /// Here T=20C, T0=-10C, t=28days
    /// Note: Does not currently use Ea, but could for Equivalent Age.
    /// Let's use Equivalent Age at 20C.
    fn estimate_maturity_index(temp_c: f32, age_days: f32, _ea: f32) -> f32 {
        let datum = -10.0;
        let maturity = (temp_c - datum) * age_days * 24.0;
        maturity
    }

    /// [HELPER] Estimate Creep Coefficient
    /// Eurocode approximation: phi(t)
    /// Stronger concrete (lower W/C) creeps less.
    fn estimate_creep_coefficient(fc: f32) -> f32 {
        // Range: 1.5 (High Strength) to 3.5 (Low Strength)
        // 50MPa -> 2.0
        // 25MPa -> 2.5
        // phi = 3.0 - 0.02 * fc
        let phi = 3.5 - 0.03 * fc;
        phi.max(1.0).min(4.0)
    }

    /// [HELPER] Estimate pH
    /// CH (pH 12.5) -> SCM consumes CH -> CSH (pH 10-11 eventual, but pore solution usually > 12.5)
    /// Actually pore solution is dominated by Alkalis (NaOH, KOH) -> pH 13+
    /// SCMs might reduce alkalis or consume OH-.
    fn estimate_ph(scm_ratio: f32) -> f32 {
        // OPC: 13.5
        // High SCM: 12.8
        13.5 - (scm_ratio * 0.7)
    }

    /// [HELPER] Estimate Sulfate Resistance
    /// Function of Permeability (W/C) and Chemistry (C3A - ignored here)
    /// Resistance Index proportional to impermeability.
    fn estimate_sulfate_resistance(wc: f32) -> f32 {
        // High resistance for low W/C
        // R = 10 * exp(-2 * wc)
        // wc 0.4 -> 10 * 0.45 = 4.5
        // wc 0.3 -> 10 * 0.55 = 5.5
        // Let's adjust scale to match previous 5.0-8.0 range
        // 12.0 * exp(-1.5 * wc)
        // wc 0.4 -> 12 * 0.55 = 6.6
        // wc 0.5 -> 12 * 0.47 = 5.6
        12.0 * (-1.5 * wc).exp()
    }

    /// [CALIBRATED] Compute hydration degree from age using Parrot's equation
    ///
    /// alpha(t) = alpha_max * (1 - exp(-k * sqrt(t)))
    /// Where:
    /// - alpha_max: Ultimate hydration degree (0.9-1.0 for OPC, lower for high SCM)
    /// - k: Reaction rate constant (calibrated per cement type)
    /// - t: Age in days
    ///
    /// Temperature effect via Arrhenius:
    /// k(T) = k_ref * exp((E/R) * (1/T_ref - 1/T))
    /// Computes the degree of cement hydration using the Avrami-Parrott model.
    ///
    /// This function implements the modified Avrami kinetics for cement hydration,
    /// accounting for temperature effects, supplementary cementitious materials,
    /// and curing time. The result is a normalized hydration degree between 0 and 1.
    ///
    /// # Arguments
    /// * `age_days` - Curing age in days (typically 1-365)
    /// * `temp_c` - Curing temperature in Celsius (typically 10-40°C)
    /// * `scm_ratio` - Supplementary cementitious material ratio (0-1)
    ///
    /// # Returns
    /// Hydration degree as a float between 0.0 (no hydration) and 1.0 (complete hydration)
    ///
    /// # Physics
    /// Based on Avrami-Parrott equation: dα/dt = k·(1-α)^n where α is hydration degree
    pub fn compute_hydration_degree(age_days: f32, temp_c: f32, scm_ratio: f32) -> f32 {
        // Ultimate hydration (SCMs may not fully hydrate)
        let alpha_max = 0.95 - scm_ratio * 0.15; // OPC=0.95, 50% SCM=0.875

        // Base reaction rate at 20°C (calibrated for typical OPC)
        let k_ref = 0.55; // Calibrated constant

        // Temperature adjustment (Arrhenius)
        let t_ref_k = 293.15; // 20°C
        let t_k = temp_c + 273.15;
        let e_over_r = 5000.0; // E/R for cement ~5000K (E~41.5 kJ/mol)
        let temp_factor = (e_over_r * (1.0 / t_ref_k - 1.0 / t_k)).exp();

        // SCM adjustment (slows reaction)
        let scm_factor = 1.0 - scm_ratio * 0.4; // 50% SCM -> 0.8x rate

        let k = k_ref * temp_factor * scm_factor;

        // Parrot's equation
        let alpha = alpha_max * (1.0 - (-k * age_days.sqrt()).exp());

        alpha.max(0.0).min(1.0)
    }

    /// [CALIBRATED] Compute strength with full maturity model
    ///
    /// Uses Powers gel-space ratio with age-dependent hydration degree.
    /// This is the "Constitutional Evolution" approach - physics over time.
    ///
    /// Arguments:
    /// - w_c: Water-cement ratio
    /// - age_days: Curing age
    /// - temp_c: Curing temperature (°C)
    /// - scm_ratio: SCM fraction of binder
    /// - k_slag: Slag efficiency factor
    /// - k_fly_ash: Fly ash efficiency factor
    /// - s_intrinsic: Intrinsic gel strength (calibrated, typically 120-180 MPa)
    pub fn compute_strength_with_maturity(
        w_c: f32,
        age_days: f32,
        temp_c: f32,
        scm_ratio: f32,
        s_intrinsic: f32,
    ) -> f32 {
        // 1. Compute hydration degree from age
        let alpha = Self::compute_hydration_degree(age_days, temp_c, scm_ratio);

        // 2. Powers gel-space ratio
        // x = 0.68 * alpha / (0.32 * alpha + w_c)
        let vg = 0.68 * alpha;
        let vc = w_c - 0.36 * alpha;
        let air = 0.02; // Assume 2% entrapped air
        let space = vg + vc.max(0.0) + air;

        if space <= 0.001 {
            return 0.0;
        }

        let x = vg / space; // Gel-space ratio

        // 3. Powers strength
        let fc = s_intrinsic * x.powi(3);

        fc.max(0.0).min(150.0)
    }
}
