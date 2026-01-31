// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
//
//! physics_compute — CLI binary for full 16-engine PhysicsKernel evaluation
//!
//! Takes a JSON mix design on stdin or as --json argument, runs ALL Rust science
//! engines (Rheology, Strength, Fracture, Transport, Thermal, ITZ, Colloidal,
//! Porosity, Sustainability, Cost, ChemoWater, Maturity), and outputs a JSON
//! object with ALL 17+ metrics.
//!
//! Usage:
//!   echo '{"cement":350,"slag":50,"fly_ash":0,"water":175,"sp":5,"coarse_agg":1000,"fine_agg":800}' | physics_compute
//!   physics_compute --json '{"cement":350,...}'
//!
//! Output: JSON with all IndustrialResult fields (fresh, hardened, durability,
//!         sustainability, mechanics, thermal, transport, chemical, economics,
//!         colloidal, itz) plus a flat reward_components object matching the
//!         Rust RewardComponents struct (17 fields).

use serde_json::{json, Value};
use std::env;
use std::io::{self, Read};

use umst_core::physics_kernel::{PhysicsConfig, PhysicsKernel};
use umst_core::tensors::MixTensor;

fn main() {
    // Parse input: either --json '...' or stdin
    let input = get_input();
    let mix_json: Value = match serde_json::from_str(&input) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ERROR: Invalid JSON input: {}", e);
            std::process::exit(1);
        }
    };

    // Extract mix components
    let cement = mix_json["cement"].as_f64().unwrap_or(350.0);
    let slag = mix_json["slag"].as_f64().unwrap_or(0.0);
    let fly_ash = mix_json["fly_ash"].as_f64().unwrap_or(0.0);
    let water = mix_json["water"].as_f64().unwrap_or(175.0);
    let sp = mix_json["sp"].as_f64().unwrap_or(5.0);
    let coarse_agg = mix_json["coarse_agg"].as_f64().unwrap_or(1000.0);
    let fine_agg = mix_json["fine_agg"].as_f64().unwrap_or(800.0);

    // Build MixTensor via JSON (same format as agent_design_benchmark.rs)
    let components_json = json!([
        {"materialId": "c",    "mass": cement},
        {"materialId": "s",    "mass": slag},
        {"materialId": "fa",   "mass": fly_ash},
        {"materialId": "w",    "mass": water},
        {"materialId": "sp",   "mass": sp},
        {"materialId": "ca",   "mass": coarse_agg},
        {"materialId": "fine", "mass": fine_agg}
    ])
    .to_string();

    // Full materials JSON with ecology (CO2), economy (cost), and physics properties
    // CO2 factors from sustainability.rs: Cement=0.85, SCM=0.10, Water=0.001, Admixture=1.5, Agg=0.005
    // Cost factors: Cement=0.12, Slag=0.06, FlyAsh=0.04, Water=0.001, SP=2.50, Agg=0.015/0.012
    let materials_json = r#"[
        {"id":"c","type":"Cement","density":3150,
         "ecology":{"embodiedCarbon":0.85},"economy":{"costPerKg":0.12},
         "properties":{"blaine":380,"shape":0.3}},
        {"id":"s","type":"SCM","density":2900,
         "ecology":{"embodiedCarbon":0.10},"economy":{"costPerKg":0.06},
         "properties":{"blaine":450,"shape":0.4}},
        {"id":"fa","type":"SCM","density":2300,
         "ecology":{"embodiedCarbon":0.05},"economy":{"costPerKg":0.04},
         "properties":{"blaine":350,"shape":0.8}},
        {"id":"w","type":"Water","density":1000,
         "ecology":{"embodiedCarbon":0.001},"economy":{"costPerKg":0.001}},
        {"id":"sp","type":"Admixture","density":1100,
         "ecology":{"embodiedCarbon":1.50},"economy":{"costPerKg":2.50}},
        {"id":"ca","type":"Aggregate","density":2650,
         "ecology":{"embodiedCarbon":0.005},"economy":{"costPerKg":0.015},
         "properties":{"fm":7.0,"shape":0.6}},
        {"id":"fine","type":"Aggregate","density":2600,
         "ecology":{"embodiedCarbon":0.005},"economy":{"costPerKg":0.012},
         "properties":{"fm":2.8,"shape":0.5}}
    ]"#;

    let tensor = match MixTensor::from_json(&components_json, materials_json) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ERROR: Failed to create MixTensor: {:?}", e);
            std::process::exit(1);
        }
    };

    // Run FULL PhysicsKernel (ALL 16 engines enabled)
    let config = PhysicsConfig::default();
    let result = PhysicsKernel::compute(&tensor, None, &config);

    // Build flat reward_components matching RewardComponents struct (17 fields)
    let reward_components = json!({
        "strength_fc": result.hardened.f28_compressive,
        "cost": result.economics.cost_per_m3,
        "co2": result.sustainability.co2_kg_m3,
        "fracture_kic": result.mechanics.fracture_toughness,
        "diffusivity": result.chemical.diffusivity,
        "damage": 0.0,  // Computed via state evolution, not single-shot
        "bond": result.mechanics.split_tensile,  // Proxy: split tensile ~ bond
        "yield_stress": result.fresh.yield_stress,
        "viscosity": result.fresh.plastic_viscosity,
        "slump_flow": result.fresh.slump_flow,
        "itz_thickness": result.itz.thickness,
        "itz_porosity": result.itz.porosity,
        "colloidal_potential": result.colloidal.zeta_potential,
        "heat_rate": result.thermal.heat_of_hydration,
        "temp_rise": result.thermal.adiabatic_rise,
        "permeability": result.transport.permeability,
        "suction": result.chemical.suction,
    });

    // Full output: industrial result + flat reward components
    let output = json!({
        "industrial": {
            "fresh": {
                "slump_flow": result.fresh.slump_flow,
                "yield_stress": result.fresh.yield_stress,
                "plastic_viscosity": result.fresh.plastic_viscosity,
                "thixotropy_index": result.fresh.thixotropy_index,
            },
            "hardened": {
                "f28_compressive": result.hardened.f28_compressive,
                "maturity_index": result.hardened.maturity_index,
                "e_modulus": result.hardened.e_modulus,
                "creep_coefficient": result.hardened.creep_coefficient,
            },
            "durability": {
                "chloride_diffusivity": result.durability.chloride_diffusivity,
                "sulfate_resistance": result.durability.sulfate_resistance,
                "asr_risk": result.durability.asr_risk,
            },
            "sustainability": {
                "co2_kg_m3": result.sustainability.co2_kg_m3,
                "embodied_energy": result.sustainability.embodied_energy,
                "lca_score": result.sustainability.lca_score,
            },
            "mechanics": {
                "fracture_toughness": result.mechanics.fracture_toughness,
                "split_tensile": result.mechanics.split_tensile,
            },
            "thermal": {
                "adiabatic_rise": result.thermal.adiabatic_rise,
                "heat_of_hydration": result.thermal.heat_of_hydration,
            },
            "transport": {
                "sorptivity": result.transport.sorptivity,
                "permeability": result.transport.permeability,
            },
            "economics": {
                "total_cost": result.economics.total_cost,
                "cost_per_m3": result.economics.cost_per_m3,
            },
            "colloidal": {
                "zeta_potential": result.colloidal.zeta_potential,
                "interparticle_distance": result.colloidal.interparticle_distance,
            },
            "itz": {
                "thickness": result.itz.thickness,
                "porosity": result.itz.porosity,
            },
            "chemical": {
                "ph": result.chemical.ph_pore_solution,
                "diffusivity": result.chemical.diffusivity,
                "suction": result.chemical.suction,
            },
        },
        "reward_components": reward_components,
        "compute_time_ms": result.compute_time_ms,
    });

    println!("{}", serde_json::to_string(&output).unwrap());
}

fn get_input() -> String {
    let args: Vec<String> = env::args().collect();

    // Check for --json argument
    for i in 0..args.len() {
        if args[i] == "--json" && i + 1 < args.len() {
            return args[i + 1].clone();
        }
    }

    // Otherwise read from stdin
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .expect("Failed to read stdin");
    input
}
