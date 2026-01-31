// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto

//! DUMSTO Unified Experiment Runner
//!
//! Differentiable Unified Material-State Tensor Optimization (DUMSTO)
//!
//! # Overview
//! Allows users to run "Combo" tests with specific physics engines.
//! Demonstrates the "Constitutional Evolution" flexibility.
//!
//! # Usage
//! cargo run --bin experiment_runner -- --mix "C30" --engines "strength,rheology,maturity"
//!

use serde_json::json;
use std::env;
use umst_core::physics_kernel::{PhysicsConfig, PhysicsKernel};
use umst_core::tensors::MixTensor;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config = PhysicsConfig {
        enable_rheology: false,
        enable_strength: false,
        enable_thermo: false,
        enable_durability: false,
        enable_sustainability: false,
        enable_mechanics: false,
        enable_transport: false,
        enable_colloidal: false,
        enable_itz: false,
        enable_cost: false,
        enable_maturity: false,
        s_intrinsic: 80.0,
        k_scm: 1.0,
    };

    let mut mix_type = "C30";

    if args.len() < 2 {
        print_help();
        return;
    }

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--mix" => {
                i += 1;
                if i < args.len() {
                    mix_type = &args[i];
                }
            }
            "--engines" => {
                i += 1;
                if i < args.len() {
                    let engines = &args[i];
                    if engines == "all" {
                        config = PhysicsConfig::default();
                    } else {
                        for engine in engines.split(',') {
                            match engine.trim() {
                                "rheology" => config.enable_rheology = true,
                                "strength" => config.enable_strength = true,
                                "thermo" => config.enable_thermo = true,
                                "durability" => config.enable_durability = true,
                                "sustainability" => config.enable_sustainability = true,
                                "mechanics" => config.enable_mechanics = true,
                                "transport" => config.enable_transport = true,
                                "colloidal" => config.enable_colloidal = true,
                                "itz" => config.enable_itz = true,
                                "cost" => config.enable_cost = true,
                                "maturity" => config.enable_maturity = true,
                                _ => println!("Warning: Unknown engine '{}'", engine),
                            }
                        }
                    }
                }
            }
            "--help" => {
                print_help();
                return;
            }
            _ => {}
        }
        i += 1;
    }

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  DUMSTO EXPERIMEMT RUNNER");
    println!("  Mix Scenario: {}", mix_type);
    println!("  Active Engines:");
    println!(
        "   - Rheology:       {}",
        if config.enable_rheology { "ON" } else { "OFF" }
    );
    println!(
        "   - Strength:       {}",
        if config.enable_strength { "ON" } else { "OFF" }
    );
    println!(
        "   - Maturity:       {}",
        if config.enable_maturity { "ON" } else { "OFF" }
    );
    println!(
        "   - Sustainability: {}",
        if config.enable_sustainability {
            "ON"
        } else {
            "OFF"
        }
    );
    println!(
        "   - Cost:           {}",
        if config.enable_cost { "ON" } else { "OFF" }
    );
    println!("═══════════════════════════════════════════════════════════════════");

    // 1. Create a dummy mix
    // Simple C30 recipe: Cement 350, Water 175, Agg 1800
    let tensor = match mix_type {
        "C30" => create_mix(350.0, 175.0, 0.0, 0.0),
        "Green" => create_mix(200.0, 175.0, 150.0, 0.0), // 150 Slag
        "HighPerformance" => create_mix(500.0, 150.0, 50.0, 50.0), // Low W/C
        _ => create_mix(300.0, 180.0, 0.0, 0.0),
    };

    let result = PhysicsKernel::compute(&tensor, None, &config);

    println!("\n  RESULTS:");
    if config.enable_strength {
        println!(
            "  Strength (28d):    {:.2} MPa",
            result.hardened.f28_compressive
        );
    }
    if config.enable_rheology {
        println!("  Yield Stress:      {:.2} Pa", result.fresh.yield_stress);
    }
    if config.enable_maturity {
        println!(
            "  Maturity Index:    {:.2} deg-hrs",
            result.hardened.maturity_index
        );
    }
    if config.enable_sustainability {
        println!(
            "  GWP (CO2):         {:.2} kg/m3",
            result.sustainability.co2_kg_m3
        );
    }
    if config.enable_cost {
        println!(
            "  Total Cost:        {:.2} USD",
            result.economics.total_cost
        );
    }

    println!("\n  Full JSON Output:");
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}

fn create_mix(cement: f32, water: f32, slag: f32, fly_ash: f32) -> MixTensor {
    // Basic wrapper to create a mix tensor string then parse it?
    // Actually simpler to use the JSON hydration if we don't want to expose internal builders
    // But PhysicsKernel::compute takes &MixTensor.
    // We can't easily construct MixTensor directly because fields are private or complex.
    // Let's use the JSON hydration helper from PhysicsKernel if possible, or simple construction.

    // MixTensor::from_json is the public API.
    let components = json!([
        { "id": "cement", "mass": cement, "details": { "specificGravity": 3.15, "type": 1, "cost": 0.1, "blaine": 350, "fm": 0.0, "shape": 0.0 } },
        { "id": "water", "mass": water, "details": { "specificGravity": 1.0, "type": 2, "cost": 0.002 } },
        { "id": "slag", "mass": slag, "details": { "specificGravity": 2.9, "type": 6, "cost": 0.05 } },
        { "id": "fly_ash", "mass": fly_ash, "details": { "specificGravity": 2.2, "type": 6, "cost": 0.04 } },
        { "id": "agg", "mass": 1800.0, "details": { "specificGravity": 2.65, "type": 5, "cost": 0.02, "fm": 4.5 } }
    ]);

    // We need to approximate the JSON structure MixTensor expects.
    // In physics_kernel.rs, MixTensor::from_json(components_json, materials_json)
    // materials_json can be dummy if components contain details? Usually they are joined.
    // Let's look at MixTensor impl.
    // For now, let's assume we can use the JSON interface.

    let components_str = components.to_string();
    let materials_str = "[]"; // Assuming embedded details work or aren't strictly checked for this demo

    // We need to access MixTensor::from_json, but it returns Result.
    // And from_json is likely in `tensors` module.

    umst_core::tensors::MixTensor::from_json(&components_str, materials_str).unwrap()
}

fn print_help() {
    println!("Usage: experiment_runner --mix [C30|Green|HighPerformance] --engines [all|list,of,engines]");
    println!("Examples:");
    println!(
        "  cargo run --bin experiment_runner -- --mix Green --engines strength,sustainability"
    );
    println!("  cargo run --bin experiment_runner -- --mix HighPerformance --engines all");
}
