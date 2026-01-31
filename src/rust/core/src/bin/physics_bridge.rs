// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto

//! DUMSTO Physics Bridge
//! Exports calibrated physics predictions and internal state variables to CSV.
//! Used by Python Hybrid Plugin.

use serde::Serialize;
use serde_json::json;
use std::env;
use std::fs::File;
use std::io::{self, BufRead};
use umst_core::physics_kernel::PhysicsKernel;
use umst_core::rl::{PPOAgent, PPOConfig, RLState, RewardType};
use umst_core::science::thermodynamic_filter::{ThermodynamicFilter, ThermodynamicState};

// --- Data Record ---
#[derive(Clone, Debug)]
struct Record {
    cement: f32,
    slag: f32,
    fly_ash: f32,
    water: f32,
    age: f32,
    strength: f32,
}

// --- Output Record (CSV) ---
#[derive(Serialize)]
struct BridgeOutput {
    cement: f32,
    slag: f32,
    fly_ash: f32,
    water: f32,
    age: f32,
    f_physics: f32,       // Prediction
    f_agent: f32,         // Agent Prediction
    alpha: f32,           // Hydration degree
    gel_space_ratio: f32, // Internal state
    is_admissible: bool,
    y_true: f32,
}

// --- Calibration per Dataset ---
#[derive(Clone)]
struct Calibration {
    s_intrinsic: f32,
    k_slag: f32,
    k_fly_ash: f32,
    k_ref: f32,
    early_boost: f32,
}

fn get_calibration(dataset: &str) -> Calibration {
    match dataset {
        "D1" => Calibration {
            s_intrinsic: 79.98,
            k_slag: 1.18,
            k_fly_ash: 1.15,
            k_ref: 0.46,
            early_boost: 1.26,
        },
        "D2" => Calibration {
            s_intrinsic: 50.69,
            k_slag: 0.20,
            k_fly_ash: 0.22,
            k_ref: 0.71,
            early_boost: 1.19,
        },
        "D3" => Calibration {
            s_intrinsic: 59.80,
            k_slag: 0.20,
            k_fly_ash: 0.20,
            k_ref: 0.48,
            early_boost: 1.90,
        },
        "D4" => Calibration {
            s_intrinsic: 79.59,
            k_slag: 0.20,
            k_fly_ash: 0.20,
            k_ref: 0.68,
            early_boost: 1.15,
        },
        "full" => Calibration {
            s_intrinsic: 60.94,
            k_slag: 1.26,
            k_fly_ash: 1.31,
            k_ref: 0.69,
            early_boost: 1.11,
        },
        _ => Calibration {
            s_intrinsic: 80.0,
            k_slag: 0.6,
            k_fly_ash: 0.4,
            k_ref: 0.55,
            early_boost: 1.0,
        },
    }
}

/// Check Clausius-Duhem admissibility via curing trajectory validation.
/// Validates that the mix's curing trajectory (day 0→7→14→21→28) satisfies
/// D_int = -ρ·ψ̇ ≥ 0 at every step.
fn check_curing_admissibility(rec: &Record, cal: &Calibration) -> bool {
    let binder = rec.cement + rec.slag + rec.fly_ash;
    if binder <= 0.0 { return false; }

    let effective_cement = rec.cement + cal.k_slag * rec.slag + cal.k_fly_ash * rec.fly_ash;
    if effective_cement <= 0.0 { return false; }

    let w_c = (rec.water / effective_cement).max(0.25).min(1.0) as f64;
    let scm_ratio = (rec.slag + rec.fly_ash) / binder;
    let s_int = cal.s_intrinsic as f64;

    let mut filter = ThermodynamicFilter::new();
    let curing_days: &[f32] = &[0.0, 7.0, 14.0, 21.0, 28.0];

    for pair in curing_days.windows(2) {
        let t_old = pair[0];
        let t_new = pair[1];
        let dt_seconds = ((t_new - t_old) * 86400.0) as f64;

        // Compute hydration degree at each curing step using calibrated k_ref
        let alpha_old = compute_hydration_degree_local(t_old, scm_ratio, cal.k_ref) as f64;
        let alpha_new = compute_hydration_degree_local(t_new, scm_ratio, cal.k_ref) as f64;

        let state_old = ThermodynamicState::from_mix_calibrated(w_c, alpha_old, 293.0, s_int);
        let state_new = ThermodynamicState::from_mix_calibrated(w_c, alpha_new, 293.0, s_int);

        let result = filter.check_transition(&state_old, &state_new, dt_seconds);
        if !result.accepted {
            return false;
        }
    }
    true
}

/// Local hydration degree computation matching physics_bridge calibration.
fn compute_hydration_degree_local(age: f32, scm_ratio: f32, k_ref: f32) -> f32 {
    let alpha_max = 0.95 - scm_ratio * 0.15;
    let scm_factor = 1.0 - scm_ratio * 0.4;
    let k = k_ref * scm_factor; // temp_factor = 1.0 at 20°C
    let alpha = alpha_max * (1.0 - (-k * age.sqrt()).exp());
    alpha.max(0.0).min(1.0)
}

// --- CSV Loading ---
fn load_csv(path: &str) -> Vec<Record> {
    let mut records = Vec::new();
    if let Ok(file) = File::open(path) {
        let lines = io::BufReader::new(file).lines();
        for (i, line) in lines.enumerate() {
            if i == 0 {
                continue;
            }
            if let Ok(l) = line {
                let cols: Vec<&str> = l.split(',').collect();
                if cols.len() < 9 {
                    continue;
                }
                records.push(Record {
                    cement: cols[0].parse().unwrap_or(0.0),
                    slag: cols[1].parse().unwrap_or(0.0),
                    fly_ash: cols[2].parse().unwrap_or(0.0),
                    water: cols[3].parse().unwrap_or(0.0),
                    age: cols[7].parse().unwrap_or(28.0),
                    strength: cols[8].parse().unwrap_or(0.0),
                });
            }
        }
    }
    records
}

// We implement strict internal calculation here to expose internals (Alpha, GelSpace)
// which are not returned by the industrial API.
fn compute_internal(rec: &Record, cal: &Calibration) -> (f32, f32, f32) {
    let binder = rec.cement + rec.slag + rec.fly_ash;
    if binder <= 0.0 {
        return (0.0, 0.0, 0.0);
    }

    let effective_cement = rec.cement + cal.k_slag * rec.slag + cal.k_fly_ash * rec.fly_ash;
    if effective_cement <= 0.0 {
        return (0.0, 0.0, 0.0);
    }

    let w_c = (rec.water / effective_cement).max(0.25).min(1.0);
    let scm_ratio = (rec.slag + rec.fly_ash) / binder;

    // Hydration
    let _alpha = PhysicsKernel::compute_hydration_degree(rec.age, 20.0, scm_ratio) * cal.k_ref; // Apply k_ref scaling linearly for calibration
                                                                                                // NOTE: PhysicsKernel::compute_hydration_degree uses a hardcoded k_ref=0.55 inside.
                                                                                                // To respect our Dataset Calibration, we need to replicate the logic or adjust.
                                                                                                // Replicating logic here for transparency and calibration access:

    let alpha_max = 0.95 - scm_ratio * 0.15;
    let t_ref_k: f32 = 293.15;
    let t_k: f32 = 293.15; // 20C
    let e_over_r: f32 = 5000.0;
    let temp_factor = (e_over_r * (1.0 / t_ref_k - 1.0 / t_k)).exp(); // = 1.0
    let scm_factor = 1.0 - scm_ratio * 0.4;
    let k = cal.k_ref * temp_factor * scm_factor;

    let alpha_calibrated = alpha_max * (1.0 - (-k * rec.age.sqrt()).exp());
    let alpha_final = alpha_calibrated.max(0.0).min(1.0);

    // Powers
    let vg = 0.68 * alpha_final;
    let vc = w_c - 0.36 * alpha_final;
    let space = vg + vc.max(0.0) + 0.02;

    if space <= 0.001 {
        return (0.0, alpha_final, 0.0);
    }

    let x = vg / space;
    let mut fc = cal.s_intrinsic * x.powi(3);

    if rec.age < 7.0 {
        fc *= cal.early_boost;
    }

    (fc.max(0.0).min(150.0), alpha_final, x)
}

fn compute_agent_strength(rec: &Record, cal: &Calibration, agent: &mut PPOAgent) -> f32 {
    let binder = rec.cement + rec.slag + rec.fly_ash;
    let scm_ratio = if binder > 0.0 {
        (rec.slag + rec.fly_ash) / binder
    } else {
        0.0
    };
    let w_c = if binder > 0.0 {
        rec.water / binder
    } else {
        0.5
    };

    let mut state = RLState::new();
    state.set_proxy(0, (rec.cement / 500.0) as f64);
    state.set_proxy(1, (rec.slag / 300.0) as f64);
    state.set_proxy(2, (rec.fly_ash / 200.0) as f64);
    state.set_proxy(3, w_c as f64);
    state.set_proxy(4, scm_ratio as f64);
    state.set_proxy(5, (rec.age / 365.0) as f64);
    state.set_proxy(6, (0.0 / 20.0) as f64); // No SP in standard record, assume 0 or infer? data has no SP col?
                                             // Wait, standard Record in bridge doesnt have SP. Original ssot_benchmark used a richer Record.
                                             // The bridge output is for Hybrid training, which didn't use SP.
                                             // But PPO expects it. Let's use 0.0 default.
    state.set_proxy(7, (1000.0 / 1200.0) as f64); // Default Coarse
    state.set_proxy(8, (800.0 / 900.0) as f64); // Default Fine
    state.set_proxy(9, (rec.water / 250.0) as f64);

    state.set_proxy(10, (cal.s_intrinsic / 100.0) as f64);
    state.set_proxy(11, cal.k_slag as f64);
    state.set_proxy(12, cal.k_fly_ash as f64);
    state.set_proxy(13, cal.k_ref as f64);
    state.set_proxy(14, (cal.early_boost - 1.0) as f64);
    state.temperature = 20.0;
    state.humidity = 0.5;

    // Physics Base (Same Calc)
    let (f_phys, _, _) = compute_internal(rec, cal);
    state.set_proxy(15, (f_phys / 100.0) as f64);
    state.fracture_kic = 1.5;
    state.diffusivity = 0.001;

    let action = agent.select_action(&state);

    let correction = action.delta_wc * 3.0 * (0.5 - w_c).signum() as f64
        + action.delta_scms * 2.0 * (scm_ratio as f64 - 0.2)
        + action.delta_sp * 1.5;

    let ppo_pred = f_phys + (correction as f32).clamp(-5.0, 5.0);
    ppo_pred.max(0.0).min(150.0)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut csv_path = "";
    let mut dataset_id = "D1";
    let mut output_path = "bridge_output.csv";

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--csv" => {
                i += 1;
                if i < args.len() {
                    csv_path = &args[i];
                }
            }
            "--dataset" => {
                i += 1;
                if i < args.len() {
                    dataset_id = &args[i];
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    output_path = &args[i];
                }
            }
            _ => {}
        }
        i += 1;
    }

    if csv_path.is_empty() {
        println!("{}", json!({"error": "No CSV"}));
        return;
    }

    let records = load_csv(csv_path);
    let cal = get_calibration(dataset_id);
    let config = PPOConfig::new();
    let mut agent = PPOAgent::new(config, RewardType::Balanced);

    let mut wtr = csv::Writer::from_path(output_path).expect("Failed to open output csv");

    // Header is handled automatically by serialize

    let mut total_duration_ns = 0u128;
    let mut count = 0;

    for rec in records {
        let start = std::time::Instant::now();
        let (f_phys, alpha, x) = compute_internal(&rec, &cal);
        // We only time the physics/internal compute as this is the kernel benchmark
        total_duration_ns += start.elapsed().as_nanos();
        count += 1;

        // Agent calculation (separate if needed, but for now we focus on physics kernel core)
        let f_agent = compute_agent_strength(&rec, &cal, &mut agent);

        // Admissibility: Clausius-Duhem curing trajectory validation
        let is_admissible = check_curing_admissibility(&rec, &cal);

        wtr.serialize(BridgeOutput {
            cement: rec.cement,
            slag: rec.slag,
            fly_ash: rec.fly_ash,
            water: rec.water,
            age: rec.age,
            f_physics: f_phys,
            f_agent: f_agent,
            alpha: alpha,
            gel_space_ratio: x,
            is_admissible: is_admissible,
            y_true: rec.strength,
        })
        .unwrap();
    }

    if count > 0 {
        let avg_ns = total_duration_ns as f64 / count as f64;
        println!("KERNEL_LATENCY_NS:{:.4}", avg_ns);
    }

    wtr.flush().unwrap();
    println!("Bridge export complete: {}", output_path);
}
