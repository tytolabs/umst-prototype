// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto

//! DUMSTO-Hybrid Benchmark - Full Dataset Evaluation
//!
//! Differentiable Unified Material-State Tensor Optimization (DUMSTO)
//!
//! # Overview
//! Runs on ALL samples for accurate comparison.
//! Evaluates the "Tensor Fidelity" hypothesis.
//!
//! # Architecture
//! - **DUMSTO-Physics**: Uses `PhysicsKernel` (Powers/Parrot models).
//! - **MC Ensemble**: Perturbs calibration parameters.
//! - **DUMSTO-Hybrid**: Weighted Physics (60%) + MC Median (40%).

use rand::Rng;
use serde_json::json;
use std::env;
use std::fs::File;
use std::io::{self, BufRead};
use std::time::Instant;

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
            s_intrinsic: 80.0,
            k_slag: 1.0,
            k_fly_ash: 1.0,
            k_ref: 0.55,
            early_boost: 1.2,
        },
        "D2" => Calibration {
            s_intrinsic: 60.0,
            k_slag: 0.2,
            k_fly_ash: 0.22,
            k_ref: 0.5,
            early_boost: 1.4,
        },
        "D3" => Calibration {
            s_intrinsic: 60.0,
            k_slag: 0.2,
            k_fly_ash: 0.2,
            k_ref: 0.5,
            early_boost: 1.6,
        },
        "D4" => Calibration {
            s_intrinsic: 81.0,
            k_slag: 0.2,
            k_fly_ash: 0.2,
            k_ref: 0.7,
            early_boost: 1.1,
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

// --- Hydration Degree (Parrot's Equation) ---
fn compute_hydration_degree(age: f32, temp_c: f32, scm_ratio: f32, k_ref: f32) -> f32 {
    let alpha_max = 0.95 - scm_ratio * 0.15;
    let t_ref_k: f32 = 293.15;
    let t_k = temp_c + 273.15;
    let e_over_r: f32 = 5000.0;
    let temp_factor = (e_over_r * (1.0 / t_ref_k - 1.0 / t_k)).exp();
    let scm_factor = 1.0 - scm_ratio * 0.4;
    let k = k_ref * temp_factor * scm_factor;
    let alpha = alpha_max * (1.0 - (-k * age.sqrt()).exp());
    alpha.max(0.0).min(1.0)
}

// --- Full Physics Strength ---
fn compute_strength(r: &Record, cal: &Calibration, temp_c: f32) -> f32 {
    let binder = r.cement + r.slag + r.fly_ash;
    if binder <= 0.0 {
        return 0.0;
    }

    let effective_cement = r.cement + cal.k_slag * r.slag + cal.k_fly_ash * r.fly_ash;
    if effective_cement <= 0.0 {
        return 0.0;
    }

    let w_c = (r.water / effective_cement).max(0.25).min(1.0);
    let scm_ratio = (r.slag + r.fly_ash) / binder;

    let alpha = compute_hydration_degree(r.age, temp_c, scm_ratio, cal.k_ref);

    let vg = 0.68 * alpha;
    let vc = w_c - 0.36 * alpha;
    let space = vg + vc.max(0.0) + 0.02;

    if space <= 0.001 {
        return 0.0;
    }

    let x = vg / space;
    let mut fc = cal.s_intrinsic * x.powi(3);

    if r.age < 7.0 {
        fc *= cal.early_boost;
    }

    fc.max(0.0).min(150.0)
}

// --- MC Sample Calibration ---
fn mc_sample_calibration(base: &Calibration, variance: f32) -> Calibration {
    let mut rng = rand::thread_rng();
    Calibration {
        s_intrinsic: base.s_intrinsic * (1.0 + rng.gen_range(-variance..variance)),
        k_slag: (base.k_slag * (1.0 + rng.gen_range(-variance..variance)))
            .max(0.1)
            .min(1.0),
        k_fly_ash: (base.k_fly_ash * (1.0 + rng.gen_range(-variance..variance)))
            .max(0.1)
            .min(1.0),
        k_ref: (base.k_ref * (1.0 + rng.gen_range(-variance..variance)))
            .max(0.3)
            .min(1.0),
        early_boost: (base.early_boost * (1.0 + rng.gen_range(-variance..variance)))
            .max(1.0)
            .min(2.0),
    }
}

// --- Run Benchmark on ALL Samples ---
fn run_full_benchmark(
    records: &[Record],
    base_cal: &Calibration,
    mc_samples: usize,
    mc_variance: f32,
) -> (f32, f32, f32, f32) {
    let start = Instant::now();

    let n = records.len();
    let mut physics_errors = Vec::with_capacity(n);
    let mut hybrid_errors = Vec::with_capacity(n);

    for rec in records.iter() {
        // Physics baseline
        let physics_pred = compute_strength(rec, base_cal, 20.0);
        physics_errors.push((physics_pred - rec.strength).abs());

        // MC ensemble
        let mut mc_preds = Vec::with_capacity(mc_samples);
        for _ in 0..mc_samples {
            let sampled_cal = mc_sample_calibration(base_cal, mc_variance);
            mc_preds.push(compute_strength(rec, &sampled_cal, 20.0));
        }
        mc_preds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mc_median = mc_preds[mc_samples / 2];

        // Ensemble: 60% physics + 40% MC
        let hybrid_pred = 0.6 * physics_pred + 0.4 * mc_median;
        hybrid_errors.push((hybrid_pred - rec.strength).abs());
    }

    let physics_mae = physics_errors.iter().sum::<f32>() / n as f32;
    let hybrid_mae = hybrid_errors.iter().sum::<f32>() / n as f32;
    let total_latency = start.elapsed().as_secs_f32() * 1000.0;
    let latency_per_sample = total_latency / n as f32;

    (physics_mae, hybrid_mae, latency_per_sample, n as f32)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut csv_path = "";
    let mut dataset_id = "D1";

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
            _ => {}
        }
        i += 1;
    }

    if csv_path.is_empty() {
        println!("{}", json!({"error": "No CSV"}));
        return;
    }

    let records = load_csv(csv_path);
    if records.is_empty() {
        println!("{}", json!({"error": "No records"}));
        return;
    }

    let cal = get_calibration(dataset_id);

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  DUMSTO HYBRID - FULL DATASET EVALUATION");
    println!(
        "  Dataset: {} ({} samples - ALL)",
        dataset_id,
        records.len()
    );
    println!("═══════════════════════════════════════════════════════════════════");

    let (physics_mae, hybrid_mae, latency, n_samples) = run_full_benchmark(&records, &cal, 20, 0.1);
    let improvement = (physics_mae - hybrid_mae) / physics_mae * 100.0;

    println!("\n  ┌──────────────────────────────────────────────────┐");
    println!(
        "  │  Physics MAE:   {:>8.2} MPa                     │",
        physics_mae
    );
    println!(
        "  │  Hybrid MAE:    {:>8.2} MPa                     │",
        hybrid_mae
    );
    println!(
        "  │  Improvement:   {:>8.2}%                       │",
        improvement
    );
    println!(
        "  │  Latency:       {:>8.3} ms/sample              │",
        latency
    );
    println!(
        "  │  Samples:       {:>8.0}                         │",
        n_samples
    );
    println!("  └──────────────────────────────────────────────────┘");

    println!(
        "\n{}",
        json!({
            "dataset": dataset_id,
            "physics_mae": physics_mae,
            "hybrid_mae": hybrid_mae,
            "improvement_pct": improvement,
            "latency_ms": latency,
            "n_samples": n_samples
        })
    );
}
