// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto

//! DUMSTO Physics Calibrator
//!
//! Differentiable Unified Material-State Tensor Optimization (DUMSTO)
//!
//! A native Rust binary for per-dataset physics calibration.
//!
//! Usage: physics_calibrator --csv <path> --dataset <D1|D2|D3|D4>
//!
//! This binary:
//! 1. Loads CSV data directly
//! 2. Uses DUMSTO PhysicsKernel::compute_strength_with_maturity
//! 3. Performs grid search calibration
//! 4. Outputs optimal parameters and MAE

use serde_json::json;
use std::env;
use std::fs::File;
use std::io::{self, BufRead};

use umst_core::physics_kernel::PhysicsKernel;

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

// --- CSV Loading ---
fn load_csv(path: &str) -> Vec<Record> {
    let mut records = Vec::new();
    if let Ok(file) = File::open(path) {
        let lines = io::BufReader::new(file).lines();
        for (i, line) in lines.enumerate() {
            if i == 0 {
                continue;
            } // Skip header
            if let Ok(l) = line {
                let cols: Vec<&str> = l.split(',').collect();
                if cols.len() < 9 {
                    continue;
                }

                let c = cols[0].parse().unwrap_or(0.0);
                let s = cols[1].parse().unwrap_or(0.0);
                let fa = cols[2].parse().unwrap_or(0.0);
                let w = cols[3].parse().unwrap_or(0.0);
                let age = cols[7].parse().unwrap_or(28.0);
                let strength = cols[8].parse().unwrap_or(0.0);

                records.push(Record {
                    cement: c,
                    slag: s,
                    fly_ash: fa,
                    water: w,
                    age,
                    strength,
                });
            }
        }
    }
    records
}

// --- Calibration Config ---
#[derive(Clone, Debug)]
struct CalibrationParams {
    s_intrinsic: f32,
    k_slag: f32,
    k_fly_ash: f32,
    _k_ref: f32,
}

impl CalibrationParams {
    fn default() -> Self {
        CalibrationParams {
            s_intrinsic: 150.0,
            k_slag: 0.6,
            k_fly_ash: 0.4,
            _k_ref: 0.55,
        }
    }
}

// --- Strength Prediction using Full Physics ---
fn predict_strength(record: &Record, params: &CalibrationParams, temp_c: f32) -> f32 {
    let binder = record.cement + record.slag + record.fly_ash;
    if binder <= 0.0 {
        return 0.0;
    }

    // Effective w/c with SCM k-values
    let effective_cement =
        record.cement + params.k_slag * record.slag + params.k_fly_ash * record.fly_ash;

    if effective_cement <= 0.0 {
        return 0.0;
    }

    let w_c = record.water / effective_cement;
    let scm_ratio = (record.slag + record.fly_ash) / binder;

    // Use the new PhysicsKernel function
    PhysicsKernel::compute_strength_with_maturity(
        w_c,
        record.age,
        temp_c,
        scm_ratio,
        params.s_intrinsic,
    )
}

// --- Compute MAE ---
fn compute_mae(records: &[Record], params: &CalibrationParams, temp_c: f32) -> f32 {
    let mut total_error = 0.0;
    for rec in records {
        let pred = predict_strength(rec, params, temp_c);
        total_error += (pred - rec.strength).abs();
    }
    total_error / records.len() as f32
}

// --- Grid Search Calibration ---
fn calibrate(records: &[Record]) -> (CalibrationParams, f32) {
    let mut best_params = CalibrationParams::default();
    let mut best_mae = f32::MAX;
    let temp_c = 20.0; // Standard curing

    // Grid search over parameter space
    for s_int in (80..=200).step_by(10) {
        for k_slag in (30..=100).step_by(10) {
            for k_fa in (20..=100).step_by(10) {
                let params = CalibrationParams {
                    s_intrinsic: s_int as f32,
                    k_slag: k_slag as f32 / 100.0,
                    k_fly_ash: k_fa as f32 / 100.0,
                    _k_ref: 0.55,
                };

                let mae = compute_mae(records, &params, temp_c);

                if mae < best_mae {
                    best_mae = mae;
                    best_params = params;
                }
            }
        }
    }

    // Fine-tune with smaller steps around best
    let fine_range = 10;
    let best_s_int = best_params.s_intrinsic as i32;
    let best_k_slag = (best_params.k_slag * 100.0) as i32;
    let best_k_fa = (best_params.k_fly_ash * 100.0) as i32;

    for s_int in (best_s_int - fine_range).max(80)..=(best_s_int + fine_range).min(200) {
        for k_slag in (best_k_slag - fine_range).max(30)..=(best_k_slag + fine_range).min(100) {
            for k_fa in (best_k_fa - fine_range).max(20)..=(best_k_fa + fine_range).min(100) {
                let params = CalibrationParams {
                    s_intrinsic: s_int as f32,
                    k_slag: k_slag as f32 / 100.0,
                    k_fly_ash: k_fa as f32 / 100.0,
                    _k_ref: 0.55,
                };

                let mae = compute_mae(records, &params, temp_c);

                if mae < best_mae {
                    best_mae = mae;
                    best_params = params;
                }
            }
        }
    }

    (best_params, best_mae)
}

// --- Sample Comparison (Debug) ---
fn print_sample_comparison(records: &[Record], params: &CalibrationParams, n: usize) {
    println!("\n  Sample Comparison (first {} records):", n);
    println!(
        "  {:>10} | {:>10} | {:>10} | {:>6}",
        "Real", "Predicted", "Error", "Age"
    );
    println!("  {}", "-".repeat(50));

    for rec in records.iter().take(n) {
        let pred = predict_strength(rec, params, 20.0);
        let err = pred - rec.strength;
        println!(
            "  {:>10.1} | {:>10.1} | {:>+10.1} | {:>6.0}",
            rec.strength, pred, err, rec.age
        );
    }
}

// --- Main ---
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
        println!("{}", json!({"error": "No CSV provided. Use --csv <path>"}));
        return;
    }

    let records = load_csv(csv_path);
    if records.is_empty() {
        println!("{}", json!({"error": "No records loaded"}));
        return;
    }

    println!("==================================================");
    println!("DUMSTO Physics Calibrator (Native Rust)");
    println!("Dataset: {} ({} records)", dataset_id, records.len());
    println!("==================================================");

    println!("\nCalibrating...");
    let (best_params, best_mae) = calibrate(&records);

    println!("\n Calibration Complete!");
    println!("  S_intrinsic: {:.1}", best_params.s_intrinsic);
    println!("  k_slag: {:.2}", best_params.k_slag);
    println!("  k_fly_ash: {:.2}", best_params.k_fly_ash);
    println!("  MAE: {:.2} MPa", best_mae);

    print_sample_comparison(&records, &best_params, 10);

    // JSON output for scripting
    println!(
        "\n{}",
        json!({
            "dataset": dataset_id,
            "mae": best_mae,
            "s_intrinsic": best_params.s_intrinsic,
            "k_slag": best_params.k_slag,
            "k_fly_ash": best_params.k_fly_ash
        })
    );
}
