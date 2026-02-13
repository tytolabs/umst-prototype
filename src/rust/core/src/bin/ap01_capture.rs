// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT

//! DUMSTO AP01: Material Characterization - Proxy Capture
//! =====================================================
//! Implements VisionToRheology proxy capture system.
//! Uses UCI Concrete dataset as proxy while real specimens are prepared.
//!
//! Trains VisionToRheology model on simulated proxy features.
//! Provides calibrated material parameters for AP02 Pareto sweep.
//!
//! Usage: cargo run --bin ap01_capture -- --dataset synthetic --output results/ap01_calibration.json

use clap::{Arg, Command};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use umst_core::metrics::{load_dataset, mean_absolute_error};

/// Proxy measurement data (simulates camera/sensor readings)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProxyMeasurement {
    pub mix_id: String,
    pub weight_total: f64,
    pub weight_paste: f64,
    pub video_flow_time: f64,  // Slump flow video duration
    pub temperature: f64,
    pub humidity: f64,
    pub surface_texture: f64,  // Roughness proxy
    pub cement_kg: f64,
    pub slag_kg: f64,
    pub fly_ash_kg: f64,
    pub water_kg: f64,
    pub sp_kg: f64,
}

/// Vision feature vector extracted from proxy data
#[derive(Clone, Debug)]
pub struct VisionFeatures {
    pub slump_flow: f64,
    pub yield_stress: f64,
    pub viscosity: f64,
    pub wc_ratio: f64,
    pub scm_pct: f64,
    pub sp_dosage: f64,
    pub temp_factor: f64,
    pub humidity_factor: f64,
    pub surface_texture: f64,
    pub flow_time: f64,
}

/// Simple gradient boosting regressor for VisionToRheology
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisionToRheologyModel {
    pub n_estimators: usize,
    pub max_depth: usize,
    pub trees: Vec<DecisionTree>,
}

/// VisionToRheology training results
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisionCalibrationResult {
    pub dataset: String,
    pub n_specimens: usize,
    pub physics_mae: f64,
    pub vision_mae: f64,
    pub hybrid_mae: f64,
    pub improvement_pct: f64,
    pub feature_importance: HashMap<String, f64>,
    pub timestamp: String,
}

/// Powers' gel-space ratio model for physics predictions
fn powers_strength(wc_ratio: f64, alpha: f64, intrinsic: f64) -> f64 {
    if wc_ratio <= 0.0 {
        return 0.0;
    }
    let x = (0.68 * alpha) / (0.32 * alpha + wc_ratio);
    intrinsic * x.powf(3.0)
}

/// Generate proxy measurements for a dataset
fn generate_proxy_measurements(data: &[(Vec<f64>, f64)]) -> Vec<ProxyMeasurement> {
    let mut rng = rand::thread_rng();
    let mut proxies = Vec::new();

    for (i, (features, _target)) in data.iter().enumerate() {
        if features.len() < 8 {
            continue; // Need cement, slag, fly_ash, water, sp, coarse, fine, age
        }

        let cement = features[0];
        let slag = features[1];
        let fly_ash = features[2];
        let water = features[3];
        let sp = features[4];

        // Calculate derived properties
        let total_weight = features.iter().sum::<f64>() * 10.0; // Scale for batch size
        let paste_weight = (cement + slag + fly_ash + water + sp) * 10.0;

        // Simulate video flow time based on W/C ratio
        let wc_ratio = water / cement;
        let base_flow_time = 10.0;
        let wc_factor = (wc_ratio - 0.4) * 10.0; // Higher W/C = longer flow time
        let flow_time = (base_flow_time + wc_factor + rng.gen_range(-2.0..2.0))
            .max(5.0).min(30.0);

        // Surface texture based on SCM content
        let scm_pct = (slag + fly_ash) / cement;
        let surface_texture = scm_pct * 50.0 + rng.gen_range(-10.0..10.0);

        let proxy = ProxyMeasurement {
            mix_id: format!("proxy_{:03}", i),
            weight_total: total_weight,
            weight_paste: paste_weight,
            video_flow_time: flow_time,
            temperature: 22.0 + rng.gen_range(-2.0..2.0),
            humidity: 65.0 + rng.gen_range(-5.0..5.0),
            surface_texture: surface_texture.max(0.0),
            cement_kg: cement * 10.0, // Scale to batch
            slag_kg: slag * 10.0,
            fly_ash_kg: fly_ash * 10.0,
            water_kg: water * 10.0,
            sp_kg: sp * 10.0,
        };

        proxies.push(proxy);
    }

    proxies
}

/// Extract vision features from proxy measurements
fn extract_vision_features(proxy: &ProxyMeasurement) -> VisionFeatures {
    let flow_time = proxy.video_flow_time;
    let _weight_ratio = proxy.weight_paste / proxy.weight_total;

    // Rheological features from vision analysis
    let slump_flow = 200.0 + (flow_time - 10.0) * 30.0; // Rough conversion
    let yield_stress = (1000.0 / (slump_flow + 10.0)).max(10.0);
    let viscosity = 50.0 + yield_stress * 0.5;

    // Material composition features
    let wc_ratio = proxy.water_kg / proxy.cement_kg;
    let scm_pct = (proxy.slag_kg + proxy.fly_ash_kg) / proxy.cement_kg * 100.0;

    // Admixture effects
    let sp_dosage = proxy.sp_kg / proxy.cement_kg * 100.0;

    // Environmental factors
    let temp_factor = (proxy.temperature - 20.0) / 10.0;
    let humidity_factor = proxy.humidity / 100.0;

    VisionFeatures {
        slump_flow,
        yield_stress,
        viscosity,
        wc_ratio,
        scm_pct,
        sp_dosage,
        temp_factor,
        humidity_factor,
        surface_texture: proxy.surface_texture,
        flow_time,
    }
}

/// Simple decision tree for gradient boosting
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionTree {
    root: Option<TreeNode>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TreeNode {
    feature_idx: Option<usize>,
    threshold: Option<f64>,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
    value: Option<f64>,
}

impl DecisionTree {
    fn new() -> Self {
        Self { root: None }
    }

    fn fit(&mut self, x: &[Vec<f64>], y: &[f64], max_depth: usize, depth: usize) {
        if depth >= max_depth || y.len() <= 1 {
            let value = y.iter().sum::<f64>() / y.len() as f64;
            self.root = Some(TreeNode {
                feature_idx: None,
                threshold: None,
                left: None,
                right: None,
                value: Some(value),
            });
            return;
        }

        // Simple splitting (find best feature)
        let n_features = x[0].len();
        let mut best_feature = 0;
        let mut best_score = f64::INFINITY;

        for feature_idx in 0..n_features {
            // Use median as split point
            let mut values: Vec<f64> = x.iter().map(|row| row[feature_idx]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let threshold = values[values.len() / 2];

            // Simple variance reduction
            let mut left_y = Vec::new();
            let mut right_y = Vec::new();

            for (row, &target) in x.iter().zip(y.iter()) {
                if row[feature_idx] <= threshold {
                    left_y.push(target);
                } else {
                    right_y.push(target);
                }
            }

            if !left_y.is_empty() && !right_y.is_empty() {
                let score = Self::variance(&left_y) + Self::variance(&right_y);
                if score < best_score {
                    best_score = score;
                    best_feature = feature_idx;
                }
            }
        }

        // Create split
        let threshold = {
            let mut values: Vec<f64> = x.iter().map(|row| row[best_feature]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            values[values.len() / 2]
        };

        let mut left_x = Vec::new();
        let mut left_y = Vec::new();
        let mut right_x = Vec::new();
        let mut right_y = Vec::new();

        for (row, &target) in x.iter().zip(y.iter()) {
            if row[best_feature] <= threshold {
                left_x.push(row.clone());
                left_y.push(target);
            } else {
                right_x.push(row.clone());
                right_y.push(target);
            }
        }

        let mut left_tree = DecisionTree::new();
        let mut right_tree = DecisionTree::new();

        left_tree.fit(&left_x, &left_y, max_depth, depth + 1);
        right_tree.fit(&right_x, &right_y, max_depth, depth + 1);

        self.root = Some(TreeNode {
            feature_idx: Some(best_feature),
            threshold: Some(threshold),
            left: left_tree.root.map(Box::new),
            right: right_tree.root.map(Box::new),
            value: None,
        });
    }

    fn variance(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
    }

    fn predict(&self, x: &[f64]) -> f64 {
        self.predict_node(&self.root.as_ref().unwrap(), x)
    }

    fn predict_node(&self, node: &TreeNode, x: &[f64]) -> f64 {
        match node.value {
            Some(val) => val,
            None => {
                let feature_val = x[node.feature_idx.unwrap()];
                if feature_val <= node.threshold.unwrap() {
                    self.predict_node(node.left.as_ref().unwrap(), x)
                } else {
                    self.predict_node(node.right.as_ref().unwrap(), x)
                }
            }
        }
    }
}

impl VisionToRheologyModel {
    fn new(n_estimators: usize, max_depth: usize) -> Self {
        Self {
            n_estimators,
            max_depth,
            trees: Vec::new(),
        }
    }

    fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        let mut predictions = vec![0.0; y.len()];

        for _ in 0..self.n_estimators {
            let residuals: Vec<f64> = y.iter()
                .zip(predictions.iter())
                .map(|(true_val, pred)| true_val - pred)
                .collect();

            let mut tree = DecisionTree::new();
            tree.fit(x, &residuals, self.max_depth, 0);

            // Update predictions
            for i in 0..predictions.len() {
                predictions[i] += 0.1 * tree.predict(&x[i]); // Learning rate = 0.1
            }

            self.trees.push(tree);
        }
    }

    fn predict(&self, x: &[f64]) -> f64 {
        self.trees.iter()
            .map(|tree| tree.predict(x))
            .sum::<f64>() * 0.1
    }

    fn feature_importance(&self) -> HashMap<String, f64> {
        let feature_names = vec![
            "slump_flow", "yield_stress", "viscosity", "wc_ratio", "scm_pct",
            "sp_dosage", "temp_factor", "humidity_factor", "surface_texture", "flow_time"
        ];

        let mut importance = HashMap::new();

        // Simple importance based on tree splits
        for tree in &self.trees {
            if let Some(ref root) = tree.root {
                Self::count_splits(&Some(Box::new(root.clone())), &mut importance);
            }
        }

        // Normalize
        let total: f64 = importance.values().sum();
        for (_feature, count) in importance.iter_mut() {
            *count /= total;
        }

        // Map to feature names
        let mut named_importance = HashMap::new();
        for (i, name) in feature_names.iter().enumerate() {
            let key = i.to_string();
            if let Some(imp) = importance.get(&key) {
                named_importance.insert(name.to_string(), *imp);
            }
        }

        named_importance
    }

    fn count_splits(node: &Option<Box<TreeNode>>, counts: &mut HashMap<String, f64>) {
        if let Some(node) = node {
            if let Some(feature_idx) = node.feature_idx {
                let key = feature_idx.to_string();
                *counts.entry(key).or_insert(0.0) += 1.0;
            }

            Self::count_splits(&node.left, counts);
            Self::count_splits(&node.right, counts);
        }
    }
}

/// Train VisionToRheology model
fn train_vision_to_rheology(
    train_proxies: &[ProxyMeasurement],
    train_targets: &[f64],
    test_proxies: &[ProxyMeasurement],
    test_targets: &[f64],
) -> VisionCalibrationResult {
    // Extract vision features
    let train_features: Vec<Vec<f64>> = train_proxies.iter()
        .map(|proxy| {
            let features = extract_vision_features(proxy);
            vec![
                features.slump_flow,
                features.yield_stress,
                features.viscosity,
                features.wc_ratio,
                features.scm_pct,
                features.sp_dosage,
                features.temp_factor,
                features.humidity_factor,
                features.surface_texture,
                features.flow_time,
            ]
        })
        .collect();

    // Train VisionToRheology model
    let mut vision_model = VisionToRheologyModel::new(100, 6);
    vision_model.fit(&train_features, train_targets);

    // Evaluate physics-only baseline
    let physics_predictions: Vec<f64> = train_proxies.iter()
        .map(|proxy| {
            let wc_ratio = proxy.water_kg / proxy.cement_kg;
            powers_strength(wc_ratio, 0.7, 150.0)
        })
        .collect();

    let physics_mae = mean_absolute_error(train_targets, &physics_predictions);

    // Evaluate VisionToRheology
    let vision_predictions: Vec<f64> = test_proxies.iter()
        .map(|proxy| {
            let features = extract_vision_features(proxy);
            let base_prediction = powers_strength(
                proxy.water_kg / proxy.cement_kg,
                0.7, 150.0
            );
            let correction = vision_model.predict(&[
                features.slump_flow,
                features.yield_stress,
                features.viscosity,
                features.wc_ratio,
                features.scm_pct,
                features.sp_dosage,
                features.temp_factor,
                features.humidity_factor,
                features.surface_texture,
                features.flow_time,
            ]);
            base_prediction + correction
        })
        .collect();

    let vision_mae = mean_absolute_error(test_targets, &vision_predictions);
    let improvement_pct = (physics_mae - vision_mae) / physics_mae * 100.0;

    let feature_importance = vision_model.feature_importance();

    VisionCalibrationResult {
        dataset: "proxy_dataset".to_string(),
        n_specimens: train_proxies.len() + test_proxies.len(),
        physics_mae,
        vision_mae,
        hybrid_mae: vision_mae, // Vision-enhanced is our hybrid
        improvement_pct,
        feature_importance,
        timestamp: chrono::Utc::now().to_rfc3339(),
    }
}

/// Export calibration results
fn export_calibration_results(
    result: &VisionCalibrationResult,
    output_path: &str
) -> Result<(), Box<dyn std::error::Error>> {
    let json_data = serde_json::to_string_pretty(result)?;
    std::fs::write(output_path, json_data)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("DUMSTO AP01: Material Characterization")
        .version("1.0")
        .author("Santhosh Shyamsundar, Prabhu S., Studio Tyto")
        .about("Trains VisionToRheology model on proxy material data")
        .arg(
            Arg::new("dataset")
                .long("dataset")
                .value_name("DATASET")
                .help("Dataset to use for training (synthetic or D1)")
                .default_value("synthetic")
        )
        .arg(
            Arg::new("output")
                .long("output")
                .value_name("OUTPUT")
                .help("Output JSON file path")
                .default_value("results/ap01_calibration.json")
        )
        .get_matches();

    let dataset_name = matches.get_one::<String>("dataset").unwrap();
    let output_path = matches.get_one::<String>("output").unwrap();

    println!("🔬 DUMSTO AP01: Material Characterization");
    println!("Training VisionToRheology on {} dataset", dataset_name);

    // Load dataset
    let data_path = if dataset_name == "synthetic" {
        "synthetic_concrete_data.csv".to_string()
    } else {
        format!("data/dataset_{}.csv", dataset_name)
    };

    let data = load_dataset(&data_path)?;
    println!("Loaded {} specimens from {}", data.len(), data_path);

    // Split data
    let train_size = (data.len() as f64 * 0.8) as usize;
    let train_data = &data[..train_size];
    let test_data = &data[train_size..];

    // Generate proxy measurements
    println!("Generating proxy measurements...");
    let train_proxies = generate_proxy_measurements(train_data);
    let test_proxies = generate_proxy_measurements(test_data);
    println!("Generated {} training and {} test proxy measurements",
             train_proxies.len(), test_proxies.len());

    // Extract targets (actual strength values)
    let train_targets: Vec<f64> = train_data.iter().map(|(_, target)| *target).collect();
    let test_targets: Vec<f64> = test_data.iter().map(|(_, target)| *target).collect();

    // Train VisionToRheology model
    println!("Training VisionToRheology model...");
    let calibration_result = train_vision_to_rheology(
        &train_proxies, &train_targets,
        &test_proxies, &test_targets
    );

    // Results
    println!("
📊 CALIBRATION RESULTS:");
    println!("Physics-only MAE: {:.3} MPa", calibration_result.physics_mae);
    println!("Vision-enhanced MAE: {:.3} MPa", calibration_result.vision_mae);
    println!("Improvement: {:.1}%", calibration_result.improvement_pct);

    println!("
🎯 TOP VISION FEATURES:");
    let mut features: Vec<_> = calibration_result.feature_importance.iter().collect();
    features.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    for (feature, importance) in features.iter().take(5) {
        println!("  {}: {:.1}%", feature, *importance * 100.0);
    }

    // Export results
    export_calibration_results(&calibration_result, output_path)?;
    println!("💾 Results exported to: {}", output_path);

    println!("
✅ AP01 calibration complete!");
    println!("📋 Next: Use calibrated VisionToRheology in AP02 Pareto sweep");

    Ok(())
}