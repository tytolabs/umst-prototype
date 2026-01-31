// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
// SPDX-License-Identifier: MIT
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TopologyConfig {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub vol_fraction: f32,
    pub penalization: f32,
    pub filter_radius: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Load {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub fx: f32,
    pub fy: f32,
    pub fz: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Boundary {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub fix_x: bool,
    pub fix_y: bool,
    pub fix_z: bool,
}

#[wasm_bindgen]
pub struct TopologyOptimizer {
    config: TopologyConfig,
    density: Vec<f32>,
    stress: Vec<f32>,
}

#[wasm_bindgen]
impl TopologyOptimizer {
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize, depth: usize, vol_fraction: f32) -> TopologyOptimizer {
        let size = width * height * depth;
        TopologyOptimizer {
            config: TopologyConfig {
                width,
                height,
                depth,
                vol_fraction,
                penalization: 3.0,
                filter_radius: 1.5,
            },
            density: vec![vol_fraction; size],
            stress: vec![0.0; size],
        }
    }

    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.config.width + z * self.config.width * self.config.height
    }

    /// Run one iteration of SIMP optimization
    /// Returns max change in density
    pub fn step(&mut self, loads_json: &str, boundaries_json: &str) -> f32 {
        let loads: Vec<Load> = serde_json::from_str(loads_json).unwrap_or_default();
        let _boundaries: Vec<Boundary> = serde_json::from_str(boundaries_json).unwrap_or_default();

        let size = self.density.len();
        let mut sensitivity = vec![0.0; size];

        // 1. "Solve" Stresses (Heuristic 1/r^2 propagation from loads)
        // Real FEA is too heavy for this single-file audit, capturing the O(N) structure
        self.stress.fill(0.1); // Base stress

        for load in &loads {
            for z in 0..self.config.depth {
                for y in 0..self.config.height {
                    for x in 0..self.config.width {
                        let idx = self.index(x, y, z);

                        let dx = (x as f32) - (load.x as f32);
                        let dy = (y as f32) - (load.y as f32);
                        let dz = (z as f32) - (load.z as f32);

                        let dist_sq = dx * dx + dy * dy + dz * dz;
                        if dist_sq < 0.1 {
                            continue;
                        }

                        let force_mag =
                            (load.fx.powi(2) + load.fy.powi(2) + load.fz.powi(2)).sqrt();

                        // Stress concentration falls off with distance
                        // Stress = Force / Area -> roughly Force / dist^2 in 3D
                        let local_stress = force_mag / dist_sq;

                        // Penalize by density (SIMP: E = E0 * rho^p)
                        // Higher density = better load bearing = less effective stress
                        let stiffness = self.density[idx].powf(self.config.penalization);

                        self.stress[idx] += local_stress / (stiffness + 0.01);
                    }
                }
            }
        }

        // 2. Compute Sensitivity
        // dC/drho = -p * rho^(p-1) * u^T * K0 * u (Strain Energy)
        // Simplified: -p * rho ^(p-1) * stress
        for i in 0..size {
            let rho = self.density[i];
            sensitivity[i] = self.config.penalization
                * rho.powf(self.config.penalization - 1.0)
                * self.stress[i];
        }

        // 3. Filter Sensitivity (Mesh independence)
        let r = self.config.filter_radius as i32;
        let mut filtered_sensitivity = sensitivity.clone();

        for z in 0..self.config.depth as i32 {
            for y in 0..self.config.height as i32 {
                for x in 0..self.config.width as i32 {
                    let idx = self.index(x as usize, y as usize, z as usize);
                    let mut weight_sum = 0.0;
                    let mut val_sum = 0.0;

                    for dz in -r..=r {
                        for dy in -r..=r {
                            for dx in -r..=r {
                                let nx = x + dx;
                                let ny = y + dy;
                                let nz = z + dz;

                                if nx >= 0
                                    && nx < self.config.width as i32
                                    && ny >= 0
                                    && ny < self.config.height as i32
                                    && nz >= 0
                                    && nz < self.config.depth as i32
                                {
                                    let dist = ((dx * dx + dy * dy + dz * dz) as f32).sqrt();
                                    if dist <= self.config.filter_radius {
                                        let weight = self.config.filter_radius - dist;
                                        let nidx =
                                            self.index(nx as usize, ny as usize, nz as usize);

                                        weight_sum += weight;
                                        val_sum += weight * sensitivity[nidx] * self.density[nidx];
                                    }
                                }
                            }
                        }
                    }

                    if weight_sum > 0.0 {
                        filtered_sensitivity[idx] =
                            val_sum / (weight_sum * self.density[idx].max(0.001));
                    }
                }
            }
        }

        // 4. Update Densities (Optimality Criteria)
        let mut max_change = 0.0;
        let move_limit = 0.2;

        // Simple fixed-step update (OC simplified)
        for i in 0..size {
            let old_rho = self.density[i];
            let mut new_rho = old_rho * filtered_sensitivity[i].sqrt(); // B_e

            // Move limits
            new_rho = new_rho.clamp(old_rho - move_limit, old_rho + move_limit);
            new_rho = new_rho.clamp(0.001, 1.0);

            self.density[i] = new_rho;
            let change = (new_rho - old_rho).abs();
            if change > max_change {
                max_change = change;
            }
        }

        // Apply volume constraint (Normalization)
        let total_vol: f32 = self.density.iter().sum();
        let target_vol = self.config.vol_fraction * (size as f32);
        let scale = target_vol / total_vol.max(0.1);

        for i in 0..size {
            self.density[i] = (self.density[i] * scale).clamp(0.001, 1.0);
        }

        max_change
    }

    pub fn get_densities(&self) -> Vec<f32> {
        self.density.clone()
    }
}
