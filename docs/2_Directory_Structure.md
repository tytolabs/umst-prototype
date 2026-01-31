# Comprehensive Directory Structure
**Exhaustive Manifest of the Reproducibility Package**

**Purpose**: This document serves as the absolute map of the repository. It details every file, script, and data artifact, explaining its function, dependencies, and outputs.
**Scope**: Covers all 4 main components: Tools, Scripts, Docs, and Core.

---

## 1. Administration & Tools (`/tools`)
*scripts handling environment lifecycle: Setup, Verification, and Diagnostics.*

| File | Type | Description |
|---|---|---|
| **`5_SETUP_TOOL.py`** | Python | **Master Installer**. Defines the standard installation path. Detects OS (Linux/Mac/Win), checks Python/Rust versions, creates venv, installs requirements.txt, and verifies the build. **Run this first.** |
| **`6_VERIFY_TOOL.sh`** | Bash | **Integrity Validator**. Runs an 8-phase audit: (1) Python Ver, (2) Rust Ver, (3) PyTorch Ver, (4) Data Existence, (5) Import Check, (6) Kernel Compilation, (7) Unit Tests, (8) E2E Run. |
| **`7_RUN_EXPERIMENTS.py`** | Python | **Unified Launcher**. The single entry point for all benchmarks. Arguments: `quick` (subset), `benchmark` (full), `platform` (info). Dispatches to `scripts/` and `src/rust/` binaries. |
| **`8_DIAGNOSTICS_TOOL.py`** | Python | **Deep Debugger**. Generates a `system_report.log`. Collects env vars, dependency versions, disk space, and runs a mini-tensor op to check float32 precision. |
| **`9_PLATFORM_TEST.sh`** | Bash | **Low-Level Platform Check**. Validates C++ compilers (gcc/clang) and Rust toolchains (`cargo`). Checks for Docker availability. |
| **`10_DOCKER_TEST.sh`** | Bash | **Container Validator**. Builds a temp Docker image to verify the `Dockerfile` works on the current host. |
| **`11_GPU_TEST.py`** | Python | **Accelerator Validator**. Checks CUDA/MPS availability. Allocates a 1GB tensor to verify VRAM stability. |

---

## 2. Documentation Suite (`/docs`)
*The Reference Manuals. Numbered 1-8 for logical reading order.*

| File | Pages | Summary |
|---|---|---|
| **`1_Overview.md`** | 5 | Executive summary, philosophy, user personas, architecture diagrams, and FAQ. |
| **`2_Directory_Structure.md`** | 5 | **You are here.** Complete file inventory and data flow map. |
| **`3_Setup_Guide.md`** | 5 | Step-by-step installation for Linux (NVIDIA/AMD), Mac (M1/Intel), and Windows. Includes troubleshooting. |
| **`4_Reproduction_Guide.md`** | 4 | The SSOT Protocol. Exact hash-verified steps to reproduce Table 2 and Table 3. |
| **`5_Technical_Methodology.md`** | 5 | Mathematical explanation of the HyperGraph Tensor, Constitutive Equations (Avrami, Powers), and Thermodynamic Filters. |
| **`6_Evaluation_Protocol.md`** | 5 | Definitions of the 18 metrics (MAE, Admissibility, Sustainability, etc.) and the "Fair Comparison" standard. |
| **`7_Supplementary_Materials.md`** | 3 | Data provenance, License info, and derivations of the Clausius-Duhem inequality. |
| **`8_Advanced_Rust_Experiments.md`** | 4 | "Advanced Guide" to the Rust kernel. How to run `ssot_benchmark`, `agent_design`, and adversarial tests. |
| **`9_Constitutional_Creativity.md`** | 4 | DUMSTO-PPO multi-objective design benchmark. 6 reward modes, 3 baselines, creativity metrics, gate statistics. |

---

## 3. Scientific Source Code (`/scripts`)
*The Python research logic. Binds the Rust kernel to PyTorch learning agents.*

### A. Benchmarking (Execution)
*   **`9_final_comparative_benchmark.py`**: **The SSOT Script**. Runs the full 8-method comparison (XGBoost, MLP, GNN, PINN, H-PINN, Physics, Hybrid, PPO) with real PyTorch training for GNN/PINN/H-PINN. 10-run averaged, 4 datasets. Outputs `results/ssot/fair_comparison_*.json`.
*   **`2_comprehensive_benchmark.py`**: Legacy benchmark (kept for archival). Uses fallback simulated baselines when PyTorch unavailable.
*   **`3_quick_benchmark.py`**: A lightweight (2-minute) version of the full benchmark. Runs only 4 methods on D1. Used for CI/CD.
*   **`1_main_benchmark.py`**: Legacy runner (kept for archival).

### B. Methodology Implementations (Logic)
*   **`7_gnn_baseline.py`**: Graph Neural Network. Implements a PyTorch Geometric `GCNConv` network over the material tensor.
*   **`8_pinn_baseline.py`**: Soft-Constraint PINN. Adds a physics regularization term `L_phy` to the MSE loss.
*   **`17_pinn_baseline.py`**: Hard-Constraint PINN. Uses a projection layer to enforce bounds.
*   **`8_hybrid_plugin.py`**: **The Bridge**. The core Hybrid Logic. Steps: (1) Calls Rust Kernel for `f_physics`, (2) Trains XGBoost on `y - f_physics`, (3) Sums outputs.
*   **`13_ppo_trainer.py`**: Reinforcement Learning loop. Trains a PPO agent `StableBaselines3` to optimize mix designs against the Physics Reward.

### C. Utilities (Support)
*   **`4_physics_calibration.py`**: **Critical**. Uses Scikit-Learn's `L-BFGS-B` to fit the physics parameters ($S_{\text{int}}, k_{\text{slag}}$) to the training data.
*   **`5_plot_results.py`**: Visualization engine. Generates the PDF plots (Parity plots, Error histograms) in `results/plots/`.
*   **`metrics_collector.py`**: Shared library. Calculates the 18 metrics (MAE, Admissibility, CO2, Cost).
*   **`merge_results.py`**: Helper to combine multiple JSON outputs into one SSOT file.
*   **`6_latency_benchmark.py`**: High-precision timer. Measures inference latency in milliseconds per sample.

---

## 4. Core Physics Kernel (`/src/rust/core`)
*The fast, differentiable backend. Check `Cargo.toml` for dependencies.*

### A. Binaries (`src/bin/`) - Executables
*   **`ssot_benchmark`**: **Fastest**. Runs the predictive power benchmark natively in Rust. <1s execution.
*   **`physics_bridge`**: **The API**. CLI tool called by Python. Input: JSON state. Output: Physics features.
*   **`physics_calibrator`**: Native parameter optimizer. Faster version of `4_physics_calibration.py`.
*   **`agent_design_benchmark`**: Evaluates the generative capabilities of the agent (Inverse Design).
*   **`full_design_benchmark`**: **Creativity Benchmark**. Runs the 4-method creativity comparison with all 6 PPO modes.
*   **`experiment_runner`**: General-purpose CLI for "Combo" experiments (mixing different physics modules).
*   **`thermodynamic_gate`**: **Safety Gate**. Clausius-Duhem admissibility filter CLI. Input: CSV predictions. Output: Gated results.
*   **`hybrid_benchmark`**: Physics + ML hybrid model benchmarking.
*   **`physics_compute`**: Direct physics engine computation (no ML layer).

### B. Science Modules (`src/science/`) - The Laws of Nature
*   **`strength.rs`**: **Powers' Law**. Calculates compressive strength from gel-space ratio.
*   **`maturity.rs`**: **Maturity & Hydration Kinetics**. Evolves hydration degree over time via maturity functions.
*   **`rheology.rs`**: **Herschel-Bulkley**. Calculates yield stress and viscosity.
*   **`thermodynamic_filter.rs`**: **The Guardrail**. Enforces Clausius-Duhem. Returns `AdmissibilityResult`.
*   **`sustainability.rs`**: Lifecycle Assessment (LCA). Calculates GWP and embodied energy.
*   **`cost.rs`**: Economic modeling.

### C. Tensor Architecture (`src/tensors/`)
*   **`hyper_graph_tensor.rs`**: The core data structure. Implements the DAG (Directed Acyclic Graph) of material nodes.
*   **`sparse.rs`**: Sparse matrix operations for GNN features.
*   **`functor.rs`**: Differentiable operations (map, reduce) over the tensor.

---

## 5. Data & Results

### Data (`/data`)
*   **`dataset_D1.csv`**: UCI Concrete (1,030 samples). Clean lab data.
*   **`dataset_D2.csv`**: Zenodo NDT (4,891 samples). Field NDT data.
*   **`dataset_D3.csv`**: Zenodo SonReb (2,780 samples). Ultrasonic + Rebound.
*   **`dataset_D4.csv`**: Zenodo RH (7,445 samples). High-variance Research data.
*   **`dataset_full.csv`**: All combined (16,000+ samples).
*   **`calibration_config.json`**: Output of `4_physics_calibration.py`. Stores the fitted $k$ values.

### Results (`/results`)
*   **`ssot/`**: **Single Source of Truth**. The verified results folder.
    *   `fair_comparison_YYYY-MM-DD.json`: The final 8-method benchmark file.
*   **`plots/`**: Generated plots (png/pdf).
*   **`archive/`**: Old results (v1, v2).

---

## 6. Data Flow Architecture

Understanding how data moves through the system:

1.  **Ingestion**: `data/*.csv` is loaded by Pandas in Python.
2.  **Calibration**: `scripts/4_physics_calibration.py` reads data, optimizes physics params, saves to `data/calibration_config.json`.
3.  **Bridge**: The Hybrid Model (`scripts/8_hybrid_plugin.py`) reads the config.
4.  **Compute**:
    *   For each sample, Python calls `src/rust/core/bin/physics_bridge`.
    *   Rust computes `f_physics` and `is_admissible`.
    *   Rust results return to Python via stdout (JSON) or FFI.
5.  **Learning**: Python trains XGBoost on the residual `(y_true - f_physics)`.
6.  **Prediction**: Final prediction = `f_physics + XGBoost_Prediction`.
7.  **Evaluation**: `scripts/metrics_collector.py` compares Final Prediction vs Ground Truth, checks Admissibility again, and writes `results/*.json`.

---

## 7. File Hashing & Integrity

To ensure exact reproduction, we track SHA-256 hashes of critical files.

*   `src/rust/core/src/science/strength.rs`: [Critical logic - Powers Law]
*   `src/rust/core/src/science/thermodynamic_filter.rs`: [Critical logic - Safety]
*   `data/dataset_D1.csv`: [Critical data - Benchmark]

Run `./tools/6_VERIFY_TOOL.sh` to automatically check these hashes against the manifest.