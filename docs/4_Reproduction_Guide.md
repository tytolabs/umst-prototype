# Reproduction Guide
**Verified "Single Source of Truth" Protocol**

**Objective**: Exact, bit-level reproduction of the experimental results.
**Scope**: Table 2 (Predictive Power), Table 3 (Generative Design), Figure 4 (Admissibility).
**Guarantees**: Deterministic outputs on x86_64 and ARM64.

---

## 1. The Benchmark Suite Overview

The reproduction suite consists of three verified experiments.

| ID | Experiment | Target Table | Description | Runtime |
|---|---|---|---|---|
| **E1** | **Predictive Power** | **Table 2** | Compares MAE of 8 methods on 4 datasets. | 30-45m |
| **E2** | **Generative Design** | **Table 3** | Tests the PPO agent's ability to solve inverse problems. | 10m |
| **E3** | **Admissibility** | **Figure 4** | Stress-tests the Thermodynamic Filter. | 5m |

---

## 2. Experiment 1: Predictive Power (The Main Result)

This is the most critical benchmark. It validates that the Hybrid architecture achieves SOTA accuracy with 100% safety.

### 2.1 Execution
From the project root:
```bash
python tools/7_RUN_EXPERIMENTS.py benchmark
```

### 2.2 Detailed Log Interpretation
The script will output progress bars for each dataset (D1-D4). Watch the logs for the verification of the "Hybrid Bridge".

```text
[INFO] Starting DUMSTO Comprehensive Benchmark v3.0...
[INFO] Methodology: Fair Comparison (Plateau Training)
----------------------------------------------------------------
[DATASET D1] Loading... (N=1030)
[CALIBRATION] D1 Parameters: s_int=80.0, k_slag=1.0... [OK]
[MODEL 1/8] XGBoost... Done. MAE=3.05
[MODEL 6/8] DUMSTO-Physics...
  -> Calling Rust Kernel... (0.4ms)
  -> Computing Admissibility... (100%)
  -> Done. MAE=6.71
[MODEL 7/8] DUMSTO-Hybrid (Ours)...
  -> Fusing Physics + XGBoost Residuals...
  -> Ensuring 100% Admissibility Guardrails...
  -> Done. MAE=2.99
...
```

### 2.3 Success Criteria
Compare your `results/ssot/fair_comparison_2026-01-28.json` with the table below.

| Method | D1 MAE | Admissibility | Interpretation |
|---|---|---|---|
| **DUMSTO-Hybrid (Ours)** | **2.99 ± 0.27** | **100%** | **Best Accuracy AND Safety.** The key result. |
| XGBoost (Baseline) | 3.05 ± 0.22 | 99.7% (D1) / 96.0% (D4) | Accurate but imperfect safety under shift. |
| PINN (Soft Constraints) | 4.56 ± 0.26 | 99.9% (D1) / 96.8% (D4) | Physics-regularized but not guaranteed safe. |
| H-PINN (Hard Constraints) | 8.33 ± 0.60 | 88.7% (D1) / 97.0% (D4) | Learned projection insufficient for safety. |

**Critical Verification Point**: The Hybrid model MUST have lower MAE than Physics, AND `Admissibility` must be exactly 100.0%. If Admissibility is <100%, the reproduction has FAILED (likely kernel version mismatch).

---

## 3. Experiment 2: Generative Design (Inverse Design)

This tests the "Creative" capacity of the AI. Can it design a mix for a specific target strength?

### 3.1 Execution
This runs entirely in the Rust kernel for performance testing.
```bash
cd src/rust/core
cargo run --release --bin agent_design_benchmark -- --dataset D1 --targets 30,40,50
```

### 3.2 Expected Output
```json
{
  "target_30": { "success_rate": 1.00, "avg_co2": 235.4, "avg_cost": 74.6 },
  "target_40": { "success_rate": 1.00, "avg_co2": 235.4, "avg_cost": 76.6 },
  "target_50": { "success_rate": 1.00, "avg_co2": 235.4, "avg_cost": 78.2 }
}
```
*Note*: The `agent_design_benchmark` binary runs the Physics Heuristic baseline by default. For the full 4-method creativity comparison (including DUMSTO-PPO), use `full_design_benchmark`.

### 3.3 Interpretation
*   **Target 30 MPa**: Analytical inversion via Powers' Law achieves 100% success.
*   **Target 50 MPa**: Requires lower w/c ratio (0.37); Physics Heuristic still achieves 100%.
*   **Physics Check**: Every design is guaranteed thermodynamically admissible by construction.
*   **Full Benchmark**: See `docs/9_Constitutional_Creativity.md` for the DUMSTO-PPO multi-objective creativity results.

---

## 4. Experiment 3: Safety & Robustness

This measures how the model behaves under "Attack" (Adversarial inputs).

### 4.1 Execution
The safety test uses the Rust thermodynamic gate binary:
```bash
cd src/rust/core
cargo run --release --bin thermodynamic_gate -- --csv ../../../data/dataset_D1.csv
```

### 4.2 Expected Output (Illustrative)
```text
[GATE] Evaluating predictions against Clausius-Duhem filter...
[RESULT]
 - Total predictions: 206
 - Accepted: 206 (100.0%)
 - Rejected: 0
 - Rejection reasons: {}
```
*Note*: The exact violation counts depend on which model's predictions are fed to the gate. All DUMSTO variants (Physics, Hybrid, PPO) produce 0 rejections by construction. ML baselines (XGBoost, PINN) may produce violations on adversarial or out-of-distribution inputs.
**Conclusion**: Only the DUMSTO models (Physics, Hybrid, PPO) survive the stress test with 0 violations.

---

## 5. Troubleshooting Reproduction Failures

### Case A: Numbers are "Way Off" (>1.0 MAE difference)
*   **Cause**: Physics Calibration mismatch.
*   **Fix**: You are likely running on a different dataset version. Run `python scripts/4_physics_calibration.py` to re-tune the $k$-values for your specific CSV file.

### Case B: Benchmark Crashes on "D4"
*   **Cause**: D4 is the "Research" dataset and contains outliers (Zero cement mixes) that cause division-by-zero in naive implementations.
*   **Fix**: Our Rust Kernel (`strength.rs`) has a guardrail `if effective_cement <= 0.0 { return 0.0; }`. Ensure you have compiled the latest Rust binary (`cargo build --release`).

### Case C: Python-Rust Bridge Error
*   **Error**: `CalledProcessError: Command '.../physics_bridge' returned non-zero exit status.`
*   **Fix**: The Python script cannot find the Rust binary. Run `python tools/5_SETUP_TOOL.py` again to ensure the binary is compiled and placed in the correct path.

---

## 6. Artifact Manifest

Upon successful reproduction, your `results/ssot/` folder should contain:

1.  **`fair_comparison_2026-01-28.json`**: (~50KB) The full 8-method predictive power results.
2.  **`design_benchmark_latest.json`**: (~50KB) The 4-method creativity comparison with PPO mode breakdown.
3.  **`data/calibration_config.json`**: (~2KB) The calibrated physics parameters.

---

## 7. Reporting Results

When citing these results, please use the precise version hash of the dataset and the calibration config used.

**Standard Citation**:
> "Results reproduced using DUMSTO v1.0 on Platform [X] with Calibration [Hash]."
