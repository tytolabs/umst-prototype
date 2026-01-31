# Advanced Rust Experiments
**The Hacker's Guide to the DUMSTO Kernel**

**Objective**: Bypass the Python wrapper and interact directly with the high-performance Rust binaries.
**Prerequisites**: Functional `cargo` toolchain (`rustc 1.75+`).
**Context**: These experiments run up to 1000x faster than their Python equivalents.

---

## 1. Kernel Architecture

The kernel defines 9 binary entry points in `src/rust/core/src/bin/`.

| Binary | Purpose |
|---|---|
| **`ssot_benchmark`** | **The Speed Demon**. Runs the D1 predictive benchmark. |
| **`agent_design_benchmark`** | **The Creator**. Inverse design optimization loop. |
| **`full_design_benchmark`** | **The Creativity Suite**. 4-method comparison with all 6 PPO reward modes. |
| **`experiment_runner`** | **The Lab**. "Combo" testing of physics modules. |
| **`thermodynamic_gate`** | **The Gatekeeper**. Clausius-Duhem admissibility filter CLI. |
| **`hybrid_benchmark`** | **The Fuser**. Physics + ML hybrid model benchmarking. |
| **`physics_bridge`** | **The API**. JSON-IO for Python. |
| **`physics_calibrator`** | **The Optimizer**. Native L-BFGS implementation. |
| **`physics_compute`** | **The Calculator**. Direct physics engine computation. |

---

## 2. Experiment 1: Native Predictive Benchmark

Run the core predictive benchmark without Python overhead.

### Command
```bash
cd src/rust/core
cargo run --release --bin ssot_benchmark
```

### Flags
*   `--release`: Critical. Without this, the debug build is 50x slower due to checked arithmetic.

### Detailed Output Analysis
```text
[RUST] Starting SSOT Benchmark...
[DATA] Loaded 1030 samples from ../../../data/dataset_D1.csv
[PHYS] Calibrating Powers Law Parameters...
       -> Iteration 1: Error=15.4 MPa
       -> Iteration 10: Error=6.71 MPa [CONVERGED]
[EVAL] running Test Set (N=206)...
       -> Sample 0: Pred=34.2, True=35.1, Valid=true
       ...
[RESULT] MAE: 6.7144 | Admissibility: 100.0% | Time: 42ms
```

**Why it matters**: This proves that the physics baseline is fast enough for real-time control (42ms for 200 samples ≈ 0.2ms per sample).

---

## 3. Experiment 2: Generative Design (Inverse Problems)

Can the kernel "invent" a concrete mix?

### Command
```bash
cargo run --release --bin agent_design_benchmark -- \
  --csv ../../../data/dataset_D1.csv \
  --targets 30,40,50,60,70
```

### How it Works
1.  **Goal**: Find vector $x$ such that $|f(x) - \text{Target}| < \epsilon$.
2.  **Constraint**: $\text{Cost}(x) < \$100/m^3$.
3.  **Algorithm**: The kernel uses a gradient-free evolutionary strategy (CMA-ES lite) to explore the mix space.

### Interpreting Success Rate
*   **30-50 MPa**: High success rate (>50%). Easy targets.
*   **70 MPa**: Low success rate (<10%). Requires specific, rare combinations of Silica Fume (not just cement).
*   **Physics Insight**: The kernel correctly "learns" that it needs lower w/c ratio to hit 70 MPa, without being told explicitly.

---

## 4. Experiment 3: "Combo" Physics (Constitutional Evolution)

This tool allows you to turn specific laws of nature "ON" or "OFF" to see their contribution.

### Command
```bash
# Test 1: Only Strength Physics (Powers Law)
cargo run --release --bin experiment_runner -- --engines strength

# Test 2: Strength + Rheology (Flow)
cargo run --release --bin experiment_runner -- --engines strength,rheology

# Test 3: The "Green" Combo (Sustainability + Cost)
cargo run --release --bin experiment_runner -- --engines sustainability,cost
```

### Output
```text
State Report:
 - Strength: 45.2 MPa (Valid)
 - Yield Stress: 120 Pa (Too Stiff! WARN)
 - CO2: 250 kg/m3 (Green!)
```

**Use Case**: Debugging specific failure modes. e.g., "Why is my mix admissible in strength but failing overall?" -> Run combo to see if it's failing the Rheology check (pumpability).

---

## 5. Debugging the Kernel

If you get a `panic!`, you can enable backtraces.

```bash
RUST_BACKTRACE=1 cargo run --release --bin ssot_benchmark
```

### Common Panics
1.  **`IndexOutOfBounds`**: Malformed CSV. Check `dataset_D1.csv` for empty lines.
2.  **`ParseFloatError`**: Non-numeric data in CSV (e.g., "N/A" strings).
3.  **`unwrap() on None`**: Missing column headers. The CSV **must** have headers matching the internal struct (`Cement`, `Water`, etc.).

---

## 6. Extending the Kernel

To add a new law (e.g., Chloride Ingress):

1.  Create `src/science/durability.rs`.
2.  Implement the trait:
    ```rust
    pub trait DurabilityModel {
        fn predict(&self, w_c: f32) -> f32;
    }
    ```
3.  Register it in `mod.rs`.
4.  Add it to the `ThermodynamicState` struct.
5.  Recompile: `cargo build --release`.
