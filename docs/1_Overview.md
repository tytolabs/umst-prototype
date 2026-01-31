# DUMSTO: Differentiable Unified Material-State Tensor Optimization
**Functional Materials Science Toolkit & Reproducibility Package**

**Version**: 1.0.0 | **License**: MIT | **Platform**: Linux / macOS / Windows / Cloud

---

## 1. Evaluation Philosophy: Why DUMSTO?

### The Crisis in Materials Informatics
Traditional machine learning (ML) applied to materials science often treats physical systems as "black box" regression problems. A neural network maps inputs (e.g., cement, water) to outputs (e.g., strength) without understanding the fundamental laws of nature.

This leads to three critical failures:
1.  **Thermodynamic Violations**: Models predict impossible states (e.g., negative entropy production, simultaneous mass creation).
2.  **Extrapolation Failure**: Data-driven models fail catastrophically outside their training distribution (e.g., high-variance field data).
3.  **Safety Risks**: In safety-critical domains like civil infrastructure, a "95% accurate" black box is unacceptable if the 5% error leads to structural collapse.

### The DUMSTO Solution: First Principles AI
The **DUMSTO** framework (Differentiable Unified Material-State Tensor Optimization) solves this by embedding physics directly into the computation graph.

*   **We do not just "add a loss term".** Soft constraints (PINNs) are insufficient because they can be violated.
*   **We enforce Hard Constraints.** The DUMSTO Kernel is a Differentiable Physics Engine that projects all ML predictions onto the manifold of thermodynamically valid states.

**Result**: An AI system designed to systematically reject any material design that violates fundamental thermodynamic constraints (within numerical tolerance).

---

## 2. System Architecture

The DUMSTO architecture consists of three distinct layers working in unison.

### Layer 1: The HyperGraph Tensor (`/src/rust/core/src/tensors`)
Unlike flat feature vectors, DUMSTO represents materials as a **HyperGraph**.
*   **Nodes**: Represent physical entities (Material Phases, Geometric Voxels, Kinematic Joints).
*   **Edges**: Represent physical couplings (Rheology constraints, Collision boundaries).
*   **Purpose**: This allows the AI to reason about *structure*. A material is not just a list of ingredients; it is a spatial arrangement of phases.

### Layer 2: The Constitutive Physics Kernel (`/src/rust/core/src/science`)
A high-performance Rust backend encoding proven laws of nature:
*   **Hydration**: Avrami-Parrott kinetics for phase evolution.
*   **Microstructure**: Powers' Gel-Space Ratio for strength development.
*   **Rheology**: Herschel-Bulkley fluid dynamics for flow.
*   **Sustainability**: Embodied carbon and Global Warming Potential (GWP) accounting.

### Layer 3: The Hybrid Learning Agent (`/scripts`)
A machine learning layer (XGBoost or PPO) that learns to *correct* the physics kernel.

$$
\text{Prediction} = f_{\text{Physics}}(x) + \mathcal{R}_{\text{ML}}(x)
$$

*   **Formula**: Physics Baseline + Residual Correction.
*   **Safety Gate**: The output is passed through a **Thermodynamic Filter** (Clausius-Duhem Inequality) before being accepted.

---

## 3. User Personas & Workflows

This toolkit is designed for three distinct types of users. Identify your persona to find the right workflow.

### Type A: The Reviewer / Auditor
**Goal**: Verify the scientific claims with minimal friction.
**Recommended Workflow**:
1.  **Setup**: Run `python tools/5_SETUP_TOOL.py` (Automated).
2.  **Verification**: Run `./tools/6_VERIFY_TOOL.sh` to certify the detailed system state.
3.  **Reproduction**: Run `python tools/7_RUN_EXPERIMENTS.py benchmark`.
4.  **Result Inspection**: Open `results/ssot/` and compare with Table 2 in the documentation.

### Type B: The Materials Scientist
**Goal**: Use the physics kernel to model a new material system (e.g., Geopolymers).
**Recommended Workflow**:
1.  **Explore**: Read `docs/5_Technical_Methodology.md` to understand the tensor structure.
2.  **Extend**: Modify `src/rust/core/src/science/mod.rs` to add a new constitutive model.
3.  **Calibrate**: Use `scripts/4_physics_calibration.py` to fit your model to your experimental data (`data/my_experiment.csv`).
4.  **Visualize**: Use `scripts/5_plot_results.py` to generate publication-quality figures.

### Type C: The AI Researcher
**Goal**: Benchmark a new ML architecture (e.g., Transformer) against DUMSTO.
**Recommended Workflow**:
1.  **Benchmark**: Run the "Fair Comparison" suite (`scripts/9_final_comparative_benchmark.py`).
2.  **Innovate**: Replace the XGBoost residual layer in `scripts/8_hybrid_plugin.py` with your custom PyTorch model.
3.  **Evaluate**: Use the rigorous `docs/6_Evaluation_Protocol.md` metrics (Admissibility, OOD Stability) to prove your model's superiority.

---

## 4. Quick Start Protocol

For the majority of users, the following sequence provides the most direct path to utilizing the package.

### Step 1: Environment Auto-Config
We provide a unified wizard that handles Python, Rust, and System dependencies.
```bash
python tools/5_SETUP_TOOL.py
```
*Action*: Checks RAM, GPU, OS, Python version, Rust compiler, and pip packages. Installs missing components.

### Step 2: System Health Check
Before running heavy compute, verify your toolchain is intact.
```bash
./tools/6_VERIFY_TOOL.sh
```
*Success Criterion*: Look for `[SUCCESS] All checks passed`.

### Step 3: Run the SSOT Benchmark
Execute the exact code used to generate the benchmark results.
```bash
python tools/7_RUN_EXPERIMENTS.py benchmark
```
*Runtime*: 30-45 minutes on GPU; 60+ minutes on CPU.
*Output*: A JSON file in `results/ssot/` containing the MAE and Admissibility scores for all 8 methods.

---

## 5. Frequently Asked Questions (FAQ)

### Q: Why is Rust required?
**A**: The core physics kernel is written in Rust for **differentiability** and **performance**. Python is too slow for the complex tensor operations required for real-time thermodynamic filtering (10,000+ checks per second). The Rust kernel compiles to a high-speed binary that the Python scripts call via subprocess or FFI.

### Q: Can I run this without a GPU?
**A**: **Yes.** The system is fully cross-compatible.
*   **Default**: PyTorch uses CUDA (NVIDIA) or MPS (Mac).
*   **Fallback**: If no GPU is found, it falls back to CPU. The results are deterministic and identical, though execution time will increase.

### Q: What is "Thermodynamic Admissibility"?
**A**: It is the mathematical assurance that a material state satisfies the Second Law of Thermodynamics. Specifically, the internal entropy production rate $\mathcal{D}_{\text{int}}$ must be non-negative. If the ML model predicts a state where entropy decreases spontaneously (beyond a tolerance of $10^{-6}$), the Kernel rejects it.

### Q: I am seeing a "Calibration Error" in the logs.
**A**: The physics models (Powers' Law) depend on coefficients like "Intrinsic Strength" ($S_{\text{int}}$). These vary by cement source. If you swap datasets (e.g., from D1 to D2) without re-running `scripts/4_physics_calibration.py`, the model will use mismatched physics parameters. Ensure you run the calibration step if using custom data.

---

## 6. Documentation Map

| ID | Document | Audience | Content Depth |
|---|---|---|---|
| **1** | [**Overview**](1_Overview.md) | General | Value proposition, architecture, FAQs. |
| **2** | [**Directory Structure**](2_Directory_Structure.md) | Developers | Exhaustive file manifest and data flow. |
| **3** | [**Setup Guide**](3_Setup_Guide.md) | Administrators | Detailed installation for Linux, Mac, Windows. |
| **4** | [**Reproduction Guide**](4_Reproduction_Guide.md) | Reviewers | Step-by-step SSOT protocol. |
| **5** | [**Technical Methodology**](5_Technical_Methodology.md) | Scientists | Math equations, Constitutive Laws, Tensors. |
| **6** | [**Evaluation Protocol**](6_Evaluation_Protocol.md) | Researchers | Metric definitions, Fair comparison standards. |
| **7** | [**Supplementary Materials**](7_Supplementary_Materials.md) | Auditors | Data provenance, derivations, legacy code. |
| **8** | [**Advanced Experiments**](8_Advanced_Rust_Experiments.md) | Advanced Users | Rust kernel binaries, generative design. |
| **9** | [**Constitutional Creativity**](9_Constitutional_Creativity.md) | Researchers | PPO design benchmark, 6 reward modes, creativity metrics. |

---

## 7. Beyond Reproduction: Building on DUMSTO

This package is designed as a **Foundation Model for Materials**.
We encourage the community to:
1.  **Fork** the repository.
2.  **Replace** `src/rust/core/src/science` with your domain physics (e.g., Metal Plasticity).
3.  **Train** the Hybrid Agent on proprietary data.
4.  **Deploy** the verified model to production with safety guarantees.
