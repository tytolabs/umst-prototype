# UMST Prototype: Physics-Gated AI for Material Design

![Teaser Figure](results/plots/fig1_teaser.png?raw=true)

## Overview
This repository contains the official implementation of the **Differentiable Unified Material-State Tensor Optimization (DUMSTO)** framework.

DUMSTO is a hybrid AI architecture that integrates rigorous thermodynamic constraints into deep learning pipelines. By enforcing the **Clausius-Duhem inequality** as a hard gate, DUMSTO ensures **100% thermodynamic admissibility** for all material predictions and designs, bridging the gap between data-driven flexibility and physics-based safety.

## Key Results (Single Source of Truth)
All results are verifiable against the Single Source of Truth (SSOT) benchmark logs located in `results/ssot/`. These metrics cover 16,146 real-world samples across four distribution-shifted datasets.

| Method | D1 Accuracy (MAE) | Global Safety (Admissibility) | Inference Speed |
|---|---|---|---|
| **DUMSTO-Hybrid (Ours)** | **2.99 MPa** | **100.0%** (Guaranteed) | < 0.01 ms |
| XGBoost (Unconstrained) | 3.05 MPa | 98.0% (2% viol.) | 2.7 µs |
| H-PINN (Hard Constr.) | 8.33 MPa | 88.7% (11% viol.) | 5.7 µs |
| GNN (MatGL) | 9.48 MPa | 97.4% (2.6% viol.) | 1.6 µs |

**Generative Creativity:**
The DUMSTO-PPO agent, operating under the constitutional gate, discovered **61 distinct Pareto-optimal designs** (Yield) across **9 SCM coverage regimes**, achieving a coverage score of **0.678**—significantly outperforming unconstrained evolutionary baselines.

## Architecture
1.  **Unified Material-State Tensor (UMST):** A sparse, multi-scale representation ($\mathbb{R}^{64}$) encoding Material, Physics, Process, Environment, and Time states.
2.  **Physics Kernel (Rust):** A high-performance, differentiable engine implementing 16 constitutive laws (Powers, hydration kinetics, gel-space ratio).
3.  **Thermodynamic Gate:** A "mathematical firewall" that rejects any state transition violating entropy production constraints ($\mathcal{D}_{\text{int}} \ge 0$).
4.  **DUMSTO-PPO:** A constitutional reinforcement learning agent with 6 reward modes (Balance, Strength, Sustainability, Durability, Cost, Printability).

## Repository Structure
```
umst-prototype/
├── data/                   # Verified Datasets (D1-D4)
├── docs/                   # Detailed Documentation
├── results/
│   ├── ssot/               # Single Source of Truth JSONs
│   └── plots/              # Generated Figures
├── scripts/                # Python Training & Analysis Scripts
│   ├── benchmark_predictive.py
│   ├── benchmark_generative.sh
│   ├── calibrate_physics.py
│   └── ...
├── src/
│   └── rust/               # Core Physics Kernel (Rust)
└── tools/                  # Verification & Setup Utilities
```

## Quick Start

### Prerequisites
- **Python 3.10+**
- **Rust (Cargo 1.75+)** for the physics kernel.

### Setup
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Build Rust Kernel
cd src/rust/core
cargo build --release
```

### Running Benchmarks
To reproduce the **Predictive Power** results (Table 2):
```bash
python scripts/benchmark_predictive.py
```
*Output: `results/ssot/fair_comparison_2026-01-28.json`*

To reproduce the **Generative Design** results (Table 4 & Figure 5):
```bash
./scripts/benchmark_generative.sh
```
*Output: `results/ssot/design_benchmark_latest.json`*

## Datasets
We utilize a **Composite Global Benchmark** ($N=16,146$) merging four sources:
- **D1 (UCI Concrete):** The canonical compressive strength dataset ($N=1,030$).
- **D2 (Zenodo NDT):** Non-destructive testing data ($N=4,891$). (License: CC-BY 4.0)
- **D3 (Zenodo Sun):** Solar reflectance and thermal mass ($N=2,780$).
- **D4 (Zenodo RH):** Relative humidity / curing data ($N=7,445$).

All data is pre-processed and located in `data/`.

## Research Publications

For the complete scientific and mathematical foundations of UMST:

- **[Scientific and Mathematical Foundations](https://github.com/studiotyto/umst-research/blob/main/foundations/UMST_Science_Mathematics.pdf)** (71 pages)  
  Complete theoretical derivation, thermodynamic proofs, and empirical validation across 16,146 samples.

- **[Technical Summary](https://github.com/studiotyto/umst-research/blob/main/foundations/UMST_Technical_Summary.pdf)** (8 pages)  
  Concise overview of the hybrid architecture, physics-gated AI approach, and benchmark results.

## Citation
If you use this code or dataset, please cite:
```bibtex
@software{dumsto2026,
  title={UMST: Unified Material-State Tensors for Physics-Gated AI},
  author={Shyamsundar, Santhosh and Prabhu, S. and Studio Tyto},
  year={2026},
  url={https://github.com/studiotyto/umst-prototype}
}
```

## License
MIT License. See `LICENSE` for details.
