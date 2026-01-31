# DUMSTO Reproducibility Package
**Differentiable Unified Material-State Tensor Optimization**

**Santhosh Shyamsundar, Prabhu S., and Studio Tyto**

---

## 🔬 Reproducibility Statement

This package provides complete reproducibility for the DUMSTO framework. All experimental results can be reproduced using the provided code and data.

### Key Claims Verified:
- ✅ **100% Thermodynamic Admissibility** across all trained DUMSTO variants (Physics, Hybrid, PPO) -- verified via Clausius-Duhem gate
- ✅ **Competitive accuracy** on 4 benchmark datasets
- ✅ **785x inference speedup** vs H-PINN (ratio of cross-dataset mean latencies: 0.0057ms / 0.000007ms; per-dataset range: 461x--1207x)
- ✅ **Deterministic results** across platforms

---

## 📋 Quick Start (5 minutes)

```bash
# 1. Setup environment
python tools/5_SETUP_TOOL.py

# 2. Verify installation
bash tools/6_VERIFY_TOOL.sh

# 3. Run main benchmark (30-45 minutes)
python tools/7_RUN_EXPERIMENTS.py benchmark
```

---

## 📊 Benchmark Results Summary

| Method | D1 MAE (MPa) | D4 MAE (MPa) | Safety (%) |
|--------|-------------|-------------|-----------|
| **DUMSTO-Hybrid (Ours)** | **2.99** | **14.61** | **100** |
| XGBoost | 3.05 | 14.62 | 96.0--99.7 |
| PINN | 4.56 | 16.04 | 96.8--99.9 |
| H-PINN | 8.33 | 16.09 | 88.7--99.9 |
| GNN | 9.48 | 17.24 | 97.4--99.9 |

*Full 8-method comparison (including MLP, Physics, PPO) available in the SSOT file.*

**Single Source of Truth**: `results/ssot/fair_comparison_2026-01-28.json`

---

## 🏗️ System Requirements

- **Python**: 3.8-3.11
- **Rust**: 1.75+
- **Memory**: 8GB RAM minimum
- **Storage**: 100MB free space
- **Platforms**: Linux, macOS, Windows (WSL2)

---

## 📁 Package Structure

```
umst-prototype/
├── docs/                 # Documentation suite
├── scripts/              # Python research code
├── src/rust/core/        # High-performance physics kernel
├── data/                 # Benchmark datasets (4.9MB)
├── results/              # Pre-computed results for verification
├── tools/                # Setup and verification utilities
└── .github/workflows/    # CI/CD pipeline
```

---

## 🔧 Detailed Reproduction Protocol

### Phase 1: Environment Setup
```bash
python tools/5_SETUP_TOOL.py  # Automated setup
bash tools/6_VERIFY_TOOL.sh   # System verification
```

### Phase 2: Data Integrity Check
```bash
python -c "
import pandas as pd
datasets = ['D1', 'D2', 'D3', 'D4']
for d in datasets:
    df = pd.read_csv(f'data/dataset_{d}.csv')
    print(f'{d}: {len(df)} samples')
"
```

### Phase 3: Execute Benchmarks
```bash
# Main comparative benchmark
python scripts/9_final_comparative_benchmark.py

# Alternative: Use the unified launcher
python tools/7_RUN_EXPERIMENTS.py benchmark
```

### Phase 4: Verify Results
```bash
# Compare against SSOT
python -c "
import json
ssot = json.load(open('results/ssot/fair_comparison_2026-01-28.json'))
hybrid_d1 = next(r for r in ssot['comparative_results'] if r['dataset'] == 'D1')
print(f'Hybrid D1 MAE: {hybrid_d1[\"hybrid_mae\"]:.2f} MPa')
"
```

---

## 🎯 Expected Runtime

| Operation | Time | Output |
|-----------|------|--------|
| Setup | 2-5 min | Environment ready |
| Verification | 1 min | System check passed |
| Data integrity | <1 min | All datasets validated |
| Main benchmark | 30-45 min | `fair_comparison_2026-01-28.json` |
| Results verification | <1 min | Claims confirmed |

---

## 🔍 Troubleshooting

### Common Issues

**Q: Rust compilation fails**
```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

**Q: Python imports fail**
```bash
# Recreate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

**Q: Memory errors during benchmark**
```bash
# Reduce batch size in scripts
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## 📈 Performance Validation

### Deterministic Results
All computations use fixed random seeds:
- **Python**: `torch.manual_seed(42)`, `np.random.seed(42)`
- **Rust**: Deterministic floating-point operations

### Platform Compatibility
- **Linux**: Primary development platform
- **macOS**: Full support with MPS acceleration
- **Windows**: WSL2 compatibility verified

---

## 🤝 Support

For questions about reproduction:
1. Check `docs/4_Reproduction_Guide.md`
2. Run `python tools/8_DIAGNOSTICS_TOOL.py`
3. File issue with diagnostic output

---

## 📜 License

MIT License - See `LICENSE` file.

**Copyright (c) 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto**