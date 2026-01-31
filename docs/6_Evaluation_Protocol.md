# Evaluation Protocol
**Metrics, Standards, and Fair Comparison Guidelines**

> **Objective**: To rigorously quantify the performance of Material-AI systems across Accuracy, Safety, and Efficiency.
> **Standard**: The "No Compromise" Benchmark (High Accuracy AND High Safety).

---

## 1. The 12-Metric Evaluation Framework

DUMSTO evaluates models using a holistic scorecard across 12 metrics in 4 categories. We do not just look at accuracy. (The physics kernel additionally tracks 18 material state properties per sample; see `docs/9_Constitutional_Creativity.md` for the full list.)

### Type A: Core Performance (Accuracy)
1.  **MAE (Mean Absolute Error)**: $\frac{1}{n} \sum |y_i - \hat{y}_i|$. The gold standard for engineering (MPa).
2.  **RMSE (Root Mean Square Error)**: Penalizes large outliers (critical for structural safety).
3.  **R² (Coefficient of Determination)**: Variance explained.
4.  **MAPE (Mean Absolute Percentage Error)**: Relative error.

### Type B: Thermodynamic Safety (The Differentiator)
5.  **Admissibility Rate (%)**: The percentage of test predictions that satisfy the Clausius-Duhem inequality.
    *   *Target*: **100.0%**. Anything less implies the model is generating "magic" materials that violate physics.
6.  **Physical Feasibility**: Percentage of predictions within known bounds (0 to 120 MPa).
7.  **Monotonicity Score**: Checks if $\partial(\text{Strength}) / \partial(\text{Cement}) > 0$. Physics dictates adding cement (binder) should generally increase strength.

### Type C: Computational Efficiency
8.  **Training Time (s)**: Wall-clock time to convergence.
9.  **Inference Latency (ms)**: Time to predict one sample. Important for real-time manufacturing control.
    *   *Baseline (cross-dataset mean from `latency_summary` in SSOT)*: GNN (~0.0016ms), XGBoost (~0.0027ms), Rust Kernel (~0.000007ms). Per-dataset values vary; see `fair_comparison_2026-01-28.json`.
10. **Model Size (KB)**: Serialized footprint.

### Type D: Sustainability
11. **CO2 Reduction Potential (%)**: Can the model find a mix with equal strength but lower cement?
12. **Embodied Carbon**: kgCO2e per cubic meter of the predicted mix.

---

## 2. Fair Comparison Protocol

Comparing an XGBoost model (trained in seconds) to a GNN (trained in hours) requires a standardized protocol.

### 2.1 The "Plateau" Standard
We do not train for a fixed number of epochs. We train until **Performance Plateau**.
*   **Neural Methods (GNN, MLP, PINN)**: Use `ReduceLROnPlateau` scheduler. Stop when Validation Loss hasn't improved for 50 epochs.
*   **Tree Methods (XGBoost)**: Use 200 estimators (empirically determined plateau).

### 2.2 Data Rigor
*   **Splits**: Fixed 80/20 Train/Test split. Random Seed = 42.
*   **Normalization**: Z-score normalization for Neural Networks; Raw features for Trees.
*   **Cross-Validation**: 5-Fold CV is used for the benchmark tables.

### 2.3 Physics Calibration
To compare "Physics" to "ML" fairly, the Physics model must be calibrated to the specific dataset materials (cement source, aggregate type).
*   **Procedure**: We use L-BFGS-B optimization to find the $S_{\text{int}}$ and $k_{\text{slag}}$ that minimize MAE on the *Training Set*.
*   **Why**: An uncalibrated physics model is like a neural network with random weights. It's an unfair strawman.

---

## 3. Dataset Characteristics

We evaluate on 4 datasets representing different data quality levels ("The Data Ladder").

| ID | Name | Type | Size | Noise Level | Challenge |
|---|---|---|---|---|---|
| **D1** | UCI Concrete | Lab | 1,030 | Low | Standard benchmark. Solving D1 is "Table Stakes". |
| **D2** | Zenodo NDT | Field | 4,891 | Medium | Real-world field noise. Multi-site variability. |
| **D3** | Zenodo SonReb | Indirect | 2,780 | Medium | Inputs are sensor readings (Ultrasound), not mix design. |
| **D4** | Zenodo RH | Research | 7,445 | **High** | Contains experimental mixes (zero cement, alkali-activated). High variance. |

**The "D4 Test"**: Many ML models fail on D4 because it contains distribution shifts (outliers). This tests Robustness.

---

## 4. Benchmark Results Interpretation

How to read the results in `fair_comparison_2026-01-28.json`:

### Scenario 1: High Accuracy, Low Admissibility
*   **Example**: XGBoost, MLP.
*   **Diagnosis**: The model is "overfitting to the noise". It finds statistical correlations that simulate physics but inevitably violates constraints in corner cases.
*   **Verdict**: Unsafe for generative design.

### Scenario 2: Low Accuracy, High Admissibility
*   **Example**: Physics Baseline (uncalibrated), H-PINN.
*   **Diagnosis**: The constraints are too tight, or the physics model is too simple (doesn't capture aggregate effects).
*   **Verdict**: Safe but not competitive.

### Scenario 3: High Accuracy, High Admissibility
*   **Example**: **DUMSTO-Hybrid**.
*   **Diagnosis**: The "Goldilocks" zone. The Physics Kernel provides the safe manifold, and the ML Component optimizes movement on that manifold.
*   **Verdict**: The desired state.

---

## 5. Metrics Implementation Details

### Calculating Admissibility
```python
def check_admissibility(y_pred, state):
    # Clausius-Duhem Check
    if state.dissipation < 0: return False
    
    # Mass Balance Check
    if abs(state.mass_in - state.mass_out) > 1e-5: return False
    
    return True
```

### Calculating Monotonicity
We perturb the input $x$ by adding $\epsilon$ to the Cement feature.
$$ \text{Score} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(f(x_i + \epsilon) > f(x_i)) $$
Ideal score is > 95% (allowing for some noise/dilution effects).

---

## 6. Reproducibility Guarantee

Our results are **Deterministic**.
*   **Python**: `torch.manual_seed(42)`, `np.random.seed(42)`.
*   **Rust**: Deterministic floating-point ops (no race conditions in reduction).
*   **Hardware**: We report `float32` precision results to account for GPU differences.

If your reproduction varies by > 0.1 MAE, check your:
1.  Python version (3.8 vs 3.12 floats).
2.  PyTorch backend (MKL vs OpenBLAS).
3.  Physics Calibration config.
