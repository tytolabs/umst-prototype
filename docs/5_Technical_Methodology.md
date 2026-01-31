# Technical Methodology
**Mathematical Foundations of the DUMSTO Framework**

**Abstract**: This document details the theoretical basis of the Differentiable Unified Material-State Tensor Optimization framework. It covers the Graph-based State Representation, the Constitutive Physics Kernel, and the Thermodynamic Admissibility Filters.

---

## 1. The Unified State Hypothesis

Classic materials informatics treats a material sample as a flat feature vector $v \in \mathbb{R}^n$.

$$
\mathbf{v} = [\text{Cement}, \text{Water}, \text{Age}, \text{Temperature}]
$$

**Hypothesis**: This representation is insufficient because it discards the *semantic* and *spatial* structure of the physical system. A material is not a list; it is a hierarchy of interacting phases.

**Solution**: We define the material state as a **HyperGraph Tensor** $\mathbf{G}$:

$$
\mathbf{G} = (\mathcal{V}, \mathcal{E}, \psi)
$$

### 1.1 The Tensor Nodes ($\mathcal{V}$)
The nodes represent physical entities at different scales.
*   **Material Nodes ($\mathcal{V}_m$)**: Represent phases (e.g., C-S-H Gel, Calcium Hydroxide, Pore Water). They hold properties like Viscosity ($\eta$) and Yield Stress ($\tau_y$).
*   **Geometric Nodes ($\mathcal{V}_g$)**: Represent spatial occupancy (e.g., Voxel grids, SDFs). They enforce boundary conditions.
*   **Kinematic Nodes ($\mathcal{V}_k$)**: Represent the actuator/robot state (e.g., 3D Printer Nozzle position).

### 1.2 The Tensor Constraints ($\mathcal{E}$)
The edges represent physical couplings.
*   **Constitutive Edges**: Connect material phases (e.g., Hydration transforms Water + Cement -> Gel).
*   **Boundary Edges**: Connect Material to Geometry (e.g., "Concrete cannot be placed outside the mold").

---

## 2. The Constitutive Physics Kernel

The Kernel is the differentiable engine that evolves the state $\mathbf{G}$ over time. It is not learned; it is derived from First Principles.

### 2.1 Hydration Kinetics (The Avrami-Parrott Model)
Hydration is the fundamental process of concrete strength gain. We model the degree of hydration $\alpha(t)$ using a modified Avrami equation.

$$
\frac{d\alpha}{dt} = \frac{k_{\text{ref}} \cdot S_{\text{int}}}{1 + (t/t_0)^n} \cdot \beta\left(\frac{w}{c}\right) \cdot \lambda(T)
$$

**Variables:**

| Symbol | Definition |
| :--- | :--- |
| $\alpha \in [0, 1]$ | Degree of hydration (State Variable) |
| $k_{\text{ref}}$ | Reference reaction rate (Calibration Parameter) |
| $\lambda(T)$ | Arrhenius thermal activation term |

$$
\lambda(T) = \exp \left[ \frac{E_a}{R} \left( \frac{1}{T_{ref}} - \frac{1}{T} \right) \right]
$$

**Implementation**: `src/rust/core/src/science/maturity.rs`

### 2.2 Microstructure Evolution (Powers' Gel-Space Ratio)
Strength is not a direct function of time, but of microstructure. We use Powers' Gel-Space Ratio theory.

The volume of gel solids $V_{\text{gel}}$ is:

$$
V_{\text{gel}} = 0.68 \cdot \alpha \cdot C_0
$$

The total available space $V_{\text{space}}$ is:

$$
V_{\text{space}} = 0.32 \cdot \alpha \cdot C_0 + V_{\text{water}} + V_{\text{air}}
$$

The **Gel-Space Ratio** $X$ is:

$$
X = \frac{V_{\text{gel}}}{V_{\text{space}}}
$$

The Compressive Strength $f_c$ is then:

$$
f_c = S_{\text{int}} \cdot X^n
$$

**Variables:**

| Symbol | Definition |
| :--- | :--- |
| $S_{\text{int}}$ | Intrinsic Strength of the C-S-H gel (approx 240 MPa, calibrated per dataset) |
| $n$ | Power law exponent (typically 3.0) |

**Why this matters**: This model naturally handles the effect of Water-Cement Ratio ($w/c$). As $w/c$ increases, $V_{\text{space}}$ increases, $X$ decreases, and Strength drops. Pure ML models must "learn" this; our Kernel "knows" it.

---

## 3. Thermodynamic Admissibility (The Guardrail)

Safety in AI is usually defined statistically. In DUMSTO, safety is defined thermodynamically.

**Theorem 1 (Admissibility)**: A material state trajectory $\mathcal{T}$ is physically valid if and only if it satisfies the Clausius-Duhem inequality at every time step.

$$
\mathcal{D}_{\text{int}} = \sigma : \dot{\varepsilon} - \rho(\dot{\psi} + s\dot{T}) - \vec{q} \cdot \frac{\nabla T}{T} \geq 0
$$

**Variables:**

| Symbol | Definition |
| :--- | :--- |
| $\mathcal{D}_{\text{int}}$ | Internal Entropy Production Rate (Must be $\geq 0$) |
| $\psi$ | Helmholtz Free Energy (Potential) |
| $s$ | Specific Entropy |

### 3.1 Implementation in Rust
In `src/rust/core/src/science/thermodynamic_filter.rs`, we simplify this for the chemo-mechanical system of concrete:

$$
\mathcal{D}_{\text{hydration}} = \mathcal{A} \cdot \dot{\xi} \geq 0
$$

Where $\mathcal{A}$ is the thermodynamic affinity of the hydration reaction and $d\xi/dt$ is the reaction rate.

**The Filter Logic**:
1.  Receive predicted state $\hat{y}$ from ML model.
2.  Compute $\mathcal{D}_{\text{int}}(\hat{y})$.
3.  If $\mathcal{D}_{\text{int}} < 0$: **REJECT**. Return gradient $\nabla \mathcal{D}_{\text{int}}$ to penalize the model.
4.  If $\mathcal{D}_{\text{int}} \geq 0$: **ACCEPT**.

---

## 4. The Hybrid Architecture

How do we combine the Physics Kernel with Machine Learning? We use a **Residual Correction** architecture.

### 4.1 The Formula

$$
f_{\text{hybrid}}(x) = f_{\text{physics}}(x; \theta_{\text{calib}}) + \mathcal{M}_{\text{ML}}(x)
$$

**Variables:**

| Symbol | Definition |
| :--- | :--- |
| $f_{\text{physics}}$ | The output of the Constitutive Kernel (Powers' Law) |
| $\mathcal{M}_{\text{ML}}$ | A Gradient Boosting Regressor (XGBoost) trained on the residual error $R = y_{\text{true}} - f_{\text{physics}}$ |

### 4.2 The Benefit
*   **Interpolation**: The ML model learns the complex 2nd-order interactions (e.g., aggregate shape, angularity) that the physics model ignores.
*   **Extrapolation**: Far from training data, the ML prediction variance typically explodes. In DUMSTO, we bound the ML residual:
    
    $$
    | \mathcal{M}_{\text{ML}}(x) | \leq \delta \cdot f_{\text{physics}}(x)
    $$
    This ensures the model falls back to the robust physics baseline in unknown territory (D4 dataset).

---

## 5. Differentiable Optimization

Because the entire Kernel is written in Rust with analytic gradients, we can perform **Inverse Design** using Gradient Descent.

**Problem**: Find mix $x$ such that Strength($x$) = 50 MPa and Cost($x$) is minimized.

$$
\min_x \mathcal{L} = (f_{\text{hybrid}}(x) - 50)^2 + \lambda \cdot \text{Cost}(x)
$$

Since $\partial f_{\text{hybrid}} / \partial x$ is computable, we can use PPO (Proximal Policy Optimization) or SGD to solve this efficiently.

---

## 6. References

1.  **Powers, T.C.** (1947). *A Working Hypothesis for Further Studies of Frost Resistance of Concrete.* formula derivation.
2.  **Parrott, L.J.** (1990). *Modeling Hydration Kinetics.*
3.  **Coleman, B.D. & Gurtin, M.E.** (1967). *Thermodynamics with Internal State Variables.* (Basis of Clausius-Duhem).
