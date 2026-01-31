# Supplementary Materials
**Derivations, Data Dictionary, and License Information**

**Purpose**: Provides deep context, theoretical proofs, and legal information supporting the repository.

---

## A. Theoretical Derivations

### A.1 Thermodynamic Admissibility (The Filter)

The core innovation of DUMSTO is the hard thermodynamic filter. Here we derive it from first principles.

**Step 1: The Second Law**
For any material volume $\Omega$, the Clausius-Duhem inequality states:

$$
\mathcal{D}_{\text{int}} = \sigma : \dot{\varepsilon} - \rho \dot{\psi} - \rho s \dot{T} - \frac{\vec{q} \cdot \nabla T}{T} \geq 0
$$

**Step 2: Isothermal Assumption**
For concrete curing at controlled temperature, $\nabla T \approx 0$ and $\dot{T} \approx 0$.

$$
\mathcal{D}_{\text{int}} = \sigma : \dot{\varepsilon} - \rho \dot{\psi} \geq 0
$$

**Step 3: Free Energy Potential**
We define the Helmholtz free energy $\psi$ as a function of the state variables (Strain $\varepsilon$, Hydration $\alpha$):

$$
\psi = \psi(\varepsilon, \alpha)
$$

Taking the time derivative:

$$
\dot{\psi} = \frac{\partial \psi}{\partial \varepsilon} \dot{\varepsilon} + \frac{\partial \psi}{\partial \alpha} \dot{\alpha}
$$

**Step 4: Substitution**
Substitute the constitutive relations into the dissipation inequality:

$$
\sigma : \dot{\varepsilon} - \rho \left(\frac{\partial \psi}{\partial \varepsilon} \dot{\varepsilon} + \frac{\partial \psi}{\partial \alpha} \dot{\alpha}\right) \geq 0
$$

Grouping terms by state variable:

$$
\left(\sigma - \rho \frac{\partial \psi}{\partial \varepsilon}\right) : \dot{\varepsilon} - \rho \frac{\partial \psi}{\partial \alpha} \dot{\alpha} \geq 0
$$

**Step 5: Constitutive Relations**
Standard thermodynamic arguments (Coleman-Gurtin) implies that the term in the parenthesis must vanish for arbitrary rates $\dot{\varepsilon}$. Thus, we defined:
*   **Stress**: $\sigma = \rho \frac{\partial \psi}{\partial \varepsilon}$
*   **Thermodynamic Affinity**: $A = -\rho \frac{\partial \psi}{\partial \alpha}$

The inequality simplifies to the *Reduced Dissipation Inequality*:

$$
\mathcal{D}_{\text{int}} = A \cdot \dot{\alpha} \geq 0
$$

**Conclusion**: Since hydration is an irreversible process, the reaction rate $\dot{\alpha}$ must always align with the thermodynamic driving force $A$.
*   **Physical Meaning**: Hydration ($\dot{\alpha} > 0$) naturally proceeds when it lowers the Free Energy of the system ($A > 0$).
*   **Implementation**: Our Rust filter explicitly checks this product $\mathcal{A} \cdot \dot{\alpha} \geq 0$.

---

## B. Data Dictionary

The benchmarks rely on 4 CSV datasets. Here is the schema.

### Schema (Common to D1-D4)
| Column | Unit | Description | Range |
|---|---|---|---|
| `Cement` | kg/m³ | Ordinary Portland Cement (OPC). Primary binder. | 100 - 540 |
| `Slag` | kg/m³ | Blast Furnace Slag. Supplementary cementitious material. | 0 - 360 |
| `FlyAsh` | kg/m³ | Pulverized Fuel Ash. SCM. | 0 - 200 |
| `Water` | kg/m³ | Mix water. | 120 - 250 |
| `Superplas` | kg/m³ | High-range water reducer. | 0 - 30 |
| `CoarseAggr` | kg/m³ | Gravel/Rock (>4mm). | 800 - 1100 |
| `FineAggr` | kg/m³ | Sand (<4mm). | 600 - 900 |
| `Age` | Days | Curing time. | 1 - 365 |
| **`Strength`** | **MPa** | **Target Variable**. Compressive Strength ($f_c$). | 0 - 90 |

### Dataset Characteristics

**D1: UCI Concrete**
*   **Origin**: Prof. I-Cheng Yeh (1998).
*   **Quality**: High (Lab controlled).
*   **Bias**: Slight over-representation of high-strength mixes.

**D2: Zenodo NDT**
*   **Origin**: Multi-site field collection.
*   **Quality**: Medium/Noisy.
*   **Feature**: Includes Non-Destructive Testing (NDT) columns (Rebound Number `R`, Pulse Velocity `V`) which are ignored in the standard benchmark but used in the "SonReb" experiments.

**D3: Zenodo SonReb**
*   **Origin**: Ultrasonic Pulse Velocity + Rebound Hammer data.
*   **Quality**: Medium (Indirect measurements).
*   **Feature**: Inputs are sensor readings rather than mix design, testing model generalization to indirect measurement data.

**D4: Zenodo RH (Research)**
*   **Origin**: Aggregated from 20+ research publications.
*   **Quality**: High Variance.
*   **Challenge**: Contains "Zero Cement" mixes (Geopolymers) where `Cement=0`. This tests the robustness of the physics kernel (avoiding divide-by-zero).

---

## C. Legacy Code & Archives

The `archive/` folder contains artifacts from the development history of DUMSTO.

*   `checkpoints_archive/`: Pre-trained PyTorch weights for the RL agent (v0.8).
*   `_archive.tar.gz`: Source code for the v1 and v2 benchmarks.
*   `adversarial_results.csv`: Raw data from the Safety Stress Test.

These are provided for transparency but are **not** required for the SSOT reproduction.

---

## D. Third-Party Licenses

This package includes/links to open-source software:
### Included Open-Source Software
| Component | License | Usage |
|---|---|---|
| **PyTorch** | BSD-Style | Deep Learning Framework |
| **Rust** | MIT / Apache-2.0 | Systems Programming Language |
| **Scikit-Learn** | BSD-3-Clause | ML Utilities |
| **PetGraph** | MIT / Apache-2.0 | Graph Algorithms |

### DUMSTO License
The DUMSTO code itself is released under the **MIT License**.
*   **Permissions**: Commercial use, Modification, Distribution, Private use.
*   **Conditions**: License and copyright notice must be included.
