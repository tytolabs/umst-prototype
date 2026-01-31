# Comprehensive Setup Guide
**Universal Installation Protocol for DUMSTO v1.0**

**Objective**: Establish a verified, deterministic reproduction environment on any supported hardware.
**Scope**: Linux (x86_64), macOS (ARM64/Intel), Windows (WSL2), Cloud (AWS/GCP).
**Difficulty**: Beginner to Advanced.

---

## 1. Architecture Support Matrix

Before starting, identify your architecture. The system adapts its build process based on this matrix.

| Platform | OS Version | CPU Arch | Accelerator | Python | Rust | Status |
|---|---|---|---|---|---|---|
| **Linux Workstation** | Ubuntu 20.04+ / Pop!_OS | x86_64 | NVIDIA CUDA 11/12 | 3.8 - 3.11 | 1.75+ | **Verified** |
| **Linux Server** | CentOS 7+ / RHEL | x86_64 | CPU / CUDA | 3.8 - 3.11 | 1.75+ | **Verified** |
| **Apple Silicon** | macOS 12+ (Monterey) | ARM64 (M1/M2/M3) | Metal (MPS) | 3.9 - 3.12 | 1.75+ | **Verified** |
| **Intel Mac** | macOS 10.15+ | x86_64 | CPU Only | 3.8 - 3.11 | 1.75+ | Legacy |
| **Windows** | Win 10/11 Pro | x86_64 | CUDA (via WSL2) | 3.8 - 3.11 | 1.75+ | **Verified** |

---

## 2. Automated Installation (The "Golden Path")

We have developed a master setup wizard that heuristics to detect your environment and configure the toolchain.

### Step 2.1: Run the Wizard
Open your terminal and execute:
```bash
python tools/5_SETUP_TOOL.py
```
*(Note: If `python` is not found, try `python3`)*

### Step 2.2: What the Wizard Does (Under the Hood)
1.  **OS Detection**: It probes `uname -a` and `sys.platform`.
2.  **Dependency Resolution**:
    *   If NVIDIA GPU detected -> Installs `torch` with CUDA.
    *   If macOS detected -> Installs `torch` with MPS support.
    *   Else -> Installs CPU-optimized `torch`.
3.  **Rust Toolchain Check**: It looks for `cargo`. If missing, it offers to install via `rustup`.
4.  **Dataset Integrity**: It calculates SHA256 hashes of files in `data/`.
5.  **Build**: It silently runs `cargo build --release` in `src/rust/core` to prime the physics kernel.

### Step 2.3: Verification
After the wizard completes, run the integrity check:
```bash
./tools/6_VERIFY_TOOL.sh
```
If you see **`[SUCCESS] All checks passed`**, you are ready.

---

## 3. Manual Installation (Deep Dive)

If the automated tool fails, or if you are on an air-gapped system, follow this rigorous manual protocol.

### Phase A: System Dependencies
**Linux (Ubuntu/Debian)**
```bash
sudo apt-get update
sudo apt-get install -y build-essential curl python3-venv python3-dev
```

**macOS**
```bash
xcode-select --install
brew install python@3.11 rust
```

**Windows (WSL2)**
*Prerequisite*: Install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and Ubuntu from the Microsoft Store.
```bash
sudo apt-get update && sudo apt-get install build-essential python3-venv
```

### Phase B: Python Environment
1.  **Create Venv**:
    ```bash
    python3 -m venv dumsto_env
    ```
2.  **Activate**:
    ```bash
    source dumsto_env/bin/activate
    # Windows: dumsto_env\Scripts\activate
    ```
3.  **Install PIP Packages**:
    ```bash
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    ```
4.  **Verify PyTorch**:
    ```bash
    python tools/11_GPU_TEST.py
    ```

### Phase C: The Rust Physics Kernel
This is the most critical step for the DUMSTO-Physics and Hybrid methods.

1.  **Install Rust** (if missing):
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source $HOME/.cargo/env
    ```
2.  **Navigate to Core**:
    ```bash
    cd src/rust/core
    ```
3.  **Compile**:
    ```bash
    cargo build --release
    ```
    *Note: This generates optimization artifacts. First build takes ~2 mins.*
4.  **Verify Binary**:
    ```bash
    ls -l target/release/ssot_benchmark
    # Should be executable and > 2MB
    ```

---

## 4. Platform-Specific Configuration

### NVIDIA GPUs (Linux/Windows)
*   **Drivers**: Ensure NVIDIA drivers v525.00+ are installed. `nvidia-smi` should work.
*   **CUDA Toolkit**: The `pip install torch` wheel includes the CUDA runtime. You generally do **not** need to install System CUDA.
*   **Multi-GPU**: The benchmark defaults to `cuda:0`. To use others, set `export CUDA_VISIBLE_DEVICES=1`.

### Apple Silicon (M1/M2/M3)
*   **MPS Backend**: We use Metal Performance Shaders.
*   **Memory Warning**: If you have 8GB RAM, run:
    ```bash
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
    ```
    This prevents `MPS out of memory` crashes by allowing more aggressive gc.

### Air-Gapped Systems (High Security)
1.  **Download Wheels**: On an internet-connected machine:
    ```bash
    pip download -r requirements.txt -d ./wheels
    ```
2.  **Transfer**: Copy the `umst-prototype` folder and `./wheels` to the secure machine.
3.  **Install**:
    ```bash
    pip install --no-index --find-links=./wheels -r requirements.txt
    ```
4.  **Rust**: Use the `rustup-init` offline installer or pre-compile the binary on a similar architecture and copy the `target/` folder.

---

## 5. Comprehensive Troubleshooting Matrix

| Symptom | Probable Cause | Fix |
|---|---|---|
| **`ModuleNotFoundError: No module named 'torch'`** | Python environment mismatch | Ensure `(dumsto_env)` is active in your prompt. Run `which python` to verify. |
| **`cargo: command not found`** | Rust not in PATH | Run `source $HOME/.cargo/env`. Restart terminal. |
| **`Linker Error` / `ld: library not found`** | Missing C++ build tools | Linux: `sudo apt install build-essential`. Mac: `xcode-select --install`. |
| **`CUDA out of memory`** | GPU VRAM too small | Reduce batch size in `scripts/9_final_comparative_benchmark.py` from 64 to 32. |
| **`MPS backend out of memory`** | Mac Shared Memory limit | Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`. Close browser tabs. |
| **`FileNotFoundError: data/dataset_D1.csv`** | Wrong working directory | Always run scripts from the **root** of the repo (`umst-prototype/`). |
| **`Admissibility: 0%`** | Physics Calibration broken | Run `python scripts/4_physics_calibration.py` to regenerate `calibration_config.json`. |
| **`Permission denied: ./tools/6_VERIFY_TOOL.sh`** | Missing execute bit | Run `chmod +x tools/*.sh`. |

---

## 6. Environment Variables

Advanced users can tune the system using these ENV vars.

*   `DUMSTO_LOG_LEVEL`: Set to `DEBUG` for verbose logging. Default: `INFO`.
*   `DUMSTO_USE_CPU`: Set to `1` to force CPU mode even if GPU is present.
*   `DUMSTO_NUM_THREADS`: Controls Rayon threads in Rust kernel. Default: `# CPUs`.
*   `DUMSTO_SEED`: Override the random seed. Default: `42`.

---

## 7. Verification Checklist

Before reporting an issue, confirm these 5 items:
1.  [ ] `python tools/11_GPU_TEST.py` prints `[OK]`.
2.  [ ] `src/rust/core/target/release/ssot_benchmark` exists.
3.  [ ] `data/calibration_config.json` exists.
4.  [ ] `requirements.txt` is fully installed.
5.  [ ] `tools/6_VERIFY_TOOL.sh` passes all phases.