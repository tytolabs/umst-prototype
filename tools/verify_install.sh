#!/bin/bash
# SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
# SPDX-License-Identifier: MIT
# DUMSTO Comprehensive Verification Tool
# Supports: Linux (x86_64/NVIDIA, x86_64/CPU) and macOS (Apple Silicon/Intel)

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}   DUMSTO PROTOTYPE VERIFICATION TOOL (Cross-Platform)          ${NC}"
echo -e "${BLUE}================================================================${NC}"
echo "Date: $(date)"
echo ""

# -------------------------------------------------------------------------
# SETUP: Locate Tools Directory & Virtual Environment
# -------------------------------------------------------------------------
TOOL_DIR=$(dirname "$0")
cd "$TOOL_DIR"/.. || exit # Go to root

# Auto-activate venv if it exists
if [ -d "dumsto_env" ]; then
    echo -e "${GREEN}Found virtual environment 'dumsto_env'. Activating...${NC}"
    source dumsto_env/bin/activate
elif [ -d "venv" ]; then
    echo -e "${GREEN}Found virtual environment 'venv'. Activating...${NC}"
    source venv/bin/activate
fi

# Go back to tools for consistency if needed, but usually we run from root
cd "$TOOL_DIR" || exit

# -------------------------------------------------------------------------
# PHASE 1: System Identification
# -------------------------------------------------------------------------
echo -e "${YELLOW}PHASE 1: System Identification${NC}"
OS="$(uname -s)"
ARCH="$(uname -m)"
echo "Operating System: $OS"
echo "Architecture: $ARCH"

PROBABLE_PLATFORM="Unknown"

if [ "$OS" == "Linux" ]; then
    if command -v nvidia-smi &> /dev/null; then
        echo "Hardware: NVIDIA GPU detected"
        PROBABLE_PLATFORM="Linux (NVIDIA GPU)"
    else
        echo "Hardware: CPU-only or non-NVIDIA GPU"
        PROBABLE_PLATFORM="Linux (CPU/Generic)"
    fi
elif [ "$OS" == "Darwin" ]; then
    if [ "$ARCH" == "arm64" ]; then
        echo "Hardware: Apple Silicon"
        PROBABLE_PLATFORM="macOS (Apple Silicon)"
    else
        echo "Hardware: Intel Mac"
        PROBABLE_PLATFORM="macOS (Intel)"
    fi
fi

echo -e "Detected Platform profile: ${GREEN}$PROBABLE_PLATFORM${NC}"
echo ""

# -------------------------------------------------------------------------
# PHASE 2: Core Dependencies Check
# -------------------------------------------------------------------------
echo -e "${YELLOW}PHASE 2: Core Dependencies Check${NC}"

# Check Python
PYTHON=${PYTHON:-python3}
if command -v $PYTHON &> /dev/null; then
    PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo -e "Python 3: ${GREEN}Found ($PY_VER)${NC}"
else
    echo -e "Python 3: ${RED}MISSING${NC}"
    exit 1
fi

# Check Pip
if command -v pip3 &> /dev/null || python3 -m pip --version &> /dev/null; then
    echo -e "Pip: ${GREEN}Found${NC}"
else
    echo -e "Pip: ${RED}MISSING${NC} (Required for setup)"
    exit 1
fi

# Check Rust (Optional but recommended)
if command -v cargo &> /dev/null; then
    RUST_VER=$(cargo --version | awk '{print $2}')
    echo -e "Rust/Cargo: ${GREEN}Found ($RUST_VER)${NC}"
else
    echo -e "Rust/Cargo: ${YELLOW}Not found (Optional - using Python-only mode)${NC}"
fi

echo ""

# -------------------------------------------------------------------------
# PHASE 3: Python Environment & Tensorcheck
# -------------------------------------------------------------------------
echo -e "${YELLOW}PHASE 3: Deep Environment Verification${NC}"

# Use 5_SETUP_TOOL.py for detailed check
if [ -f "5_SETUP_TOOL.py" ]; then
    echo "Running internal setup check..."
    $PYTHON 5_SETUP_TOOL.py --check
    if [ $? -eq 0 ]; then
        echo -e "Internal setup check: ${GREEN}PASSED${NC}"
    else
        echo -e "Internal setup check: ${RED}FAILED${NC}"
        echo "Run '$PYTHON 5_SETUP_TOOL.py' to fix."
    fi
else
    echo -e "${RED}ERROR: 5_SETUP_TOOL.py not found!${NC}"
    exit 1
fi

# Hardware Acceleration Check
echo "Checking Hardware Acceleration..."
$PYTHON -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
if torch.cuda.is_available():
    print('Accelerator: CUDA (NVIDIA)')
    print(f'Device: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('Accelerator: MPS (Apple Silicon)')
else:
    print('Accelerator: CPU (Fallback)')
"
echo ""

# -------------------------------------------------------------------------
# PHASE 4: Functional Verification via Quick Benchmark
# -------------------------------------------------------------------------
echo -e "${YELLOW}PHASE 4: Functional Verification (Running Quick Benchmark)${NC}"
echo "Running '$PYTHON 7_RUN_EXPERIMENTS.py quick'..."
echo "This tests the full pipeline: Data Loading -> Physics Kernel -> ML Training -> Validation"

$PYTHON 7_RUN_EXPERIMENTS.py quick

if [ $? -eq 0 ]; then
    echo -e "Quick Benchmark: ${GREEN}SUCCESS${NC}"
else
    echo -e "Quick Benchmark: ${RED}FAILED${NC}"
    echo "Check error logs above."
    exit 1
fi

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}   VERIFICATION COMPLETE - SYSTEM READY FOR REPRODUCTION        ${NC}"
echo -e "${BLUE}================================================================${NC}"
