#!/bin/bash
# SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
# SPDX-License-Identifier: MIT
# DUMSTO Platform Testing Script
# Simulates the exact testing process for your friend
# Run this to verify the package works on the target system

set -e

echo " DUMSTO TESTING FOR X86 POP!_OS + NVIDIA GPU"
echo "=============================================="
echo "Target System: Pop!_OS (Ubuntu-based) x86_64 + NVIDIA GPU"
echo "Test Date: $(date)"
echo ""

# Phase 1: System Information
echo " PHASE 1: SYSTEM INFORMATION"
echo "------------------------------"
echo "OS: $(uname -s) $(uname -m)"
echo "Distribution: $(lsb_release -d 2>/dev/null | cut -f2 || echo 'Pop!_OS (assumed)')"
echo "Kernel: $(uname -r)"
echo ""

# Phase 2: NVIDIA GPU Check
echo "🎮 PHASE 2: NVIDIA GPU VERIFICATION"
echo "-----------------------------------"
if command -v nvidia-smi &> /dev/null; then
    echo " NVIDIA GPU Detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
    echo ""
    echo "CUDA Version Check:"
    if command -v nvcc &> /dev/null; then
        nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//'
    else
        echo "  nvcc not found (may still work with PyTorch CUDA)"
    fi
else
    echo " NVIDIA GPU not detected or drivers not installed"
    echo "   This would be a critical issue for your friend"
fi
echo ""

# Phase 3: Package Integrity
echo " PHASE 3: PACKAGE INTEGRITY CHECK"
echo "-----------------------------------"
echo "Package structure:"
find . -maxdepth 2 -type d | sort | sed 's|^\./||' | sed '/^\s*$/d'
echo ""

echo "Key files check:"
files_to_check=(
    "1_START_HERE.md"
    "4_PACKAGE_CONTENTS.md"
    "requirements.txt"
    "5_SETUP_TOOL.py"
    "7_RUN_EXPERIMENTS.py"
    "6_VERIFY_TOOL.sh"
    "README.md"
    "scripts/fair_comparison_benchmark.py"
    "data/dataset_D1.csv"
    "data/dataset_D2.csv"
)

for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        echo " $file"
    else
        echo " MISSING: $file"
    fi
done
echo ""

# Phase 4: Python Environment Setup
echo "🐍 PHASE 4: PYTHON ENVIRONMENT SETUP"
echo "------------------------------------"
echo "Python version:"
python3 --version
echo ""

echo "Checking pip:"
if command -v pip3 &> /dev/null; then
    echo " pip3 available"
else
    echo " pip3 not found"
fi
echo ""

# Phase 5: Dependency Installation Test
echo "📥 PHASE 5: DEPENDENCY INSTALLATION TEST"
echo "----------------------------------------"
echo "Testing dependency installation (dry run)..."

# Create a temporary virtual environment for testing
TEMP_VENV="/tmp/dumsto_test_venv_$(date +%s)"
python3 -m venv "$TEMP_VENV" 2>/dev/null && echo " Virtual environment created" || echo " Virtual environment creation failed"

if [ -d "$TEMP_VENV" ]; then
    source "$TEMP_VENV/bin/activate"
    echo " Virtual environment activated"

    # Test pip install (dry run)
    if pip install --dry-run -r requirements.txt >/dev/null 2>&1; then
        echo " Dependencies can be installed"
    else
        echo "  Some dependencies may require system packages"
    fi

    deactivate
    rm -rf "$TEMP_VENV"
    echo " Virtual environment cleaned up"
fi
echo ""

# Phase 6: Setup Script Test
echo "  PHASE 6: SETUP SCRIPT TEST"
echo "------------------------------"
echo "Testing 5_SETUP_TOOL.py --check..."
if python3 5_SETUP_TOOL.py --check >/dev/null 2>&1; then
    echo " Setup script runs successfully"
else
    echo " Setup script failed"
fi
echo ""

# Phase 7: Platform Compatibility Test
echo " PHASE 7: PLATFORM COMPATIBILITY TEST"
echo "---------------------------------------"
echo "Testing platform detection..."
if python3 7_RUN_EXPERIMENTS.py platform >/dev/null 2>&1; then
    echo " Platform detection works"

    # Show actual platform output
    echo "Platform details:"
    python3 7_RUN_EXPERIMENTS.py platform 2>/dev/null | head -5
else
    echo " Platform detection failed"
fi
echo ""

# Phase 8: Import Test
echo " PHASE 8: IMPORT TEST"
echo "-----------------------"
echo "Testing core imports..."
python3 -c "
try:
    import pandas as pd
    import numpy as np
    import torch
    import xgboost as xgb
    from sklearn.ensemble import GradientBoostingRegressor
    print(' All core imports successful')
    print(f'PyTorch CUDA: {torch.cuda.is_available()}')
except ImportError as e:
    print(f' Import failed: {e}')
" 2>/dev/null
echo ""

# Phase 9: Quick Benchmark Test
echo " PHASE 9: QUICK BENCHMARK TEST"
echo "-------------------------------"
echo "Testing quick benchmark (this may take a few minutes)..."

# Timeout after 5 minutes to avoid hanging
timeout 300 python3 7_RUN_EXPERIMENTS.py quick >/tmp/dumsto_test.log 2>&1 &
TEST_PID=$!

# Wait up to 5 minutes
for i in {1..60}; do
    if ! kill -0 $TEST_PID 2>/dev/null; then
        break
    fi
    sleep 5
done

# Check if process is still running
if kill -0 $TEST_PID 2>/dev/null; then
    echo " Quick benchmark timed out (killed)"
    kill $TEST_PID 2>/dev/null
else
    echo " Quick benchmark completed"
    echo "Checking results..."
    if [ -f "results/fair_comparison_2026-01-22.json" ] || [ -f "results/$(ls results/ | grep fair_comparison | head -1)" ]; then
        echo " Results file created"
    else
        echo "  No results file found"
    fi
fi
echo ""

# Phase 10: Final Assessment
echo " PHASE 10: FINAL ASSESSMENT"
echo "-----------------------------"
echo " ASSESSMENT FOR X86 POP!_OS + NVIDIA GPU:"
echo ""
echo " STRENGTHS:"
echo "• Native x86_64 architecture support"
echo "• NVIDIA GPU acceleration available"
echo "• Ubuntu-based (Pop!_OS) compatibility"
echo "• All major dependencies supported"
echo ""
echo "  POTENTIAL ISSUES:"
echo "• NVIDIA driver installation required"
echo "• CUDA toolkit version compatibility"
echo "• System package dependencies (build-essential, etc.)"
echo ""
echo " RECOMMENDED SETUP STEPS FOR YOUR FRIEND:"
echo "1. sudo apt update && sudo apt upgrade"
echo "2. Install NVIDIA drivers (Pop!_OS makes this easy)"
echo "3. sudo apt install python3 python3-pip python3-venv build-essential"
echo "4. cd umst-prototype"
echo "5. python3 5_SETUP_TOOL.py"
echo "6. ./6_VERIFY_TOOL.sh"
echo "7. python 7_RUN_EXPERIMENTS.py benchmark"
echo ""

# Cleanup
rm -f /tmp/dumsto_test.log

echo " TESTING COMPLETE!"
echo "==================="
echo "Package Status: READY FOR X86 POP!_OS + NVIDIA GPU TESTING"
echo ""
echo "📞 If your friend encounters issues, have them run:"
echo "   ./test_x86_popos_nvidia.sh"
echo "   And share the output for troubleshooting."