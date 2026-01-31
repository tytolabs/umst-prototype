#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
# SPDX-License-Identifier: MIT
"""
DUMSTO Reproducibility Package Launcher
============================================

This script provides easy access to all DUMSTO experiments and benchmarks.

Usage:
    python 7_RUN_EXPERIMENTS.py [experiment_name]

Available experiments:
    benchmark    - Run the main 8-method fair comparison benchmark
    quick        - Run quick 4-method benchmark (faster)
    calibration  - Run physics parameter calibration
    demo         - Run metrics demonstration
    platform     - Test cross-platform compatibility
    help         - Show this help message

UMST Prototype: Physics-Informed ML Toolkit
Cross-platform deterministic results guaranteed on: Linux, macOS, WASM
"""

import sys
import os
import subprocess
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

def run_command(cmd, description, cwd=None):
    """Run a command with proper error handling."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    if cwd is None:
        cwd = scripts_dir

    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=False, text=True)
        if result.returncode == 0:
            print(f" {description} completed successfully")
        else:
            print(f" {description} failed with return code {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f" Error running {description}: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        experiment = "help"
    else:
        experiment = sys.argv[1].lower()

    print("DUMSTO Reproducibility Package")
    print("=" * 50)

    if experiment == "benchmark":
        # Main 8-method benchmark
        # FILE MAP: 2_comprehensive_benchmark.py
        success = run_command(
            [sys.executable, "2_comprehensive_benchmark.py"],
            "Main 8-Method Fair Comparison Benchmark"
        )

    elif experiment == "quick":
        # Quick 4-method benchmark
        # FILE MAP: 3_quick_benchmark.py
        success = run_command(
            [sys.executable, "3_quick_benchmark.py"],
            "Quick 4-Method Benchmark"
        )

    elif experiment == "calibration":
        # Physics calibration
        # FILE MAP: 4_physics_calibration.py
        success = run_command(
            [sys.executable, "4_physics_calibration.py"],
            "Physics Parameter Calibration"
        )

    elif experiment == "demo":
        # Metrics demonstration
        # FILE MAP: 10_metrics_demo.py
        success = run_command(
            [sys.executable, "10_metrics_demo.py"],
            "Comprehensive Metrics Demonstration"
        )

    elif experiment == "rust":
        # Rust kernel compilation check
        rust_dir = Path(__file__).parent.parent / "src" / "rust" / "core"
        if not rust_dir.exists():
             print(f"Error: Rust directory not found at {rust_dir}")
             return

        success = run_command(
            ["cargo", "build", "--release"],
            "Rust Physics Kernel Compilation",
            cwd=rust_dir
        )

    elif experiment == "platform":
        # Cross-platform compatibility test
        import platform
        print(f"Platform: {platform.system()} {platform.machine()}")
        print(f"Python: {sys.version.split()[0]}")

        # Test PyTorch CPU availability
        try:
            import torch
            print(f"PyTorch: {torch.__version__} (CPU available)")
        except ImportError:
            print("PyTorch: Not available")

        # Test basic imports
        try:
            import pandas, numpy, sklearn
            print("Core dependencies: Available")
        except ImportError as e:
            print(f"Missing dependency: {e}")

        # Test Rust compilation
        rust_dir = Path(__file__).parent.parent / "src" / "rust" / "core"
        if rust_dir.exists():
            success = run_command(
                ["cargo", "--version"],
                "Rust toolchain check",
                cwd=rust_dir
            )
        else:
            print("Rust kernel: Directory not found")

        return True

    elif experiment == "help" or experiment == "--help" or experiment == "-h":
        print(__doc__)
        return

    else:
        print(f" Unknown experiment: {experiment}")
        print("\nAvailable experiments:")
        print("  benchmark    - Main 8-method fair comparison")
        print("  quick        - Quick 4-method benchmark")
        print("  calibration  - Physics parameter calibration")
        print("  demo         - Metrics demonstration")
        print("  rust         - Test Rust kernel compilation")
        print("  help         - Show this help")
        return

    if 'success' in locals() and success:
        print(f"\n Experiment '{experiment}' completed successfully!")
    else:
        print(f"\n💥 Experiment '{experiment}' failed. Check output above.")

if __name__ == "__main__":
    main()