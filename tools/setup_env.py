#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
# SPDX-License-Identifier: MIT
"""
UMST Prototype Setup
============================================

This setup script ensures all dependencies are correctly installed
for reproducing DUMSTO (Differentiable Unified Material-State Tensor Optimization) results.

Usage:
    python 5_SETUP_TOOL.py          # Interactive setup
    python 5_SETUP_TOOL.py --auto   # Automatic setup
    python 5_SETUP_TOOL.py --check  # Verify installation
"""

import sys
import os
import subprocess
import platform
import argparse
from pathlib import Path

class DUMSTOSetup:
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        self.system = platform.system()
        self.machine = platform.machine()
        self.python_version = sys.version_info

    def print_header(self):
        print("=" * 70)
        print(" UMST Prototype Setup")
        print("=" * 70)
        print(f"Platform: {self.system} {self.machine}")
        print(f"Python: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        print("=" * 70)

    def check_python_version(self):
        """Check if Python version is compatible."""
        print(" Checking Python version...")
        if self.python_version >= (3, 8):
            print(f" Python {self.python_version.major}.{self.python_version.minor} is compatible")
            return True
        else:
            print(f" Python {self.python_version.major}.{self.python_version.minor} is too old. Need Python 3.8+")
            return False

    def check_and_install_dependencies(self):
        """Check and install Python dependencies."""
        print("\n Checking Python dependencies...")

        try:
            import pip
            print(" pip is available")
        except ImportError:
            print(" pip is not available. Please install pip first.")
            return False

        # Check if requirements.txt exists
        req_file = self.root_dir / "requirements.txt"
        if not req_file.exists():
            print(f" requirements.txt not found at {req_file}")
            return False

        print("📥 Installing dependencies from requirements.txt...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(req_file)
            ], capture_output=True, text=True, cwd=self.root_dir)

            if result.returncode == 0:
                print(" Dependencies installed successfully")
                return True
            else:
                print(" Failed to install dependencies:")
                print(result.stderr)
                return False

        except Exception as e:
            print(f" Error installing dependencies: {e}")
            return False

    def check_pytorch_setup(self):
        """Check PyTorch installation and CUDA availability."""
        print("\n Checking PyTorch setup...")

        try:
            import torch
            print(f" PyTorch {torch.__version__} installed")

            cuda_available = torch.cuda.is_available()
            if cuda_available:
                print(f" CUDA available: {torch.cuda.get_device_name(0)}")
                print(f"   CUDA version: {torch.version.cuda}")
                print(f"   GPU devices: {torch.cuda.device_count()}")
            else:
                print("  CUDA not available (CPU-only mode)")
                print("   This is normal on systems without NVIDIA GPUs")

            return True

        except ImportError:
            print(" PyTorch not installed")
            return False

    def check_rust_setup(self):
        """Check Rust installation (optional for physics kernel)."""
        print("\n🦀 Checking Rust setup (optional)...")

        try:
            result = subprocess.run(["rustc", "--version"],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip().split()[1]
                print(f" Rust {version} available")

                # Check if physics kernel can be built
                rust_dir = self.root_dir / "src" / "rust" / "core"
                if rust_dir.exists():
                    print("🔨 Testing physics kernel compilation...")
                    compile_result = subprocess.run(
                        ["cargo", "check"], cwd=rust_dir,
                        capture_output=True, text=True
                    )
                    if compile_result.returncode == 0:
                        print(" Physics kernel compilation successful")
                    else:
                        print("  Physics kernel compilation failed (optional)")

                return True
            else:
                print("  Rust not installed (optional for physics kernel)")
                print("   Python benchmarks will still work")
                return True  # Not required

        except FileNotFoundError:
            print("  Rust not installed (optional for physics kernel)")
            return True

    def check_datasets(self):
        """Check if benchmark datasets are available."""
        print("\n Checking benchmark datasets...")

        data_dir = self.root_dir / "data"
        if not data_dir.exists():
            print(f" Data directory not found: {data_dir}")
            return False

        expected_datasets = [
            "dataset_D1.csv", "dataset_D2.csv", "dataset_D3.csv", "dataset_D4.csv"
        ]

        missing_datasets = []
        for dataset in expected_datasets:
            if not (data_dir / dataset).exists():
                missing_datasets.append(dataset)

        if missing_datasets:
            print(f" Missing datasets: {', '.join(missing_datasets)}")
            print("   Please ensure all dataset files are present in the data/ directory")
            return False
        else:
            print(" All benchmark datasets present")
            return True

    def run_basic_tests(self):
        """Run basic functionality tests."""
        print("\n Running basic functionality tests...")

        # Test platform detection
        try:
            result = subprocess.run([
                sys.executable, "tools/7_RUN_EXPERIMENTS.py", "platform"
            ], cwd=self.root_dir, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                print(" Platform detection works")
                return True
            else:
                print(" Platform detection failed:")
                print(result.stderr)
                return False

        except subprocess.TimeoutExpired:
            print(" Platform detection timed out")
            return False
        except Exception as e:
            print(f" Error running platform test: {e}")
            return False

    def create_results_directory(self):
        """Create results directory if it doesn't exist."""
        results_dir = self.root_dir / "results"
        results_dir.mkdir(exist_ok=True)
        print(" Results directory ready")

    def print_success_message(self):
        """Print success message with next steps."""
        print("\n" + "=" * 70)
        print(" DUMSTO SETUP COMPLETE!")
        print("=" * 70)
        print("Your environment is ready for DUMSTO reproducibility.")
        print("")
        print(" NEXT STEPS:")
        print("1. Run verification:     ./6_VERIFY_TOOL.sh")
        print("2. Quick test:          python 7_RUN_EXPERIMENTS.py quick")
        print("3. Full benchmark:      python 7_RUN_EXPERIMENTS.py benchmark")
        print("")
        print(" DOCUMENTATION:")
        print("- Setup guide:          docs/3_Setup_Guide.md")
        print("- Reproduction:         docs/4_Reproduction_Guide.md")
        print("- Methodology:          docs/5_Technical_Methodology.md")
        print("")
        print(" SUPPORT:")
        print("- Directory Map:        docs/2_Directory_Structure.md")
        print("- Evaluation:           docs/6_Evaluation_Protocol.md")
        print("=" * 70)

    def run_setup(self, auto_mode=False):
        """Run the complete setup process."""
        self.print_header()

        # Run all checks
        checks = [
            ("Python Version", self.check_python_version),
            ("Dependencies", self.check_and_install_dependencies),
            ("PyTorch Setup", self.check_pytorch_setup),
            ("Rust Setup", self.check_rust_setup),
            ("Datasets", self.check_datasets),
            ("Basic Tests", self.run_basic_tests),
        ]

        all_passed = True
        for check_name, check_func in checks:
            try:
                if not check_func():
                    all_passed = False
                    if not auto_mode:
                        response = input(f" {check_name} check failed. Continue anyway? (y/N): ")
                        if response.lower() != 'y':
                            print("Setup aborted.")
                            return False
            except Exception as e:
                print(f" Error during {check_name} check: {e}")
                all_passed = False

        self.create_results_directory()

        if all_passed:
            self.print_success_message()
            return True
        else:
            print("\n  Setup completed with some issues.")
            print("You may still be able to run some experiments, but full reproducibility is not guaranteed.")
            return False

def main():
    parser = argparse.ArgumentParser(description="DUMSTO Setup Script")
    parser.add_argument("--auto", action="store_true",
                       help="Run setup automatically without prompts")
    parser.add_argument("--check", action="store_true",
                       help="Only check current installation status")

    args = parser.parse_args()

    setup = DUMSTOSetup()

    if args.check:
        # Just run checks without installation
        setup.print_header()
        print(" CHECK MODE - Verifying current installation")
        checks = [
            ("Python Version", setup.check_python_version),
            ("PyTorch Setup", setup.check_pytorch_setup),
            ("Rust Setup", setup.check_rust_setup),
            ("Datasets", setup.check_datasets),
            ("Basic Tests", setup.run_basic_tests),
        ]

        for check_name, check_func in checks:
            check_func()

        print("\n Check complete. Run 'python 5_SETUP_TOOL.py' to install missing components.")
        return

    # Run full setup
    success = setup.run_setup(auto_mode=args.auto)

    if success:
        print("\n Ready to reproduce DUMSTO results!")
    else:
        print("\n💥 Setup encountered issues. Check output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()