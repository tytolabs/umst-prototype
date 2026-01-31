#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
# SPDX-License-Identifier: MIT
"""
UMST Prototype Verification Script
=============================================

This script performs comprehensive verification of the UMST prototype toolkit
to ensure it meets research-grade standards for materials science applications.

Usage:
    python 8_DIAGNOSTICS_TOOL.py        # Full verification
    python 8_DIAGNOSTICS_TOOL.py quick  # Quick check only
"""

import sys
import os
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

class ReproducibilityVerifier:
    def __init__(self):
        self.root = Path(__file__).parent
        self.issues = []
        self.passed = 0
        self.failed = 0

    def log(self, status, message, details=None):
        """Log a verification result."""
        if status == "PASS":
            self.passed += 1
            print(f" {message}")
        elif status == "FAIL":
            self.failed += 1
            self.issues.append(message)
            print(f" {message}")
        elif status == "WARN":
            print(f"  {message}")
        else:
            print(f"ℹ  {message}")

        if details:
            print(f"   {details}")

    def check_file_exists(self, file_path, description):
        """Check if a required file exists."""
        path = self.root / file_path
        if path.exists():
            self.log("PASS", f"{description}: {file_path}")
            return True
        else:
            self.log("FAIL", f"{description}: {file_path} - MISSING")
            return False

    def check_directory_exists(self, dir_path, description):
        """Check if a required directory exists."""
        path = self.root / dir_path
        if path.exists() and path.is_dir():
            self.log("PASS", f"{description}: {dir_path}")
            return True
        else:
            self.log("FAIL", f"{description}: {dir_path} - MISSING")
            return False

    def check_script_executable(self, script_path, description):
        """Check if a script is executable."""
        path = self.root / script_path
        if path.exists() and os.access(path, os.X_OK):
            self.log("PASS", f"{description}: {script_path}")
            return True
        else:
            self.log("WARN", f"{description}: {script_path} - Not executable")
            return False

    def check_python_imports(self):
        """Check if key Python packages can be imported."""
        required_packages = [
            'pandas', 'numpy', 'torch', 'xgboost', 'sklearn'
        ]

        failed_imports = []
        for package in required_packages:
            try:
                __import__(package)
                self.log("PASS", f"Python import: {package}")
            except ImportError:
                failed_imports.append(package)
                self.log("FAIL", f"Python import: {package}")

        return len(failed_imports) == 0

    def check_datasets(self):
        """Check if all required datasets are present."""
        required_datasets = [
            'data/dataset_D1.csv',
            'data/dataset_D2.csv',
            'data/dataset_D3.csv',
            'data/dataset_D4.csv'
        ]

        all_present = True
        for dataset in required_datasets:
            if not self.check_file_exists(dataset, "Dataset"):
                all_present = False

        return all_present

    def check_scripts(self):
        """Check if all required scripts are present."""
        required_scripts = [
            '7_RUN_EXPERIMENTS.py',
            '5_SETUP_TOOL.py',
            '6_VERIFY_TOOL.sh',
            'scripts/fair_comparison_benchmark.py'
        ]

        all_present = True
        for script in required_scripts:
            if not self.check_file_exists(script, "Script"):
                all_present = False
            elif script.endswith('.sh'):
                self.check_script_executable(script, "Executable script")

        return all_present

    def check_documentation(self):
        """Check if all required documentation is present."""
        required_docs = [
            '1_START_HERE.md',
            '4_PACKAGE_CONTENTS.md',
            'README.md',
            'requirements.txt'
        ]

        all_present = True
        for doc in required_docs:
            if not self.check_file_exists(doc, "Documentation"):
                all_present = False

        return all_present

    def run_platform_check(self):
        """Run the platform compatibility check."""
        try:
            result = subprocess.run([
                sys.executable, '7_RUN_EXPERIMENTS.py', 'platform'
            ], cwd=self.root, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                self.log("PASS", "Platform compatibility check")
                return True
            else:
                self.log("FAIL", "Platform compatibility check failed")
                return False
        except Exception as e:
            self.log("FAIL", f"Platform check error: {e}")
            return False

    def check_file_sizes(self):
        """Check that files are reasonable sizes."""
        large_files = []
        for file_path in self.root.rglob('*'):
            if file_path.is_file() and file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
                large_files.append(f"{file_path.name} ({file_path.stat().st_size // (1024*1024)}MB)")

        if large_files:
            self.log("WARN", "Large files detected", ", ".join(large_files))
        else:
            self.log("PASS", "File sizes reasonable")

        return len(large_files) == 0

    def generate_report(self):
        """Generate a verification report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "package_version": "1.0.0",
            "verification_results": {
                "total_checks": self.passed + self.failed,
                "passed": self.passed,
                "failed": self.failed,
                "issues": self.issues
            },
            "system_info": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform
            }
        }

        # Save report
        report_path = self.root / "results" / f"verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.log("PASS", f"Verification report saved: {report_path}")

        return report

    def run_full_verification(self):
        """Run complete verification suite."""
        print("=" * 70)
        print(" UMST Prototype Verification")
        print("=" * 70)
        print(f"Package: {self.root.name}")
        print(f"Python: {sys.version.split()[0]}")
        print("=" * 70)

        # Core file checks
        print("\n FILE STRUCTURE VERIFICATION")
        self.check_directory_exists("data", "Data directory")
        self.check_directory_exists("scripts", "Scripts directory")
        self.check_directory_exists("docs", "Documentation directory")
        self.check_directory_exists("results", "Results directory")

        # Required files
        print("\n REQUIRED FILES CHECK")
        scripts_ok = self.check_scripts()
        docs_ok = self.check_documentation()
        datasets_ok = self.check_datasets()

        # Functionality checks
        print("\n  FUNCTIONALITY CHECKS")
        imports_ok = self.check_python_imports()
        platform_ok = self.run_platform_check()

        # Quality checks
        print("\n QUALITY CHECKS")
        self.check_file_sizes()

        # Generate report
        print("\n GENERATING VERIFICATION REPORT")
        report = self.generate_report()

        # Final summary
        print("\n" + "=" * 70)
        if self.failed == 0:
            print(" VERIFICATION: ALL CHECKS PASSED")
            print("   UMST prototype is ready for research use!")
        else:
            print(f"  VERIFICATION: {self.failed} ISSUES FOUND")
            print("   See issues above and verification report")
        print("=" * 70)

        print(f"\n Summary: {self.passed} passed, {self.failed} failed")

        return self.failed == 0

def main():
    verifier = ReproducibilityVerifier()

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick check only
        print("Running quick verification...")
        success = (
            verifier.check_scripts() and
            verifier.check_documentation() and
            verifier.check_datasets() and
            verifier.check_python_imports()
        )
    else:
        # Full verification
        success = verifier.run_full_verification()

    if success:
        print("\n Ready for research use!")
        sys.exit(0)
    else:
        print("\n Issues found. Check output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()