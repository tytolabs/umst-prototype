#!/bin/bash
# SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
# SPDX-License-Identifier: MIT
# ==============================================================================
# DUMSTO - Full Design Benchmark Runner
# ==============================================================================
# This is the canonical entry point for running the "Constitutional Creativity"
# benchmark (6 PPO modes + Baselines).
#
# Usage:
#   ./scripts/benchmark_generative.sh [release|debug]
#
# Output:
#   results/ssot/design_benchmark_latest.json
# ==============================================================================

set -e

MODE=${1:-release}
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="results/ssot"
OUTPUT_FILE="${OUTPUT_DIR}/design_benchmark_${TIMESTAMP}.json"
LATEST_LINK="${OUTPUT_DIR}/design_benchmark_latest.json"

echo "========================================================================"
echo "Running DUMSTO Design Benchmark in ${MODE} mode..."
echo "========================================================================"

mkdir -p ${OUTPUT_DIR}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ "$MODE" == "release" ]; then
    echo "Building release binary..."
    cargo build --release --bin full_design_benchmark --manifest-path src/rust/core/Cargo.toml
    BINARY="src/rust/core/target/release/full_design_benchmark"
else
    echo "Building debug binary..."
    cargo build --bin full_design_benchmark --manifest-path src/rust/core/Cargo.toml
    BINARY="src/rust/core/target/debug/full_design_benchmark"
fi

echo "Executing benchmark..."
echo "Output will be saved to: ${OUTPUT_FILE}"

$BINARY > "${OUTPUT_FILE}"

# Create/Update 'latest' symlink (or copy)
cp "${OUTPUT_FILE}" "${LATEST_LINK}"

echo "========================================================================"
echo "Benchmark Complete."
echo "Results: ${LATEST_LINK}"
echo "========================================================================"
