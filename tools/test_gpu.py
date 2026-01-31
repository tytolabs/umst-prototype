#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
# SPDX-License-Identifier: MIT
"""
DUMSTO GPU Verification Tool
============================

Tests PyTorch and CUDA/MPS availability for hardware acceleration.
"""
import torch
import sys
import platform

print("="*60)
print(" DUMSTO GPU VERIFICATION TOOL")
print("="*60)

print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print("-" * 60)

if torch.cuda.is_available():
    print("✅ CUDA Available: YES")
    print(f"   Device Count: {torch.cuda.device_count()}")
    print(f"   Current Device: {torch.cuda.current_device()}")
    print(f"   Device Name: {torch.cuda.get_device_name(0)}")
    try:
        x = torch.rand(5, 3).cuda()
        print("   Tensor Allocation Test: PASSED")
    except Exception as e:
        print(f"   Tensor Allocation Test: FAILED ({e})")
else:
    print("⚠️ CUDA Available: NO")
    print("   Mode: CPU-Only")
    if platform.system() == "Linux":
        print("\n   [SUGGESTION] To enable GPU on Pop!_OS/Linux:")
        print("   Run: ./setup_gpu.sh")

print("="*60)
