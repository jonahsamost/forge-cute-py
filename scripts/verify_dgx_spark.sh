#!/usr/bin/env bash
#
# DGX Spark (GB10) verification script
# Checks all pre-work environment requirements
#

set -euo pipefail

echo "==================================================================="
echo "DGX Spark (GB10) Environment Verification"
echo "==================================================================="
echo

# Check Python
echo "[1/6] Checking Python..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "  ✓ Python ${python_version}"
echo

# Check PyTorch and CUDA
echo "[2/6] Checking PyTorch and CUDA..."
python - <<'PY'
import torch
print(f"  ✓ PyTorch: {torch.__version__}")
print(f"  ✓ CUDA Runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    print("  ❌ CUDA not available")
    raise SystemExit(1)
props = torch.cuda.get_device_properties(0)
print(f"  ✓ GPU: {props.name}")
print(f"  ✓ Compute Capability: {props.major}.{props.minor}")
arch_list = torch.cuda.get_arch_list()
if 'sm_120' in arch_list or 'compute_120' in arch_list:
    print("  ✓ sm_120/compute_120 in arch list")
else:
    print("  ❌ sm_120/compute_120 NOT in arch list")
    raise SystemExit(1)
PY
echo

# Check CUTLASS
echo "[3/6] Checking CUTLASS (CuTe DSL)..."
python - <<'PY'
try:
    import cutlass
    import cutlass.cute
    print("  ✓ CUTLASS import successful")
except Exception as e:
    print(f"  ❌ CUTLASS import failed: {e}")
    raise SystemExit(1)
PY
echo

# Check Nsight tools
echo "[4/6] Checking Nsight tools..."
if command -v ncu >/dev/null 2>&1; then
    ncu_version=$(ncu --version 2>&1 | grep "Version" | head -1)
    echo "  ✓ Nsight Compute: ${ncu_version}"
else
    echo "  ❌ ncu not found in PATH"
fi

if command -v nsys >/dev/null 2>&1; then
    nsys_version=$(nsys --version 2>&1 | head -1)
    echo "  ✓ Nsight Systems: ${nsys_version}"
else
    echo "  ❌ nsys not found in PATH"
fi

if command -v compute-sanitizer >/dev/null 2>&1; then
    sanitizer_version=$(compute-sanitizer --version 2>&1 | grep "Version" | head -1)
    echo "  ✓ compute-sanitizer: ${sanitizer_version}"
else
    echo "  ❌ compute-sanitizer not found in PATH"
fi
echo

# Check forge_cute_py installation
echo "[5/6] Checking forge_cute_py installation..."
if python -c "import forge_cute_py" 2>/dev/null; then
    echo "  ✓ forge_cute_py module installed"
else
    echo "  ❌ forge_cute_py module not found"
    echo "     Run: uv sync"
    exit 1
fi
echo

# Run full env_check
echo "[6/6] Running forge_cute_py.env_check..."
python -m forge_cute_py.env_check
echo

echo "==================================================================="
echo "Verification PASSED ✓"
echo "==================================================================="
echo
echo "Next steps:"
echo "  1. Complete layout drills (notes/layout_drills.{py,md})"
echo "  2. Document JIT option experiment (notes/jit_option.md)"
echo "  3. Profile with Nsight Compute (save to profiles/)"
echo "  4. Write bottleneck analysis (notes/copy_transpose_limits.md)"
echo "  5. Run bench smoke: uv run python bench/run.py --suite smoke --out results.json"
