#!/usr/bin/env bash
set -euo pipefail

is_sourced() {
  [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  if is_sourced; then
    return 0
  fi
  exit 0
fi

capability=""
if command -v python >/dev/null 2>&1; then
  capability="$(python - <<'PY' 2>/dev/null || true
import torch
if not torch.cuda.is_available():
    raise SystemExit(1)
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
)"
fi

arch_list=""
case "${capability}" in
  12.*)
    arch_list="12.0+PTX"
    ;;
  10.*)
    arch_list="10.0"
    ;;
  9.*)
    arch_list="9.0"
    ;;
  *)
    arch_list="12.0+PTX;10.0;9.0"
    ;;
esac

if is_sourced; then
  export TORCH_CUDA_ARCH_LIST="${arch_list}"
  export CMAKE_CUDA_ARCHITECTURES="${arch_list//+PTX/}"
else
  echo "export TORCH_CUDA_ARCH_LIST=\"${arch_list}\""
  echo "export CMAKE_CUDA_ARCHITECTURES=\"${arch_list//+PTX/}\""
fi
