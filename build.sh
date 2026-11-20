#!/usr/bin/env bash
set -euo pipefail

# Build custom RDNA2/RDNA3 Wave32 HIP kernels for YuE2-3B.
# Override target with: GPU_ARCH=gfx1100 ./build.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${GPU_ARCH:-}" ]]; then
  if command -v rocminfo >/dev/null 2>&1; then
    GPU_ARCH="$(rocminfo | awk '/Name:[[:space:]]+gfx/{print $2; exit}')"
  fi
fi
GPU_ARCH="${GPU_ARCH:-gfx1030}"

echo "Building libv620_kernels.so for ${GPU_ARCH}..."
hipcc -O3 --offload-arch="${GPU_ARCH}" -shared -fPIC \
  "${DIR}/v620_kernels.hip" -o "${DIR}/libv620_kernels.so"
ls -lh "${DIR}/libv620_kernels.so"
