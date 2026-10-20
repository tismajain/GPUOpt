#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${GPUOPT_BUILD_DIR:-$ROOT/build}"
mkdir -p "$BUILD"
nvcc -O3 "$ROOT/cuda/stage2_coalescing.cu" -o "$BUILD/stage2_coalescing"
"$BUILD/stage2_coalescing"
