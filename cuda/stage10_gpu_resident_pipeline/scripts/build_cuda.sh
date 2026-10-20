#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${GPUOPT_BUILD_DIR:-$ROOT_DIR/build}"

command -v nvcc >/dev/null 2>&1 || {
  echo "nvcc was not found. Install the CUDA toolkit or use an environment that provides nvcc." >&2
  exit 1
}

mkdir -p "$BUILD_DIR"

nvcc -O3 -shared -Xcompiler -fPIC \
  "$ROOT_DIR/kernels/nv12_yolo_preprocess.cu" \
  -o "$BUILD_DIR/libnv12_yolo.so"

nvcc -O3 -shared -Xcompiler -fPIC \
  "$ROOT_DIR/kernels/rgbp_yolo_preprocess.cu" \
  -o "$BUILD_DIR/librgbp_yolo.so"

echo "Built $BUILD_DIR/libnv12_yolo.so"
echo "Built $BUILD_DIR/librgbp_yolo.so"
