#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

nvcc -O3 -shared -Xcompiler -fPIC \
  "$ROOT_DIR/kernels/nv12_yolo_preprocess.cu" \
  -o /kaggle/working/libnv12_yolo.so

nvcc -O3 -shared -Xcompiler -fPIC \
  "$ROOT_DIR/kernels/rgbp_yolo_preprocess.cu" \
  -o /kaggle/working/librgbp_yolo.so

echo "Built /kaggle/working/libnv12_yolo.so"
echo "Built /kaggle/working/librgbp_yolo.so"
