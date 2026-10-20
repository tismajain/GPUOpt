#!/usr/bin/env bash
set -euo pipefail

nvcc -O3 -shared -Xcompiler -fPIC \
  nv12_yolo_preprocess.cu \
  -o /kaggle/working/libnv12_yolo.so

nvcc -O3 -shared -Xcompiler -fPIC \
  rgbp_yolo_preprocess.cu \
  -o /kaggle/working/librgbp_yolo.so
