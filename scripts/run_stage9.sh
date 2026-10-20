#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/1920x1080_video.mp4 [yolov8n.pt]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${GPUOPT_BUILD_DIR:-$ROOT/build}"
VIDEO="$1"
MODEL="${2:-yolov8n.pt}"
mkdir -p "$BUILD"
nvcc -O3 -shared -Xcompiler -fPIC \
  "$ROOT/cuda/stage9_yolo_integration/fused_yolo_preprocess.cu" \
  -o "$BUILD/libfused_yolo_preprocess.so"
python "$ROOT/cuda/stage9_yolo_integration/reproduce_stage9.py" \
  --video "$VIDEO" --model "$MODEL" --library "$BUILD/libfused_yolo_preprocess.so"
