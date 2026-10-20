#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/video.mp4 [model]" >&2
  exit 1
fi

VIDEO="$1"
MODEL="${2:-yolov8n.pt}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$ROOT_DIR/scripts/check_environment.py"
bash "$ROOT_DIR/scripts/build_cuda.sh"

python "$ROOT_DIR/benchmarks/stage10_final_pipeline.py" \
  --video "$VIDEO" \
  --model "$MODEL" \
  --frames 300 \
  --warmup 30 \
  --output "$ROOT_DIR/results/stage10_final_pipeline_rerun.csv"
