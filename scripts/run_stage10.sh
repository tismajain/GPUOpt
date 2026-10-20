#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/1920x1080_video.mp4 [yolov8n.pt]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO="$1"
MODEL="${2:-yolov8n.pt}"
bash "$ROOT/cuda/stage10_gpu_resident_pipeline/scripts/recreate_stage10.sh" "$VIDEO" "$MODEL"
