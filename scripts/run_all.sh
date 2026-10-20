#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO="${1:-}"
MODEL="${2:-yolov8n.pt}"

for s in 1 2 3 4 5; do
  echo
  echo "================ STAGE $s ================"
  bash "$ROOT/scripts/run_stage${s}.sh"
done

echo
echo "Stage 6 is profiler-dependent and is not run automatically. Use: bash scripts/run_stage6_ncu.sh"

if [[ -n "$VIDEO" ]]; then
  echo
  echo "================ STAGES 7-8 ================"
  bash "$ROOT/scripts/run_stage7_8.sh" "$VIDEO"
  echo
  echo "================ STAGE 9 ================"
  bash "$ROOT/scripts/run_stage9.sh" "$VIDEO" "$MODEL"
  echo
  echo "================ STAGE 10 ================"
  bash "$ROOT/scripts/run_stage10.sh" "$VIDEO" "$MODEL"
else
  echo
  echo "No video path supplied, so stages 7-10 were skipped."
  echo "Run again with: bash scripts/run_all.sh /path/to/video.mp4 [yolov8n.pt]"
fi
