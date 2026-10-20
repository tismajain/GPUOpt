#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/1920x1080_video.mp4" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${GPUOPT_BUILD_DIR:-$ROOT/build}"
VIDEO="$1"
FRAME="$BUILD/stage8_frame_6000.bgr"
PATCHED="$BUILD/stage8_yolo_real_portable.cu"
BIN="$BUILD/stage8_yolo_real"
mkdir -p "$BUILD"
command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg not found." >&2; exit 1; }

ffmpeg -y -v error -i "$VIDEO" -vf "select=eq(n\,6000)" -frames:v 1 -pix_fmt bgr24 -f rawvideo "$FRAME"

python - "$ROOT/cuda/stage7_8_fused_real_video/stage8_yolo_real.cu" "$PATCHED" <<'PY'
from pathlib import Path
import sys
src = Path(sys.argv[1]).read_text()
src = src.replace('int main() {', 'int main(int argc, char** argv) {')
old = '''    const char* framePath =\n        "/kaggle/working/GPUOpt/data/boxing_frames/frame_6000.bgr";'''
new = '''    const char* framePath = (argc > 1) ? argv[1] : "frame_6000.bgr";'''
if old not in src:
    raise SystemExit("Could not patch Stage 8 frame path; source layout changed.")
src = src.replace(old, new)
Path(sys.argv[2]).write_text(src)
PY

nvcc -O3 "$PATCHED" -o "$BIN"
"$BIN" "$FRAME"
