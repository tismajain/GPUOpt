#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${GPUOPT_BUILD_DIR:-$ROOT/build}"
mkdir -p "$BUILD"
command -v ncu >/dev/null 2>&1 || { echo "Nsight Compute CLI (ncu) not found." >&2; exit 1; }
nvcc -O3 -lineinfo "$ROOT/cuda/stage4_register_blocking.cu" -o "$BUILD/stage4_register_blocking_profile"
set +e
ncu --set full "$BUILD/stage4_register_blocking_profile"
STATUS=$?
set -e
if [[ $STATUS -ne 0 ]]; then
  echo
  echo "Profiling failed. On Kaggle this is expected because GPU performance counters are restricted (ERR_NVGPUCTRPERM)."
fi
exit $STATUS
