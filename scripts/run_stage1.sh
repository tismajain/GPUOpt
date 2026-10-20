#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${GPUOPT_BUILD_DIR:-$ROOT/build}"
mkdir -p "$BUILD"
g++ -O3 -std=c++17 "$ROOT/cpu/cpu_blocked.cpp" -o "$BUILD/stage1_cpu_blocked"
"$BUILD/stage1_cpu_blocked"
