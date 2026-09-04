#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_ROOT="$ROOT/llama-hip"
BUILD_DIR="${BUILD_DIR:-$ENGINE_ROOT/build-hip}"
JOBS="${JOBS:-$(nproc)}"

cmake -S "$ENGINE_ROOT" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1100 \
    -DGGML_CCACHE=ON
cmake --build "$BUILD_DIR" --target llama-server llama-cli llama-mtmd-cli -j "$JOBS"
