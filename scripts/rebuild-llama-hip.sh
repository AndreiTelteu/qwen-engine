#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_ROOT="$ROOT/llama-hip"
BUILD_DIR="${BUILD_DIR:-$ENGINE_ROOT/build-hip}"
JOBS="${JOBS:-$(nproc)}"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm}"

cmake -S "$ENGINE_ROOT" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang++" \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_ROOT" \
    '-DCMAKE_HIP_COMPILE_OPTIONS_EXPLICIT_LANGUAGE=-x;hip' \
    -DCMAKE_HIP_FLAGS_RELEASE=-O3 \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS=gfx1100 \
    -DGGML_CCACHE=ON
cmake --build "$BUILD_DIR" --target llama-server llama-cli llama-mtmd-cli -j "$JOBS"
