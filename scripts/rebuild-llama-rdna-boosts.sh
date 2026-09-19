#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_ROOT="$ROOT/llama-rdna-boosts"
BUILD_DIR="${BUILD_DIR:-$ENGINE_ROOT/build-rocm-gfx1100}"
JOBS="${JOBS:-$(nproc)}"
ROCM_ROOT="${ROCM_ROOT:-}"

if [ -z "$ROCM_ROOT" ]; then
    ROCM_ROOT="$(/opt/rocm/bin/hipconfig --path 2>/dev/null || true)"
fi
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm}"

for compiler in clang clang++; do
    if [ ! -x "$ROCM_ROOT/lib/llvm/bin/$compiler" ]; then
        printf 'Missing ROCm compiler: %s\n' "$ROCM_ROOT/lib/llvm/bin/$compiler" >&2
        exit 1
    fi
done

if [ ! -f "$ENGINE_ROOT/CMakeLists.txt" ]; then
    printf 'Missing patched llama.cpp checkout: %s\n' "$ENGINE_ROOT" >&2
    exit 1
fi

# Isolate the ROCm HIP headers from older distro headers in /usr/include.
mkdir -p "$BUILD_DIR/rocm-include"
cp -a "$ROCM_ROOT/include/hip" "$BUILD_DIR/rocm-include/"

cmake --fresh -S "$ENGINE_ROOT" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_PREFIX_PATH="$ROCM_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang++" \
    -DCMAKE_HIP_COMPILER="$ROCM_ROOT/lib/llvm/bin/clang" \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_ROOT" \
    '-DCMAKE_HIP_COMPILE_OPTIONS_EXPLICIT_LANGUAGE=-x;hip' \
    -DCMAKE_HIP_FLAGS_RELEASE=-O3 \
    -DCMAKE_HIP_FLAGS="-I$BUILD_DIR/rocm-include -mllvm --amdgpu-unroll-threshold-local=600" \
    -DGGML_HIP=ON \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_HIP_RCCL=OFF \
    -DAMDGPU_TARGETS=gfx1100 \
    -DLLAMA_BUILD_TESTS=ON

cmake --build "$BUILD_DIR" --config Release --target llama-server llama-cli llama-bench -j "$JOBS"
