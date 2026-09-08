#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT/llama-hip/models/qwen3.8-27b-q4_0"

: "${MODEL:=$MODEL_DIR/Qwen3.8-27B-Q4_0.gguf}"
: "${UPSTREAM_BENCH:=$ROOT/llama-hip/build-hip/bin/llama-bench}"
: "${FORK_BENCH:=$ROOT/llama-fork/build-rocm-gfx1100-portable/bin/llama-bench}"
: "${RESULTS_DIR:=$ROOT/artifacts/llama-fork/benchmarks}"
: "${REPETITIONS:=3}"

for path in "$MODEL" "$UPSTREAM_BENCH" "$FORK_BENCH"; do
    if [ ! -e "$path" ]; then
        printf 'Missing required file: %s\n' "$path" >&2
        exit 1
    fi
done

ROCM_ROOT="${ROCM_ROOT:-$(/opt/rocm/bin/hipconfig --path 2>/dev/null || true)}"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm}"
export LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROCM_ROOT/lib/llvm/lib:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export GGML_CUDA_GDN_CHUNKED_BF16="${GGML_CUDA_GDN_CHUNKED_BF16:-1}"

mkdir -p "$RESULTS_DIR"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
SUMMARY="$RESULTS_DIR/pp8192-$RUN_ID.csv"

run_bench() {
    local engine=$1 binary=$2
    local output="$RESULTS_DIR/pp8192-$RUN_ID-$engine.json"
    "$binary" -m "$MODEL" -p 8192 -n 0 -b 4096 -ub 1024 -ngl 99 -fa on \
        -r "$REPETITIONS" -o json > "$output"

    python3 - "$engine" "$output" <<'PY'
import json
import sys

engine, path = sys.argv[1:]
row = json.load(open(path, encoding="utf-8"))[0]
print(f"{engine},{row['avg_ts']:.2f},{row['stddev_ts']:.2f}")
PY
}

printf '%s\n' "engine,pp8192_tok_per_s,stddev" > "$SUMMARY"
run_bench upstream "$UPSTREAM_BENCH" >> "$SUMMARY"
run_bench rdna3-fork "$FORK_BENCH" >> "$SUMMARY"

cat "$SUMMARY"
