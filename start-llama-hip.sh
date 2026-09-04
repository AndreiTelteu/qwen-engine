#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_ROOT="$ROOT/llama-hip"
MODEL_DIR="$ENGINE_ROOT/models/qwen3.8-27b-q4_0"
MODEL="$MODEL_DIR/Qwen3.8-27B-Q4_0.gguf"
DRAFT="$MODEL_DIR/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
SERVER="$ENGINE_ROOT/build-hip/bin/llama-server"

: "${CTX_SIZE:=131072}"
: "${SPEC_DRAFT_N_MAX:=2}"
: "${PORT:=8080}"
: "${REASONING:=off}"

for required in "$SERVER" "$MODEL" "$DRAFT"; do
    if [ ! -e "$required" ]; then
        printf "Missing required file: %s\\n" "$required" >&2
        printf "Run scripts/rebuild-llama-hip.sh or scripts/download-qwen3.8-27b.sh.\\n" >&2
        exit 1
    fi
done

export LD_LIBRARY_PATH="$ENGINE_ROOT/build-hip/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$SERVER" \
    --model "$MODEL" \
    --model-draft "$DRAFT" \
    --device ROCm0 \
    --gpu-layers all \
    --ctx-size "$CTX_SIZE" \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --spec-type draft-mtp \
    --spec-draft-n-max "$SPEC_DRAFT_N_MAX" \
    --flash-attn on \
    --fit off \
    --batch-size 2048 \
    --ubatch-size 512 \
    --parallel 1 \
    --reasoning "$REASONING" \
    --threads 16 \
    --threads-batch 16 \
    --host 127.0.0.1 \
    --port "$PORT"
