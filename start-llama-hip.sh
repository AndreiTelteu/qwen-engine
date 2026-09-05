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
: "${REASONING:=auto}"
: "${VERBOSITY:=3}"
: "${FLASH_ATTN:=on}"
: "${UBATCH_SIZE:=512}"
: "${JINJA:=off}"
: "${REASONING_FORMAT:=auto}"
: "${MMAP:=on}"

# llama.cpp cannot load a quantized V cache without Flash Attention. Retain the
# Q8 default for normal runs, but make the Flash Attention-off benchmark valid.
if [ -z "${CACHE_TYPE_V:-}" ]; then
    if [ "$FLASH_ATTN" = "off" ]; then
        CACHE_TYPE_V=f16
    else
        CACHE_TYPE_V=q8_0
    fi
fi

case "$FLASH_ATTN:$CACHE_TYPE_V" in
    off:q* | off:iq* | off:tq*)
        printf "A quantized V cache (%s) requires FLASH_ATTN to be on or auto.\n" \
            "$CACHE_TYPE_V" >&2
        printf "Use CACHE_TYPE_V=f16 when FLASH_ATTN=off.\n" >&2
        exit 2
        ;;
esac

case "$JINJA" in
    on)
        jinja_args=(--jinja)
        ;;
    off)
        jinja_args=()
        ;;
    *)
        printf "JINJA must be 'on' or 'off', got: %s\n" "$JINJA" >&2
        exit 2
        ;;
esac

case "$MMAP" in
    on)
        mmap_args=()
        ;;
    off)
        mmap_args=(--no-mmap)
        ;;
    *)
        printf "MMAP must be 'on' or 'off', got: %s\n" "$MMAP" >&2
        exit 2
        ;;
esac

for required in "$SERVER" "$MODEL" "$DRAFT"; do
    if [ ! -e "$required" ]; then
        printf "Missing required file: %s\\n" "$required" >&2
        printf "Run scripts/rebuild-llama-hip.sh or scripts/download-qwen3.8-27b.sh.\\n" >&2
        exit 1
    fi
done

export LD_LIBRARY_PATH="$ENGINE_ROOT/build-hip/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$SERVER" \
    --verbosity "$VERBOSITY" \
    --model "$MODEL" \
    --model-draft "$DRAFT" \
    --device ROCm0 \
    --gpu-layers all \
    --ctx-size "$CTX_SIZE" \
    --cache-type-k q8_0 \
    --cache-type-v "$CACHE_TYPE_V" \
    --spec-type draft-mtp \
    --spec-draft-n-max "$SPEC_DRAFT_N_MAX" \
    --flash-attn "$FLASH_ATTN" \
    --fit off \
    --batch-size 2048 \
    --ubatch-size "$UBATCH_SIZE" \
    --parallel 1 \
    "${jinja_args[@]}" \
    --reasoning-format "$REASONING_FORMAT" \
    --reasoning "$REASONING" \
    "${mmap_args[@]}" \
    --threads 16 \
    --threads-batch 16 \
    --host 127.0.0.1 \
    --port "$PORT"
