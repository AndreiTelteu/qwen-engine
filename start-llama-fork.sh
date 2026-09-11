#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_ROOT="$ROOT/llama-fork"
MODEL_DIR="$ROOT/llama-hip/models/qwen3.8-27b-q4_0"

: "${MODEL:=$MODEL_DIR/Qwen3.8-27B-Q4_0.gguf}"
: "${DRAFT:=$MODEL_DIR/DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf}"
: "${SERVER:=$ENGINE_ROOT/build-rocm-gfx1100-portable/bin/llama-server}"
: "${ALIAS:=qwen3.8-27b}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8080}"
: "${CTX_SIZE:=131072}"
: "${SPEC_TYPE:=draft-dflash}"
: "${SPEC_DRAFT_N_MAX:=3}"
: "${SPEC_DRAFT_P_MIN:=0.20}"
: "${DEVICE:=ROCm0}"
: "${DEVICE_DRAFT:=ROCm0}"
: "${GPU_LAYERS:=all}"
: "${SPLIT_MODE:=none}"
: "${FLASH_ATTN:=on}"
: "${BATCH_SIZE:=2048}"
: "${UBATCH_SIZE:=512}"
: "${MOE_EXPERT_CACHE:=0}"
: "${CACHE_TYPE_K:=q8_0}"
: "${CACHE_TYPE_V:=q8_0}"
: "${DRAFT_CACHE_TYPE:=f16}"
: "${TEMPERATURE:=0.6}"
: "${TOP_P:=0.95}"
: "${TOP_K:=20}"
: "${MIN_P:=0.00}"
: "${PRESENCE_PENALTY:=0.0}"
: "${REASONING:=auto}"
: "${REASONING_EFFORT:=medium}"
: "${PARALLEL:=1}"
: "${FIT:=off}"
: "${FIT_TARGET:=2800}"
: "${LOAD_MODE:=none}"
: "${LAZY_MODE:=on-direct}"
: "${HIP_VISIBLE_DEVICES:=0}"
: "${GGML_CUDA_GDN_CHUNKED_BF16:=1}"

required=("$SERVER" "$MODEL")
if [ "$SPEC_TYPE" != "none" ]; then
    required+=("$DRAFT")
fi
for path in "${required[@]}"; do
    if [ ! -e "$path" ]; then
        printf 'Missing required file: %s\n' "$path" >&2
        exit 1
    fi
done

ROCM_ROOT="${ROCM_ROOT:-$(/opt/rocm/bin/hipconfig --path 2>/dev/null || true)}"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm}"
export LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROCM_ROOT/lib/llvm/lib:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIP_VISIBLE_DEVICES
export GGML_CUDA_GDN_CHUNKED_BF16

draft_args=()
if [ "$SPEC_TYPE" != "none" ]; then
    draft_args=(
        --model-draft "$DRAFT"
        --spec-type "$SPEC_TYPE"
        --spec-draft-n-max "$SPEC_DRAFT_N_MAX"
        --spec-draft-p-min "$SPEC_DRAFT_P_MIN"
        --device-draft "$DEVICE_DRAFT"
    )
fi

command=(
    "$SERVER"
    --model "$MODEL"
    --alias "$ALIAS"
    --host "$HOST"
    --port "$PORT"
    --ctx-size "$CTX_SIZE"
    "${draft_args[@]}"
    --device "$DEVICE"
    --gpu-layers "$GPU_LAYERS"
    --split-mode "$SPLIT_MODE"
    --flash-attn "$FLASH_ATTN"
    --batch-size "$BATCH_SIZE"
    --ubatch-size "$UBATCH_SIZE"
    --moe-expert-cache "$MOE_EXPERT_CACHE"
    --cache-type-k "$CACHE_TYPE_K"
    --cache-type-v "$CACHE_TYPE_V"
    --cache-type-k-draft "$DRAFT_CACHE_TYPE"
    --cache-type-v-draft "$DRAFT_CACHE_TYPE"
    --temp "$TEMPERATURE"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --min-p "$MIN_P"
    --presence-penalty "$PRESENCE_PENALTY"
    --reasoning "$REASONING"
    --reasoning-effort "$REASONING_EFFORT"
    --parallel "$PARALLEL"
    --fit "$FIT"
    --fit-target "$FIT_TARGET"
    --load-mode "$LOAD_MODE"
    --lazy-mode "$LAZY_MODE"
)

printf 'Starting fork llama-server:\n  ' >&2
printf 'HIP_VISIBLE_DEVICES=%q GGML_CUDA_GDN_CHUNKED_BF16=%q LD_LIBRARY_PATH=%q ' \
    "$HIP_VISIBLE_DEVICES" "$GGML_CUDA_GDN_CHUNKED_BF16" "$LD_LIBRARY_PATH" >&2
printf '%q ' "${command[@]}" >&2
printf '\n' >&2

exec "${command[@]}"
