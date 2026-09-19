#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_ROOT="$ROOT/llama-hip"
MODEL_DIR="$ENGINE_ROOT/models/qwen3.8-27b-q4_0"
source "$ROOT/scripts/qwen3.8-model-profile.sh"
: "${DRAFT:=$MODEL_DIR/DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf}"
: "${SERVER:=$ENGINE_ROOT/build-hip/bin/llama-server}"
: "${DEVICE:=ROCm0}"
: "${DEVICE_DRAFT:=$DEVICE}"
: "${CACHE_TYPE_K:=q8_0}"

: "${CTX_SIZE:=131072}"
: "${SPEC_TYPE:=draft-dflash}"
: "${SPEC_DRAFT_N_MAX:=3}"
: "${DRAFT_CACHE_TYPE:=q8_0}"
: "${SPEC_DRAFT_P_MIN:=}"
: "${SPEC_NGRAM_MOD_N_MATCH:=}"
: "${SPEC_NGRAM_MOD_N_MIN:=}"
: "${PORT:=8080}"
: "${REASONING:=auto}"
: "${REASONING_EFFORT:=medium}"
: "${VERBOSITY:=3}"
: "${FLASH_ATTN:=on}"
: "${UBATCH_SIZE:=512}"
: "${PARALLEL:=2}"
: "${JINJA:=on}"
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

required_files=("$SERVER" "$MODEL")
if [ "$SPEC_TYPE" != "none" ] && [ "$DRAFT" != "embedded" ]; then
    required_files+=("$DRAFT")
fi
for required in "${required_files[@]}"; do
    if [ ! -e "$required" ]; then
        printf "Missing required file: %s\\n" "$required" >&2
        printf "Run scripts/rebuild-llama-hip.sh or scripts/download-qwen3.8-27b{,-byteshape}.sh.\\n" >&2
        exit 1
    fi
done

export LD_LIBRARY_PATH="$ENGINE_ROOT/build-hip/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

speculative_args=(
    --spec-type "$SPEC_TYPE"
    --spec-draft-n-max "$SPEC_DRAFT_N_MAX"
)
if [ -n "$SPEC_DRAFT_P_MIN" ]; then
    speculative_args+=(--spec-draft-p-min "$SPEC_DRAFT_P_MIN")
fi
if [ -n "$SPEC_NGRAM_MOD_N_MATCH" ]; then
    speculative_args+=(--spec-ngram-mod-n-match "$SPEC_NGRAM_MOD_N_MATCH")
fi
if [ -n "$SPEC_NGRAM_MOD_N_MIN" ]; then
    speculative_args+=(--spec-ngram-mod-n-min "$SPEC_NGRAM_MOD_N_MIN")
fi
if [ -n "$DRAFT_CACHE_TYPE" ]; then
    speculative_args+=(
        --cache-type-k-draft "$DRAFT_CACHE_TYPE"
        --cache-type-v-draft "$DRAFT_CACHE_TYPE"
    )
fi

draft_args=()
if [ "$SPEC_TYPE" != "none" ]; then
    draft_args=(--device-draft "$DEVICE_DRAFT")
    if [ "$DRAFT" != "embedded" ]; then
        draft_args+=(--model-draft "$DRAFT")
    fi
fi

command=(
    "$SERVER"
    --verbosity "$VERBOSITY"
    --model "$MODEL"
    "${draft_args[@]}"
    --device "$DEVICE"
    --gpu-layers all
    --ctx-size "$CTX_SIZE"
    --cache-type-k "$CACHE_TYPE_K"
    --cache-type-v "$CACHE_TYPE_V"
    "${speculative_args[@]}"
    --flash-attn "$FLASH_ATTN"
    --fit off
    --batch-size 2048
    --ubatch-size "$UBATCH_SIZE"
    --parallel "$PARALLEL"
    --kv-unified
    "${jinja_args[@]}"
    --reasoning-format "$REASONING_FORMAT"
    --reasoning "$REASONING"
    --reasoning-effort "$REASONING_EFFORT"
    "${mmap_args[@]}"
    --threads 16
    --threads-batch 16
    --host 127.0.0.1
    --port "$PORT"
)

# %q makes this a pasteable Bash command, including paths with whitespace.
printf 'Starting llama-server:\n  LD_LIBRARY_PATH=%q ' "$LD_LIBRARY_PATH" >&2
printf '%q ' "${command[@]}" >&2
printf '\n' >&2

exec "${command[@]}"
