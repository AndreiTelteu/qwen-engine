#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# MODEL_QUANT=byteshape selects IQ4_XS-3.84bpw through the common launcher.
# Explicit MODEL overrides the quant profile; standalone drafts remain shared.
: "${SERVER:=$ROOT/llama-fork/build-rocm-dual/bin/llama-server}"
: "${HIP_VISIBLE_DEVICES:=0,1}"
: "${DEVICE:=ROCm0}"
: "${DEVICE_DRAFT:=ROCm1}"
: "${TARGET_GPU_MATCH:=AMD Radeon RX 7900 XTX}"
: "${DRAFT_GPU_MATCH:=AMD Radeon RX 6800}"
: "${CTX_SIZE:=200000}"
: "${CACHE_TYPE_K:=q8_0}"
: "${CACHE_TYPE_V:=q8_0}"
: "${SPLIT_MODE:=none}"

if [ ! -x "$SERVER" ]; then
    printf 'Missing dual-GPU server: %s\n' "$SERVER" >&2
    printf 'Build it with the dual ROCm instructions in llama-fork/AGENTS.md.\n' >&2
    exit 1
fi

export HIP_VISIBLE_DEVICES

device_list="$($SERVER --list-devices 2>&1)"
target_device="${DEVICE%%,*}"

if ! grep -Fq "$target_device: $TARGET_GPU_MATCH" <<< "$device_list"; then
    printf 'Expected first target device %s to be %s. Available devices:\n%s\n' "$target_device" "$TARGET_GPU_MATCH" "$device_list" >&2
    exit 1
fi

if ! grep -Fq "$DEVICE_DRAFT: $DRAFT_GPU_MATCH" <<< "$device_list"; then
    printf 'Expected %s to be %s. Available devices:\n%s\n' "$DEVICE_DRAFT" "$DRAFT_GPU_MATCH" "$device_list" >&2
    printf 'Adjust HIP_VISIBLE_DEVICES after checking the physical GPU order.\n' >&2
    exit 1
fi

export SERVER DEVICE DEVICE_DRAFT CTX_SIZE CACHE_TYPE_K CACHE_TYPE_V SPLIT_MODE

exec "$ROOT/start-llama-fork.sh"
