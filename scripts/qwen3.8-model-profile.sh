#!/usr/bin/env bash
# Sourced by launchers after ROOT and MODEL_DIR are set. Explicit MODEL wins.
# MODEL_DIR continues to hold the existing standalone DFlash2 and MTP drafts.
: "${MODEL_QUANT:=q4_0}"
if [ -z "${MODEL:-}" ]; then
    case "$MODEL_QUANT" in
        q4_0) MODEL="$MODEL_DIR/Qwen3.8-27B-Q4_0.gguf" ;;
        byteshape)
            MODEL="$ROOT/llama-hip/models/qwen3.8-27b-byteshape/Qwen3.8-27B-IQ4_XS-3.84bpw.gguf"
            ;;
        *)
            printf 'Unknown MODEL_QUANT: %s (use q4_0 or byteshape, or set MODEL).\n' "$MODEL_QUANT" >&2
            exit 2
            ;;
    esac
fi

# Embedded MTP reuses the target GGUF; omit --model-draft in the launchers.
if [ "${DRAFT:-}" = "embedded" ]; then
    case "${SPEC_TYPE:-draft-dflash}" in
        draft-mtp | draft-mtp-adaptive) ;;
        *) printf 'DRAFT=embedded requires SPEC_TYPE=draft-mtp or draft-mtp-adaptive.\n' >&2; exit 2 ;;
    esac
fi
