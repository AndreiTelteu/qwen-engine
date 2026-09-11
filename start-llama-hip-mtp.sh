#!/usr/bin/env bash
# Original MTP comparison profile retained after DFlash2 became the default.
set -euo pipefail

profile_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export DRAFT="$profile_root/llama-hip/models/qwen3.8-27b-q4_0/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
export SPEC_TYPE=draft-mtp
export SPEC_DRAFT_N_MAX=2
export DRAFT_CACHE_TYPE=
export UBATCH_SIZE=2048

exec "$profile_root/start-llama-hip.sh" "$@"
