#!/usr/bin/env bash
# Long-context RX 7900 XTX profile: depth 7 measured best at 100K occupied
# tokens. It can be slower than the balanced profile on short generations.
set -euo pipefail

profile_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export DRAFT="$profile_root/llama-hip/models/qwen3.8-27b-q4_0/DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
export SPEC_TYPE=draft-dflash
export SPEC_DRAFT_N_MAX=7
export DRAFT_CACHE_TYPE=q8_0
export UBATCH_SIZE=512

exec "$profile_root/start-llama-hip.sh" "$@"
