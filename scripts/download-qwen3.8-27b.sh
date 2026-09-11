#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT/llama-hip/models/qwen3.8-27b-q4_0"
REPO_URL="https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main"
mkdir -p "$MODELS_DIR"

download() {
    local name="$1"
    local sha256="$2"
    local repo_url="${3:-$REPO_URL}"
    local remote_name="${4:-$name}"
    local destination="$MODELS_DIR/$name"
    local partial="$destination.partial"

    mkdir -p "$(dirname "$destination")"

    if [ -f "$destination" ] && echo "$sha256  $destination" | sha256sum --check --status; then
        printf "Verified existing %s\\n" "$name"
        return
    fi

    if [ -f "$partial" ] && echo "$sha256  $partial" | sha256sum --check --status; then
        printf "Verified completed partial download %s\\n" "$name"
        mv "$partial" "$destination"
        return
    fi

    rm -f "$destination"
    printf "Downloading %s\\n" "$name"
    curl --fail --location --continue-at - --retry 8 --retry-all-errors \
        --output "$partial" "$repo_url/$remote_name?download=true"
    echo "$sha256  $partial" | sha256sum --check --status
    mv "$partial" "$destination"
}

download "Qwen3.8-27B-Q4_0.gguf" \
    "ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d"
download "MTP/mtp-Qwen3.8-27B-Q4_0.gguf" \
    "50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e"
download "DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf" \
    "e44b99d7f4bce8ad5b573190c95e20678613fec9c9f455b073b6d9ef99c241e1" \
    "https://huggingface.co/Akicou/Qwen3.8-27B-DFlash2-GGUF/resolve/main" \
    "Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
