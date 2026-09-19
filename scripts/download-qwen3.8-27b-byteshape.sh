#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$ROOT/llama-hip/models/qwen3.8-27b-byteshape}"
REPO_URL="https://huggingface.co/byteshape/Qwen3.8-27B-GGUF/resolve/c9cc5b2ae2520ab77d1e5d3fae395b4140bb37e8"
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
        printf "Verified existing %s\n" "$name"
        return
    fi

    if [ -f "$partial" ] && echo "$sha256  $partial" | sha256sum --check --status; then
        printf "Verified completed partial download %s\n" "$name"
        mv "$partial" "$destination"
        return
    fi

    rm -f "$destination"
    printf "Downloading %s\n" "$name"
    curl --fail --location --continue-at - --retry 8 --retry-all-errors \
        --output "$partial" "$repo_url/$remote_name?download=true"
    echo "$sha256  $partial" | sha256sum --check --status
    mv "$partial" "$destination"
}

download "Qwen3.8-27B-IQ4_XS-3.84bpw.gguf" \
    "89434f23dc89c5f990894e3fe9fdad19d88c370f0d3638a176f29933f218b78b"
