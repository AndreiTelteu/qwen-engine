#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_ROOT="$ROOT/llama-hip"
TARGET="${1:-origin/master}"

if [ -n "$(git -C "$ENGINE_ROOT" status --porcelain)" ]; then
    printf "%s\\n" "llama-hip has local changes; refusing to update." >&2
    printf "%s\\n" "Commit, stash, or discard them first." >&2
    exit 1
fi

git -C "$ENGINE_ROOT" fetch --tags origin
git -C "$ENGINE_ROOT" switch --detach "$TARGET"
printf "llama-hip is now at %s\\n" "$(git -C "$ENGINE_ROOT" rev-parse --short HEAD)"
printf "%s\\n" "Rebuild next with: scripts/rebuild-llama-hip.sh"
printf "%s\\n" "Record the new submodule revision with: git add llama-hip && git commit"
