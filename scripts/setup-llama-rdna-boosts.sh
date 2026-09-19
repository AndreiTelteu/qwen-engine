#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_ROOT="$ROOT/llama-rdna-boosts"
DELIVERY_ROOT="$ROOT/artifacts/rdna-boosts-v16-ebbb18522-r2"
RELEASE_TAG="v16-ebbb18522-r2"
EXPECTED_BASE="ebbb18522"
EXPECTED_TREE="7dc63cb3c93aa1cd74435698f045f93d2ee3a9e6"

if [ -e "$ENGINE_ROOT" ]; then
    printf 'Refusing to replace existing checkout: %s\n' "$ENGINE_ROOT" >&2
    exit 1
fi

if [ ! -d "$DELIVERY_ROOT/.git" ]; then
    git clone --branch "$RELEASE_TAG" --depth 1 \
        https://github.com/stew675/llama-cpp-rdna-boosts.git "$DELIVERY_ROOT"
fi

base="$(jq -r .base "$DELIVERY_ROOT/release.json")"
tree="$(jq -r .tree "$DELIVERY_ROOT/release.json")"
if [ "$base" != "$EXPECTED_BASE" ] || [ "$tree" != "$EXPECTED_TREE" ]; then
    printf 'Unexpected release manifest: base=%s tree=%s\n' "$base" "$tree" >&2
    exit 1
fi

while IFS=$'\t' read -r name expected; do
    patch="$DELIVERY_ROOT/patches/$name"
    actual="$(sha256sum "$patch" | awk '{print $1}')"
    if [ "$actual" != "$expected" ]; then
        printf 'Patch checksum mismatch: %s\n' "$name" >&2
        exit 1
    fi
done < <(jq -r '.patches | to_entries[] | [.key, .value] | @tsv' "$DELIVERY_ROOT/release.json")

clone_args=()
if [ -d "$ROOT/llama-hip/.git" ] || [ -f "$ROOT/llama-hip/.git" ]; then
    clone_args+=(--reference-if-able "$ROOT/llama-hip")
fi
git clone "${clone_args[@]}" https://github.com/ggml-org/llama.cpp.git "$ENGINE_ROOT"
git -C "$ENGINE_ROOT" checkout --detach "$base"
bash "$DELIVERY_ROOT/scripts/apply-all.sh" "$ENGINE_ROOT"

actual_tree="$(git -C "$ENGINE_ROOT" rev-parse 'HEAD^{tree}')"
if [ "$actual_tree" != "$EXPECTED_TREE" ]; then
    printf 'Applied tree mismatch: expected %s, got %s\n' "$EXPECTED_TREE" "$actual_tree" >&2
    exit 1
fi

printf 'RDNA boosts checkout ready: %s\n' "$ENGINE_ROOT"
printf 'Release: %s; base: %s; tree: %s\n' "$RELEASE_TAG" "$base" "$actual_tree"
