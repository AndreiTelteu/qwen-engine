#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
: "${QWEN_ENGINE_PROFILE:=dflash-balanced}"
export QWEN_ENGINE_PROFILE
mkdir -p bin
GOTOOLCHAIN=local go build -o bin/agent-evals ./cmd/agent-evals
exec ./bin/agent-evals
