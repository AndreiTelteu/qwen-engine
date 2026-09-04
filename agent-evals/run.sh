#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p bin
GOTOOLCHAIN=local go build -o bin/agent-evals ./cmd/agent-evals
exec ./bin/agent-evals
