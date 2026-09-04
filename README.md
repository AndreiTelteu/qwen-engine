# Qwen engine workspace

Local Qwen inference and reproducible coding-agent evaluations on an RX 7900 XTX under WSL2.

## Layout

- `llama-hip/` — pinned upstream `llama.cpp` Git submodule, built with ROCm/HIP for `gfx1100`.
- `start-llama-hip.sh` — the one foreground launcher for the local OpenAI-compatible server.
- `scripts/` — model download, benchmark, rebuild, and explicit upstream update commands.
- `agent-evals/` — the coding-agent evaluation TUI, TOML definitions, and isolated run machinery.
- `artifacts/` — local logs and benchmark output, deliberately excluded from Git.

## Run the engine

```bash
./start-llama-hip.sh
```

Defaults: Qwen3.8-27B Q4_0 + its MTP draft model, ROCm0, 128K context, Q8 KV cache, draft depth 2, reasoning off, and `127.0.0.1:8080`.

```bash
REASONING=auto ./start-llama-hip.sh
CTX_SIZE=32768 ./start-llama-hip.sh
```

## Engine maintenance

```bash
# Download/verify the two GGUF files, resumably.
./scripts/download-qwen3.8-27b.sh

# Reconfigure and compile the HIP build for gfx1100.
./scripts/rebuild-llama-hip.sh

# Explicitly move the submodule to a fetched upstream revision, then rebuild.
./scripts/update-llama-hip.sh
```

`update-llama-hip.sh` never silently updates at launch. It preserves reproducible benchmarks; after validating an update, commit the new Gitlink from this root repository.

## Measured baseline

At 32K context, MTP draft depth 2 measured 45.92 decode tok/s with 60.04% draft acceptance. The same setup served 128K capacity at 44.13 decode tok/s. Historical local output is under `artifacts/llama-hip/`.

## Agent evaluations

Start the inference engine in one terminal, then run the evaluation cockpit in another:

```bash
./start-llama-hip.sh
cd agent-evals && cp .env.example .env && $EDITOR .env && ./run.sh
```

See [`agent-evals/README.md`](agent-evals/README.md) for TOML authoring, sample submodules, isolated worktrees, and live server telemetry.
