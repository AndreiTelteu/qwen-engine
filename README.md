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

Defaults: Qwen3.8-27B Q4_0 + its MTP draft model, ROCm0, 128K context, Q8 KV cache, draft depth 2, reasoning auto, and `127.0.0.1:8080`.

```bash
# Disable thinking only when maximum throughput matters more than quality.
REASONING=off ./start-llama-hip.sh
CTX_SIZE=32768 ./start-llama-hip.sh

# Override the llama.cpp flags benchmarked below.
FLASH_ATTN=off UBATCH_SIZE=2048 JINJA=on REASONING_FORMAT=auto MMAP=off ./start-llama-hip.sh
```

`FLASH_ATTN` accepts `on`, `off`, or `auto` (`-fa`); `UBATCH_SIZE` sets `-ub`.
`JINJA=on` enables `--jinja`, `REASONING_FORMAT` sets `--reasoning-format`, and
`MMAP=off` adds `--no-mmap`. `REASONING` controls whether the model thinks;
`REASONING_FORMAT` only controls how thought content is returned. A quantized
V-cache requires Flash Attention: the launcher automatically uses `f16` V-cache
when `FLASH_ATTN=off` (or set `CACHE_TYPE_V=f16` explicitly).

## Compare llama.cpp flags

```bash
./scripts/benchmark-llama-hip-flags.sh
```

The script uses the fixed prompt `Fă-mi în Python un calculator TUI care să
meargă și cu mouse-ul.` and starts a fresh server for every case. It compares a
current working baseline, Flash Attention off, `-ub` values 256/512/1024/2048,
Jinja, `--reasoning-format none`, `--no-mmap`, and all selected flags together. The
CSV records V-cache type because Flash Attention-off cases must use `f16` rather than
Q8. A configuration that exceeds available VRAM is recorded in a separate failures
CSV and does not stop the remaining cases.

It writes raw samples and a median summary under
`artifacts/llama-hip/benchmarks/`. By default each case has one warm-up request
and three measured requests; repeated prompts explicitly disable KV prompt-cache
reuse. Tune a run without editing the script:

```bash
RUNS=5 WARMUP_RUNS=2 CTX_SIZE=32768 MAX_TOKENS=512 ./scripts/benchmark-llama-hip-flags.sh
```

### Compare mmap with the required agent settings

```bash
./scripts/benchmark-llama-hip-mmap.sh
```

This test holds Flash Attention on, `-ub 2048`, Jinja on, `REASONING=auto`, and
`--reasoning-format auto` for both cases. It compares only mmap on versus
`--no-mmap`, alternates their startup order over two server rounds, and writes a
separate raw/median/failures CSV set. Increase confidence with more alternating
rounds:

```bash
SERVER_ROUNDS=4 RUNS=3 ./scripts/benchmark-llama-hip-mmap.sh
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
