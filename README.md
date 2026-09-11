# Qwen engine workspace

## Clone with submodules

This workspace uses Git submodules for `llama-fork`, `llama-hip`, and the
Laravel evaluation sample. Clone everything in one command:

```bash
git clone --recurse-submodules https://github.com/AndreiTelteu/qwen-engine.git
```

If you have already cloned the repository without submodules, initialize them
afterward:

```bash
git submodule update --init --recursive
```

To update every submodule to the commit recorded by this repository:

```bash
git submodule update --recursive
```

Local Qwen inference and reproducible coding-agent evaluations on an RX 7900 XTX under native Linux.

## Layout

- `llama-hip/` — pinned upstream `llama.cpp` Git submodule, built with ROCm/HIP for `gfx1100`.
- `llama-fork/` — pinned RDNA3/RX 7900 XTX optimized fork.
- `start-llama-hip.sh` — the default balanced DFlash2 launcher.
- `start-llama-hip-mtp.sh` — the original MTP comparison profile.
- `start-llama-hip-dflash-balanced.sh` / `start-llama-hip-dflash-long.sh` — tested RX 7900 XTX DFlash2 profiles.
- `scripts/` — model download, benchmark, rebuild, and explicit upstream update commands.
- `agent-evals/` — the coding-agent evaluation TUI, TOML definitions, and isolated run machinery.
- `artifacts/` — local logs and benchmark output, deliberately excluded from Git.

## Run the engine

```bash
./start-llama-hip.sh
```

Defaults: Qwen3.8-27B Q4_0 + Q4_K_M DFlash2, ROCm0, a shared 128K unified KV
pool, two server slots, Q8 target and draft KV caches, draft depth 3, Flash
Attention on, `-ub 512`, Jinja on, reasoning auto with
`--reasoning-format auto`, mmap on, and `127.0.0.1:8080`. The unified pool lets
one slot use the full context; two simultaneous slots share that capacity.

The DFlash2 launchers keep the same Q4_0 target, 128K context, Q8 target KV,
reasoning mode, and API. Both use the Q4_K_M DFlash2 controller with Q8 draft
KV and `-ub 512`. The main launcher and `dflash-balanced` use draft depth 3;
this is also the default for the Agent Evals Start button. Native measurements
showed that depth 3 also beats depth 7 with 100K occupied tokens. The normal model
download script installs and verifies the DFlash2 controller too.

```bash
./start-llama-hip-dflash-balanced.sh
./start-llama-hip-dflash-long.sh
./start-llama-hip-mtp.sh
```

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

## Run and compare the RDNA3 fork

```bash
./scripts/rebuild-llama-fork.sh
./start-llama-fork.sh
```

The fork launcher defaults to the fastest balanced native profile found on this
RX 7900 XTX: Q4_0 target weights, the Q4_K_M DFlash2 controller, draft depth 3,
`--spec-draft-p-min 0.20`, Q8 target KV, F16 draft KV, Flash Attention,
`-b 2048`, `-ub 512`, a 128K context, one server slot, explicit ROCm0 placement,
all model layers on the GPU, `split-mode=none`, and fit disabled. Restricting HIP
to the discrete GPU prevents the integrated GPU from being included in tensor
or AllReduce initialization.

To compare adaptive MTP explicitly:

```bash
SPEC_TYPE=draft-mtp-adaptive \
DRAFT=llama-hip/models/qwen3.8-27b-q4_0/MTP/mtp-Qwen3.8-27B-Q4_0.gguf \
SPEC_DRAFT_N_MAX=4 SPEC_DRAFT_P_MIN=0 DRAFT_CACHE_TYPE=f16 ./start-llama-fork.sh
```

Run controlled no-speculation, fixed MTP, adaptive MTP, and DFlash2 depth 4 /
trained block maximum comparisons on both prose and code:

```bash
./scripts/benchmark-llama-fork-speculators.sh
```

Results, server logs, and failures are written under
`artifacts/llama-fork/`. Use more samples with `RUNS=5 WARMUP_RUNS=2`.

Compare upstream and fork prompt processing with the author's PP8192 shape:

```bash
./scripts/benchmark-llama-fork-pp8192.sh
```

On native Linux, five PP8192 repetitions measured **923.18 +/- 1.69 tok/s
upstream** and **951.54 +/- 0.61 tok/s in the fork**, a 3.1% fork uplift.

The balanced fork DFlash2 profile was validated with three 512-token code runs
at 128K capacity. Median code decode throughput was **62.62 tok/s**. The
equivalent standard DFlash2 profile without the tuned probability threshold
measured **56.35 tok/s**, so the optimized profile improved code throughput by
11.1% while retaining Q8 target KV.

With 100K occupied tokens, the fork measured **422.39 prompt tok/s and 31.28
decode tok/s with Q8 target KV**. Q4 target KV increased those figures to
**427.67 prompt tok/s and 33.77 decode tok/s**, but is retained only as an
explicit quality/performance tradeoff. The native upstream Q8 profile remained
faster at this length, at 456.81 prompt tok/s and 36.32 decode tok/s.

### Fork context allocation ceiling

An allocation-only sweep started a fresh optimized fork server at each 10K
increment, without filling or processing the advertised context. A single
19-token `ping` request was then used only to verify that each admitted server
could perform inference:

| Requested context | Startup/allocation | Minimal inference |
| ---: | :---: | :---: |
| 150K | pass | `pong` |
| 160K | pass | `pong` |
| 170K | pass | `pong` |
| 180K | pass | `pong` |
| **190K** | **pass** | **`pong`** |
| 200K | fail twice | not run |

The largest validated allocation is therefore **190K requested tokens**
(190,208 cells after internal alignment). At 200K the target and draft contexts
were created far enough to expose the limiting allocation, then ROCm failed to
reserve a 549.17 MiB draft compute buffer. This is an allocation/startup ceiling,
not proof that 190K occupied tokens can be processed reliably. The 128K launcher
default intentionally retains operating headroom.

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

### Compare MTP with MTP + ngram-mod

```bash
./scripts/benchmark-llama-hip-speculative.sh
```

The baseline is the launcher's complete default configuration, including 128K
context, Flash Attention, `-ub 2048`, Jinja, reasoning auto, and mmap. The
experiment changes speculative decoding only:

```text
--spec-type draft-mtp,ngram-mod
--spec-draft-p-min 0.82
--spec-draft-n-max 5
--spec-ngram-mod-n-match 24
--spec-ngram-mod-n-min 8
```

It alternates profile order over two server rounds and reports native draft
acceptance alongside prompt and decode throughput. Set `SERVER_ROUNDS=4` for a
more stable median.

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

## Native measured baseline

At 128K capacity, upstream DFlash2 depth 3 measured median code decode
throughput of **63.05 tok/s** over three 512-token runs. Autoregressive code
decode measured approximately **36.46 tok/s** in the screening run. Raw native
output is under `artifacts/`.

## Agent evaluations

Run the evaluation cockpit; its Start button uses the balanced profile by default:

```bash
cd agent-evals && cp .env.example .env && $EDITOR .env && ./run.sh

# Select another managed launcher for this invocation.
QWEN_ENGINE_PROFILE=dflash-long ./run.sh
QWEN_ENGINE_PROFILE=baseline ./run.sh
```

See [`agent-evals/README.md`](agent-evals/README.md) for TOML authoring, sample submodules, isolated worktrees, and live server telemetry.
