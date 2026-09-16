# Qwen Engine Workspace Instructions

## Purpose

This workspace runs and benchmarks Qwen3.8-27B locally through llama.cpp on AMD GPUs with ROCm. It also contains a Go-based coding-agent evaluation harness. Preserve reproducible launch profiles and benchmark results when changing builds or runtime flags.

## Repository Layout

- `llama-fork/`: pinned RDNA3/RX 7900 XTX optimized llama.cpp fork from `nasone32/llama.cpp-RDNA3-7900xtx-opt`.
- `llama-hip/`: pinned upstream llama.cpp HIP build and local model storage.
- `agent-evals/`: coding-agent evaluation TUI and isolated evaluation samples.
- `scripts/`: rebuild, download, benchmark, and analysis tools.
- `artifacts/`: ignored benchmark logs and research reports.
- `start-llama-fork.sh`: primary single-discrete-GPU launcher.
- `start-llama-fork-dual.sh`: dual-discrete-GPU launcher with the target on the RX 7900 XTX and the draft model on the RX 6800 XT.

Nested repositories have their own `AGENTS.md` files. Follow those instructions when working inside them. Do not commit, push, open a PR, or write PR/reviewer text unless the applicable repository instructions and the user request allow it.

## Hardware and Platform

- OS: native Linux, CachyOS kernel series 7.2.
- CPU: AMD Ryzen 9 7950X3D, 16 cores and 32 threads.
- Motherboard: MSI B650 Gaming Plus WiFi, MS-7E26.
- Primary GPU: AMD Radeon RX 7900 XTX, 24 GiB, `gfx1100`, PCIe 4.0 x16 from the CPU.
- Secondary GPU: AMD Radeon RX 6800 XT, 16 GiB, `gfx1030`, connected on the motherboard PCI_E3 link through a PCIe 4.0 riser. ROCm reports its marketing name as the generic `AMD Radeon RX 6800`.
- PCI_E3 is physically x16 but electrically PCIe 4.0 x4 through the B650 chipset.
- Integrated GPU: Ryzen `gfx1036`. Never select it as the target or draft GPU.
- PSU: Corsair RM1000x 1000 W with separate modular PCIe power cables for the secondary GPU.
- ROCm/HIP: `/opt/rocm`, version series 7.2. Use the ROCm Clang toolchain, not distro HIP headers or compilers.

After any GPU, riser, BIOS, driver, or visibility change, verify devices before inference:

```sh
./llama-fork/build-rocm-dual/bin/llama-server --list-devices
```

Do not assume device indices remain stable. The dual launcher checks that the configured target is an RX 7900 XTX and the configured draft device matches ROCm's RX 6800 family name.

## Models

The active model directory is `llama-hip/models/qwen3.8-27b-q4_0`:

- Target: `Qwen3.8-27B-Q4_0.gguf`, about 15 GiB.
- DFlash2 draft: `DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, about 1.1 GiB.
- MTP draft: `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`, about 1.3 GiB.

DFlash2 is the default speculative decoder. Keep reasoning enabled unless a benchmark explicitly studies reasoning-off behavior.

## ROCm Builds

The optimized fork has two independent build directories:

- `llama-fork/build-rocm-gfx1100-portable`: single-GPU build for `gfx1100`.
- `llama-fork/build-rocm-dual`: fat HIP build for `gfx1100;gfx1030`.

Use the rebuild scripts as the source of truth because they isolate ROCm headers and preserve the tuned compiler flags:

```sh
./scripts/rebuild-llama-fork.sh
./scripts/rebuild-llama-fork-dual.sh
```

The dual build must contain both targets in one executable. Confirm after rebuilding:

```sh
roc-obj-ls llama-fork/build-rocm-dual/bin/libggml-hip.so | rg 'gfx(1030|1100)'
```

Keep `GGML_HIP_RCCL=OFF` initially. The secondary card uses a chipset PCIe 4.0 x4 link, so do not make `row` or experimental `tensor` split the default. Test them only as explicit experiments.

## Runtime Profiles

Single GPU:

```sh
./start-llama-fork.sh
```

The single-GPU baseline uses the RX 7900 XTX for target and DFlash2, 131072 context, Q8 target KV, F16 draft KV, Flash Attention, batch 2048, ubatch 512, one slot, reasoning auto/medium, and split mode `none`.

Dual GPU:

```sh
./start-llama-fork-dual.sh
```

The initial dual topology is:

- RX 7900 XTX: target weights, target KV, recurrent state, and target compute.
- RX 6800 XT: DFlash2 or MTP weights, draft KV, and draft compute.
- Split mode: `none`.
- Initial validation profile: 200000 context with Q8 target KV.

The dual launcher delegates all common model, reasoning, sampling, and speculative flags to `start-llama-fork.sh`. Keep shared defaults in the single-GPU script and only dual-specific overrides in the wrapper.

For a lower-memory long-context profile:

```sh
CACHE_TYPE_K=q4_0 CACHE_TYPE_V=q4_0 CTX_SIZE=200000 ./start-llama-fork-dual.sh
```

For the validated maximum-context Q8 profile, split target layers and their KV 80/20 while keeping DFlash2 on the secondary GPU:

```sh
DEVICE=ROCm0,ROCm1 SPLIT_MODE=layer LLAMA_ARG_TENSOR_SPLIT=80,20 \
  CTX_SIZE=262144 ./start-llama-fork-dual.sh
```

Keep the default dual profile at `split-mode=none`: layer splitting is a capacity profile, not a general performance optimization. The 80/20 ratio has been filled-context validated; a 90/10 split was substantially slower at decode and is not recommended.

## Measured Context Results

On the RX 7900 XTX with DFlash2 and reasoning enabled:

- Q8 KV at 100K: 430.45 prompt tok/s, 34.29 generation tok/s, about 21.56 GiB after the run.
- Q4 KV at 100K: 433.89 prompt tok/s, 40.36 generation tok/s, about 19.73 GiB after the run.
- Q4 KV at 198K occupied tokens: 271.92 prompt tok/s, 29.35 generation tok/s, about 21.95 GiB after the run.
- Q8 KV at 200K fails on the single GPU when the draft compute buffer requests another 549 MiB. Moving the draft workload to the RX 6800 XT is intended to remove this specific limit.
- Q4 KV at 262144 allocates on one GPU but has little safety margin. Draft offload should improve that margin, but it still requires a full occupied-context validation.

Dual-GPU smoke validation on 2026-09-14 used 32768 context, Q8 target KV, F16 draft KV, DFlash2 depth 3, reasoning auto/medium, target `ROCm0`, and draft `ROCm1`:

- The target workload used the RX 7900 XTX and the draft allocation used about 1985 MiB on the RX 6800 XT. The iGPU remained unused.
- A 192-token request reached 48.60 generation tok/s with 123 of 199 draft tokens accepted, or 61.8 percent.
- A second 384-token request drove the RX 6800 XT to 99 percent busy and about 70 W. It reached 32.40 generation tok/s with 200 of 515 draft tokens accepted, or 38.8 percent.
- The content and acceptance rates differed, so these smoke numbers are functional evidence, not a controlled single-vs-dual performance comparison.
- No AMDGPU reset, GPU fault, timeout, or PCIe AER error appeared. After server shutdown, draft VRAM returned to about 17 MiB and the RX 6800 XT entered runtime suspend.

Controlled 100K single-vs-dual tests on 2026-09-14 used identical prompts, seeds, outputs, Q8/F16 caches, DFlash2 depth 3, threshold 0.20, reasoning auto/medium, batch 2048, and ubatch 512:

- Single-GPU median decode was 64.88 tok/s for code and 72.13 tok/s for reasoning.
- Draft-on-6800 median decode was 40.84 tok/s for code and 45.41 tok/s for reasoning, about 37 percent slower in both scenarios.
- Median DFlash acceptance remained effectively unchanged: 57.46 vs 56.99 percent for code and 68.09 percent for reasoning.
- Matching runs produced identical output hashes. The dual topology preserves behavior but trades speed for target-GPU VRAM headroom.

A filled-context Q8 dual test on 2026-09-14 processed 198112 prompt tokens in a 200000 requested context without OOM:

- Prefill was 253.82 tok/s and took 780.54 seconds.
- Generation at the filled context was 15.38 tok/s for 64 tokens.
- DFlash accepted 43 of 58 draft tokens, or 74.1 percent.
- The RX 7900 XTX memory breakdown reported 823 MiB free after the run; transient system readings during prefill showed as little as about 100 MiB free.
- The RX 6800 XT held about 2.0 GiB during the run. Its logged allocations were 1079.61 MiB model, 50 MiB KV, and 539.14 MiB draft compute.
- The stress test passed memory and runtime stability. Retrieval quality was not fully validated because the 64-token output limit was consumed by reasoning before the final answer.

Layer-split tests on 2026-09-14 used both GPUs for the target model while DFlash2 remained on the RX 6800 XT:

- At 100K, the 80/20 target split placed 11084.32 MiB of target weights on the RX 7900 XTX and 3282.90 MiB on the RX 6800 XT. Median decode was 41.97 tok/s for code and 36.28 tok/s for reasoning.
- At 100K, a 90/10 split produced only 19.80 tok/s for code and 21.80 tok/s for reasoning. The unbalanced pipeline was slower and is not recommended.
- The 80/20 Q8 profile fully processed 260113 prompt tokens and generated 64 more in a 262144 allocated context. Prefill was 163.84 tok/s and generation at the filled context was 8.16 tok/s; DFlash accepted 44 of 57 drafts, or 77.2 percent.
- The final 262K memory breakdown reported 3211 MiB free on the RX 7900 XTX and 7074 MiB free on the RX 6800 XT. No OOM, GPU fault/reset/timeout, or PCIe AER error occurred.
- The 64-token output budget again ended during reasoning after locating ALPHA, so the run validates maximum-context allocation, prefill, and generation stability rather than completed three-value retrieval quality.

Long-context numbers come from synthetic repeated-token stress prompts. Do not compare speculative generation speeds without also reporting draft acceptance, prompt contents, run count, and cache type.

The current dual-GPU report is `artifacts/research/dual-gpu-rx7900xtx-rx6800xt-20260914.md`. The older TurboQuant research is in `artifacts/research/turboquant-rx7900xtx-benchmark.md`; its source clones were removed, so do not recreate or integrate them unless the user explicitly revives that work.

## Change Safety

- Preserve unrelated root changes, including current `.gitmodules` and `vllm-fork` work.
- Build directories and benchmark artifacts are local outputs; do not add them to Git.
- Do not alter the proven single-GPU launcher when a dual-only wrapper or environment override is sufficient.
- Before a long benchmark, verify no stale server is using VRAM and record both GPUs separately.
- For performance comparisons, keep model, prompt, reasoning, speculative settings, batch, ubatch, and cache types fixed unless the changed variable is the subject of the test.
