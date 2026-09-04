# Agent Evals

A live, keyboard-first evaluation cockpit for coding agents. It never starts or stops llama.cpp: start the local engine yourself from the workspace root, then launch this TUI.

```bash
cd ~/qwen-engine
./start-llama-hip.sh

# In a second terminal:
cd ~/qwen-engine/agent-evals
cp .env.example .env
$EDITOR .env
./run.sh
```

## First setup

1. Put the local Qwen OpenAI endpoint and model ID in the `AGENT_*` variables in `.env`.
2. Put a separate OpenAI-compatible judge endpoint, key, provider ID, and model ID in `JUDGE_*`.
3. Register each clean Laravel/React sample as a Git submodule:

   ```bash
   cd ~/qwen-engine
   git submodule add <repository-url> agent-evals/samples/laravel-users
   git commit -m "Add Laravel evaluation sample"
   ```

4. Add or edit `[[eval]]` blocks in `evals.toml`. The `e` key opens it in `$EDITOR`; `n` appends a complete editable template and opens it.

## Safety and repeatability

The source checkout in `samples/` is never used as the agent workspace. The TUI creates a detached Git worktree at `runs/<evaluation-id>/`, then before every run executes:

```text
git reset --hard
git clean -ffd
git gc --prune=now
```

This discards tracked and untracked changes made in the prior run but leaves ignored dependency folders such as `vendor/` and `node_modules/` intact. Results are written to `results/<evaluation-id>/` and intentionally ignored by the outer repository.

## Definition anatomy

```toml
[[eval]]
id = "laravel-add-user-filter"
title = "Laravel · user status filter"
sample = "laravel-users"
timeout = "15m"
agent_prompt = """What the coding agent must do."""
expected = """Observable definition of a good result."""
setup = ["composer install --no-interaction --prefer-dist"]
verify = ["php artisan test --filter=UserFilterTest", "vendor/bin/pint --test"]
judge_prompt = """Additional evaluation instructions for the independent judge."""
```

## Managed engine and authoritative telemetry

At startup, Agent Evals starts `../start-llama-hip.sh` with `VERBOSITY=4`, owns that child process, and stops only that child when the TUI exits. It writes the raw llama.cpp output to `results/engine/`.

It does **not** guess throughput from API chunks and has no HTTP metrics proxy. It parses the actual llama.cpp timing lines instead:

- **PP** is `prompt eval time` / `prompt processing` tokens per second.
- **GEN 3S** is llama.cpp’s `tg_3s`, already its measured rolling three-second generation speed.
- **CR** is cached tokens divided by cached plus fresh prompt tokens. Debug verbosity emits the cached-token count; final prompt timings emit fresh tokens.
- **DA** is MTP draft acceptance from `draft acceptance`.

If an engine is already listening on the configured port, Agent Evals leaves it untouched and marks it external. Stop that process, then relaunch the TUI, when managed logs are required.
