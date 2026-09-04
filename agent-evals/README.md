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

## Live telemetry

The TUI places a transparent local proxy between Pi and llama-hip for the coding-agent stage. It adds llama.cpp `return_progress` to streaming requests and reads server-sent `prompt_progress`, `gen_second`, and final timing data. Prompt throughput and **generation tok/s** therefore update while Pi performs tool loops, rather than only after the final response.

### Metric semantics

- **PP FRESH** is shown only when llama.cpp processes at least 16 non-cached prompt tokens. Cached prompt tokens are excluded, so a one-token cache continuation cannot appear as a misleading multi-thousand tok/s prefill result.
- **CR** is the cache-reuse ratio for the current model request: cached prompt tokens divided by total prompt tokens.
- **GEN 10S** is a rolling ten-second, time-weighted speed. It follows real slowdowns during generation and fades toward zero after generation stops instead of freezing on the final token speed.
