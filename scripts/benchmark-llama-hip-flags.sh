#!/usr/bin/env bash
# Compare llama-server runtime flags against one fixed coding request.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START="$ROOT/start-llama-hip.sh"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/artifacts/llama-hip/benchmarks}"
LOGS_DIR="${LOGS_DIR:-$ROOT/artifacts/llama-hip/logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RESULTS="$RESULTS_DIR/flags-$RUN_ID.csv"
SUMMARY="$RESULTS_DIR/flags-$RUN_ID-summary.csv"

: "${PORT:=8081}"
: "${CTX_SIZE:=32768}"
: "${MAX_TOKENS:=512}"
: "${RUNS:=3}"
: "${WARMUP_RUNS:=1}"
: "${STARTUP_TIMEOUT_SECONDS:=600}"

# Keep this request identical for every case. `cache_prompt: false` prevents one
# sample from reusing the prompt KV cache of a previous sample.
PROMPT="Fă-mi în Python un calculator TUI care să meargă și cu mouse-ul."

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
printf '%s\n' \
    "case,run,flash_attn,cache_type_v,ubatch_size,jinja,reasoning_format,mmap,startup_s,prompt_tok_per_s,decode_tok_per_s,prompt_tokens,completion_tokens,draft_acceptance,accepted,drafted" \
    > "$RESULTS"

server_pid=""

stop_server() {
    if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid"
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=""
}
trap stop_server EXIT INT TERM

now_ms() {
    date +%s%3N
}

wait_for_server() {
    local deadline=$(( $(date +%s) + STARTUP_TIMEOUT_SECONDS ))
    while ! curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            printf 'llama-server exited before becoming ready; log follows:\n' >&2
            cat "$current_log" >&2
            exit 1
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            printf 'Timed out after %ss waiting for llama-server; log: %s\n' \
                "$STARTUP_TIMEOUT_SECONDS" "$current_log" >&2
            exit 1
        fi
        sleep 2
    done
}

record_request() {
    local case_name=$1 run=$2 flash_attn=$3 cache_type_v=$4 ubatch_size=$5 jinja=$6 reasoning_format=$7 mmap=$8
    local response="$RESULTS_DIR/flags-$RUN_ID-$case_name-$run.json"

    PROMPT="$PROMPT" MAX_TOKENS="$MAX_TOKENS" \
        python3 - <<'PY' | curl --fail --silent --show-error \
            "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            --data-binary @- > "$response"
import json
import os

print(json.dumps({
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "temperature": 0,
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "cache_prompt": False,
}))
PY

    python3 - "$case_name" "$run" "$flash_attn" "$cache_type_v" "$ubatch_size" "$jinja" \
        "$reasoning_format" "$mmap" "$startup_seconds" "$response" >> "$RESULTS" <<'PY'
import json
import sys

case_name, run, flash_attn, cache_type_v, ubatch_size, jinja, reasoning_format, mmap, startup, response = sys.argv[1:]
payload = json.load(open(response, encoding="utf-8"))
timings = payload["timings"]
usage = payload.get("usage", {})
drafted = timings.get("draft_n", 0)
accepted = timings.get("draft_n_accepted", 0)
acceptance = accepted / drafted if drafted else 0

def number(name):
    return float(timings.get(name, 0))

print(",".join(map(str, [
    case_name, run, flash_attn, cache_type_v, ubatch_size, jinja, reasoning_format, mmap,
    startup, f"{number('prompt_per_second'):.2f}",
    f"{number('predicted_per_second'):.2f}",
    usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
    f"{acceptance:.4f}", accepted, drafted,
])))
PY
}

run_case() {
    local case_name=$1 flash_attn=$2 ubatch_size=$3 jinja=$4 reasoning_format=$5 mmap=$6
    local run start_ms ready_ms startup_seconds cache_type_v=q8_0
    if [ "$flash_attn" = "off" ]; then
        cache_type_v=f16
    fi

    current_log="$LOGS_DIR/flags-$RUN_ID-$case_name.log"
    start_ms="$(now_ms)"
    CTX_SIZE="$CTX_SIZE" PORT="$PORT" REASONING=off \
        FLASH_ATTN="$flash_attn" CACHE_TYPE_V="$cache_type_v" UBATCH_SIZE="$ubatch_size" JINJA="$jinja" \
        REASONING_FORMAT="$reasoning_format" MMAP="$mmap" \
        "$START" > "$current_log" 2>&1 &
    server_pid=$!
    wait_for_server
    ready_ms="$(now_ms)"
    startup_seconds="$(python3 - "$start_ms" "$ready_ms" <<'PY'
import sys
print(f"{(int(sys.argv[2]) - int(sys.argv[1])) / 1000:.3f}")
PY
)"

    # Exercise one request after startup to avoid timing one-time HIP setup.
    for run in $(seq 1 "$WARMUP_RUNS"); do
        record_request "$case_name" "warmup-$run" "$flash_attn" "$cache_type_v" "$ubatch_size" \
            "$jinja" "$reasoning_format" "$mmap"
    done

    for run in $(seq 1 "$RUNS"); do
        record_request "$case_name" "$run" "$flash_attn" "$cache_type_v" "$ubatch_size" \
            "$jinja" "$reasoning_format" "$mmap"
    done
    stop_server
}

# Baseline turns off the tested features. Each following case changes precisely
# one dimension, except `all-flags`, which represents the proposed full setup.
run_case baseline off 512 off none on
run_case flash-attn-on on 512 off none on
run_case ubatch-256 off 256 off none on
run_case ubatch-1024 off 1024 off none on
run_case ubatch-2048 off 2048 off none on
run_case jinja-on off 512 on none on
run_case reasoning-format-auto off 512 off auto on
run_case no-mmap off 512 off none off
run_case all-flags on 2048 on auto off

python3 - "$RESULTS" "$SUMMARY" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

results, summary = sys.argv[1:]
groups = defaultdict(list)
with open(results, newline="", encoding="utf-8") as source:
    for row in csv.DictReader(source):
        if not row["run"].startswith("warmup-"):
            groups[row["case"]].append(row)

with open(summary, "w", newline="", encoding="utf-8") as destination:
    fields = [
        "case", "samples", "startup_s", "prompt_tok_per_s_median",
        "decode_tok_per_s_median", "completion_tokens_median",
        "draft_acceptance_median",
    ]
    writer = csv.DictWriter(destination, fieldnames=fields)
    writer.writeheader()
    for case_name, rows in groups.items():
        median = lambda field: statistics.median(float(row[field]) for row in rows)
        writer.writerow({
            "case": case_name,
            "samples": len(rows),
            "startup_s": f"{median('startup_s'):.3f}",
            "prompt_tok_per_s_median": f"{median('prompt_tok_per_s'):.2f}",
            "decode_tok_per_s_median": f"{median('decode_tok_per_s'):.2f}",
            "completion_tokens_median": f"{median('completion_tokens'):.0f}",
            "draft_acceptance_median": f"{median('draft_acceptance'):.4f}",
        })
PY

printf '\nRaw samples: %s\nSummary: %s\n\n' "$RESULTS" "$SUMMARY"
column -s, -t < "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
