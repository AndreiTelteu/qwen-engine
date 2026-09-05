#!/usr/bin/env bash
# Compare mmap model loading while holding the required serving configuration fixed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START="$ROOT/start-llama-hip.sh"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/artifacts/llama-hip/benchmarks}"
LOGS_DIR="${LOGS_DIR:-$ROOT/artifacts/llama-hip/logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RESULTS="$RESULTS_DIR/mmap-$RUN_ID.csv"
SUMMARY="$RESULTS_DIR/mmap-$RUN_ID-summary.csv"
FAILURES="$RESULTS_DIR/mmap-$RUN_ID-failures.csv"

: "${PORT:=8081}"
: "${CTX_SIZE:=32768}"
: "${MAX_TOKENS:=512}"
: "${RUNS:=3}"
: "${WARMUP_RUNS:=1}"
: "${SERVER_ROUNDS:=2}"
: "${STARTUP_TIMEOUT_SECONDS:=600}"

# These settings are intentionally fixed. The only variable under test is MMAP.
FLASH_ATTN=on
UBATCH_SIZE=2048
JINJA=on
REASONING=auto
REASONING_FORMAT=auto
PROMPT="Fă-mi în Python un calculator TUI care să meargă și cu mouse-ul."

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
printf '%s\n' \
    "case,server_round,run,mmap,startup_s,prompt_tok_per_s,decode_tok_per_s,prompt_tokens,completion_tokens,draft_acceptance,accepted,drafted" \
    > "$RESULTS"
printf '%s\n' "case,server_round,mmap,stage,log" > "$FAILURES"

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
            return 1
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            printf 'Timed out after %ss waiting for llama-server; log: %s\n' \
                "$STARTUP_TIMEOUT_SECONDS" "$current_log" >&2
            return 1
        fi
        sleep 2
    done
}

record_failure() {
    local case_name=$1 round=$2 mmap=$3 stage=$4
    printf '%s,%s,%s,%s,%s\n' \
        "$case_name" "$round" "$mmap" "$stage" "$current_log" >> "$FAILURES"
}

record_request() {
    local case_name=$1 round=$2 run=$3 mmap=$4 startup_seconds=$5
    local response="$RESULTS_DIR/mmap-$RUN_ID-$case_name-$round-$run.json"

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

    python3 - "$case_name" "$round" "$run" "$mmap" "$startup_seconds" "$response" >> "$RESULTS" <<'PY'
import json
import sys

case_name, round_number, run, mmap, startup, response = sys.argv[1:]
payload = json.load(open(response, encoding="utf-8"))
timings = payload["timings"]
usage = payload.get("usage", {})
drafted = timings.get("draft_n", 0)
accepted = timings.get("draft_n_accepted", 0)
acceptance = accepted / drafted if drafted else 0

print(",".join(map(str, [
    case_name, round_number, run, mmap, startup,
    f"{float(timings.get('prompt_per_second', 0)):.2f}",
    f"{float(timings.get('predicted_per_second', 0)):.2f}",
    usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
    f"{acceptance:.4f}", accepted, drafted,
])))
PY
}

run_case() {
    local case_name=$1 round=$2 mmap=$3
    local run start_ms ready_ms startup_seconds

    current_log="$LOGS_DIR/mmap-$RUN_ID-$case_name-$round.log"
    start_ms="$(now_ms)"
    CTX_SIZE="$CTX_SIZE" PORT="$PORT" REASONING="$REASONING" \
        FLASH_ATTN="$FLASH_ATTN" UBATCH_SIZE="$UBATCH_SIZE" JINJA="$JINJA" \
        REASONING_FORMAT="$REASONING_FORMAT" MMAP="$mmap" \
        "$START" > "$current_log" 2>&1 &
    server_pid=$!
    if ! wait_for_server; then
        record_failure "$case_name" "$round" "$mmap" server_start
        stop_server
        return 0
    fi
    ready_ms="$(now_ms)"
    startup_seconds="$(python3 - "$start_ms" "$ready_ms" <<'PY'
import sys
print(f"{(int(sys.argv[2]) - int(sys.argv[1])) / 1000:.3f}")
PY
)"

    for run in $(seq 1 "$WARMUP_RUNS"); do
        if ! record_request "$case_name" "$round" "warmup-$run" "$mmap" "$startup_seconds"; then
            record_failure "$case_name" "$round" "$mmap" warmup_request
            stop_server
            return 0
        fi
    done

    for run in $(seq 1 "$RUNS"); do
        if ! record_request "$case_name" "$round" "$run" "$mmap" "$startup_seconds"; then
            record_failure "$case_name" "$round" "$mmap" measured_request
            stop_server
            return 0
        fi
    done
    stop_server
}

# Alternate order every round so mmap-on does not always get a cold or warm page cache.
for round in $(seq 1 "$SERVER_ROUNDS"); do
    if [ $((round % 2)) -eq 1 ]; then
        run_case mmap-on "$round" on
        run_case no-mmap "$round" off
    else
        run_case no-mmap "$round" off
        run_case mmap-on "$round" on
    fi
done

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
        "case", "samples", "startup_s_median", "prompt_tok_per_s_median",
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
            "startup_s_median": f"{median('startup_s'):.3f}",
            "prompt_tok_per_s_median": f"{median('prompt_tok_per_s'):.2f}",
            "decode_tok_per_s_median": f"{median('decode_tok_per_s'):.2f}",
            "completion_tokens_median": f"{median('completion_tokens'):.0f}",
            "draft_acceptance_median": f"{median('draft_acceptance'):.4f}",
        })
PY

printf '\nFixed configuration: -fa on, -ub 2048, --jinja, --reasoning auto, --reasoning-format auto\n'
printf 'Raw samples: %s\nSummary: %s\nFailures: %s\n\n' "$RESULTS" "$SUMMARY" "$FAILURES"
column -s, -t < "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
if [ "$(wc -l < "$FAILURES")" -gt 1 ]; then
    printf '\nConfigurations that did not complete:\n'
    column -s, -t < "$FAILURES" 2>/dev/null || cat "$FAILURES"
fi
