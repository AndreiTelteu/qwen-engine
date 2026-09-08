#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START="$ROOT/start-llama-fork.sh"
MODEL_DIR="$ROOT/llama-hip/models/qwen3.8-27b-q4_0"

: "${MODEL:=$MODEL_DIR/Qwen3.8-27B-Q4_0.gguf}"
: "${MTP_DRAFT:=$MODEL_DIR/MTP/mtp-Qwen3.8-27B-Q4_0.gguf}"
: "${DFLASH_DRAFT:=$MODEL_DIR/DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf}"
: "${RESULTS_DIR:=$ROOT/artifacts/llama-fork/benchmarks}"
: "${LOGS_DIR:=$ROOT/artifacts/llama-fork/logs}"
: "${RUN_ID:=$(date +%Y%m%d-%H%M%S)}"
: "${PORT:=8081}"
: "${CTX_SIZE:=50144}"
: "${MAX_TOKENS:=512}"
: "${RUNS:=3}"
: "${WARMUP_RUNS:=1}"
: "${STARTUP_TIMEOUT_SECONDS:=600}"

RESULTS="$RESULTS_DIR/speculators-$RUN_ID.csv"
SUMMARY="$RESULTS_DIR/speculators-$RUN_ID-summary.csv"
FAILURES="$RESULTS_DIR/speculators-$RUN_ID-failures.csv"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
printf '%s\n' \
    "profile,scenario,run,spec_type,draft_n_max,prompt_tok_per_s,decode_tok_per_s,prompt_tokens,completion_tokens,draft_acceptance,accepted,drafted" \
    > "$RESULTS"
printf '%s\n' "profile,stage,log" > "$FAILURES"

server_pid=""
current_log=""

stop_server() {
    if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid"
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=""
}
trap stop_server EXIT INT TERM

wait_for_server() {
    local deadline=$(( $(date +%s) + STARTUP_TIMEOUT_SECONDS ))
    while ! curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            return 1
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            return 1
        fi
        sleep 2
    done
}

record_request() {
    local profile=$1 scenario=$2 run=$3 spec_type=$4 draft_n_max=$5 prompt=$6
    local response="$RESULTS_DIR/speculators-$RUN_ID-$profile-$scenario-$run.json"

    PROMPT="$prompt" MAX_TOKENS="$MAX_TOKENS" \
        python3 - <<'PY' | curl --fail --silent --show-error \
            "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" --data-binary @- > "$response"
import json
import os

print(json.dumps({
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "temperature": 0,
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "cache_prompt": False,
}))
PY

    python3 - "$profile" "$scenario" "$run" "$spec_type" "$draft_n_max" "$response" >> "$RESULTS" <<'PY'
import json
import sys

profile, scenario, run, spec_type, draft_n_max, response = sys.argv[1:]
payload = json.load(open(response, encoding="utf-8"))
timings = payload["timings"]
usage = payload.get("usage", {})
drafted = timings.get("draft_n", 0)
accepted = timings.get("draft_n_accepted", 0)
acceptance = accepted / drafted if drafted else 0
print(",".join(map(str, [
    profile, scenario, run, spec_type, draft_n_max,
    f"{float(timings.get('prompt_per_second', 0)):.2f}",
    f"{float(timings.get('predicted_per_second', 0)):.2f}",
    usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
    f"{acceptance:.4f}", accepted, drafted,
])))
PY
}

run_profile() {
    local profile=$1 spec_type=$2 draft=$3 draft_n_max=$4
    local scenario run prompt

    current_log="$LOGS_DIR/speculators-$RUN_ID-$profile.log"
    MODEL="$MODEL" DRAFT="$draft" SPEC_TYPE="$spec_type" SPEC_DRAFT_N_MAX="$draft_n_max" \
        CTX_SIZE="$CTX_SIZE" PORT="$PORT" REASONING=off \
        "$START" > "$current_log" 2>&1 &
    server_pid=$!

    if ! wait_for_server; then
        printf '%s,%s,%s\n' "$profile" server_start "$current_log" >> "$FAILURES"
        cat "$current_log" >&2
        stop_server
        return
    fi

    for run in $(seq 1 "$WARMUP_RUNS"); do
        record_request "$profile" code "warmup-$run" "$spec_type" "$draft_n_max" \
            "Write a quicksort algorithm in Python. Write code only." || {
            printf '%s,%s,%s\n' "$profile" warmup_request "$current_log" >> "$FAILURES"
            stop_server
            return
        }
    done

    for scenario in prose code; do
        if [ "$scenario" = prose ]; then
            prompt="Explain in detail how speculative decoding works in transformer inference."
        else
            prompt="Implement an asynchronous LRU cache in Python with tests. Return code only."
        fi
        for run in $(seq 1 "$RUNS"); do
            record_request "$profile" "$scenario" "$run" "$spec_type" "$draft_n_max" "$prompt" || {
                printf '%s,%s,%s\n' "$profile" measured_request "$current_log" >> "$FAILURES"
                stop_server
                return
            }
        done
    done
    stop_server
}

run_profile no-speculation none "" 0
run_profile mtp-4 draft-mtp "$MTP_DRAFT" 4
run_profile mtp-adaptive-4 draft-mtp-adaptive "$MTP_DRAFT" 4
run_profile dflash2-4 draft-dflash "$DFLASH_DRAFT" 4
run_profile dflash2-block-max draft-dflash "$DFLASH_DRAFT" 7

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
            groups[(row["profile"], row["scenario"])].append(row)

with open(summary, "w", newline="", encoding="utf-8") as destination:
    fields = ["profile", "scenario", "samples", "prompt_tok_per_s_median",
              "decode_tok_per_s_median", "draft_acceptance_median"]
    writer = csv.DictWriter(destination, fieldnames=fields)
    writer.writeheader()
    for (profile, scenario), rows in groups.items():
        median = lambda field: statistics.median(float(row[field]) for row in rows)
        writer.writerow({
            "profile": profile,
            "scenario": scenario,
            "samples": len(rows),
            "prompt_tok_per_s_median": f"{median('prompt_tok_per_s'):.2f}",
            "decode_tok_per_s_median": f"{median('decode_tok_per_s'):.2f}",
            "draft_acceptance_median": f"{median('draft_acceptance'):.4f}",
        })
PY

printf '\nRaw samples: %s\nSummary: %s\nFailures: %s\n\n' "$RESULTS" "$SUMMARY" "$FAILURES"
column -s, -t < "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
