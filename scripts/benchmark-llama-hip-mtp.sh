#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START="$ROOT/start-llama-hip.sh"
RESULTS_DIR="$ROOT/artifacts/llama-hip/benchmarks"
LOGS_DIR="$ROOT/artifacts/llama-hip/logs"
RESULTS="$RESULTS_DIR/mtp-32k.csv"
PORT=8081
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"
printf "draft_n_max,prompt_tok_per_s,decode_tok_per_s,draft_acceptance,accepted,drafted\\n" > "$RESULTS"

stop_server() {
    if [ -n "${server_pid:-}" ] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid"
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap stop_server EXIT INT TERM

for draft_n_max in 1 2 3 4 5 6; do
    log="$LOGS_DIR/benchmark-mtp-${draft_n_max}.log"
    response="$RESULTS_DIR/mtp-${draft_n_max}.json"
    CTX_SIZE=32768 SPEC_DRAFT_N_MAX="$draft_n_max" PORT="$PORT" REASONING=off \
        "$START" > "$log" 2>&1 &
    server_pid=$!

    for _ in $(seq 1 48); do
        if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            cat "$log"
            exit 1
        fi
        sleep 5
    done

    curl --fail --silent --show-error "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\\"messages\\":[{\\"role\\":\\"user\\",\\"content\\":\\"Explică, într-un text coerent și tehnic de cel puțin 400 de cuvinte, rolul unui cache KV în inferența unui model transformer.\\"}],\\"temperature\\":0,\\"max_tokens\\":512}" \
        > "$response"

    python3 - "$draft_n_max" "$response" >> "$RESULTS" <<"PY"
import json
import sys
n, response = sys.argv[1:]
timings = json.load(open(response))["timings"]
drafted = timings.get("draft_n", 0)
accepted = timings.get("draft_n_accepted", 0)
acceptance = accepted / drafted if drafted else 0
print(f"{n},{timings[prompt_per_second]:.2f},{timings[predicted_per_second]:.2f},{acceptance:.4f},{accepted},{drafted}")
PY

    stop_server
    unset server_pid
done
cat "$RESULTS"
