#!/usr/bin/env bash
# Phase E: full contradictory noise pool (3000 records).
#
# Usage:
#   bash scripts/run_phase_e.sh
#
# Wraps the canonical build_noise.py invocation so paste/auto-correct
# artifacts (em-dash, line-wrap) cannot mangle the CLI arguments. Logs
# go to outputs/logs/noise_contradictory.log; the process is detached
# via nohup + disown so it survives terminal exit.

set -euo pipefail

cd "$(dirname "$0")/.."

OUT="corpus/noise/contradictory_passages.jsonl"
LOG="outputs/logs/noise_contradictory.log"
EXCLUDE="data/sciknoweval/main_test_sampled.json"

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

EXCLUDE_ARG=()
if [[ -f "$EXCLUDE" ]]; then
    EXCLUDE_ARG=(--exclude-split "$EXCLUDE")
    echo "[run_phase_e] using --exclude-split $EXCLUDE"
else
    echo "[run_phase_e] WARN: $EXCLUDE not found; running WITHOUT --exclude-split (Phase C may not be done yet)"
fi

nohup python scripts/build_noise.py contradictory \
    --model qwen2.5-7b \
    --target 3000 \
    --seed 42 \
    --batch-size 16 \
    "${EXCLUDE_ARG[@]}" \
    --output "$OUT" \
    > "$LOG" 2>&1 &

PID=$!
disown

echo "[run_phase_e] started PID=$PID, logging to $LOG"
echo "[run_phase_e] tail -f $LOG     # to watch progress"
echo "[run_phase_e] wc -l $OUT       # to check record count when done"
