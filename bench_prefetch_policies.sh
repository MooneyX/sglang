#!/bin/bash
# End-to-end benchmark: compare hicache prefetch-stop policies.
# Runs a shared-prefix workload; for each policy: warm L3, restart to force
# prefetch-on-queue, then measure. All inside the sglang-dev container.
set -u

MODEL=/root/.cache/huggingface/qwen05b
PORT=31000
HICACHE_DIR=/tmp/hicache_bench
RESULT_DIR=/tmp/bench_results
LOG_DIR=/tmp/bench_logs
mkdir -p "$RESULT_DIR" "$LOG_DIR"

# Shared-prefix workload knobs (small, for a quick but meaningful run)
GSP_GROUPS=8
GSP_PER_GROUP=8
GSP_SYS_LEN=2048      # long shared prefix -> big KV to prefetch
GSP_Q_LEN=128
GSP_OUT_LEN=32
NUM_PROMPTS=$((GSP_GROUPS * GSP_PER_GROUP))
REQ_RATE=8            # create queuing so prefetch overlaps waiting

start_server () {
  local policy="$1"
  local logf="$LOG_DIR/server_${policy}.log"
  rm -rf "$HICACHE_DIR"; mkdir -p "$HICACHE_DIR"
  SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$HICACHE_DIR" \
  python -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --tp 1 --mem-fraction-static 0.6 \
    --enable-hierarchical-cache \
    --hicache-ratio 1.2 \
    --hicache-storage-backend file \
    --hicache-storage-prefetch-policy "$policy" \
    --hicache-storage-backend-extra-config '{"prefetch_threshold":32,"cost_aware_gamma":1.0}' \
    --log-level info \
    > "$logf" 2>&1 &
  echo $!
}

wait_ready () {
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  return 1
}

run_bench () {
  local policy="$1" tag="$2" outfile="$3"
  python -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port "$PORT" \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups "$GSP_GROUPS" \
    --gsp-prompts-per-group "$GSP_PER_GROUP" \
    --gsp-system-prompt-len "$GSP_SYS_LEN" \
    --gsp-question-len "$GSP_Q_LEN" \
    --gsp-output-len "$GSP_OUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQ_RATE" \
    --output-file "$outfile" \
    > "$LOG_DIR/bench_${policy}_${tag}.log" 2>&1
}

stop_server () {
  local pid="$1"
  kill "$pid" 2>/dev/null
  for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || return 0; sleep 1; done
  kill -9 "$pid" 2>/dev/null
}

POLICIES="best_effort wait_complete timeout cost_aware"
echo "=== BENCH START $(date) ==="
for pol in $POLICIES; do
  echo "########## POLICY=$pol ##########"
  # Phase 1: warm L3 (server writes shared-prefix KV to file backend)
  PID=$(start_server "$pol")
  if ! wait_ready; then echo "[$pol] server failed to start (warm)"; tail -n 30 "$LOG_DIR/server_${pol}.log"; stop_server "$PID"; continue; fi
  run_bench "$pol" warm "$RESULT_DIR/${pol}_warm.json"
  stop_server "$PID"
  sleep 3
  # Phase 2: restart (GPU/host cache cleared, L3 files persist) -> prefetch on queue
  PID=$(start_server "$pol")
  if ! wait_ready; then echo "[$pol] server failed to start (measure)"; tail -n 30 "$LOG_DIR/server_${pol}.log"; stop_server "$PID"; continue; fi
  run_bench "$pol" measure "$RESULT_DIR/${pol}_measure.json"
  # capture prefetch-related server log lines
  grep -iE "prefetch|hicache|storage" "$LOG_DIR/server_${pol}.log" | tail -n 40 > "$LOG_DIR/prefetch_${pol}.log"
  stop_server "$PID"
  sleep 3
  echo "[$pol] done"
done
echo "=== BENCH DONE $(date) ==="
ls -l "$RESULT_DIR"
