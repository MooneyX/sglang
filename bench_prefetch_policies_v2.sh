#!/bin/bash
# End-to-end benchmark v2: compare hicache prefetch-stop policies on a large
# model with a heavy shared-prefix load, to expose the cost_aware trade-off.
# For each policy: warm L3, restart to force prefetch-on-queue, then measure.
set -u

MODEL="${BENCH_MODEL:-/root/.cache/huggingface/qwen14b}"
PORT=31000
HICACHE_DIR=/tmp/hicache_bench2
RESULT_DIR=/tmp/bench_results2
LOG_DIR=/tmp/bench_logs2
mkdir -p "$RESULT_DIR" "$LOG_DIR"

# Heavy shared-prefix workload: long prefix (big KV to prefetch) + high concurrency.
GSP_GROUPS="${GSP_GROUPS:-16}"
GSP_PER_GROUP="${GSP_PER_GROUP:-12}"
GSP_SYS_LEN="${GSP_SYS_LEN:-8192}"     # long shared prefix -> large KV transfer
GSP_Q_LEN="${GSP_Q_LEN:-256}"
GSP_OUT_LEN="${GSP_OUT_LEN:-64}"
NUM_PROMPTS=$((GSP_GROUPS * GSP_PER_GROUP))
REQ_RATE="${REQ_RATE:-24}"             # high rate -> real queuing

start_server () {
  local policy="$1"
  local logf="$LOG_DIR/server_${policy}.log"
  rm -rf "$HICACHE_DIR"; mkdir -p "$HICACHE_DIR"
  SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$HICACHE_DIR" \
  python -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    --tp 1 --mem-fraction-static 0.75 \
    --enable-hierarchical-cache \
    --hicache-ratio 1.5 \
    --hicache-storage-backend file \
    --hicache-storage-prefetch-policy "$policy" \
    --hicache-storage-backend-extra-config '{"prefetch_threshold":32,"cost_aware_gamma":1.0}' \
    --max-running-requests 32 \
    --log-level info \
    > "$logf" 2>&1 &
  echo $!
}

wait_ready () {
  for i in $(seq 1 180); do
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
  for i in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || return 0; sleep 1; done
  kill -9 "$pid" 2>/dev/null
}

POLICIES="${POLICIES:-best_effort wait_complete timeout cost_aware}"
echo "=== BENCH2 START $(date) model=$MODEL sys_len=$GSP_SYS_LEN rate=$REQ_RATE n=$NUM_PROMPTS ==="
for pol in $POLICIES; do
  echo "########## POLICY=$pol ##########"
  PID=$(start_server "$pol")
  if ! wait_ready; then echo "[$pol] server failed (warm)"; tail -n 30 "$LOG_DIR/server_${pol}.log"; stop_server "$PID"; continue; fi
  run_bench "$pol" warm "$RESULT_DIR/${pol}_warm.json"
  stop_server "$PID"
  sleep 3
  PID=$(start_server "$pol")
  if ! wait_ready; then echo "[$pol] server failed (measure)"; tail -n 30 "$LOG_DIR/server_${pol}.log"; stop_server "$PID"; continue; fi
  run_bench "$pol" measure "$RESULT_DIR/${pol}_measure.json"
  grep -iE "prefetch|storage_hit|revok|terminate" "$LOG_DIR/server_${pol}.log" | tail -n 60 > "$LOG_DIR/prefetch_${pol}.log"
  stop_server "$PID"
  sleep 3
  echo "[$pol] done"
done
echo "=== BENCH2 DONE $(date) ==="
ls -l "$RESULT_DIR"
