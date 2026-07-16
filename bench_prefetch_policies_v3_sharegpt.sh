#!/bin/bash
# End-to-end benchmark v3: ShareGPT mixed-load, compare hicache prefetch-stop
# policies on the 14B model. ShareGPT gives realistic mixed conversation
# lengths; multi-turn reuse exercises the L3 prefix cache.
set -u

MODEL="${BENCH_MODEL:-/root/.cache/huggingface/qwen14b}"
DATASET_PATH="${DATASET_PATH:-/root/.cache/sglang_datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
PORT=31000
HICACHE_DIR=/tmp/hicache_bench3
RESULT_DIR=/tmp/bench_results3
LOG_DIR=/tmp/bench_logs3
mkdir -p "$RESULT_DIR" "$LOG_DIR"

NUM_PROMPTS="${NUM_PROMPTS:-400}"
REQ_RATE="${REQ_RATE:-24}"

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
    --dataset-name sharegpt \
    --dataset-path "$DATASET_PATH" \
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
echo "=== BENCH3 START $(date) model=$MODEL dataset=sharegpt n=$NUM_PROMPTS rate=$REQ_RATE ==="
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
echo "=== BENCH3 DONE $(date) ==="
ls -l "$RESULT_DIR"
