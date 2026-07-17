#!/bin/bash
# A/B probe: does L3 read-throttle actually slow prefetch, and is prefetch even
# triggered? Same workload, only GET_DELAY_MS differs. Debug logging on so the
# "Prefetch <rid> completed with N tokens" lines are visible.
set -u
MODEL=/root/.cache/huggingface/qwen14b
PORT=31000
LOG_DIR=/tmp/ab_logs
RES_DIR=/tmp/ab_results
mkdir -p "$LOG_DIR" "$RES_DIR"
rm -f "$RES_DIR"/*.json

GSP_GROUPS=8; GSP_PER_GROUP=10; GSP_SYS_LEN=4096; GSP_Q_LEN=128; GSP_OUT_LEN=64
NUM_PROMPTS=$((GSP_GROUPS*GSP_PER_GROUP)); REQ_RATE=6

start_server () {
  local delay="$1" wipe="${2:-0}" hdir="/tmp/hicache_ab_${delay}"
  # Only wipe L3 on the warm (first) start; the measure restart MUST keep the
  # L3 files populated by warm, otherwise prefetch queries hit an empty L3.
  if [ "$wipe" = "1" ]; then rm -rf "$hdir"; fi
  mkdir -p "$hdir"
  SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$hdir" \
  SGLANG_HICACHE_FILE_BACKEND_GET_DELAY_MS="$delay" \
  python -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --tp 1 --mem-fraction-static 0.75 --enable-hierarchical-cache --hicache-ratio 1.5 \
    --hicache-storage-backend file --hicache-storage-prefetch-policy cost_aware \
    --hicache-storage-backend-extra-config '{"prefetch_threshold":32,"cost_aware_gamma":1.0}' \
    --max-running-requests 32 --log-level debug > "$LOG_DIR/server_d${delay}.log" 2>&1 &
  echo $!
}
wait_ready () { for i in $(seq 1 180); do curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }
bench () { python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port "$PORT" \
  --dataset-name generated-shared-prefix --gsp-num-groups "$GSP_GROUPS" --gsp-prompts-per-group "$GSP_PER_GROUP" \
  --gsp-system-prompt-len "$GSP_SYS_LEN" --gsp-question-len "$GSP_Q_LEN" --gsp-output-len "$GSP_OUT_LEN" \
  --num-prompts "$NUM_PROMPTS" --request-rate "$REQ_RATE" --output-file "$1" > "$2" 2>&1; }
stop () { kill "$1" 2>/dev/null; for i in $(seq 1 40); do kill -0 "$1" 2>/dev/null || return 0; sleep 1; done; kill -9 "$1" 2>/dev/null; }

for delay in 0 30; do
  echo "########## DELAY=${delay}ms ##########"
  PID=$(start_server "$delay" 1); wait_ready || { echo "start failed d=$delay"; tail -n 20 "$LOG_DIR/server_d${delay}.log"; stop "$PID"; continue; }
  bench "$RES_DIR/warm_d${delay}.json" "$LOG_DIR/bench_warm_d${delay}.log"; stop "$PID"; sleep 3
  echo "[d=$delay] L3_files_after_warm: $(ls /tmp/hicache_ab_${delay} 2>/dev/null | wc -l)"
  PID=$(start_server "$delay" 0); wait_ready || { echo "restart failed d=$delay"; stop "$PID"; continue; }
  echo "[d=$delay] L3_files_at_measure_start: $(ls /tmp/hicache_ab_${delay} 2>/dev/null | wc -l)"
  bench "$RES_DIR/measure_d${delay}.json" "$LOG_DIR/bench_measure_d${delay}.log"
  echo "[d=$delay] prefetch_completed_loglines: $(grep -c 'Prefetch .* completed with' "$LOG_DIR/server_d${delay}.log")"
  echo "[d=$delay] prefetch_nonzero_tokens: $(grep -oE 'completed with [1-9][0-9]* tokens' "$LOG_DIR/server_d${delay}.log" | wc -l)"
  echo "[d=$delay] throttle_enabled_log: $(grep -c 'read throttle enabled' "$LOG_DIR/server_d${delay}.log")"
  stop "$PID"; sleep 3
done
echo "=== AB DONE ==="
python3 -c "
import json,glob
for f in sorted(glob.glob('$RES_DIR/*.json')):
    L=[l for l in open(f) if l.strip()]
    d=json.loads(L[-1])
    print(f.split('/')[-1], 'ttft_mean', round(d['mean_ttft_ms'],1), 'e2e_mean', round(d['mean_e2e_latency_ms'],1), 'thrpt', round(d['request_throughput'],2))
"
