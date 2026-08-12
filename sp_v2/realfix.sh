#!/bin/bash
# realfix.sh - redo cells lost to server crashes / pool limits (runs INSIDE container)
# MEMFRAC=0.925: pool ~70.9K tokens (>65600 needed) with 7.5% workspace headroom.
# Output: /tmp/bigrun/realfix_{label}_{pol}.json ; buckets per config below.
set -u
PORT=31000
TAG=big
SERVER_LOG=/tmp/sp_server_${TAG}.log
DELAY_FILE=/tmp/hicache_read_delay_us
WORK=/sgl-workspace/sglang/sp_v2
OUT=/tmp/bigrun
MEMFRAC=${MEMFRAC:-0.925}
CHUNK=${CHUNK:-8192}
MC_MASTER=${MC_MASTER:-127.0.0.1:50052}
FILE_DIR=/data1/hicache_v2
BUCKETS_FILE=/sgl-workspace/sglang/sp_v2/realdata/realdata_buckets.json
mkdir -p "$OUT"

log(){ echo "[realfix $(date +%m%d-%H:%M:%S)] $*"; }

kill_server(){
  pkill -f '[l]aunch_server' 2>/dev/null
  for _ in $(seq 1 72); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ "${used:-0}" -lt 3000 ] && { sleep 20; return 0; }
    sleep 5
  done
  log "WARN: GPU still busy (max ${used:-?}MiB)"
}

ensure_master(){
  if ss -tln 2>/dev/null | grep -q ':50052 '; then
    log "master 50052 alive"
  else
    MC_SEG=64gb nohup mooncake_master -rpc_port 50052 -metrics_port 19004 > /tmp/mc_master_50052.log 2>&1 &
    sleep 5
    log "started master 64gb"
  fi
  export MC_SEG=${MC_SEG:-64gb}
}

start_server(){ # backend policy delay clean
  local backend=$1 policy=$2 delay=$3 clean=$4
  kill_server
  if [ "$clean" = yes ] && [ "$backend" = file ]; then
    log "cleaning $FILE_DIR"
    rm -rf "${FILE_DIR:?}"/* 2>/dev/null
  fi
  echo "$delay" > "$DELAY_FILE"
  log "start server backend=$backend policy=$policy delay=${delay}us MEMFRAC=$MEMFRAC"
  MC_MASTER=$MC_MASTER MC_SEG=${MC_SEG:-} nohup bash "$WORK/start_dsv3.sh" \
      "$PORT" "$TAG" "$MEMFRAC" "$CHUNK" "$policy" 0 "$delay" "$backend" \
      > /tmp/realfix_start.log 2>&1 &
  sleep 60
  if ! pgrep -f '[l]aunch_server' >/dev/null; then
    log "FATAL: server died"; tail -20 "$SERVER_LOG" 2>/dev/null; exit 1
  fi
}

wait_healthy(){
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 10
  done
  log "FATAL: not healthy after 30min"; exit 1
}

check_pool(){
  local toks
  toks=$(grep -oE '#tokens: [0-9]+' "$SERVER_LOG" | tail -1 | awk '{print $2}')
  log "KV pool tokens=${toks:-unknown}"
  if [ -n "$toks" ] && [ "$toks" -lt 66000 ]; then
    log "FATAL: pool $toks < 66000"
    exit 1
  fi
}

run_real(){ # label buckets
  [ -s "$OUT/realfix_$1.json" ] && { log "skip realfix_$1"; return; }
  python -u "$WORK/realdata_test.py" "$PORT" --rounds 2 --settle 30 \
      --buckets "$2" --buckets-file "$BUCKETS_FILE" \
      --server-log "$SERVER_LOG" --out "$OUT/realfix_$1.json"
}

log "=== realfix start (MEMFRAC=$MEMFRAC) ==="
ensure_master
FIRST=1
# label backend policy delay clean buckets
run_one(){
  [ -s "$OUT/realfix_$1.json" ] && { log "skip realfix_$1"; return; }
  start_server "$2" "$3" "$4" "$5"
  wait_healthy
  [ "$FIRST" = 1 ] && { check_pool; FIRST=0; }
  run_real "$1" "$6"
}

run_one "rdma_wait_complete" mooncake wait_complete 0 no "65536"
run_one "rdma_suffix_race"   mooncake suffix_race   0 no "65536"
run_one "rdma_best_effort"   mooncake best_effort   0 no "32768"
run_one "file0_wait_complete" file wait_complete 0 yes "65536"
run_one "file0_suffix_race"   file suffix_race   0 no "49152,65536"
kill_server
log "=== realfix DONE ==="
