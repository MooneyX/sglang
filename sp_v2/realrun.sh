#!/bin/bash
# realrun.sh - real-text full matrix: 3 backends x 3 policies (runs INSIDE container)
# rdma / file0 / file100  x  wait_complete / suffix_race / best_effort
# Idempotent: skips configs whose output json exists.
set -u
PORT=31000
TAG=big
SERVER_LOG=/tmp/sp_server_${TAG}.log
DELAY_FILE=/tmp/hicache_read_delay_us
WORK=/sgl-workspace/sglang/sp_v2
OUT=/tmp/bigrun
MEMFRAC=${MEMFRAC:-0.90}
CHUNK=${CHUNK:-8192}
MC_MASTER=${MC_MASTER:-127.0.0.1:50052}
FILE_DIR=/data1/hicache_v2
mkdir -p "$OUT"

log(){ echo "[realrun $(date +%m%d-%H:%M:%S)] $*"; }
kill_fluct(){ pkill -f '[f]luctuate2' 2>/dev/null; }

kill_server(){
  kill_fluct
  pkill -f '[l]aunch_server' 2>/dev/null
  for _ in $(seq 1 72); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ "${used:-0}" -lt 3000 ] && { sleep 20; return 0; }
    sleep 5
  done
  log "WARN: GPU still busy (max ${used:-?}MiB)"
}

ensure_master(){
  local want_seg=64gb   # one bucket max ~37GB (64K x 6); 32gb too small
  if ss -tln 2>/dev/null | grep -q ':50052 '; then
    local cur
    cur=$(grep -oE 'global_segment_size[^,]*' /tmp/mc_master_50052.log 2>/dev/null | head -1)
    log "master 50052 alive"
  else
    log "starting mooncake_master 50052 seg=$want_seg"
    MC_SEG=$want_seg nohup mooncake_master -rpc_port 50052 -metrics_port 19004 > /tmp/mc_master_50052.log 2>&1 &
    sleep 5
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
  log "start server backend=$backend policy=$policy delay=${delay}us"
  MC_MASTER=$MC_MASTER MC_SEG=${MC_SEG:-} nohup bash "$WORK/start_dsv3.sh" \
      "$PORT" "$TAG" "$MEMFRAC" "$CHUNK" "$policy" 0 "$delay" "$backend" \
      > /tmp/realrun_start.log 2>&1 &
  sleep 60
  if ! pgrep -f '[l]aunch_server' >/dev/null; then
    log "FATAL: server died"; tail -30 "$SERVER_LOG" 2>/dev/null; exit 1
  fi
}

wait_healthy(){
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 10
  done
  log "FATAL: not healthy after 30min"; tail -30 "$SERVER_LOG" 2>/dev/null; exit 1
}

run_real(){ # label
  [ -s "$OUT/real_$1.json" ] && { log "skip real_$1"; return; }
  python -u "$WORK/realdata_test.py" "$PORT" --rounds 2 --settle 30 \
      --server-log "$SERVER_LOG" --out "$OUT/real_$1.json"
}

log "=== real run start ==="
ensure_master
for combo in "mooncake rdma 0" "file file0 0" "file file100 100"; do
  set -- $combo
  local_backend=$1; label=$2; delay=$3
  for pol in wait_complete suffix_race best_effort; do
    [ -s "$OUT/real_${label}_${pol}.json" ] && { log "skip real_${label}_${pol}"; continue; }
    start_server "$local_backend" "$pol" "$delay" yes
    wait_healthy
    run_real "${label}_${pol}"
  done
done
kill_server
log "=== real run DONE ==="
