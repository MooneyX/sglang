#!/bin/bash
# bigrun.sh - SuffixPrefetchV2 large-scale A+B test orchestrator.
# Runs INSIDE the container (sglang-SuffixPrefetchV2) on the GPU host.
#
# Final code only (f58af3d3c+): suffix_race has per-request wait/race mode switch.
#
# Module A (steady-state matrix):
#   rdma:  policy {wc,race,be} x probe 8K n=32 + sweep {16,32,48,64}K x 8 reps
#   file:  policy {wc,race} x delay {0,100}us x probe 8K n=24 + sweep {16,32,64}K x 6
#          + be @ delay0 (be never reads L3; delay irrelevant)
#   [cancelled: file@300us slow-disk configs]
# Module B1 (random fluctuation, n=3 seeds x 40 rows/cell):
#   backend {rdma,file} x policy {wc,race} x seeds {11,22,33}, fluctuate2 1-8s / 0-800us
# Module B2 (fixed bandwidth point scan):
#   backend {rdma,file} x policy {wc,race} x delay {0,50,100,200}us x 12 rows/length
#   [cancelled: 400/800us slow levels]
#
# B1/B2 reuse the same fluct_test prefixes (seed 777) written once per backend;
# delay is changed via the runtime control file, no server restart needed.
#
# Usage:  bash bigrun.sh [A|B|all]      # default all; idempotent resume (skips existing json)
# Env:    MEMFRAC (0.90) CHUNK (8192) MC_MASTER (127.0.0.1:50052) MC_SEG (auto by RAM)
#         PAGE (1) tokens per L3 page; page>1 -> output files get a p{PAGE} prefix.
#         With PAGE>1 the per-page delay knob no longer equals "slow disk"
#         (delay is a per-page sleep, see PAGE16_TEST_PLAN §3.2), so Module A
#         automatically drops the file delay=100 cells and runs file0 only.
set -u
MODULE=${1:-all}
PORT=31000
TAG=big
SERVER_LOG=/tmp/sp_server_${TAG}.log
DELAY_FILE=/tmp/hicache_read_delay_us
WORK=/sgl-workspace/sglang/sp_v2
OUT=/tmp/bigrun
MEMFRAC=${MEMFRAC:-0.90}
CHUNK=${CHUNK:-8192}
PAGE=${PAGE:-1}
MC_MASTER=${MC_MASTER:-127.0.0.1:50052}
FILE_DIR=/data1/hicache_v2
mkdir -p "$OUT"

log(){ echo "[bigrun $(date +%m%d-%H:%M:%S)] $*"; }

kill_fluct(){ pkill -f '[f]luctuate2' 2>/dev/null; }

kill_server(){
  kill_fluct
  pkill -f '[l]aunch_server' 2>/dev/null
  for _ in $(seq 1 72); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ "${used:-0}" -lt 2000 ] && return 0
    sleep 5
  done
  log "WARN: GPU still busy (max used=${used:-?}MiB) - continuing anyway"
}

ensure_master(){
  # If MC_SEG is unset, pick from free RAM; a running master keeps its config
  # (yesterday's master used 32gb - if it is still up we must match it).
  if [ -z "${MC_SEG:-}" ]; then
    if pgrep -f 'mooncake_master.*50052' >/dev/null; then
      MC_SEG=32gb
      log "master 50052 already running - assuming seg=32gb (set MC_SEG to override)"
    else
      local freeg
      freeg=$(free -g | awk '/^Mem:/{print $7}')
      MC_SEG=32gb
      [ "${freeg:-0}" -ge 150 ] && MC_SEG=64gb
      [ "${freeg:-0}" -ge 300 ] && MC_SEG=128gb
      log "starting mooncake_master 50052 seg=$MC_SEG (free=${freeg}G)"
      nohup mooncake_master -rpc_port 50052 -metrics_port 19004 > /tmp/mc_master_50052.log 2>&1 &
      sleep 5
    fi
  elif ! pgrep -f 'mooncake_master.*50052' >/dev/null; then
    log "starting mooncake_master 50052 seg=$MC_SEG"
    nohup mooncake_master -rpc_port 50052 -metrics_port 19004 > /tmp/mc_master_50052.log 2>&1 &
    sleep 5
  fi
  export MC_SEG
}

start_server(){ # backend policy delay clean
  local backend=$1 policy=$2 delay=$3 clean=$4
  kill_server
  if [ "$clean" = yes ] && [ "$backend" = file ]; then
    log "cleaning $FILE_DIR"
    rm -rf "${FILE_DIR:?}"/* 2>/dev/null
  fi
  echo "$delay" > "$DELAY_FILE"
  log "starting server backend=$backend policy=$policy delay=${delay}us page=$PAGE (mem=$MEMFRAC chunk=$CHUNK)"
  MC_MASTER=$MC_MASTER MC_SEG=${MC_SEG:-} PAGE=$PAGE nohup bash "$WORK/start_dsv3.sh" \
      "$PORT" "$TAG" "$MEMFRAC" "$CHUNK" "$policy" 0 "$delay" "$backend" \
      > /tmp/bigrun_start.log 2>&1 &
  sleep 60
  if ! pgrep -f '[l]aunch_server' >/dev/null; then
    log "FATAL: server died at startup"; tail -30 "$SERVER_LOG" 2>/dev/null; exit 1
  fi
}

wait_healthy(){
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 10
  done
  log "FATAL: server not healthy after 30min"; tail -30 "$SERVER_LOG" 2>/dev/null; exit 1
}

check_page(){
  # Watchdog: some attention backends (FlashMLA=64, Cutlass=128, TRTLLM=32/64)
  # silently override page_size. If the running server disagrees with PAGE,
  # warn so results labeled p{PAGE} aren't misread. No match = stay silent.
  local actual
  actual=$(grep -aoiE "page_size[=: ]+[0-9]+" "$SERVER_LOG" | tail -1 | grep -aoE "[0-9]+")
  if [ -n "$actual" ] && [ "$actual" != "$PAGE" ]; then
    log "WARN: server page_size=$actual != requested PAGE=$PAGE (backend forced it); results labeled p${PAGE} are actually p${actual}"
  fi
}

run_probe(){ # label n
  local label=$1 n=$2
  [ -s "$OUT/probe_p${PAGE}_${label}.json" ] && { log "skip probe_p${PAGE}_${label} (exists)"; return; }
  python -u "$WORK/probe_v2.py" "$PORT" --n "$n" --prefix-len 8192 --mode sequential \
      --server-log "$SERVER_LOG" --out "$OUT/probe_p${PAGE}_${label}.json"
}

run_sweep(){ # label lengths repeats
  local label=$1 lengths=$2 reps=$3
  [ -s "$OUT/sweep_p${PAGE}_${label}.json" ] && { log "skip sweep_p${PAGE}_${label} (exists)"; return; }
  python -u "$WORK/sweep_v2.py" "$PORT" --lengths "$lengths" --repeats "$reps" \
      --server-log "$SERVER_LOG" --out "$OUT/sweep_p${PAGE}_${label}.json"
}

run_fluct(){ # label rounds seed skipwrite
  local label=$1 rounds=$2 seed=$3 skip=$4
  [ -s "$OUT/fluct_p${PAGE}_${label}.json" ] && { log "skip fluct_p${PAGE}_${label} (exists)"; return; }
  local extra=""
  [ "$skip" = yes ] && extra="--skip-write"
  python -u "$WORK/fluct_test.py" "$PORT" --lengths 32768,65536 --n 4 --rounds "$rounds" \
      --seed "$seed" --settle 30 $extra \
      --server-log "$SERVER_LOG" --out "$OUT/fluct_p${PAGE}_${label}.json"
}

# ---------- Module A ----------
module_A(){
  # rdma: 3 policies, full lengths, n=32/8 reps
  ensure_master
  for pol in wait_complete suffix_race best_effort; do
    start_server mooncake "$pol" 0 yes
    wait_healthy
    check_page
    run_probe "rdma_${pol}" 32
    run_sweep "rdma_${pol}" "16384,32768,49152,65536" 8
  done
  if [ "$PAGE" -gt 1 ]; then
    # page>1: delay knob is a per-page sleep, so 100us no longer emulates a
    # slow disk (page=16 -> ~7.6GB/s equivalent). Run file0 only; slow-disk
    # semantics need delay*PAGE and are covered by a dedicated run.
    log "PAGE=$PAGE: skipping file delay=100 cells (delay semantics changed, see PAGE16_TEST_PLAN)"
  else
    for delay in 0 100; do
      for pol in wait_complete suffix_race; do
        start_server file "$pol" "$delay" yes
        wait_healthy
        check_page
        run_probe "file${delay}_${pol}" 24
        run_sweep "file${delay}_${pol}" "16384,32768,65536" 6
      done
    done
  fi
  start_server file best_effort 0 yes
  wait_healthy
  check_page
  run_probe "file0_best_effort" 24
  run_sweep "file0_best_effort" "16384,32768,65536" 6
}

# ---------- B shared: write fluct prefixes once per backend ----------
write_fluct_prefixes(){ # backend_label
  # writes under the current server (any policy; write-through is policy-independent)
  run_fluct "$1_write" 0 777 no
}

# ---------- Module B1: random fluctuation ----------
module_B1(){
  ensure_master
  # rdma
  for pol in wait_complete suffix_race; do
    start_server mooncake "$pol" 0 no
    wait_healthy
    check_page
    write_fluct_prefixes rdma
    for seed in 11 22 33; do
      kill_fluct
      nohup python -u "$WORK/fluctuate2.py" "$seed" 1 8 0 800 > "$OUT/fluctuator_rdma_${pol}_s${seed}.log" 2>&1 &
      sleep 2
      run_fluct "rdma_${pol}_s${seed}" 5 777 yes
    done
    kill_fluct
  done
  # file: first policy cleans + writes prefixes, second reuses (same seed 777)
  local first=yes
  for pol in wait_complete suffix_race; do
    start_server file "$pol" 0 "$first"
    wait_healthy
    check_page
    write_fluct_prefixes file
    for seed in 11 22 33; do
      kill_fluct
      nohup python -u "$WORK/fluctuate2.py" "$seed" 1 8 0 800 > "$OUT/fluctuator_file_${pol}_s${seed}.log" 2>&1 &
      sleep 2
      run_fluct "file_${pol}_s${seed}" 5 777 yes
    done
    kill_fluct
    first=no
  done
}

# ---------- Module B2: fixed delay point scan ----------
module_B2(){
  ensure_master
  for be_pair in "rdma mooncake" "file file"; do
    set -- $be_pair
    local label=$1 backend=$2
    for pol in wait_complete suffix_race; do
      # reuse prefixes written by B1 (same seed 777); clean only if missing
      local clean=no
      [ "$backend" = file ] && [ ! -s "$OUT/fluct_p${PAGE}_${label}_write.json" ] && clean=yes
      start_server "$backend" "$pol" 0 "$clean"
      wait_healthy
      check_page
      [ ! -s "$OUT/fluct_p${PAGE}_${label}_write.json" ] && write_fluct_prefixes "$label"
      for d in 0 50 100 200; do   # 400/800us slow levels cancelled
        echo "$d" > "$DELAY_FILE"
        sleep 2
        run_fluct "${label}_${pol}_d${d}" 3 777 yes
      done
      echo 0 > "$DELAY_FILE"
    done
  done
}

case "$MODULE" in
  A)   module_A ;;
  B1)  module_B1 ;;
  B2)  module_B2 ;;
  B)   module_B1; module_B2 ;;
  all) module_A; module_B1; module_B2 ;;
  *)   echo "usage: bash bigrun.sh [A|B1|B2|B|all]"; exit 2 ;;
esac
kill_server
log "DONE module=$MODULE"
