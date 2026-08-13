#!/bin/bash
# verify_store_node.sh - Route A verification: does the sglang hicache mooncake
# backend put KV onto a SEPARATE memory-only store node (M2), or does it stay in
# the sglang process (M1)? Runs INSIDE containers (same digest image).
#
# Usage (M1 = inference machine, M2 = memory-only backend machine):
#   M1: ROLE=master bash verify_store_node.sh     # start master w/ HTTP metadata (8080)
#   M2: ROLE=store  bash verify_store_node.sh     # start mooncake_store_service (50Gi, no GPU)
#   M1: ROLE=sglang bash verify_store_node.sh     # start sglang hicache mooncake (P2PHANDSHAKE)
#   M1: ROLE=write  bash verify_store_node.sh     # wave1 populate, then judge placement
#   M2: ROLE=watch  bash verify_store_node.sh     # poll store_service RSS / RDMA traffic
#
# Env: M1_IP (reachable IP of M1, used by master RPC + HTTP metadata),
#      M2_IP (bond2 IP of M2 for store local_hostname),
#      MASTER_PORT (50052), HTTP_PORT (8080), SEG (50gb),
#      DEV (mlx5_bond_1), WORK (/sgl-workspace/sglang/sp_v2), OUT (/tmp/store_verify)
#
# Judge (after ROLE=write): data is on M2 iff
#   (a) master log "Mem Storage: X GB / SEG" rises with X close to written bytes, AND
#   (b) M2 store_service RSS rises (ROLE=watch), AND
#   (c) M1 sglang RSS stays flat / low.
# If (a)-(c) all hold -> Route A works, no code change needed. Else -> Route B.
set -u
M1_IP=${M1_IP:?set M1_IP (master host)}
M2_IP=${M2_IP:?set M2_IP (store node bond2 IP)}
MASTER_PORT=${MASTER_PORT:-50052}
HTTP_PORT=${HTTP_PORT:-8080}
SEG=${SEG:-50gb}
DEV=${DEV:-mlx5_bond_1}
WORK=${WORK:-/sgl-workspace/sglang/sp_v2}
OUT=${OUT:-/tmp/store_verify}
MASTER_LOG=/tmp/mc_master_50052.log
mkdir -p "$OUT"

log(){ echo "[$(date +%m%d-%H:%M:%S) verify_store] $*"; }

precheck(){
  log "precheck mooncake components"
  python3 - <<'PY'
import importlib
for m in ("mooncake.mooncake_store_service", "mooncake.store"):
    try:
        importlib.import_module(m)
        print("OK", m)
    except Exception as e:
        print("MISSING", m, str(e)[:120])
PY
  ls /usr/local/bin/ | grep -iE "mooncake" || true
}

case "${ROLE:-}" in
  master)
    # master with HTTP metadata server (needed by mooncake_store_service)
    precheck
    pkill -9 -f "mooncake_master.*${MASTER_PORT}" 2>/dev/null
    sleep 2
    log "starting mooncake_master rpc=${MASTER_PORT} http=${HTTP_PORT} seg=${SEG}"
    MC_SEG=${SEG} nohup mooncake_master \
      --enable_http_metadata_server=true \
      --rpc_address=0.0.0.0 --rpc_port=${MASTER_PORT} \
      --http_metadata_server_host=0.0.0.0 --http_metadata_server_port=${HTTP_PORT} \
      --metrics_port=19004 > "${MASTER_LOG}" 2>&1 &
    sleep 6
    grep -E "Master service started|http_metadata" "${MASTER_LOG}" | tail -2
    ;;
  store)
    # memory-only backend node: no GPU, registers SEG of DRAM into the pool
    precheck
    pkill -9 -f "mooncake_store_service" 2>/dev/null
    sleep 2
    log "starting mooncake_store_service master=${M1_IP}:${MASTER_PORT} http=${M1_IP}:${HTTP_PORT} seg=${SEG}"
    MOONCAKE_MASTER="${M1_IP}:${MASTER_PORT}" \
    MOONCAKE_TE_META_DATA_SERVER="http://${M1_IP}:${HTTP_PORT}/metadata" \
    MOONCAKE_GLOBAL_SEGMENT_SIZE="${SEG}" \
    MOONCAKE_LOCAL_HOSTNAME="${M2_IP}" \
    MOONCAKE_PROTOCOL=rdma \
    MOONCAKE_DEVICE="${DEV}" \
    nohup python3 -m mooncake.mooncake_store_service > "${OUT}/store_service.log" 2>&1 &
    sleep 8
    tail -5 "${OUT}/store_service.log"
    log "store_service pid $(pgrep -f mooncake_store_service | head -1)"
    ;;
  sglang)
    # inference server: hicache mooncake backend, P2PHANDSHAKE to the same master.
    # MC_HOST must be THIS host's RDMA-reachable IP (cross-host 29.x here, NOT bond2).
    log "starting sglang hicache mooncake (master=${M1_IP}:${MASTER_PORT}, dev=${DEV}, host=${SGLANG_HOST:-${M1_IP}})"
    MC_MASTER="${M1_IP}:${MASTER_PORT}" MC_SEG="${SEG}" \
    MC_DEV="${DEV:-mlx5_bond_1}" MC_HOST="${SGLANG_HOST:-${M1_IP}}" PAGE=${PAGE:-16} \
    bash "${WORK}/start_dsv3.sh" 31000 storev 0.90 8192 wait_complete 0 0 mooncake
    ;;
  write)
    # wave1 populate (4 x 8K prefixes) then sleep; judge from master log + M2 watch
    log "populating prefixes (wave1)"
    python3 -u "${WORK}/probe_v2.py" 31000 --n 4 --prefix-len 8192 --mode sequential \
        --server-log /tmp/sp_server_storev.log --out "${OUT}/wave1.json"
    log "write done; now check:"
    log "  1) master log: grep 'Mem Storage' ${MASTER_LOG}  (M2 segment should rise)"
    log "  2) M2: ROLE=watch output (store_service RSS should rise)"
    log "  3) M1: ps -o rss= -p <sglang pid> (should stay flat)"
    ;;
  watch)
    # on M2: sample store_service RSS + RDMA iface bytes every 5s for 120s
    log "watching store_service RSS / rdma traffic on ${M2_IP}"
    PID=$(pgrep -f mooncake_store_service | head -1)
    [ -z "$PID" ] && { log "no store_service process"; exit 1; }
    IF=$(ls /sys/class/net | grep -iE "mlx5|ib|bond" | head -1)
    R1=$(cat /sys/class/net/$IF/statistics/rx_bytes 2>/dev/null)
    for i in $(seq 1 24); do
      RSS=$(ps -o rss= -p "$PID" | tr -d ' ')
      R2=$(cat /sys/class/net/$IF/statistics/rx_bytes 2>/dev/null)
      [ -n "$R1" ] && [ -n "$R2" ] && DELTA=$(( (R2-R1)/1024/1024 )) || DELTA=0
      log "t=${i} rss_mb=$(( RSS/1024 )) rdma_rx_delta_mb=${DELTA}"
      R1=$R2
      sleep 5
    done
    ;;
  *)
    echo "usage: ROLE=master|store|sglang|write|watch bash verify_store_node.sh (M1_IP/M2_IP required)"
    exit 2
    ;;
esac
