#!/bin/bash
# start_dsv3.sh - DeepSeek-V3 hicache server for SuffixPrefetchV2 measurement
# Usage: bash start_dsv3.sh [PORT] [TAG] [MEMFRAC] [CHUNK] [POLICY] [CGBS] [DELAY_US] [BACKEND]
# CGBS: cuda graph max bs decode; 0 = disable cuda graph entirely
# DELAY_US: artificial per-page L3 read delay in microseconds (file backend only)
# BACKEND: file | mooncake (mooncake requires mooncake_master on 127.0.0.1:50051)
# PAGE: tokens per L3 page (env, default 1 = server_args default on non-MUSA).
#       page>1 amortizes per-page Python/syscall overhead and enlarges each
#       storage batch (128 pages * PAGE * 70KB for MLA) -> higher L3 bandwidth.
#       NOTE: changing PAGE invalidates the on-disk hash layout -> clean storage
#       dir first (bigrun.sh does this via clean=yes).
PORT=${1:-31000}
TAG=${2:-v2}
MEMFRAC=${3:-0.90}
CHUNK=${4:-8192}
POLICY=${5:-wait_complete}
CGBS=${6:-0}
DELAY_US=${7:-0}
BACKEND=${8:-file}
PAGE=${PAGE:-1}
LOG=/tmp/sp_server_${TAG}.log
if [ "${CGBS}" -gt 0 ]; then
  GRAPH_FLAG="--cuda-graph-max-bs-decode ${CGBS}"
else
  GRAPH_FLAG="--disable-cuda-graph"
fi
if [ "${BACKEND}" = "mooncake" ]; then
  # RDMA validated on mlx5_bond_1 (RoCEv2, ~12 GB/s). local_hostname is the
  # RDMA storage-network IP (bond2); auto-detected per machine, override
  # with MC_HOST if needed. Use MC_PROTO=tcp to fall back (~0.56 GB/s).
  MC_PROTO=${MC_PROTO:-rdma}
  MC_MASTER=${MC_MASTER:-127.0.0.1:50051}
  MC_SEG=${MC_SEG:-8gb}
  if [ -z "${MC_HOST}" ]; then
    MC_HOST=$(ifconfig bond2 2>/dev/null | grep -oE 'inet [0-9.]+' | awk '{print $2}')
  fi
  if [ "${MC_PROTO}" = "rdma" ]; then
    EXTRA_CFG="{\"master_server_address\":\"${MC_MASTER}\",\"local_hostname\":\"${MC_HOST}\",\"protocol\":\"rdma\",\"device_name\":\"mlx5_bond_1\",\"global_segment_size\":\"${MC_SEG}\",\"prefetch_threshold\":64}"
  else
    EXTRA_CFG="{\"master_server_address\":\"${MC_MASTER}\",\"local_hostname\":\"127.0.0.1\",\"protocol\":\"tcp\",\"global_segment_size\":\"${MC_SEG}\",\"prefetch_threshold\":64}"
  fi
else
  EXTRA_CFG='{"prefetch_threshold":64}'
fi
SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/data1/hicache_v2 \
SGLANG_HICACHE_FILE_READ_DELAY_US=${DELAY_US} \
python -m sglang.launch_server \
  --model-path /data1/models/DeepSeek-V3 \
  --tp 8 \
  --host 127.0.0.1 --port ${PORT} \
  --mem-fraction-static ${MEMFRAC} \
  --chunked-prefill-size ${CHUNK} \
  --page-size ${PAGE} \
  ${GRAPH_FLAG} \
  --enable-hierarchical-cache \
  --hicache-storage-backend ${BACKEND} \
  --hicache-storage-prefetch-policy ${POLICY} \
  --hicache-storage-backend-extra-config "${EXTRA_CFG}" \
  --log-level info > ${LOG} 2>&1
