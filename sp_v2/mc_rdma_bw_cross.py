#!/usr/bin/env python3
"""Cross-host mooncake TE probe: put/get of N objects of given size.

Usage: python3 mc_rdma_bw_cross.py <PAGE_BYTES> <N> [B]
Connects to the existing master (29.209.114.88:50052) as a P2PHANDSHAKE
client on 29.209.114.88 (dev mlx5_9), writes/reads N objects via bare TE
batch_put_from / batch_get_into. Reports GB/s and us/object.
"""
import sys
import time

import torch

PAGE = int(sys.argv[1]) if len(sys.argv) > 1 else 71680
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
B = int(sys.argv[3]) if len(sys.argv) > 3 else 512
# local segment size in GiB; 0/None -> tiny segment so master places objects
# on the remote memholder segment (true cross-host test)
SEG = int(sys.argv[4]) * 1024**3 if len(sys.argv) > 4 and int(sys.argv[4]) > 0 else 0

from mooncake.store import MooncakeDistributedStore

store = MooncakeDistributedStore()
# local_hostname=29.209.114.88, P2PHANDSHAKE, 8GB seg, 16MB local buf,
# rdma, mlx5_9, master on 88:50052
ret = store.setup(
    "29.209.114.88",
    "P2PHANDSHAKE",
    SEG or (16 * 1024**2),
    16 * 1024**2,
    "rdma",
    "mlx5_9",
    "29.209.114.88:50052",
    None,
)
assert ret == 0, f"setup ret={ret}"

buf = torch.zeros(N * PAGE, dtype=torch.uint8).pin_memory()
r = store.register_buffer(buf.data_ptr(), buf.numel())
assert r == 0, f"register ret={r}"

keys = [f"bw_probe_cross_{i}" for i in range(N)]
ptrs = [buf.data_ptr() + i * PAGE for i in range(N)]
sizes = [PAGE] * N

# warmup 1 batch
store.batch_put_from(keys[: min(128, N)], ptrs[: min(128, N)], sizes[: min(128, N)])

t0 = time.perf_counter()
for i in range(0, N, B):
    store.batch_put_from(keys[i : i + B], ptrs[i : i + B], sizes[i : i + B])
t_put = time.perf_counter() - t0

t0 = time.perf_counter()
for i in range(0, N, B):
    store.batch_get_into(keys[i : i + B], ptrs[i : i + B], sizes[i : i + B])
t_get = time.perf_counter() - t0

gb = N * PAGE / 1e9
print(
    f"obj={PAGE} n={N} B={B} total={gb:.2f}GB "
    f"put={gb/t_put:.2f}GB/s({t_put/N*1e6:.1f}us/obj) "
    f"get={gb/t_get:.2f}GB/s({t_get/N*1e6:.1f}us/obj)"
)
