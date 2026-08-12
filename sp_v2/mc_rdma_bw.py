#!/usr/bin/env python3
"""Mooncake RDMA bandwidth probe: batch put/get of N x 70KB pages."""
import sys
import time

import torch

proto = sys.argv[1] if len(sys.argv) > 1 else "rdma"
dev = sys.argv[2] if len(sys.argv) > 2 else "mlx5_bond_1"
host = sys.argv[3] if len(sys.argv) > 3 else "28.31.3.220"
N = int(sys.argv[4]) if len(sys.argv) > 4 else 8255
PAGE = 71680  # 70KB per MLA page

from mooncake.store import MooncakeDistributedStore

store = MooncakeDistributedStore()
ret = store.setup(host, "P2PHANDSHAKE", 2 * 1024**3, 16 * 1024**2,
                  proto, dev, "127.0.0.1:50051", None)
assert ret == 0

buf = torch.zeros(N * PAGE, dtype=torch.uint8).pin_memory()
r = store.register_buffer(buf.data_ptr(), buf.numel())
assert r == 0

keys = [f"bw_probe_key_{i}" for i in range(N)]
ptrs = [buf.data_ptr() + i * PAGE for i in range(N)]
sizes = [PAGE] * N

# warmup
store.batch_put_from(keys[:128], ptrs[:128], sizes[:128])

t0 = time.perf_counter()
B = 1024
for i in range(0, N, B):
    store.batch_put_from(keys[i:i+B], ptrs[i:i+B], sizes[i:i+B])
t_put = time.perf_counter() - t0

t0 = time.perf_counter()
for i in range(0, N, B):
    store.batch_get_into(keys[i:i+B], ptrs[i:i+B], sizes[i:i+B])
t_get = time.perf_counter() - t0

gb = N * PAGE / 1e9
print(f"pages={N} total={gb:.2f}GB")
print(f"put: {t_put:.3f}s -> {gb/t_put:.2f} GB/s ({t_put/N*1e6:.1f} us/page)")
print(f"get: {t_get:.3f}s -> {gb/t_get:.2f} GB/s ({t_get/N*1e6:.1f} us/page)")
