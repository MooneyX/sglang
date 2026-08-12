#!/usr/bin/env python3
"""mc_load.py - background bandwidth load injector for mooncake (real
contention instead of synthetic delay).

Sets up a MooncakeDistributedStore client once, then loops doing
batch_put_from / batch_get_into of N pages with random sizes and random
gaps, competing for master/transfer bandwidth with the sglang server.

Usage: python -u mc_load.py [proto] [device] [host_ip] [master] [max_pages] [seed]
  defaults: rdma mlx5_bond_1 28.31.3.220 127.0.0.1:50052 8000 1
"""

import random
import sys
import time

import torch

PROTO = sys.argv[1] if len(sys.argv) > 1 else "rdma"
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "mlx5_bond_1"
HOST = sys.argv[3] if len(sys.argv) > 3 else "28.31.3.220"
MASTER = sys.argv[4] if len(sys.argv) > 4 else "127.0.0.1:50052"
MAX_PAGES = int(sys.argv[5]) if len(sys.argv) > 5 else 8000
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 7

PAGE_BYTES = 61 * 576 * 2  # DeepSeek-V3 MLA per-token KV (bf16)

from mooncake.store import MooncakeDistributedStore

rng = random.Random(SEED)
store = MooncakeDistributedStore()
local_buffer = torch.empty(MAX_PAGES * PAGE_BYTES, dtype=torch.uint8).pin_memory()
ret = store.setup(
    HOST, "P2PHANDSHAKE",
    8 * 1024**3, 16 * 1024**2,
    PROTO, DEVICE if PROTO == "rdma" else "",
    MASTER, None,
)
assert ret == 0, f"setup failed: {ret}"
buf_ptr = local_buffer.data_ptr()
reg = store.register_buffer(buf_ptr, MAX_PAGES * PAGE_BYTES)
assert reg == 0, f"register_buffer failed: {reg}"
print(f"[mc_load] setup ok proto={PROTO} dev={DEVICE} max_pages={MAX_PAGES}", flush=True)

n = 0
while True:
    pages = rng.randint(500, MAX_PAGES)
    size = pages * PAGE_BYTES
    key = f"load_key_{rng.randint(0, 3)}"  # few keys -> repeated get hits
    t0 = time.perf_counter()
    r1 = store.batch_put_from([key], [buf_ptr], [size])
    r2 = store.batch_get_into([key], [buf_ptr], [size])
    dt = time.perf_counter() - t0
    n += 1
    if n % 5 == 0:
        print(f"[mc_load] iter={n} pages={pages} r=({r1},{r2}) "
              f"{size / 1e9 / dt:.2f}GB/s", flush=True)
    time.sleep(rng.uniform(0.05, 0.5))
