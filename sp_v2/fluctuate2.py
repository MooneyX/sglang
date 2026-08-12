#!/usr/bin/env python3
"""fluctuate2.py - random-interval random-value L3 bandwidth fluctuator.

Runs inside the container. Every random interval (uniform [lo, hi] seconds)
writes a random per-page delay (microseconds) to the runtime control file
read by the storage backends. 25% of draws are 0 (full speed), the rest
uniform in [vmin, vmax].

Usage: python -u fluctuate2.py [seed] [lo_s] [hi_s] [vmin_us] [vmax_us]
"""

import random
import sys
import time

FILE = "/tmp/hicache_read_delay_us"


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260730
    lo = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    hi = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    vmin = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    vmax = float(sys.argv[5]) if len(sys.argv) > 5 else 800.0
    rng = random.Random(seed)
    print(f"[fluct2] start seed={seed} interval=[{lo},{hi}]s value=[{vmin},{vmax}]us", flush=True)
    while True:
        dt = rng.uniform(lo, hi)
        time.sleep(dt)
        d = 0 if rng.random() < 0.25 else round(rng.uniform(vmin, vmax))
        with open(FILE, "w") as f:
            f.write(str(d))
        print(f"[fluct2] {time.strftime('%H:%M:%S')} next={dt:.1f}s delay={d}us", flush=True)


if __name__ == "__main__":
    main()
