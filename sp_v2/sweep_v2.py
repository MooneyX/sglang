#!/usr/bin/env python3
"""sweep_v2.py - prefix-length sweep: recompute vs L3-prefetch latency.

For each prefix length L and each repeat:
  1. send a FRESH prefix (never seen, nothing in L3) -> cold recompute TTFT
  2. flush_cache (L1+L2 cleared, L3 keeps the copy written by write-through)
  3. resend the SAME prefix -> L3 prefetch path (prefetch_dur from server log)
Both divided by token count -> per-token latency vs position/length.

Runs INSIDE the container. Usage:
  python -u sweep_v2.py PORT [--lengths 4096,8192,16384,32768,65536]
      [--repeats 2] [--suffix-len 64] [--out /tmp/sweep_v2_result.json]
"""

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request

MEASURE_RE = re.compile(r"\[PrefetchMeasure\] (.+)$")
FLUSH_WAIT_S = 5


def post_json(url, payload, timeout=900):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return body, time.perf_counter() - t0


def flush_cache(port, retries=10):
    url = f"http://127.0.0.1:{port}/flush_cache"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=b"", method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
            return True
        except Exception:
            time.sleep(2)
    return False


def log_line_count(path):
    try:
        with open(path, "r", errors="ignore") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def read_new_measures(path, baseline):
    """New [PrefetchMeasure] lines after baseline, deduped by rid."""
    out, seen = [], set()
    kv_re = re.compile(r"(\w+)=([\w.\-]+)")
    with open(path, "r", errors="ignore") as f:
        for ln, line in enumerate(f, 1):
            if ln <= baseline:
                continue
            m = MEASURE_RE.search(line)
            if not m:
                continue
            fields = dict(kv_re.findall(m.group(1)))
            if fields.get("rid") in seen:
                continue
            seen.add(fields.get("rid"))
            for k in ("arrival_qlen", "device_hit", "host_hit", "prefetch_len", "l3_loaded"):
                if k in fields:
                    fields[k] = int(fields[k])
            for k in ("prefetch_dur", "queue_dur", "prefetch_wait", "sched_wait"):
                if k in fields:
                    fields[k] = float(fields[k])
            out.append(fields)
    return out


def send(port, input_ids):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
    }
    body, lat = post_json(f"http://127.0.0.1:{port}/generate", payload)
    mi = body.get("meta_info", {})
    return lat, mi.get("cached_tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("--lengths", default="4096,8192,16384,32768,65536")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--suffix-len", type=int, default=64)
    ap.add_argument("--server-log", default="/tmp/sp_server_v2.log")
    ap.add_argument("--out", default="/tmp/sweep_v2_result.json")
    ap.add_argument("--seed", type=int, default=None,
                    help="prefix RNG seed; default None = nondeterministic (avoid L3 reuse across runs)")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    rng = random.Random(args.seed)
    rows = []

    for L in lengths:
        for rep in range(args.repeats):
            ids = [rng.randrange(1000, 100000) for _ in range(L + args.suffix_len)]

            # ---- path 1: cold recompute (prefix unseen, not in L3) ----
            t0 = time.perf_counter()
            lat_cold, cached = send(args.port, ids)
            rec = {
                "L": L, "rep": rep,
                "recompute_ttft": round(lat_cold, 4),
                "cold_cached_tokens": cached,
            }
            print(f"[sweep] L={L} rep={rep} recompute_ttft={lat_cold:.3f}s "
                  f"(cached={cached})", flush=True)

            # ---- flush L1+L2, keep L3 ----
            ok = flush_cache(args.port)
            if not ok:
                print("[sweep] WARN flush failed", flush=True)
            time.sleep(FLUSH_WAIT_S)

            # ---- path 2: L3 prefetch ----
            base = log_line_count(args.server_log)
            lat_warm, cached2 = send(args.port, ids)
            time.sleep(2)
            ms = [m for m in read_new_measures(args.server_log, base)
                  if m.get("prefetch_len", 0) > 0]
            m = ms[-1] if ms else {}
            rec.update({
                "prefetch_ttft": round(lat_warm, 4),
                "prefetch_dur": m.get("prefetch_dur"),
                "l3_loaded": m.get("l3_loaded"),
                "prefetch_wait": m.get("prefetch_wait"),
                "sched_wait": m.get("sched_wait"),
                # per-token metrics (microseconds)
                "recompute_us_per_tok": round(lat_cold / L * 1e6, 1),
                "prefetch_us_per_tok": (
                    round(m["prefetch_dur"] / L * 1e6, 1)
                    if m.get("prefetch_dur") else None
                ),
            })
            print(f"[sweep] L={L} rep={rep} prefetch_dur={m.get('prefetch_dur')}s "
                  f"l3={m.get('l3_loaded')} "
                  f"us/tok rec={rec['recompute_us_per_tok']} "
                  f"pf={rec['prefetch_us_per_tok']}", flush=True)
            rows.append(rec)

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)
    print(f"[sweep] result -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
