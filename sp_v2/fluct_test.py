#!/usr/bin/env python3
"""fluct_test.py - L3 bandwidth fluctuation end-to-end test (SuffixPrefetchV2)

Runs INSIDE the container against a local sglang server. Companion to the
runtime delay knob (SGLANG_HICACHE_FILE_READ_DELAY_FILE): a separate
fluctuator process rewrites that file every few seconds with a random delay
while this script measures.

Protocol per length L:
  write:  send N distinct prefixes once (populates L3; TTFT here is the
          cold-recompute reference), settle --settle s for write-back drain
  measure: rounds x N requests; before EACH request flush_cache (drop L1/L2)
          so the request must go through L3 prefetch. Record per-request
          client latency, cached_tokens, and the delay knob value seen at
          request start.

Usage:
  python -u fluct_test.py PORT [--lengths 32768,65536] [--n 4] [--rounds 5]
      [--settle 30] [--seed 777] [--out /tmp/fluct_result.json]
"""

import argparse
import json
import random
import re
import time
import urllib.request
import urllib.error

PAGE = 64
MEASURE_RE = re.compile(r"\[PrefetchMeasure\] (.+)$")
DELAY_FILE = "/tmp/hicache_read_delay_us"


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


def wait_healthy(port, timeout_s=1800):
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(10)
    return False


def cur_delay_us():
    try:
        with open(DELAY_FILE) as f:
            return float(f.read().strip())
    except Exception:
        return 0.0


def gen_prefixes(n, prefix_len, suffix_len, seed):
    assert prefix_len % PAGE == 0
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        prefix = [rng.randrange(1000, 100000) for _ in range(prefix_len)]
        suffix = [rng.randrange(1000, 100000) for _ in range(suffix_len)]
        reqs.append({"idx": i, "input_ids": prefix + suffix})
    return reqs


def send_one(port, item, L, rnd):
    payload = {
        "input_ids": item["input_ids"],
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
    }
    delay = cur_delay_us()
    try:
        body, lat = post_json(f"http://127.0.0.1:{port}/generate", payload)
        mi = body.get("meta_info", {})
        return {
            "L": L, "round": rnd, "idx": item["idx"], "ok": True,
            "delay_us": delay,
            "client_latency": round(lat, 4),
            "cached_tokens": mi.get("cached_tokens"),
        }
    except Exception as e:
        return {"L": L, "round": rnd, "idx": item["idx"], "ok": False,
                "delay_us": delay, "error": str(e)[:200]}


def log_line_count(path):
    try:
        with open(path, "r", errors="ignore") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def read_measures(path, baseline):
    out = []
    kv_re = re.compile(r"(\w+)=([\w.\-]+)")
    seen = set()
    with open(path, "r", errors="ignore") as f:
        for ln, line in enumerate(f, 1):
            if ln <= baseline:
                continue
            m = MEASURE_RE.search(line)
            if not m:
                continue
            fields = dict(kv_re.findall(m.group(1)))
            rid = fields.get("rid")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            for k in ("arrival_qlen", "device_hit", "host_hit", "prefetch_len", "l3_loaded", "pf_qdepth"):
                if k in fields:
                    fields[k] = int(fields[k])
            for k in ("prefetch_dur", "queue_dur", "prefetch_wait", "total_wait"):
                if k in fields:
                    fields[k] = float(fields[k])
            out.append(fields)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("--lengths", default="32768,65536")
    ap.add_argument("--n", type=int, default=4, help="prefixes per length")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--suffix-len", type=int, default=64)
    ap.add_argument("--settle", type=int, default=30)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--skip-write", action="store_true",
                    help="skip write+settle; reuses deterministic prefixes already in L3")
    ap.add_argument("--server-log", default="/tmp/sp_server_v2.log")
    ap.add_argument("--out", default="/tmp/fluct_result.json")
    args = ap.parse_args()

    assert wait_healthy(args.port), "server not healthy"
    lengths = [int(x) for x in args.lengths.split(",")]
    result = {"args": vars(args), "write": [], "measure": [], "measures": []}

    for L in lengths:
        reqs = gen_prefixes(args.n, L, args.suffix_len, args.seed + L)
        if args.skip_write:
            print(f"[fluct] L={L}: skip-write, reusing {args.n} prefixes", flush=True)
        else:
            print(f"[fluct] L={L}: writing {args.n} prefixes ...", flush=True)
            for r in reqs:
                res = send_one(args.port, r, L, -1)
                result["write"].append(res)
                print(f"[fluct]   write idx={r['idx']} lat={res.get('client_latency')}s", flush=True)
            print(f"[fluct] L={L}: settle {args.settle}s for write-back", flush=True)
            time.sleep(args.settle)

        for rnd in range(args.rounds):
            for r in reqs:
                if not flush_cache(args.port):
                    print("[fluct] WARNING: flush failed", flush=True)
                base = log_line_count(args.server_log)
                res = send_one(args.port, r, L, rnd)
                time.sleep(1)  # let server flush the measure line
                ms = read_measures(args.server_log, base)
                if ms:
                    res["mode"] = ms[-1].get("mode")
                    res["l3_loaded"] = ms[-1].get("l3_loaded")
                    res["prefetch_dur"] = ms[-1].get("prefetch_dur")
                result["measure"].append(res)
                print(f"[fluct]   L={L} r={rnd} idx={r['idx']} delay={res['delay_us']:.0f}us "
                      f"mode={res.get('mode')} lat={res.get('client_latency')}s "
                      f"l3={res.get('l3_loaded')}", flush=True)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[fluct] result -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
