#!/usr/bin/env python3
"""probe_v2.py - flush-mode L3 prefetch measurement (SuffixPrefetchV2)

Runs INSIDE the container against a local sglang server started with
--enable-hierarchical-cache. Protocol:
  wave1: send N distinct long prefixes sequentially (populates L1/L2/L3)
  flush: POST /flush_cache, sleep (clears L1+L2, keeps L3)
  wave2: resend the same prefixes (sequential or burst) -> triggers L3 prefetch

Collects per-request client latency + meta_info, then parses [PrefetchMeasure]
lines from the server log (visible because we run in the same container).

Usage:
  python -u probe_v2.py PORT [--n 8] [--prefix-len 8192] [--suffix-len 64]
      [--mode sequential|burst] [--burst 8] [--seed 1234]
      [--server-log /tmp/sp_server_v2.log] [--out /tmp/probe_v2_result.json]
"""

import argparse
import concurrent.futures as cf
import json
import random
import re
import time
import urllib.request

PAGE = 64
MEASURE_RE = re.compile(r"\[PrefetchMeasure\] (.+)$")
FLUSH_WAIT_S = 5


def post_json(url, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return body, time.perf_counter() - t0


def flush_cache(port, retries=10):
    """POST /flush_cache; 400 means scheduler busy -> retry with backoff."""
    url = f"http://127.0.0.1:{port}/flush_cache"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=b"", method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
            print(f"[probe] flush ok (attempt {attempt + 1})", flush=True)
            return True
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")[:120].replace("\n", " ")
            print(f"[probe] flush attempt {attempt + 1}: HTTP {e.code} {body}", flush=True)
            time.sleep(2)
        except Exception as e:
            print(f"[probe] flush attempt {attempt + 1}: {e}", flush=True)
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


def gen_prefixes(n, prefix_len, suffix_len, seed):
    """Deterministic distinct prefixes, page-aligned, with unique suffixes."""
    assert prefix_len % PAGE == 0
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        prefix = [rng.randrange(1000, 100000) for _ in range(prefix_len)]
        suffix = [rng.randrange(1000, 100000) for _ in range(suffix_len)]
        reqs.append({"idx": i, "input_ids": prefix + suffix})
    return reqs


def send_one(port, item):
    payload = {
        "input_ids": item["input_ids"],
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
    }
    t_submit = time.perf_counter()
    try:
        body, lat = post_json(f"http://127.0.0.1:{port}/generate", payload)
        mi = body.get("meta_info", {})
        return {
            "idx": item["idx"],
            "ok": True,
            "client_latency": round(lat, 4),
            "cached_tokens": mi.get("cached_tokens"),
            "cached_tokens_details": mi.get("cached_tokens_details"),
            "completion_tokens": mi.get("completion_tokens"),
        }
    except Exception as e:
        return {"idx": item["idx"], "ok": False, "error": str(e),
                "client_latency": round(time.perf_counter() - t_submit, 4)}


def log_line_count(path):
    try:
        with open(path, "r", errors="ignore") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def read_measures(path, baseline):
    """Parse [PrefetchMeasure] lines after `baseline` line number.
    All 8 TP ranks log the same line per request -> dedupe by rid."""
    out = []
    seen_rids = set()
    kv_re = re.compile(r"(\w+)=([\w.\-]+)")
    with open(path, "r", errors="ignore") as f:
        for ln, line in enumerate(f, 1):
            if ln <= baseline:
                continue
            m = MEASURE_RE.search(line)
            if not m:
                continue
            fields = dict(kv_re.findall(m.group(1)))
            rid = fields.get("rid")
            if rid in seen_rids:
                continue
            seen_rids.add(rid)
            for k in ("arrival_qlen", "device_hit", "host_hit", "prefetch_len", "l3_loaded", "pf_qdepth"):
                if k in fields:
                    fields[k] = int(fields[k])
            for k in ("prefetch_dur", "queue_dur", "prefetch_wait", "total_wait"):
                if k in fields:
                    fields[k] = float(fields[k])
            out.append(fields)
    return out


def stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {}
    n = len(xs)
    return {
        "n": n,
        "mean": round(sum(xs) / n, 4),
        "p50": xs[n // 2],
        "min": xs[0],
        "max": xs[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("--n", type=int, default=8, help="number of distinct prefixes")
    ap.add_argument("--prefix-len", type=int, default=8192)
    ap.add_argument("--suffix-len", type=int, default=64)
    ap.add_argument("--mode", choices=["sequential", "burst"], default="sequential")
    ap.add_argument("--burst", type=int, default=8, help="concurrency for burst mode")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--server-log", default="/tmp/sp_server_v2.log")
    ap.add_argument("--out", default="/tmp/probe_v2_result.json")
    ap.add_argument("--flush-wait", type=int, default=FLUSH_WAIT_S,
                    help="seconds to settle after flush (let write-back drain)")
    args = ap.parse_args()

    assert wait_healthy(args.port), "server not healthy"
    reqs = gen_prefixes(args.n, args.prefix_len, args.suffix_len, args.seed)
    total_tok = args.n * (args.prefix_len + args.suffix_len)
    print(f"[probe] n={args.n} prefix={args.prefix_len} suffix={args.suffix_len} "
          f"total_tokens={total_tok} mode={args.mode}", flush=True)

    # ---------- wave 1: populate ----------
    w1_base = log_line_count(args.server_log)
    t0 = time.perf_counter()
    wave1 = [send_one(args.port, r) for r in reqs]
    w1_wall = time.perf_counter() - t0
    print(f"[probe] wave1 done wall={w1_wall:.1f}s "
          f"ok={sum(1 for r in wave1 if r['ok'])}/{args.n}", flush=True)

    # ---------- flush L1+L2 ----------
    if not flush_cache(args.port):
        print("[probe] WARNING: flush_cache failed, wave2 may hit L1/L2", flush=True)
    print(f"[probe] flushed, sleeping {args.flush_wait}s", flush=True)
    time.sleep(args.flush_wait)

    # ---------- wave 2: trigger L3 prefetch ----------
    w2_base = log_line_count(args.server_log)
    t0 = time.perf_counter()
    if args.mode == "sequential":
        wave2 = [send_one(args.port, r) for r in reqs]
    else:
        with cf.ThreadPoolExecutor(max_workers=args.burst) as ex:
            wave2 = list(ex.map(lambda r: send_one(args.port, r), reqs))
    w2_wall = time.perf_counter() - t0
    time.sleep(3)  # let server flush log lines
    measures = read_measures(args.server_log, w2_base)
    print(f"[probe] wave2 done wall={w2_wall:.1f}s "
          f"ok={sum(1 for r in wave2 if r['ok'])}/{args.n} "
          f"measure_lines={len(measures)}", flush=True)

    result = {
        "args": vars(args),
        "wave1": {"wall_s": round(w1_wall, 2), "requests": wave1,
                  "latency": stats([r["client_latency"] for r in wave1 if r["ok"]])},
        "wave2": {"wall_s": round(w2_wall, 2), "requests": wave2,
                  "latency": stats([r["client_latency"] for r in wave2 if r["ok"]]),
                  "measures": measures,
                  "prefetch_dur": stats([m.get("prefetch_dur") for m in measures]),
                  "l3_loaded": stats([m.get("l3_loaded") for m in measures]),
                  "queue_dur": stats([m.get("queue_dur") for m in measures]),
                  "prefetch_wait": stats([m.get("prefetch_wait") for m in measures]),
                  "total_wait": stats([m.get("total_wait") for m in measures]),
                  "pf_qdepth": stats([m.get("pf_qdepth") for m in measures]),
                  "arrival_qlen": stats([m.get("arrival_qlen") for m in measures])},
        "log_baselines": {"wave1": w1_base, "wave2": w2_base},
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[probe] result -> {args.out}", flush=True)
    print("[probe] wave2 summary: "
          f"prefetch_dur={result['wave2']['prefetch_dur']} "
          f"l3_loaded={result['wave2']['l3_loaded']} "
          f"queue_dur={result['wave2']['queue_dur']}", flush=True)


if __name__ == "__main__":
    main()
