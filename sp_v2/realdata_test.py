#!/usr/bin/env python3
"""realdata_test.py - real-text prefix-cache TTFT measurement.

Runs INSIDE the container against a local sglang server. Mirrors fluct_test.py
protocol but with per-bucket write->measure cycles (mooncake 64gb pool holds
one bucket at a time):

  for each bucket L (ascending):
    write:   send the n prefixes once (cold recompute reference), settle
    measure: rounds x n requests; flush_cache before EACH request so it must
             go through L3 prefetch. Record client latency, cached_tokens,
             and the [PrefetchMeasure] fields from the server log.

Usage:
  python -u realdata_test.py PORT [--rounds 2] [--settle 30]
      [--buckets 8192,16384,32768,49152,65536] [--out /tmp/real_x.json]
"""
import argparse, json, re, time, urllib.request

MEASURE_RE = re.compile(r"\[PrefetchMeasure\] (.+)$")
DATA = "/sgl-workspace/sglang/sp_v2/realdata"


def post_json(url, payload, timeout=900):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return body, time.perf_counter() - t0


def flush_cache(port, retries=10):
    url = f"http://127.0.0.1:{port}/flush_cache"
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, data=b"", method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
            return True
        except Exception:
            time.sleep(2)
    return False


def wait_healthy(port, timeout_s=1800):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(10)
    return False


def send_one(port, item, L, rnd):
    payload = {"input_ids": item["input_ids"],
               "sampling_params": {"max_new_tokens": 1, "temperature": 0}}
    try:
        body, lat = post_json(f"http://127.0.0.1:{port}/generate", payload)
        mi = body.get("meta_info", {})
        return {"L": L, "round": rnd, "idx": item["idx"], "source": item["source"],
                "ok": True, "client_latency": round(lat, 4),
                "cached_tokens": mi.get("cached_tokens")}
    except Exception as e:
        return {"L": L, "round": rnd, "idx": item["idx"], "source": item["source"],
                "ok": False, "error": str(e)[:200]}


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
            out.append(fields)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--settle", type=int, default=30)
    ap.add_argument("--buckets", default="8192,16384,32768,49152,65536")
    ap.add_argument("--buckets-file", default=f"{DATA}/realdata_buckets.json")
    ap.add_argument("--server-log", default="/tmp/sp_server_big.log")
    ap.add_argument("--out", default="/tmp/real_result.json")
    args = ap.parse_args()

    assert wait_healthy(args.port), "server not healthy"
    all_buckets = json.load(open(args.buckets_file))
    lens = [int(x) for x in args.buckets.split(",")]
    result = {"args": vars(args), "write": [], "measure": []}

    for L in lens:
        items = all_buckets[str(L)]
        print(f"[real] L={L}: writing {len(items)} prefixes ...", flush=True)
        for r in items:
            res = send_one(args.port, r, L, -1)
            result["write"].append(res)
            print(f"[real]   write {r['source']} lat={res.get('client_latency')}s", flush=True)
        time.sleep(args.settle)

        for rnd in range(args.rounds):
            for r in items:
                if not flush_cache(args.port):
                    print("[real] WARNING: flush failed", flush=True)
                base = log_line_count(args.server_log)
                res = send_one(args.port, r, L, rnd)
                time.sleep(1)
                ms = read_measures(args.server_log, base)
                if ms:
                    res["mode"] = ms[-1].get("mode")
                    res["l3_loaded"] = ms[-1].get("l3_loaded")
                    res["prefetch_dur"] = ms[-1].get("prefetch_dur")
                result["measure"].append(res)
                print(f"[real]   L={L} r={rnd} {r['source']} mode={res.get('mode')} "
                      f"lat={res.get('client_latency')}s l3={res.get('l3_loaded')}", flush=True)
        json.dump(result, open(args.out, "w"), indent=1)  # checkpoint per bucket

    json.dump(result, open(args.out, "w"), indent=1)
    print(f"[real] result -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
