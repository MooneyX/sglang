#!/usr/bin/env python3
# Real-medium sweep: (L3 medium: shm/disk) x request-rate x 4 prefetch-stop policies.
# NO artificial sleep -- L3 read speed comes from the real underlying medium:
#   - shm  : /dev/shm  (tmpfs, in-memory, fast tier)
#   - disk : /tmp/hicache_real (overlay/disk, slow tier)
# Real per-page transfer time is measured via SGLANG_HICACHE_FILE_BACKEND_PROFILE=1
# (logged as "observed L3 read bandwidth: X GB/s").
# Model defaults to Qwen2.5-32B to raise per-token prefill cost (t_save).
import os, sys, json, time, subprocess, itertools, queue, threading

MODEL = os.environ.get("SCAN_MODEL", "/root/.cache/huggingface/qwen32b")
# medium name -> base dir on that filesystem
MEDIA = {
    "shm":  "/dev/shm/hicache_real",
    "disk": "/tmp/hicache_real",
}
MEDIA_LIST = os.environ.get("SCAN_MEDIA", "shm,disk").split(",")
RATES  = [int(x) for x in os.environ.get("SCAN_RATES", "4,8,12").split(",")]
POLICIES = os.environ.get("SCAN_POLICIES", "best_effort,wait_complete,timeout,cost_aware").split(",")
NGPU = int(os.environ.get("SCAN_NGPU", "4"))
HICACHE_RATIO = os.environ.get("SCAN_HICACHE_RATIO", "1.2")
HICACHE_SIZE_GB = float(os.environ.get("SCAN_HICACHE_SIZE_GB", "0"))  # >0 overrides ratio; small -> force L3 eviction
MEM_FRAC = os.environ.get("SCAN_MEM_FRAC", "0.85")   # 32B weights ~62GB; leave room for KV

# Longer shared prefix -> heavier prefill (bigger t_save). Keep prompt count modest
# because 32B + long prefix is memory/compute heavy.
GSP_GROUPS = int(os.environ.get("SCAN_GSP_GROUPS", "4"))
GSP_PER    = int(os.environ.get("SCAN_GSP_PER", "8"))
SYS        = int(os.environ.get("SCAN_SYS", "8192"))
QLEN, OUT  = 128, 32
NUM = GSP_GROUPS * GSP_PER
MEASURE_TIMEOUT = int(os.environ.get("SCAN_MEASURE_TIMEOUT", "1200"))
WARM_TIMEOUT = int(os.environ.get("SCAN_WARM_TIMEOUT", "900"))

RES = "/tmp/scan_real_results"; LOG = "/tmp/scan_real_logs"
os.makedirs(RES, exist_ok=True); os.makedirs(LOG, exist_ok=True)

def sh(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode

def start_server(gpu, port, hdir, policy, logf):
    # NO GET_DELAY -- real medium speed only. PROFILE=1 to log observed bandwidth.
    # hicache-size (GB, absolute) overrides ratio when >0. A SMALL host pool forces
    # prefixes to be evicted from L2 down to L3-only, so the request must rely on the
    # background prefetch to bring KV back from L3 -> this is what actually exercises
    # the prefetch-stop policies (nz>0).
    if HICACHE_SIZE_GB > 0:
        cap = f"--hicache-size {int(HICACHE_SIZE_GB)}"
    else:
        cap = f"--hicache-ratio {HICACHE_RATIO}"
    cmd = (f"CUDA_VISIBLE_DEVICES={gpu} "
           f"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR={hdir} "
           f"SGLANG_HICACHE_FILE_BACKEND_PROFILE=1 "
           f"python -m sglang.launch_server --model-path {MODEL} "
           f"--host 127.0.0.1 --port {port} --tp 1 --mem-fraction-static {MEM_FRAC} "
           f"--enable-hierarchical-cache {cap} --hicache-storage-backend file "
           f"--hicache-storage-prefetch-policy {policy} "
           f"--hicache-storage-backend-extra-config '{{\"prefetch_threshold\":32,\"cost_aware_gamma\":1.0}}' "
           f"--max-running-requests 32 --log-level info > {logf} 2>&1 &")
    subprocess.run(cmd, shell=True)

def wait_ready(port, tries=240):
    for _ in range(tries):
        if sh(f"curl -sf http://127.0.0.1:{port}/health") == 0:
            return True
        time.sleep(2)
    return False

def stop_server(port):
    sh(f"pkill -9 -f 'port {port}'")
    time.sleep(3)

def bench(port, rate, outfile, logf, timeout):
    cmd = (f"python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port {port} "
           f"--dataset-name generated-shared-prefix --gsp-num-groups {GSP_GROUPS} "
           f"--gsp-prompts-per-group {GSP_PER} --gsp-system-prompt-len {SYS} "
           f"--gsp-question-len {QLEN} --gsp-output-len {OUT} --num-prompts {NUM} "
           f"--request-rate {rate} --output-file {outfile} > {logf} 2>&1")
    try:
        subprocess.run(cmd, shell=True, timeout=timeout)
        return 0
    except subprocess.TimeoutExpired:
        return -1

def start_and_wait(gpu, port, hdir, policy, slog, tries=2):
    for attempt in range(tries):
        stop_server(port)
        start_server(gpu, port, hdir, policy, slog)
        if wait_ready(port):
            return True
        time.sleep(5 + gpu * 3)
    return False

def observed_bw(slog):
    # last "observed L3 read bandwidth: X GB/s" line
    out = subprocess.run(
        f"grep -oE 'observed L3 read bandwidth: [0-9.]+ GB/s' {slog} | tail -n1",
        shell=True, capture_output=True, text=True).stdout.strip()
    return out or "NA"

def task(gpu, medium, rate, policy):
    port = 31000 + gpu
    tag = f"{policy}_{medium}_r{rate}"
    hdir = f"{MEDIA[medium]}/{tag}"
    sh(f"rm -rf {hdir}"); os.makedirs(hdir, exist_ok=True)
    warm_out = f"{RES}/{tag}_warm.json"; meas_out = f"{RES}/{tag}_measure.json"
    sh(f"rm -f {warm_out} {meas_out}")
    slog = f"{LOG}/server_{tag}.log"

    # warm: fill L3 on the target medium
    if not start_and_wait(gpu, port, hdir, policy, slog):
        stop_server(port); return (tag, "WARM_START_FAIL")
    bench(port, rate, warm_out, f"{LOG}/bench_{tag}_warm.log", WARM_TIMEOUT)
    stop_server(port)
    nfiles = len(os.listdir(hdir)) if os.path.isdir(hdir) else 0

    # measure: KEEP L3 (no wipe), same real medium
    if not start_and_wait(gpu, port, hdir, policy, slog):
        stop_server(port); return (tag, f"MEASURE_START_FAIL L3={nfiles}")
    rc = bench(port, rate, meas_out, f"{LOG}/bench_{tag}_measure.log", MEASURE_TIMEOUT)
    nz = int(subprocess.run(
        f"grep -oE 'completed with [1-9][0-9]* tokens' {slog} | wc -l",
        shell=True, capture_output=True, text=True).stdout.strip() or "0")
    bw = observed_bw(slog)
    stop_server(port)
    status = "TIMEOUT" if rc == -1 else "ok"
    return (tag, f"{status} L3={nfiles} nz={nz} bw={bw}")

def worker(gpu, task_q, results, lock):
    time.sleep(gpu * 25)  # stagger heavy 32B server startup
    while True:
        try:
            m, r, p = task_q.get_nowait()
        except queue.Empty:
            return
        t0 = time.time()
        tag, st = task(gpu, m, r, p)
        with lock:
            results.append((tag, st, round(time.time()-t0)))
            print(f"[gpu{gpu}] {tag}: {st} ({round(time.time()-t0)}s)  "
                  f"[{len(results)}/{len(MEDIA_LIST)*len(RATES)*len(POLICIES)}]", flush=True)
        task_q.task_done()

def main():
    print(f"=== SCAN_REAL START {time.strftime('%F %T')} model={MODEL} media={MEDIA_LIST} "
          f"rates={RATES} policies={POLICIES} ngpu={NGPU} sys={SYS} n={NUM} ===", flush=True)
    task_q = queue.Queue()
    for m, r, p in itertools.product(MEDIA_LIST, RATES, POLICIES):
        task_q.put((m, r, p))
    results = []; lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(g, task_q, results, lock))
               for g in range(NGPU)]
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"=== SCAN_REAL DONE {time.strftime('%F %T')} ===", flush=True)
    summarize()

def summarize():
    rows = []
    for m, r, p in itertools.product(MEDIA_LIST, RATES, POLICIES):
        tag = f"{p}_{m}_r{r}"; f = f"{RES}/{tag}_measure.json"
        rec = {"policy": p, "medium": m, "rate": r,
               "ttft": None, "e2e": None, "thrpt": None}
        try:
            L = [x for x in open(f) if x.strip()]
            j = json.loads(L[-1])
            rec["ttft"] = round(j["mean_ttft_ms"], 1)
            rec["e2e"] = round(j["mean_e2e_latency_ms"], 1)
            rec["thrpt"] = round(j["request_throughput"], 2)
        except Exception:
            pass
        rows.append(rec)
    json.dump(rows, open(f"{RES}/summary.json", "w"), indent=2)
    print("\n=== SUMMARY (measure) ===")
    print("medium rate  policy          ttft_ms    e2e_ms  thrpt")
    for rec in sorted(rows, key=lambda x: (x["medium"], x["rate"], x["policy"])):
        print(f"{rec['medium']:>5} {rec['rate']:>4}  {rec['policy']:<15} "
              f"{str(rec['ttft']):>9} {str(rec['e2e']):>9} {str(rec['thrpt']):>6}")

if __name__ == "__main__":
    main()
