#!/usr/bin/env python3
# 2D sweep (L3 read-delay x request-rate) x 4 prefetch-stop policies, 8-GPU parallel.
# Each task: warm (delay=0, fill L3) -> restart keeping L3 -> measure (target delay).
# Finds the "sweet spot" where cost_aware beats both best_effort (gives up too early)
# and wait_complete (waits too long).
import os, sys, json, time, subprocess, itertools, queue, threading

MODEL = "/root/.cache/huggingface/qwen14b"
DELAYS = [int(x) for x in os.environ.get("SCAN_DELAYS", "0,2,4").split(",")]
RATES  = [int(x) for x in os.environ.get("SCAN_RATES", "4,8,12").split(",")]
POLICIES = os.environ.get("SCAN_POLICIES", "best_effort,wait_complete,timeout,cost_aware").split(",")
NGPU = int(os.environ.get("SCAN_NGPU", "8"))

# Compact shared-prefix load to keep each task short.
GSP_GROUPS, GSP_PER, SYS, QLEN, OUT = 6, 8, 4096, 128, 32
NUM = GSP_GROUPS * GSP_PER  # 48
MEASURE_TIMEOUT = int(os.environ.get("SCAN_MEASURE_TIMEOUT", "900"))  # crash guard
WARM_TIMEOUT = 600

RES = "/tmp/scan_results"; LOG = "/tmp/scan_logs"
os.makedirs(RES, exist_ok=True); os.makedirs(LOG, exist_ok=True)

def sh(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode

def start_server(gpu, port, hdir, delay, policy, logf):
    cmd = (f"CUDA_VISIBLE_DEVICES={gpu} "
           f"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR={hdir} "
           f"SGLANG_HICACHE_FILE_BACKEND_GET_DELAY_MS={delay} "
           f"python -m sglang.launch_server --model-path {MODEL} "
           f"--host 127.0.0.1 --port {port} --tp 1 --mem-fraction-static 0.75 "
           f"--enable-hierarchical-cache --hicache-ratio 1.5 --hicache-storage-backend file "
           f"--hicache-storage-prefetch-policy {policy} "
           f"--hicache-storage-backend-extra-config '{{\"prefetch_threshold\":32,\"cost_aware_gamma\":1.0}}' "
           f"--max-running-requests 32 --log-level info > {logf} 2>&1 &")
    subprocess.run(cmd, shell=True)

def wait_ready(port, tries=180):
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

def task(gpu, delay, rate, policy):
    port = 31000 + gpu
    tag = f"{policy}_d{delay}_r{rate}"
    hdir = f"/tmp/hicache_scan/{tag}"
    sh(f"rm -rf {hdir}"); os.makedirs(hdir, exist_ok=True)
    warm_out = f"{RES}/{tag}_warm.json"; meas_out = f"{RES}/{tag}_measure.json"
    sh(f"rm -f {warm_out} {meas_out}")
    slog = f"{LOG}/server_{tag}.log"

    # warm: delay=0 to fill L3 fast (write path unaffected by read delay)
    start_server(gpu, port, hdir, 0, policy, slog)
    if not wait_ready(port):
        stop_server(port); return (tag, "WARM_START_FAIL")
    bench(port, rate, warm_out, f"{LOG}/bench_{tag}_warm.log", WARM_TIMEOUT)
    stop_server(port)
    nfiles = len(os.listdir(hdir)) if os.path.isdir(hdir) else 0

    # measure: target delay, KEEP L3 (no wipe)
    start_server(gpu, port, hdir, delay, policy, slog)
    if not wait_ready(port):
        stop_server(port); return (tag, f"MEASURE_START_FAIL L3={nfiles}")
    rc = bench(port, rate, meas_out, f"{LOG}/bench_{tag}_measure.log", MEASURE_TIMEOUT)
    nz = int(subprocess.run(
        f"grep -oE 'completed with [1-9][0-9]* tokens' {slog} | wc -l",
        shell=True, capture_output=True, text=True).stdout.strip() or "0")
    stop_server(port)
    status = "TIMEOUT" if rc == -1 else "ok"
    return (tag, f"{status} L3={nfiles} nz={nz}")

def worker(gpu, task_q, results, lock):
    while True:
        try:
            d, r, p = task_q.get_nowait()
        except queue.Empty:
            return
        t0 = time.time()
        tag, st = task(gpu, d, r, p)
        with lock:
            results.append((tag, st, round(time.time()-t0)))
            print(f"[gpu{gpu}] {tag}: {st} ({round(time.time()-t0)}s)  "
                  f"[{len(results)}/{len(DELAYS)*len(RATES)*len(POLICIES)}]", flush=True)
        task_q.task_done()

def main():
    print(f"=== SCAN START {time.strftime('%F %T')} delays={DELAYS} rates={RATES} "
          f"policies={POLICIES} ngpu={NGPU} n={NUM} ===", flush=True)
    task_q = queue.Queue()
    for d, r, p in itertools.product(DELAYS, RATES, POLICIES):
        task_q.put((d, r, p))
    results = []; lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(g, task_q, results, lock))
               for g in range(NGPU)]
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"=== SCAN DONE {time.strftime('%F %T')} ===", flush=True)
    summarize()

def summarize():
    rows = []
    for d, r, p in itertools.product(DELAYS, RATES, POLICIES):
        tag = f"{p}_d{d}_r{r}"; f = f"{RES}/{tag}_measure.json"
        rec = {"policy": p, "delay": d, "rate": r,
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
    print("delay rate  policy          ttft_ms    e2e_ms  thrpt")
    for rec in sorted(rows, key=lambda x: (x["delay"], x["rate"], x["policy"])):
        print(f"{rec['delay']:>4} {rec['rate']:>4}  {rec['policy']:<15} "
              f"{str(rec['ttft']):>9} {str(rec['e2e']):>9} {str(rec['thrpt']):>6}")

if __name__ == "__main__":
    main()
