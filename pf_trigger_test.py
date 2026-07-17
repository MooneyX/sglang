#!/usr/bin/env python3
# Single-server (NO restart) L3-prefetch trigger test.
# Key idea (per user): keep the server running so the host radix tree persists.
#   phase 1 (fill): send MANY distinct long prefixes -> overflows a modest L2,
#                   LRU-evicts the earliest prefixes down to L3-only (files remain,
#                   tree node popped but hash still queryable via storage).
#   phase 2 (replay): WITHOUT restart, replay the EARLIEST prefixes. Their L2 is gone,
#                     so match_prefix falls back to an ancestor and prefetch_from_storage
#                     issues a real _storage_hit_query -> background L3->L2 prefetch fires.
# Success signal: _storage_hit_query calls > 0 AND "completed with N>0 tokens" (nz>0).
#
# Uses the OpenAI-compatible /generate endpoint directly with hand-built prompts so we
# fully control which prefix is replayed and in what order (bench_serving shuffles).
import os, sys, json, time, subprocess, urllib.request, threading

MODEL = os.environ.get("PF_MODEL", "/root/.cache/huggingface/qwen32b")
PORT = int(os.environ.get("PF_PORT", "31000"))
HICACHE_SIZE_GB = int(os.environ.get("PF_HICACHE_SIZE_GB", "8"))  # modest L2
MEM_FRAC = os.environ.get("PF_MEM_FRAC", "0.85")
POLICY = os.environ.get("PF_POLICY", "cost_aware")
MEDIUM_DIR = os.environ.get("PF_HDIR", "/dev/shm/hicache_pf")
# Each distinct prefix ~ PREFIX_TOKENS tokens; NUM_PREFIXES distinct ones.
PREFIX_TOKENS = int(os.environ.get("PF_PREFIX_TOKENS", "3000"))
NUM_PREFIXES = int(os.environ.get("PF_NUM_PREFIXES", "30"))
QUESTION = "\n\nQuestion: summarize the above in one sentence.\nAnswer:"
OUT_TOKENS = int(os.environ.get("PF_OUT", "16"))

LOG = "/tmp/pf_test"
os.makedirs(LOG, exist_ok=True)
SLOG = f"{LOG}/server.log"

def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, **kw)

def start_server():
    sh(f"rm -rf {MEDIUM_DIR}; mkdir -p {MEDIUM_DIR}")
    cmd = (f"CUDA_VISIBLE_DEVICES=0 "
           f"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR={MEDIUM_DIR} "
           f"SGLANG_HICACHE_FILE_BACKEND_PROFILE=1 "
           f"python -m sglang.launch_server --model-path {MODEL} "
           f"--host 127.0.0.1 --port {PORT} --tp 1 --mem-fraction-static {MEM_FRAC} "
           f"--enable-hierarchical-cache --hicache-size {HICACHE_SIZE_GB} "
           f"--hicache-storage-backend file "
           f"--hicache-storage-prefetch-policy {POLICY} "
           f"--hicache-storage-backend-extra-config '{{\"prefetch_threshold\":32,\"cost_aware_gamma\":1.0}}' "
           f"--max-running-requests 32 --log-level info > {SLOG} 2>&1 &")
    sh(cmd)

def wait_ready(tries=240):
    for _ in range(tries):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            return True
        except Exception:
            time.sleep(2)
    return False

def make_prefix(i):
    # Deterministic distinct long prefix. Repeat a unique sentence to reach ~PREFIX_TOKENS.
    sentence = f"This is document number {i}. It contains unique reference token {i}xyz. "
    # ~1 token per ~0.75 word; sentence ~ 15 tokens. repeat to reach target.
    reps = max(1, PREFIX_TOKENS // 15)
    return sentence * reps

def gen(prefix_text):
    body = json.dumps({
        "text": prefix_text + QUESTION,
        "sampling_params": {"max_new_tokens": OUT_TOKENS, "temperature": 0},
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        urllib.request.urlopen(req, timeout=300).read()
        return time.time() - t0
    except Exception as e:
        return -1

def count(pattern, f=SLOG):
    r = subprocess.run(f"grep -cE '{pattern}' {f} 2>/dev/null", shell=True,
                       capture_output=True, text=True)
    return int(r.stdout.strip() or "0")

def count_nz():
    r = subprocess.run(f"grep -oE 'completed with [1-9][0-9]* tokens' {SLOG} | wc -l",
                       shell=True, capture_output=True, text=True)
    return int(r.stdout.strip() or "0")

def snapshot(tag):
    hitq = count(r"__storage_hit_query__|storage_hit")
    nz = count_nz()
    l3 = subprocess.run(f"ls {MEDIUM_DIR} 2>/dev/null | wc -l", shell=True,
                        capture_output=True, text=True).stdout.strip()
    print(f"[{tag}] storage_hit_query_lines={hitq} nz(prefetch>0)={nz} L3_files={l3}", flush=True)

def main():
    print(f"=== PF TEST START {time.strftime('%F %T')} model={MODEL} "
          f"L2={HICACHE_SIZE_GB}GB prefixes={NUM_PREFIXES}x{PREFIX_TOKENS}tok policy={POLICY} ===", flush=True)
    start_server()
    if not wait_ready():
        print("SERVER_FAILED"); sh(f"pkill -9 -f 'port {PORT}'"); return
    print("server ready", flush=True)

    prefixes = [make_prefix(i) for i in range(NUM_PREFIXES)]

    # Phase 1 (fill): send each distinct prefix once, in order 0..N-1.
    # Early ones (0,1,2,...) get evicted from L2 as later ones fill it, but persist in L3.
    print("--- phase1: fill (write distinct prefixes to L2/L3) ---", flush=True)
    t0 = time.time()
    for i, p in enumerate(prefixes):
        dt = gen(p)
        if i % 10 == 0:
            print(f"  fill {i}/{NUM_PREFIXES} dt={dt:.1f}s", flush=True)
    print(f"phase1 done in {time.time()-t0:.0f}s", flush=True)
    snapshot("after_fill")

    # brief pause to let write_through flush to L3
    time.sleep(10)
    snapshot("after_flush")

    # Phase 2 (replay): WITHOUT restart, replay the EARLIEST prefixes (0..k), whose L2
    # should now be evicted -> must be prefetched back from L3.
    print("--- phase2: replay earliest prefixes (L2 evicted -> force L3 prefetch) ---", flush=True)
    k = min(10, NUM_PREFIXES)
    hitq_before = count(r"__storage_hit_query__|storage_hit")
    nz_before = count_nz()
    t0 = time.time()
    for i in range(k):
        dt = gen(prefixes[i])
        print(f"  replay prefix#{i} dt={dt:.1f}s", flush=True)
    print(f"phase2 done in {time.time()-t0:.0f}s", flush=True)
    snapshot("after_replay")
    print(f"DELTA storage_hit_query_lines=+{count(r'__storage_hit_query__|storage_hit')-hitq_before} "
          f"nz=+{count_nz()-nz_before}", flush=True)

    sh(f"pkill -9 -f 'port {PORT}'")
    print(f"=== PF TEST DONE {time.strftime('%F %T')} ===", flush=True)

if __name__ == "__main__":
    main()
