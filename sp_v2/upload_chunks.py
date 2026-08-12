# -*- coding: utf-8 -*-
"""Chunked base64 uploader for large files -> container path via ioa-ssh-cli exec."""
import base64, hashlib, subprocess, sys, time

HOST = "29.209.104.16"
PORT = "56000"
CHUNK = 400 * 1024  # raw bytes per exec

def sh(cmd, timeout=120):
    r = subprocess.run(["ioa-ssh-cli", "exec", HOST, "--remote-port", PORT,
                        "--timeout", str(timeout), cmd, "--output", "json"],
                       capture_output=True, text=True)
    import json
    d = json.loads(r.stdout)
    if not d.get("ok"):
        raise RuntimeError(str(d)[:300])
    return d["data"]["stdout"]

def upload(local, remote_in_container):
    raw = open(local, "rb").read()
    b64 = base64.b64encode(raw).decode()
    md5 = hashlib.md5(raw).hexdigest()
    n = (len(b64) + CHUNK - 1) // CHUNK
    t0 = time.time()
    for i in range(n):
        part = b64[i * CHUNK:(i + 1) * CHUNK]
        op = ">" if i == 0 else ">>"
        # decode on the CONTAINER side via docker exec
        sh(f"sudo docker exec sglang-SuffixPrefetchV2 bash -lc 'echo {part} | base64 -d {op} {remote_in_container}'")
        print(f"  chunk {i+1}/{n}", flush=True)
    remote_md5 = sh(f"sudo docker exec sglang-SuffixPrefetchV2 bash -lc 'md5sum {remote_in_container}'").split()[0]
    ok = remote_md5 == md5
    print(f"{local} -> {remote_in_container}: {'OK' if ok else 'MD5 MISMATCH'} ({len(raw)} bytes, {time.time()-t0:.0f}s)")
    return ok

if __name__ == "__main__":
    pairs = eval(sys.argv[1])  # [(local, remote), ...]
    results = [upload(l, r) for l, r in pairs]
    sys.exit(0 if all(results) else 1)
