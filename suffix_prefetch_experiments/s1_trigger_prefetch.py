import urllib.request, json, time

BASE = "http://127.0.0.1:31000"


def gen(text, max_new=8):
    data = json.dumps(
        {"text": text, "sampling_params": {"max_new_tokens": max_new, "temperature": 0}}
    ).encode()
    req = urllib.request.Request(
        BASE + "/generate", data=data, headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req).read())


def health():
    try:
        urllib.request.urlopen(BASE + "/health", timeout=3)
        return True
    except Exception:
        return False


# unique long prefixes to fill and churn the host cache
def make_prefix(tag):
    return (f"Document {tag}: " + ("alpha beta gamma delta epsilon zeta eta theta. " * 30))


print("health:", health())

# Phase 1: write many distinct long prefixes (each >=256 tokens) -> write_through to L3,
# and their sheer number should evict earlier ones out of host memory.
tags = [f"T{i:02d}" for i in range(12)]
for t in tags:
    r = gen(make_prefix(t) + f" unique question for {t}?")
    print(f"WRITE {t} prompt_tokens={r['meta_info']['prompt_tokens']} cached={r['meta_info'].get('cached_tokens')}")

time.sleep(4)  # let write_through settle

# Phase 2: re-request the EARLIEST prefixes (likely evicted from host) with new suffixes
# -> should miss host, hit L3 storage, trigger real prefetch_from_storage with hits.
for t in tags[:6]:
    r = gen(make_prefix(t) + f" a totally different follow-up for {t}!")
    print(f"REHIT {t} prompt_tokens={r['meta_info']['prompt_tokens']} cached={r['meta_info'].get('cached_tokens')}")
    time.sleep(0.5)

print("done")
