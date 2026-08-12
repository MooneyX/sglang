#!/usr/bin/env python3
"""prep_realdata.py - build real-text prefix buckets for realdata_test.py.

Runs INSIDE the container. Tokenizes real texts with the DeepSeek-V3 tokenizer
and builds (prefix, suffix) pairs per length bucket:

  ShareGPT multi-turn   -> 8K, 16K   (history turns = prefix, next turn = suffix)
  LongBench doc QA      -> 8K, 16K, 32K (document = prefix, question = suffix)
  Gutenberg books       -> 48K, 64K  (disjoint chapters = prefix, next chunk = suffix)

All prefix lengths are page-aligned (multiples of 64). Output: realdata_buckets.json
  { "8192": [{"idx":0, "source":"sharegpt#12", "input_ids":[...]}], ... }

Usage: python -u prep_realdata.py [--n 6] [--out /sgl-workspace/sglang/sp_v2/realdata/realdata_buckets.json]
"""
import argparse, json, os, random

PAGE = 64
SUFFIX = 64
DATA = os.environ.get("REALDATA_DIR", "/tmp/realdata")
MODEL = "/data1/models/DeepSeek-V3"


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def enc(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def sharegpt_items(tok, buckets):
    """Multi-turn conversations (ShareGPT and/or UltraChat) -> 8K/16K buckets."""
    convs = []
    for fn, key, valkey in (("sharegpt_subset.json", "conversations", "value"),
                            ("ultrachat_subset.json", "messages", "content")):
        path = os.path.join(DATA, fn)
        if not os.path.exists(path):
            continue
        for conv in json.load(open(path)):
            turns = [t.get(valkey, "") for t in conv.get(key, [])]
            if len(turns) >= 2:
                convs.append((fn.split("_")[0], turns))
    if not convs:
        print("[prep] no conversation data, skip")
        return
    out = {b: [] for b in buckets}
    for ci, (src, turns) in enumerate(convs):
        if len(turns) < 2:
            continue
        # accumulate turns; for each bucket, if the running text covers it, emit
        ids = enc(tok, "\n".join(turns))
        for b in buckets:
            need = b + SUFFIX
            if len(ids) >= need and len(out[b]) < 999:
                out[b].append((f"{src}#{ci}", ids[:b], ids[b:b + SUFFIX]))
    return out


def longbench_items(tok, buckets):
    """LongBench doc+question -> 8K/16K/32K buckets."""
    out = {b: [] for b in buckets}
    for fn in os.listdir(DATA):
        if not fn.startswith("longbench_"):
            continue
        task = fn[len("longbench_"):-len(".jsonl")]
        for li, line in enumerate(open(os.path.join(DATA, fn), errors="ignore")):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            ctx = row.get("context", "")
            q = row.get("input", "")
            if not ctx:
                continue
            ids = enc(tok, ctx)
            qids = enc(tok, "\n\nQuestion: " + q)[:SUFFIX]
            qids = (qids + [0] * SUFFIX)[:SUFFIX]
            for b in buckets:
                if len(ids) >= b and len(out[b]) < 999:
                    out[b].append((f"{task}#{li}", ids[:b], qids))
    return out


def gutenberg_items(tok, buckets):
    """Book slices -> 48K/64K buckets, disjoint slices from different offsets."""
    out = {b: [] for b in buckets}
    for fn in os.listdir(DATA):
        if not fn.startswith("gutenberg_"):
            continue
        book = fn[len("gutenberg_"):-len(".txt")]
        text = open(os.path.join(DATA, fn), errors="ignore").read()
        ids = enc(tok, text)
        print(f"[prep] {book}: {len(ids)} tokens")
        for b in buckets:
            need = b + SUFFIX
            # disjoint slices evenly spaced across the book
            usable = len(ids) - need
            if usable <= 0:
                continue
            for k in range(12):
                off = int(usable * k / 12)
                out[b].append((f"{book}#s{k}", ids[off:off + b], ids[off + b:off + b + SUFFIX]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="prefixes per bucket")
    ap.add_argument("--out", default=os.path.join(DATA, "realdata_buckets.json"))
    args = ap.parse_args()

    tok = load_tokenizer()
    rng = random.Random(42)
    buckets = {}

    sg = sharegpt_items(tok, (8192, 16384))
    lb = longbench_items(tok, (8192, 16384, 32768))
    gb = gutenberg_items(tok, (49152, 65536))

    for b in (8192, 16384):
        pool = (sg or {}).get(b, []) + (lb or {}).get(b, [])
        rng.shuffle(pool)
        buckets[str(b)] = [{"idx": i, "source": s, "input_ids": p + sfx}
                           for i, (s, p, sfx) in enumerate(pool[:args.n])]
    pool = (lb or {}).get(32768, [])
    rng.shuffle(pool)
    buckets["32768"] = [{"idx": i, "source": s, "input_ids": p + sfx}
                        for i, (s, p, sfx) in enumerate(pool[:args.n])]
    for b in (49152, 65536):
        pool = (gb or {}).get(b, [])
        rng.shuffle(pool)
        buckets[str(b)] = [{"idx": i, "source": s, "input_ids": p + sfx}
                           for i, (s, p, sfx) in enumerate(pool[:args.n])]

    for b, items in buckets.items():
        lens = [len(x["input_ids"]) for x in items]
        srcs = [x["source"].split("#")[0] for x in items]
        print(f"[prep] bucket {b}: n={len(items)} len={lens[0] if lens else 0} sources={sorted(set(srcs))}")
    json.dump(buckets, open(args.out, "w"))
    print(f"[prep] -> {args.out}")


if __name__ == "__main__":
    main()
