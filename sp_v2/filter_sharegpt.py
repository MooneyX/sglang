#!/usr/bin/env python3
"""Filter long conversations from sharegpt_full.json -> sharegpt_subset.json."""
import json, os

DATA = os.environ.get("REALDATA_DIR", "/tmp/realdata")
src = f"{DATA}/sharegpt_full.json"
dst = f"{DATA}/sharegpt_subset.json"

convs = json.load(open(src))
print("total conversations:", len(convs))
kept = []
for c in convs:
    turns = c.get("conversations", [])
    chars = sum(len(t.get("value", "")) for t in turns)
    if chars >= 30000 and len(turns) >= 4:  # ~8K+ tokens, multi-turn
        kept.append(c)
kept.sort(key=lambda c: -sum(len(t.get("value", "")) for t in c["conversations"]))
kept = kept[:200]
json.dump(kept, open(dst, "w"))
print("kept:", len(kept))
lens = sorted(sum(len(t.get("value", "")) for t in c["conversations"]) for c in kept)
if lens:
    print("chars min/med/max:", lens[0], lens[len(lens) // 2], lens[-1])
