# -*- coding: utf-8 -*-
"""Fetch long multi-turn conversations from UltraChat200k via datasets-server rows API."""
import json, time, urllib.request, sys

OUT = r"D:\Documents\SuffixPrefetchV2\offline_experiments\realdata\ultrachat_subset.json"
BASE = ("https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4%2Fultrachat_200k"
        "&config=default&split=train_sft&offset={off}&length=100")

def text_len(msgs):
    return sum(len(m.get("content", "")) for m in msgs)

kept = []
scanned = 0
for off in range(0, 6000, 100):
    url = BASE.format(off=off)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                d = json.loads(r.read())
            break
        except Exception as e:
            print(f"off={off} retry{attempt}: {str(e)[:60]}", flush=True)
            time.sleep(3)
    else:
        continue
    rows = d.get("rows", [])
    scanned += len(rows)
    for row in rows:
        msgs = row["row"].get("messages", [])
        tl = text_len(msgs)
        if tl >= 60000 and len(msgs) >= 4:  # ~15K+ tokens, multi-turn
            kept.append({"id": row["row"].get("id", f"uc{off}"), "messages": msgs, "chars": tl})
    if off % 500 == 0:
        print(f"off={off} scanned={scanned} kept={len(kept)}", flush=True)
    if len(kept) >= 80:
        break
    time.sleep(0.3)

json.dump(kept, open(OUT, "w"))
print(f"DONE scanned={scanned} kept={len(kept)} -> {OUT}")
lens = sorted(k["chars"] for k in kept)
if lens:
    print("chars min/med/max:", lens[0], lens[len(lens)//2], lens[-1])
