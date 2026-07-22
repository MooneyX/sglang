"""
SuffixPrefetch S2 集成测试 — 真实模型上「前缀重算 vs 前缀复用」等价性
=====================================================================

目标：在真实 sglang 全模型（不止 attention 层）上验证 SuffixPrefetch 的地基前提：
  同一段前缀，无论「这次现算 KV」还是「从缓存复用 KV」，
  产出的下游 token 序列与 logprobs 数值一致。

这是整个 SuffixPrefetch 敢于「把前缀改成重算」的正确性依据的真机版
（阶段0已在纯 causal attention 层证过，S2 升级到端到端真实模型）。

方法（黑盒，用现成 radix cache，无需 hicache）：
  路径A（重算）：先 flush_cache 清空缓存 → 发请求，前缀被完整重新计算。
  路径B（复用）：立即再发相同前缀的请求 → 前缀命中 radix cache，KV 被复用。
  两次都用 temperature=0（贪心）+ return_logprob，对比：
    (1) 输出 token id 序列完全一致；
    (2) 每步 output logprob 数值一致（容差）。

若一致 → 真机上「前缀重算」与「前缀复用」等价，SuffixPrefetch 地基成立。
"""

import urllib.request
import json
import time

BASE = "http://127.0.0.1:31000"
ATOL = 1e-2  # logprob 容差（bf16 + 不同 batch 形状下的可接受浮点差）


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req).read())


def flush():
    try:
        urllib.request.urlopen(BASE + "/flush_cache", timeout=10).read()
    except Exception as e:
        print("flush warn:", e)
    time.sleep(1.5)


def gen(text):
    return post(
        "/generate",
        {
            "text": text,
            "sampling_params": {"max_new_tokens": 24, "temperature": 0},
            "return_logprob": True,
            "logprob_start_len": 0,
        },
    )


def out_tokens(r):
    # output token ids
    return r["meta_info"].get("output_token_ids") or [
        tid for tid, _ in _pairs(r["meta_info"].get("output_token_logprobs", []))
    ]


def _pairs(lst):
    # sglang output_token_logprobs entries look like [logprob, token_id, ...]
    res = []
    for e in lst:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            res.append((e[1], e[0]))
    return res


def out_logprobs(r):
    return [e[0] for e in r["meta_info"].get("output_token_logprobs", []) if isinstance(e, (list, tuple))]


# 用足够长的共享前缀（>=数百 token，跨多个 page），后接不同问题。
# radix cache 复用的是"已提交的前缀"，所以让 SHARED 作为公共前缀，
# 两个请求 SHARED+Q1 / SHARED+Q2 的公共部分 SHARED 会被第二个请求命中复用。
SHARED = (
    "In the field of machine learning, attention mechanisms have become "
    + ("a fundamental building block for modern neural architectures. " * 30)
)
Q = " Question: summarize the key idea above in exactly one clear sentence. Answer:"
PROMPT = SHARED + Q

print("=== 路径A: flush 缓存后，前缀完全重算 ===")
flush()
rA = gen(PROMPT)
tokA = [e[1] for e in rA["meta_info"]["output_token_logprobs"]]
lpA = out_logprobs(rA)
print("A output tokens:", tokA[:12], "...")
print("A prompt_tokens:", rA["meta_info"]["prompt_tokens"], "cached:", rA["meta_info"].get("cached_tokens"))

# 路径B: 先用一个"种子"请求把 SHARED 前缀写入 radix cache（不同后缀），
# 再发 PROMPT，使其 SHARED 部分命中缓存被复用。
print("=== 路径B: 先种入前缀缓存，再复用 ===")
flush()
_seed = gen(SHARED + " Seed different tail to commit the shared prefix into cache.")
time.sleep(1.0)
rB = gen(PROMPT)
tokB = [e[1] for e in rB["meta_info"]["output_token_logprobs"]]
lpB = out_logprobs(rB)
print("B output tokens:", tokB[:12], "...")
print("B prompt_tokens:", rB["meta_info"]["prompt_tokens"], "cached:", rB["meta_info"].get("cached_tokens"))

print("\n=== 对比 ===")
tok_match = tokA == tokB
print(f"[{'PASS' if tok_match else 'FAIL'}] 输出 token 序列一致: A={len(tokA)}tok B={len(tokB)}tok")

max_lp_diff = 0.0
if len(lpA) == len(lpB) and lpA:
    max_lp_diff = max(abs(a - b) for a, b in zip(lpA, lpB))
lp_match = len(lpA) == len(lpB) and max_lp_diff < ATOL
print(f"[{'PASS' if lp_match else 'FAIL'}] output logprobs 一致: 最大差={max_lp_diff:.2e} (容差{ATOL})")

# 关键验证：B 确实复用了缓存（cached>0），A 基本没复用
reuse_ok = (rB["meta_info"].get("cached_tokens") or 0) > (rA["meta_info"].get("cached_tokens") or 0)
print(f"[{'PASS' if reuse_ok else 'WARN'}] B 复用缓存 > A: A_cached={rA['meta_info'].get('cached_tokens')} B_cached={rB['meta_info'].get('cached_tokens')}")

print("\n==== S2 结论 ====")
if tok_match and lp_match:
    print("真机上『前缀重算』与『前缀复用』输出等价 ✅ — SuffixPrefetch 地基成立")
else:
    print("存在不一致 ❌ — 需进一步排查")
