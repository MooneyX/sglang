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


def gen_ids(ids):
    # 直接用 input_ids 传入，绕开分词/模板差异，保证两次前缀 token 完全一致
    return post(
        "/generate",
        {
            "input_ids": ids,
            "sampling_params": {"max_new_tokens": 24, "temperature": 0},
            "return_logprob": True,
            "logprob_start_len": 0,
        },
    )


# 构造确定性 token 序列：共享前缀 SHARED_IDS（384 token，6×page64）+ 不同后缀。
# 用词表内安全 id（qwen 词表很大，用小范围可打印 token id 拼接）。
import itertools

base_cycle = [785, 3974, 9887, 21296, 5867, 1052, 1207, 8412]  # 任意固定 token id
SHARED_IDS = list(itertools.islice(itertools.cycle(base_cycle), 384))
Q_IDS = [40, 1128, 279, 1376, 4522, 30]      # 问题后缀
SEED_TAIL = [7985, 264, 2155, 9789, 13]       # 种子用的不同后缀
PROMPT_IDS = SHARED_IDS + Q_IDS

print("=== 路径A: flush 缓存后，前缀完全重算 ===")
flush()
rA = gen_ids(PROMPT_IDS)
tokA = [e[1] for e in rA["meta_info"]["output_token_logprobs"]]
lpA = out_logprobs(rA)
print("A output tokens:", tokA[:12], "...")
print("A prompt_tokens:", rA["meta_info"]["prompt_tokens"], "cached:", rA["meta_info"].get("cached_tokens"))

print("=== 路径B: 先种入 SHARED 前缀缓存，再复用 ===")
flush()
_seed = gen_ids(SHARED_IDS + SEED_TAIL)  # 种子请求：把 SHARED 前缀写入 radix
time.sleep(1.0)
rB = gen_ids(PROMPT_IDS)                  # SHARED 部分应命中缓存
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
