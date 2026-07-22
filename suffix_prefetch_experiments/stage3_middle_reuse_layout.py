"""
SuffixPrefetch Stage3 离线验证 — 中间段复用的 req_to_token 填表策略
=====================================================================

背景：sglang prefill 靠 req_to_token[req_idx, pos] = kv_slot 间接寻址，
attention 对每个 query 位置 p，用 req_to_token[req_idx, 0:p+1] 收集 K/V。
现有 write_cache_indices 假设「复用=连续头部 [0,prefix_len)」。

SuffixPrefetch 要的是「中间段复用」：
  - 复用 [x*, N) 的 KV（从缓存/L3 load 到 device 槽位）
  - 重算 [0, x*) 和 [N, seq) 的 KV（GPU 现算，写新槽位）
本测试离线复刻这套填表 + 位置生成，用参考 causal attention 验证：
  「中间复用布局」的逐层输出 == 「全量重算」baseline（数值等价）。

这是把「中间复用到底怎么填 req_to_token 表 + 怎么给 position」在离线钉死，
作为改真实 write_cache_indices / forward_batch_info 的可信依据。

判据：每层每位置 hidden 的最大绝对误差 < 1e-5。
"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)

# ---- 配置 ----
SEQ = 40        # 总序列长度
N = 32          # 可复用前缀长度（page 对齐后）
XSTAR = 16      # 分界点：重算 [0,16)，复用 [16,32)，新算 [32,40)
D, H, HD, L = 64, 4, 16, 3
POOL = 256      # kv-pool 槽位数
ATOL = 1e-5


class Layer:
    def __init__(self):
        self.Wq = torch.randn(D, D) * 0.1
        self.Wk = torch.randn(D, D) * 0.1
        self.Wv = torch.randn(D, D) * 0.1
        self.Wo = torch.randn(D, D) * 0.1

    def qkv(self, h):
        q = (h @ self.Wq).view(-1, H, HD)
        k = (h @ self.Wk).view(-1, H, HD)
        v = (h @ self.Wv).view(-1, H, HD)
        return q, k, v

    def attn(self, q, k, v):
        # 标准 causal attention; q,k,v: [S,H,HD] -> [S,D]
        S = q.shape[0]
        q_, k_, v_ = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        scores = q_ @ k_.transpose(-1, -2) / (HD ** 0.5)
        mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        o = (F.softmax(scores, dim=-1) @ v_).transpose(0, 1).reshape(S, D)
        return o @ self.Wo


layers = [Layer() for _ in range(L)]
embed = torch.randn(SEQ, D) * 0.5  # 每位置 token embedding（只依赖 token+绝对位置）


def full_forward(h0):
    """全量 baseline：逐层前向，记录每层每位置的 K,V ground truth。"""
    h = h0
    kv = []
    for lyr in layers:
        q, k, v = lyr.qkv(h)
        kv.append((k.clone(), v.clone()))
        h = h + lyr.attn(q, k, v)
    return h, kv


# ============ Baseline：全量 forward，得到每层 ground-truth KV ============
h_full, kv_gt = full_forward(embed)


# ============ 模拟「缓存里存的 KV」：复用段 [XSTAR,N) 的历史完整 KV ============
# 这些是之前完整 prefill 算出、存进缓存的历史值（数值上 == kv_gt 对应段）。
cached_kv = [(k[XSTAR:N].clone(), v[XSTAR:N].clone()) for (k, v) in kv_gt]


# ============ 中间复用填表 + prefill 模拟 ============
# 关键：模拟 req_to_token 表。为每个位置分配一个 kv-pool 槽位。
#   - 复用段 [XSTAR,N)：槽位由 load_back 分配（这里模拟为把 cached_kv 写入指定槽位）
#   - 重算段 [0,XSTAR) 和 [N,SEQ)：新算 KV 写新槽位（out_cache_loc）
# req_to_token[p] = 该位置 KV 所在的物理槽位。attention 靠它间接寻址。

def build_req_to_token():
    """分配槽位并填 req_to_token 表（模拟 write_cache_indices 的中间复用版）。"""
    slot_of = [-1] * SEQ
    free = list(range(POOL))
    torch.manual_seed(123)
    perm = torch.randperm(POOL).tolist()  # 打乱，证明物理槽位顺序无关
    free = perm[:]
    # 复用段先占槽（load 段）
    for p in range(XSTAR, N):
        slot_of[p] = free.pop()
    # 重算段占槽（extend 段）：[0,XSTAR) + [N,SEQ)
    for p in list(range(0, XSTAR)) + list(range(N, SEQ)):
        slot_of[p] = free.pop()
    return slot_of


slot_of = build_req_to_token()

# kv-pool（物理存储）
poolK = [torch.zeros(POOL, H, HD) for _ in range(L)]
poolV = [torch.zeros(POOL, H, HD) for _ in range(L)]

# 1) 复用段：把缓存的历史 KV 写入其槽位（模拟 load_back）
for l in range(L):
    ck, cv = cached_kv[l]
    for idx, p in enumerate(range(XSTAR, N)):
        poolK[l][slot_of[p]] = ck[idx]
        poolV[l][slot_of[p]] = cv[idx]

# 2) 重算段：GPU 现算 [0,XSTAR) 和 [N,SEQ)。
# 关键正确性点：重算某位置 p 的 K/V 需要该位置的「层输入 hidden」，
# 而 hidden 依赖 attention over [0,p]，其中就包含复用段。
# prefill 里重算段和复用段在同一次 forward：attention 通过 req_to_token 读全体 KV。
# 这里逐层模拟这次「混合 forward」：
recompute_positions = list(range(0, XSTAR)) + list(range(N, SEQ))
h_layer_input = embed.clone()  # 第0层输入 = embedding（每位置独立）

for l, lyr in enumerate(layers):
    # 先算重算段各位置的 q,k,v（只有重算段是「新算」）
    q_all, k_all, v_all = lyr.qkv(h_layer_input)  # 对所有位置算 q（q 每步都要）
    # 把重算段的 k,v 写入 pool（复用段的 k,v 已在 pool 里，不覆盖）
    for p in recompute_positions:
        poolK[l][slot_of[p]] = k_all[p]
        poolV[l][slot_of[p]] = v_all[p]
    # attention：每个位置 p 通过 req_to_token 收集 [0,p] 的 KV（含复用+重算混合）
    idx = torch.tensor([slot_of[p] for p in range(SEQ)])
    Kg = poolK[l][idx]   # [SEQ,H,HD] gather 回逻辑顺序
    Vg = poolV[l][idx]
    o = lyr.attn(q_all, Kg, Vg)
    h_layer_input = h_layer_input + o


# ============ 验证：混合布局 vs baseline ============
# (a) 复用段+重算段 gather 回来的每层 KV 是否 == ground truth
maxerr_kv = 0.0
for l in range(L):
    idx = torch.tensor([slot_of[p] for p in range(SEQ)])
    Kg, Vg = poolK[l][idx], poolV[l][idx]
    Kt, Vt = kv_gt[l]
    maxerr_kv = max(maxerr_kv, (Kg - Kt).abs().max().item(), (Vg - Vt).abs().max().item())

# (b) 最后位置 hidden（喂给首个 decode）是否一致
err_hidden = (h_layer_input[-1] - h_full[-1]).abs().max().item()
# (c) 全部位置 hidden 一致
err_all = (h_layer_input - h_full).abs().max().item()

print(f"[中间复用] 布局: 重算[0,{XSTAR}) + 复用[{XSTAR},{N}) + 新算[{N},{SEQ})")
print(f"[验证a] 每层gather KV vs baseline 最大误差 = {maxerr_kv:.2e} -> {'PASS' if maxerr_kv<ATOL else 'FAIL'}")
print(f"[验证b] 最后位置 hidden 误差 = {err_hidden:.2e} -> {'PASS' if err_hidden<ATOL else 'FAIL'}")
print(f"[验证c] 全部位置 hidden 误差 = {err_all:.2e} -> {'PASS' if err_all<ATOL else 'FAIL'}")

ok = maxerr_kv < ATOL and err_hidden < ATOL and err_all < ATOL
print("\n==== Stage3 离线结论 ====")
print("中间段复用填表策略数值正确:", "PASS ✅" if ok else "FAIL ❌")
print("(证明: req_to_token 填「复用段→缓存槽/重算段→新槽」+ 位置按绝对下标 == 全量重算)")
