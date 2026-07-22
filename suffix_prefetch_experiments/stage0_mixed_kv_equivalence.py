"""
SuffixPrefetch 阶段0 可行性桩 — 混合 KV 拼接数值等价性验证
==============================================================

目标（不依赖 sglang 调度，纯 PyTorch 自证）：
  验证 "前段 [0,x) 重算 + 后段 [x,N) 从历史完整KV加载" 的混合布局，
  在数值上等价于 "全量 prefill baseline"。

为什么这能证明 SuffixPrefetch 成立：
  1. attention 的 KV 通过 req_to_token 风格的"索引表"间接寻址，
     只要索引正确，KV 的物理槽位顺序不影响结果  -> 验证A。
  2. 后段 token 的 KV 是"历史某次完整 prefill"算出的值（模拟 L3 缓存），
     它已经编码了对前缀的注意力，加载回来即为正确历史值；
     前段 token 的 KV 由当前重算得到（causal 自洽）。
     两段拼成完整 0..N 的 KV，做后续 attention == baseline  -> 验证B。

关键事实（本实验要坐实的）：
  - 第 L 层的 k_i,v_i = W_k/W_v @ h_i^L，h_i^L 是该层输入。
  - 多层下 h_i^L 依赖前面所有位置（attention 混合），所以后段KV必须是
    "完整序列历史值"才正确 —— 本实验用 ground-truth KV 模拟 L3 里存的正是它。
  - 前段 [0,x) 重算逐层自洽：causal 使前段只 attend 前段，可独立逐层前向。

判据：完整 KV（逐层逐位置）与 baseline 的最大绝对误差 < 1e-5。
"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)

# ---- 配置 ----
N = 32          # 可复用前缀总长度
D = 64          # hidden dim
H = 4           # heads
HD = D // H     # head dim
L = 3           # 层数（多层才能暴露"后段KV依赖前缀"的本质）
CAP = 128       # KV pool 容量（槽位）
X = 20          # 分界点：前段 [0,X) 重算，后段 [X,N) 从历史加载
ATOL = 1e-5


# ---- 模型：L 层 causal multi-head self-attention（无 MLP，足以验证KV机制）----
class Layer:
    def __init__(self):
        self.Wq = torch.randn(D, D) * 0.1
        self.Wk = torch.randn(D, D) * 0.1
        self.Wv = torch.randn(D, D) * 0.1
        self.Wo = torch.randn(D, D) * 0.1

    def qkv(self, h):
        # h: [S, D] -> q,k,v: [S, H, HD]
        q = (h @ self.Wq).view(-1, H, HD)
        k = (h @ self.Wk).view(-1, H, HD)
        v = (h @ self.Wv).view(-1, H, HD)
        return q, k, v

    def attn_out(self, q, k, v):
        # 标准 causal attention（q,k,v: [S,H,HD]），返回 [S,D]
        S = q.shape[0]
        q_ = q.transpose(0, 1)            # [H,S,HD]
        k_ = k.transpose(0, 1)
        v_ = v.transpose(0, 1)
        scores = q_ @ k_.transpose(-1, -2) / (HD ** 0.5)   # [H,S,S]
        mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        p = F.softmax(scores, dim=-1)
        o = p @ v_                        # [H,S,HD]
        o = o.transpose(0, 1).reshape(S, D)
        return o @ self.Wo


layers = [Layer() for _ in range(L)]
embed = torch.randn(N, D) * 0.5   # 模拟 token embedding（含位置），只依赖各自 token


def full_prefill(h0):
    """全量 prefill baseline：逐层前向，记录每层 K,V（ground truth）。"""
    h = h0
    kv_per_layer = []
    for lyr in layers:
        q, k, v = lyr.qkv(h)
        kv_per_layer.append((k.clone(), v.clone()))
        h = h + lyr.attn_out(q, k, v)     # 残差
    return h, kv_per_layer


# ================= Baseline =================
h_full, kv_gt = full_prefill(embed)   # kv_gt[l] = (K[N,H,HD], V[N,H,HD])


# ============ 验证 A：乱序槽位 + 索引表 gather 无关性 ============
# 把 baseline 的 KV 打散到乱序物理槽位，用 req_to_token 索引表还原，重算 attention。
def build_pool_and_table(kv_layers, slot_ids):
    """把每层 KV 写入乱序槽位；返回 pool[L] 和 req_to_token 表。"""
    poolK = [torch.zeros(CAP, H, HD) for _ in range(L)]
    poolV = [torch.zeros(CAP, H, HD) for _ in range(L)]
    for l in range(L):
        K, V = kv_layers[l]
        for pos in range(N):
            poolK[l][slot_ids[pos]] = K[pos]
            poolV[l][slot_ids[pos]] = V[pos]
    return poolK, poolV


def gather_kv(poolK, poolV, table, l):
    """按 req_to_token 表把槽位 gather 回 0..N-1 逻辑顺序。"""
    idx = torch.tensor(table)                 # [N] -> slot
    return poolK[l][idx], poolV[l][idx]


# 乱序槽位：故意打乱，证明物理顺序无关
perm = torch.randperm(CAP)[:N].tolist()
poolK, poolV = build_pool_and_table(kv_gt, perm)

# 用 gather 回来的 KV 重算每层 attention 的 Q（Q 仍来自 baseline hidden）
maxerr_A = 0.0
h = embed
for l, lyr in enumerate(layers):
    q, _, _ = lyr.qkv(h)
    Kg, Vg = gather_kv(poolK, poolV, perm, l)
    o = lyr.attn_out(q, Kg, Vg)
    h_next = h + o
    # 对比该层 attn 用 gather-KV vs ground-truth-KV 的输出
    Kt, Vt = kv_gt[l]
    o_ref = lyr.attn_out(q, Kt, Vt)
    maxerr_A = max(maxerr_A, (o - o_ref).abs().max().item())
    h = h_next

print(f"[验证A] 乱序槽位 gather vs 连续 KV，最大误差 = {maxerr_A:.2e}  "
      f"-> {'PASS' if maxerr_A < ATOL else 'FAIL'}")


# ============ 验证 B：前段重算 + 后段历史加载 拼接等价性 ============
# 后段 [X,N) 的 KV 直接用 ground-truth（模拟从 L3 加载历史完整值）。
# 前段 [0,X) 的 KV 由"重算"得到：逐层只前向前段 token（causal 自洽）。
# 拼成完整 KV 后，逐层逐位置对比 baseline。

# --- 前段重算：只跑 [0,X) 这段 token，逐层前向 ---
recomputed_kv = []   # 每层前段的 (K[0:X], V[0:X])
h_pre = embed[:X]    # 前段输入（第0层输入只依赖 token 自身）
for l, lyr in enumerate(layers):
    q, k, v = lyr.qkv(h_pre)           # 前段的 K,V
    recomputed_kv.append((k.clone(), v.clone()))
    # 前段做 attention 只 attend 前段（causal），自洽推进到下一层
    h_pre = h_pre + lyr.attn_out(q, k, v)

# --- 拼接：完整 KV = 前段重算 ⊕ 后段历史加载，写入乱序槽位 ---
mix_slots = torch.randperm(CAP)[:N].tolist()
mixK = [torch.zeros(CAP, H, HD) for _ in range(L)]
mixV = [torch.zeros(CAP, H, HD) for _ in range(L)]
for l in range(L):
    Krec, Vrec = recomputed_kv[l]          # 前段 [0,X)
    Kgt, Vgt = kv_gt[l]                     # 用于后段 [X,N)
    for pos in range(N):
        if pos < X:
            k_src, v_src = Krec[pos], Vrec[pos]        # 重算
        else:
            k_src, v_src = Kgt[pos], Vgt[pos]          # 历史加载
        mixK[l][mix_slots[pos]] = k_src
        mixV[l][mix_slots[pos]] = v_src

# --- 逐层逐位置对比：拼接 KV vs baseline ground-truth KV ---
maxerr_B_pre = 0.0    # 前段重算 vs 历史
maxerr_B_full = 0.0   # 完整拼接 KV vs baseline
for l in range(L):
    idx = torch.tensor(mix_slots)
    Kmix = mixK[l][idx]     # gather 回 0..N-1
    Vmix = mixV[l][idx]
    Kgt, Vgt = kv_gt[l]
    maxerr_B_pre = max(maxerr_B_pre,
                       (Kmix[:X] - Kgt[:X]).abs().max().item(),
                       (Vmix[:X] - Vgt[:X]).abs().max().item())
    maxerr_B_full = max(maxerr_B_full,
                        (Kmix - Kgt).abs().max().item(),
                        (Vmix - Vgt).abs().max().item())

print(f"[验证B] 前段重算 KV vs 历史 KV，最大误差 = {maxerr_B_pre:.2e}  "
      f"-> {'PASS' if maxerr_B_pre < ATOL else 'FAIL'}")
print(f"[验证B] 混合拼接完整 KV vs baseline，最大误差 = {maxerr_B_full:.2e}  "
      f"-> {'PASS' if maxerr_B_full < ATOL else 'FAIL'}")


# ============ 验证 C：端到端 —— 用混合 KV 算"下一个 token"的 attention ============
# prefill 的真正产出之一：最后位置的 hidden（喂给首个 decode）。
# 用混合拼接的完整 KV，逐层推进 Q，得到最后位置 hidden，对比 baseline。
h = embed.clone()
for l, lyr in enumerate(layers):
    q, _, _ = lyr.qkv(h)
    idx = torch.tensor(mix_slots)
    Kmix, Vmix = mixK[l][idx], mixV[l][idx]
    h = h + lyr.attn_out(q, Kmix, Vmix)
err_C = (h[-1] - h_full[-1]).abs().max().item()
print(f"[验证C] 混合KV最后位置 hidden vs baseline，最大误差 = {err_C:.2e}  "
      f"-> {'PASS' if err_C < ATOL else 'FAIL'}")

all_pass = (maxerr_A < ATOL and maxerr_B_pre < ATOL
            and maxerr_B_full < ATOL and err_C < ATOL)
print("\n==== 阶段0 结论 ====")
print("混合布局（前段重算+后段历史加载+乱序槽位）数值等价性:",
      "全部 PASS ✅" if all_pass else "存在 FAIL ❌")
