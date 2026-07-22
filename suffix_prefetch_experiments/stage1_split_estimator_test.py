"""
SuffixPrefetch 阶段1 单测 — SuffixSplitEstimator (x* 估计器) 正确性验证
=====================================================================

纯 Python、无第三方依赖，直接导入被测模块（把 repo 的 python/ 加入 sys.path）。
验证点：
  T1 α/β 线性拟合：喂合成的 c(i)=α·i+β 采样，拟合结果应接近真值。
  T2 τ 传输速率采样：EMA 收敛到真值。
  T3 x* 落在理论交点：α·x+β=τ 的解，page 对齐后一致。
  T4 边界：τ 处处高于重算 -> x*=N（全重算）；τ 处处低于重算 -> x*=0（全预取）。
  T5 fallback：未标定时返回 None；样本不足时返回 None。
  T6 gamma 偏置：gamma>1 使 x* 增大（更保守、少预取）。
  T7 page 对齐：x* 向下对齐到 page_size 边界。
"""

import os
import sys
import importlib.util

# 直接按文件路径加载被测模块，绕开 sglang 包 __init__（含 torch / Linux-only
# 的 resource 等重依赖），保证该单测在任意平台纯 Python 下可跑。
HERE = os.path.dirname(os.path.abspath(__file__))
MOD_PATH = os.path.abspath(os.path.join(
    HERE, "..", "python", "sglang", "srt", "mem_cache", "suffix_prefetch.py"))
_spec = importlib.util.spec_from_file_location("suffix_prefetch", MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
# dataclass 处理需要模块已在 sys.modules 中（解析类型注解时会回查模块命名空间）
sys.modules["suffix_prefetch"] = _mod
_spec.loader.exec_module(_mod)
SuffixSplitConfig = _mod.SuffixSplitConfig
SuffixSplitEstimator = _mod.SuffixSplitEstimator

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}  {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def feed_recompute_curve(est, alpha_true, beta_true, positions, step=8):
    """按真值曲线 c(i)=alpha*i+beta 喂采样。用 step 个 token 的一步，
    start_pos=p，则 step_gpu_time = 平均per-token * step。"""
    for p in positions:
        mid = p + step / 2.0
        per_token = alpha_true * mid + beta_true  # 真值曲线在中点
        est.record_prefill_step(step_gpu_time=per_token * step,
                                step_tokens=step, start_pos=p)


print("T1: α/β 线性拟合准确性")
est = SuffixSplitEstimator(SuffixSplitConfig(ema_alpha=0.3, min_fit_samples=3))
alpha_true, beta_true = 2e-6, 1e-4
# 喂多种位置，制造 x 方差
feed_recompute_curve(est, alpha_true, beta_true,
                     positions=[0, 64, 128, 256, 512, 1024, 2048] * 5)
fit = est.fit_alpha_beta()
check("fit 非 None", fit is not None)
if fit:
    a, b = fit
    check("alpha 接近真值", abs(a - alpha_true) < 0.3 * alpha_true,
          f"got alpha={a:.3e} true={alpha_true:.3e}")
    check("beta 接近真值", abs(b - beta_true) < 0.5 * beta_true,
          f"got beta={b:.3e} true={beta_true:.3e}")

print("T2: τ 传输速率 EMA 收敛")
est2 = SuffixSplitEstimator(SuffixSplitConfig(ema_alpha=0.2))
rate_true = 500000.0  # tokens/s
for _ in range(50):
    est2.record_transfer_rate(tokens=1024, seconds=1024 / rate_true)
tau = est2.transfer_seconds_per_token()
check("tau 非 None", tau is not None)
if tau:
    check("tau 接近 1/rate", abs(tau - 1.0 / rate_true) < 0.05 / rate_true,
          f"got tau={tau:.3e} true={1.0/rate_true:.3e}")

print("T3: x* 落在理论交点 α·x+β=τ")
# 用真值构造：交点 x_cross=(tau-beta)/alpha 落在中间
est3 = SuffixSplitEstimator(SuffixSplitConfig(ema_alpha=0.3, gamma=1.0))
alpha_t, beta_t = 1e-6, 1e-4
feed_recompute_curve(est3, alpha_t, beta_t,
                     positions=[0, 64, 128, 256, 512, 1024, 2048, 4096] * 6)
# 设 tau 使交点在 ~ (tau-beta)/alpha
N = 4096
x_cross_target = 2000
tau_target = alpha_t * x_cross_target + beta_t
rate_t = 1.0 / tau_target
for _ in range(60):
    est3.record_transfer_rate(tokens=1024, seconds=1024 / rate_t)
x_star = est3.compute_split_point(prefix_len=N, page_size=1)
check("x* 非 None", x_star is not None)
if x_star is not None:
    check("x* 接近理论交点(±10%)",
          abs(x_star - x_cross_target) < 0.1 * x_cross_target,
          f"got x*={x_star} target={x_cross_target}")

print("T4: 边界情形")
# τ 处处高于重算成本 -> 预取贵 -> 全重算 x*=N
est4 = SuffixSplitEstimator(SuffixSplitConfig(ema_alpha=0.3))
feed_recompute_curve(est4, 1e-7, 1e-5, positions=[0, 128, 512, 2048] * 6)
slow_rate = 1.0 / 1e-2  # tau=1e-2, 远高于重算(~1e-5)
for _ in range(40):
    est4.record_transfer_rate(tokens=1024, seconds=1024 / slow_rate)
x4 = est4.compute_split_point(prefix_len=1024, page_size=1)
check("τ贵 -> x*=N(全重算)", x4 == 1024, f"got x*={x4}")

# τ 处处低于重算成本 -> 预取便宜 -> 全预取 x*=0
est5 = SuffixSplitEstimator(SuffixSplitConfig(ema_alpha=0.3))
feed_recompute_curve(est5, 1e-6, 1e-3, positions=[0, 128, 512, 2048] * 6)
fast_rate = 1.0 / 1e-6  # tau=1e-6, 远低于重算(>=1e-3)
for _ in range(40):
    est5.record_transfer_rate(tokens=1024, seconds=1024 / fast_rate)
x5 = est5.compute_split_point(prefix_len=1024, page_size=1)
check("τ便宜 -> x*=0(全预取)", x5 == 0, f"got x*={x5}")

print("T5: 未标定 fallback 返回 None")
est6 = SuffixSplitEstimator()
check("无任何采样 -> None", est6.compute_split_point(1024) is None)
est7 = SuffixSplitEstimator()
est7.record_transfer_rate(1024, 1024 / 500000.0)  # 只有 tau，无重算模型
check("仅 tau 无重算 -> None", est7.compute_split_point(1024) is None)
est8 = SuffixSplitEstimator(SuffixSplitConfig(min_fit_samples=3))
est8.record_prefill_step(1e-3, 100, start_pos=0)  # 只有 1 个样本
check("样本不足 fit -> None", est8.fit_alpha_beta() is None)

print("T6: gamma 偏置方向")
def make_est(gamma):
    e = SuffixSplitEstimator(SuffixSplitConfig(ema_alpha=0.3, gamma=gamma))
    feed_recompute_curve(e, 1e-6, 1e-4, positions=[0, 128, 512, 1024, 2048, 4096] * 6)
    for _ in range(60):
        e.record_transfer_rate(1024, 1024 / (1.0 / (1e-6 * 2000 + 1e-4)))
    return e
x_g1 = make_est(1.0).compute_split_point(4096)
x_g2 = make_est(2.0).compute_split_point(4096)   # gamma 大 -> tau 有效值变大 -> x* 变大
check("gamma>1 使 x* 增大(更保守)", x_g2 > x_g1, f"g1 x*={x_g1}, g2 x*={x_g2}")

print("T7: page 对齐(向下)")
est9 = SuffixSplitEstimator(SuffixSplitConfig(ema_alpha=0.3))
feed_recompute_curve(est9, 1e-6, 1e-4, positions=[0, 128, 512, 1024, 2048, 4096] * 6)
for _ in range(60):
    est9.record_transfer_rate(1024, 1024 / (1.0 / (1e-6 * 2000 + 1e-4)))
xp = est9.compute_split_point(prefix_len=4096, page_size=64)
check("x* 是 page_size 的倍数", xp is not None and xp % 64 == 0, f"got x*={xp}")

print("\n==== 阶段1 单测结论 ====")
print(f"PASS={PASS} FAIL={FAIL} ->",
      "全部通过 ✅" if FAIL == 0 else "存在失败 ❌")
sys.exit(1 if FAIL else 0)
