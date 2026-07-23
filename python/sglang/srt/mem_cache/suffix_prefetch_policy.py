"""SuffixPrefetch 成本模型：计算前缀重算/后缀预取的分界点 x*。

理论（详见项目 SuffixPrefetch_ANALYSIS_REPORT.md / IMPL_SPEC.md）：
  - 每 token 边际重算成本随位置线性增长：c(i) = alpha * i + beta （秒/token）
  - 每 token 从 L3 预取成本近似恒定：tau （秒/token）
  - 位置 i 处：c(i) < tau 时重算更快、c(i) > tau 时预取更快。
  - 分界点 x*：令 c(x*) = tau  ==>  x* = (tau - beta) / alpha

在可复用前缀 [0, N) 内，[0, x*) 走 GPU 重算、[x*, N) 从 L3 预取复用。
x* 必须约束在 [h, N)：h = 已在 device+host 命中的长度（matched_len），
[0, h) 已在快速层、无 tau 成本，永远直接复用，不重算。

真机标定参考（qwen14b + file backend SSD，见 2026-07-23 memory）：
  c(front)≈0.28ms, c(back)≈0.40ms, tau≈0.24ms → x* 落在前缀内部（甜点区）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SuffixPrefetchCostModel:
    """线性边际重算成本模型 + 恒定预取成本，用于求分界点 x*。

    单位统一为 秒/token（alpha 为 秒/token^2）。数值来自真机标定或经验默认。
    """

    alpha: float = 0.0        # c(i) 关于位置 i 的斜率（秒/token^2）
    beta: float = 0.0         # c(i) 截距（秒/token），≈ 与位置无关的投影/FFN 边际
    tau: float = 0.0          # 每 token L3→host 预取成本（秒/token）
    enabled: bool = False     # 参数是否有效（alpha>0 且 tau>0）

    def __post_init__(self) -> None:
        self.enabled = self.alpha > 0.0 and self.tau > 0.0

    def compute_x_star(self, h: int, n: int, page_size: int = 1) -> int:
        """返回分界点 x*（page 对齐、clamp 到 [h, n]）。

        Args:
            h: 已在 device+host 命中的长度（matched_len），x* 不应小于它。
            n: 可复用前缀总长度 N。
            page_size: 页大小，x* 向下对齐到页边界。

        Returns:
            分界点 x*：[h, x*) 建议重算、[x*, n) 建议从 L3 预取。
            - 参数无效 → 返回 h（退化为"全部预取 [h,n)"，即现状）。
            - x* <= h（c 全程 > tau，大模型/慢重算）→ 返回 h（全预取）。
            - x* >= n（c 全程 < tau，小模型/快重算）→ 返回 n（全重算，不预取）。
        """
        if not self.enabled or n <= h:
            return h

        x_star_f = (self.tau - self.beta) / self.alpha
        # clamp 到 [h, n]
        x_star = int(max(h, min(x_star_f, n)))
        # 向下对齐到 page 边界（保证与 radix / prefetch 的页粒度一致）
        if page_size > 1:
            x_star -= x_star % page_size
            if x_star < h:
                x_star = h
        return x_star

    @classmethod
    def from_extra_config(cls, extra_config: dict) -> "SuffixPrefetchCostModel":
        """从 hicache_storage_backend_extra_config 解析成本参数。

        识别键（均为可选，单位 秒/token）：
          suffix_prefetch_alpha, suffix_prefetch_beta, suffix_prefetch_tau
        """
        return cls(
            alpha=float(extra_config.pop("suffix_prefetch_alpha", 0.0)),
            beta=float(extra_config.pop("suffix_prefetch_beta", 0.0)),
            tau=float(extra_config.pop("suffix_prefetch_tau", 0.0)),
        )
