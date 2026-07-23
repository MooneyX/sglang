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

import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SuffixPrefetchCostModel:
    """线性边际重算成本模型 + 恒定预取成本，用于求分界点 x*。

    单位统一为 秒/token（alpha 为 秒/token^2）。数值来自真机标定或经验默认。

    支持「在线拟合」：推理运行时用真实观测持续修正 alpha/beta/tau，使 x* 逐步
    逼近当前机器/负载/后端的真实成本，而不是死守启动时的离线标定值。
      - tau：用每次 prefetch 完成的 (completed_tokens, elapsed) 做 EMA。
      - alpha/beta：用每个 prefill chunk 的 (位置中点 i, 每 token 耗时 c) 做
        在线加权最小二乘（维护 Σw, Σwi, Σwc, Σwi², Σwic）拟合直线 c=alpha*i+beta。
    online_fit=True 时，只要样本量达到最小阈值即用拟合值覆盖初始参数。
    """

    alpha: float = 0.0        # c(i) 关于位置 i 的斜率（秒/token^2）
    beta: float = 0.0         # c(i) 截距（秒/token），≈ 与位置无关的投影/FFN 边际
    tau: float = 0.0          # 每 token L3→host 预取成本（秒/token）
    enabled: bool = False     # 参数是否有效（alpha>0 且 tau>0）

    # ---- 在线拟合开关与超参 ----
    online_fit: bool = False          # 是否启用运行时在线拟合
    tau_ema_gamma: float = 0.2        # tau EMA 平滑系数（新样本权重）
    ls_decay: float = 0.98            # 最小二乘的遗忘因子（越小越看重近期样本）
    min_recompute_samples: int = 8    # alpha/beta 生效所需最小 chunk 样本数
    min_prefetch_samples: int = 2     # tau 生效所需最小 prefetch 样本数
    # 采样过滤：只接受 token 数 >= 该阈值的 prefill chunk 作为重算样本。
    # 小 chunk（尤其 tokens<=1 的 decode/收尾前向）的 gap_latency 里混入了大量
    # 调度/等待时间，会算出荒谬的每 token 成本（真机见过 739ms、5489ms/tok），
    # 毒化最小二乘并把 beta 拉高到 c(i)≫tau、恒定误判全预取。默认过滤 <64 token。
    min_chunk_tokens_for_fit: int = 64

    # ---- 在线统计的内部状态（不参与 __init__ 的位置参数） ----
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _n_recompute: float = 0.0         # 加权样本数（重算 chunk，随遗忘因子衰减）
    _n_recompute_raw: int = 0         # 真实累计样本数（不衰减，仅用于阈值判断）
    _sw: float = 0.0                  # Σw
    _swi: float = 0.0                 # Σ w*i
    _swc: float = 0.0                 # Σ w*c
    _swii: float = 0.0                # Σ w*i*i
    _swic: float = 0.0                # Σ w*i*c
    _n_prefetch: int = 0              # tau 样本计数
    _tau_ema: float = 0.0             # tau 的 EMA 当前值

    def __post_init__(self) -> None:
        self.enabled = self.alpha > 0.0 and self.tau > 0.0
        # 用初始 tau 作为 EMA 起点（若有）
        self._tau_ema = self.tau

    # ---------------- 在线观测入口 ----------------

    def observe_prefetch(self, completed_tokens: int, elapsed_s: float) -> None:
        """记录一次真实 prefetch：completed_tokens 个 token 花了 elapsed_s 秒。

        用 EMA 更新 tau（每 token 预取成本）。仅在 online_fit 时生效。
        """
        if not self.online_fit:
            return
        if completed_tokens <= 0 or elapsed_s <= 0.0:
            return
        tau_obs = elapsed_s / completed_tokens
        with self._lock:
            self._n_prefetch += 1
            # First real observation seeds tau directly (avoid dragging the EMA
            # from the cold-start guess); subsequent ones smooth via EMA.
            if self._n_prefetch == 1:
                self._tau_ema = tau_obs
            else:
                g = self.tau_ema_gamma
                self._tau_ema = (1.0 - g) * self._tau_ema + g * tau_obs
            self._maybe_apply_locked()
        logger.debug(
            "[SuffixPrefetch][obs-tau] tokens=%d elapsed_ms=%.1f "
            "tau_obs_ms/tok=%.4f tau_ema_ms/tok=%.4f n_prefetch=%d",
            completed_tokens,
            elapsed_s * 1e3,
            tau_obs * 1e3,
            self._tau_ema * 1e3,
            self._n_prefetch,
        )

    def observe_recompute_chunk(
        self, start_pos: int, num_tokens: int, elapsed_s: float
    ) -> None:
        """记录一次真实的重算 chunk：从 start_pos 起算了 num_tokens 个 token，
        耗时 elapsed_s 秒。位置中点作为 i，每 token 耗时作为 c，喂给在线最小二乘。
        仅在 online_fit 时生效。
        """
        if not self.online_fit:
            return
        if num_tokens <= 0 or elapsed_s <= 0.0:
            return
        # 过滤小 chunk：其 gap_latency 混入调度/等待时间，每 token 成本不可信。
        if num_tokens < self.min_chunk_tokens_for_fit:
            logger.debug(
                "[SuffixPrefetch][obs-recompute] SKIP small chunk tokens=%d "
                "(<%d) c_raw_ms/tok=%.2f",
                num_tokens,
                self.min_chunk_tokens_for_fit,
                (elapsed_s / num_tokens) * 1e3,
            )
            return
        i_mid = start_pos + num_tokens / 2.0
        c_obs = elapsed_s / num_tokens
        with self._lock:
            d = self.ls_decay
            # 先对已有统计做遗忘衰减，再并入新样本（权重 1）
            self._sw = self._sw * d + 1.0
            self._swi = self._swi * d + i_mid
            self._swc = self._swc * d + c_obs
            self._swii = self._swii * d + i_mid * i_mid
            self._swic = self._swic * d + i_mid * c_obs
            self._n_recompute = self._n_recompute * d + 1.0
            self._n_recompute_raw += 1
            self._maybe_apply_locked()
        logger.debug(
            "[SuffixPrefetch][obs-recompute] i=%.0f tokens=%d elapsed_ms=%.1f "
            "c_obs_ms/tok=%.4f tau_ms/tok=%.4f alpha=%.3e beta_ms/tok=%.4f "
            "n_raw=%d n_weighted=%.1f",
            i_mid,
            num_tokens,
            elapsed_s * 1e3,
            c_obs * 1e3,
            self.tau * 1e3,
            self.alpha,
            self.beta * 1e3,
            self._n_recompute_raw,
            self._n_recompute,
        )

    def _maybe_apply_locked(self) -> None:
        """在持锁状态下，根据当前统计尝试更新 alpha/beta/tau。"""
        # 更新 tau
        if self._n_prefetch >= self.min_prefetch_samples and self._tau_ema > 0.0:
            self.tau = self._tau_ema

        # 拟合 alpha/beta（用真实累计样本数判阈值，避免遗忘衰减后的 _n_recompute
        # 永远差一点到阈值导致 alpha/beta 从不更新；拟合本身仍用衰减统计量）
        if self._n_recompute_raw >= self.min_recompute_samples and self._sw > 0.0:
            mean_i = self._swi / self._sw
            mean_c = self._swc / self._sw
            var_i = self._swii / self._sw - mean_i * mean_i
            cov_ic = self._swic / self._sw - mean_i * mean_c
            if var_i > 1e-9:
                alpha_fit = cov_ic / var_i
                beta_fit = mean_c - alpha_fit * mean_i
                # 斜率必须为正才有物理意义（c 随位置增长）；否则只更新 beta
                if alpha_fit > 0.0:
                    self.alpha = alpha_fit
                    self.beta = beta_fit
                elif mean_c > 0.0:
                    # 位置不敏感：退化为常数成本 c≈mean_c（alpha 置极小正数）
                    self.beta = mean_c
        # 重新评估 enabled
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
            logger.debug(
                "[SuffixPrefetch][x*] h=%d N=%d -> x*=%d (disabled/empty) "
                "alpha=%.3e beta_ms=%.4f tau_ms=%.4f",
                h, n, h, self.alpha, self.beta * 1e3, self.tau * 1e3,
            )
            return h

        x_star_f = (self.tau - self.beta) / self.alpha
        # clamp 到 [h, n]
        x_star = int(max(h, min(x_star_f, n)))
        # 向下对齐到 page 边界（保证与 radix / prefetch 的页粒度一致）
        if page_size > 1:
            x_star -= x_star % page_size
            if x_star < h:
                x_star = h
        # 决策日志：记录每次 x* 计算的输入(h,N)、原始解、clamp 后结果、当前参数，
        # 以及三段划分 [h,x*)重算 / [x*,N)预取，便于分析 x* 漂移与最优策略的偏差。
        if x_star <= h:
            regime = "ALL_PREFETCH"      # c 全程 > tau
        elif x_star >= n:
            regime = "ALL_RECOMPUTE"     # c 全程 < tau
        else:
            regime = "SPLIT"             # 甜点：前段重算+后段预取
        logger.debug(
            "[SuffixPrefetch][x*] h=%d N=%d x*_raw=%.1f x*=%d regime=%s | "
            "recompute[%d,%d)=%d tok, prefetch[%d,%d)=%d tok | "
            "alpha=%.3e beta_ms/tok=%.4f tau_ms/tok=%.4f "
            "c(h)_ms=%.4f c(N)_ms=%.4f",
            h, n, x_star_f, x_star, regime,
            h, x_star, max(0, x_star - h),
            x_star, n, max(0, n - x_star),
            self.alpha, self.beta * 1e3, self.tau * 1e3,
            (self.alpha * h + self.beta) * 1e3,
            (self.alpha * n + self.beta) * 1e3,
        )
        return x_star

    @classmethod
    def from_extra_config(cls, extra_config: dict) -> "SuffixPrefetchCostModel":
        """从 hicache_storage_backend_extra_config 解析成本参数。

        识别键（均为可选，单位 秒/token）：
          suffix_prefetch_alpha, suffix_prefetch_beta, suffix_prefetch_tau
        在线拟合相关（可选）：
          suffix_prefetch_online_fit (bool，默认 False)
          suffix_prefetch_tau_ema_gamma, suffix_prefetch_ls_decay
          suffix_prefetch_min_recompute_samples, suffix_prefetch_min_prefetch_samples
        """
        return cls(
            alpha=float(extra_config.pop("suffix_prefetch_alpha", 0.0)),
            beta=float(extra_config.pop("suffix_prefetch_beta", 0.0)),
            tau=float(extra_config.pop("suffix_prefetch_tau", 0.0)),
            online_fit=bool(extra_config.pop("suffix_prefetch_online_fit", False)),
            tau_ema_gamma=float(
                extra_config.pop("suffix_prefetch_tau_ema_gamma", 0.2)
            ),
            ls_decay=float(extra_config.pop("suffix_prefetch_ls_decay", 0.98)),
            min_recompute_samples=int(
                extra_config.pop("suffix_prefetch_min_recompute_samples", 8)
            ),
            min_prefetch_samples=int(
                extra_config.pop("suffix_prefetch_min_prefetch_samples", 2)
            ),
            min_chunk_tokens_for_fit=int(
                extra_config.pop("suffix_prefetch_min_chunk_tokens_for_fit", 64)
            ),
        )
