# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SuffixPrefetch split-point (x*) estimator.

Standalone, side-effect-free calibration + split-point math for the
``suffix_prefetch`` HiCache prefetch policy. Kept independent of the rest of
HiCache so it can be unit-tested offline and toggled on/off cleanly.

Idea
----
For a reusable KV prefix of length ``N`` (page-aligned), recomputing token *i*
costs roughly ``c(i) = alpha * i + beta`` seconds (attention makes later tokens
more expensive -> approximately linear growth), while fetching one token from L3
storage costs a roughly constant ``tau = 1 / transfer_rate`` seconds.

Because per-token recompute cost grows with position while fetch cost is flat,
there is a crossover position ``x*`` where they are equal::

    alpha * x* + beta = tau   =>   x* = (tau - beta) / alpha

Recompute the cheap prefix ``[0, x*)`` on the GPU and prefetch the expensive
suffix ``[x*, N)`` from L3. This is the *opposite* split direction from the
existing ``cost_aware_endpoint`` policy (which fetches the prefix and recomputes
the suffix); flipping the direction turns the TTFT objective from concave (only
endpoint optima) into convex (a genuine interior optimum at ``x*``).

This module only *computes* ``x*``. Actually issuing a suffix-only prefetch and
laying out the mixed KV in prefill are later stages.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SuffixSplitConfig:
    """Tunables for the suffix-prefetch split estimator.

    All are overridable via ``hicache_storage_backend_extra_config`` JSON.
    """

    # EMA weight for new samples (both recompute-cost fit and transfer rate).
    # Higher = adapt faster to drift, lower = smoother. 0 < ema_alpha <= 1.
    ema_alpha: float = 0.1
    # Safety factor on the crossover. gamma > 1 biases toward recomputing more
    # (prefetch less / more conservatively); gamma < 1 biases toward prefetching
    # more. x* solves alpha*x + beta = gamma * tau.
    gamma: float = 1.0
    # Minimum number of position-spread samples before the linear fit is trusted.
    min_fit_samples: int = 3


@dataclass
class SuffixSplitEstimator:
    """Online estimator of the recompute/prefetch crossover point ``x*``.

    Two independently-calibrated quantities:

    * Recompute cost model ``c(i) = alpha * i + beta`` (seconds/token at
      position ``i``), fit by EMA-decayed least squares over
      (position, per-token-time) samples fed from measured prefill steps.
    * Transfer cost ``tau`` (seconds/token) = 1 / EMA(transfer rate tokens/s),
      fed from measured prefetch throughput.

    Side-effect-free and lock-free: the scheduler thread owns it.
    """

    config: SuffixSplitConfig = field(default_factory=SuffixSplitConfig)

    # --- recompute-cost linear fit running sums (EMA-decayed) ---
    _fit_n: float = 0.0
    _fit_sx: float = 0.0
    _fit_sy: float = 0.0
    _fit_sxx: float = 0.0
    _fit_sxy: float = 0.0
    # flat fallback: EMA of per-token recompute time regardless of position
    _flat_per_token: Optional[float] = None

    # --- transfer rate (tokens/s) EMA ---
    _transfer_rate: Optional[float] = None

    # -- diagnostics --
    _split_calls: int = 0

    # ------------------------------------------------------------------ #
    # Calibration inputs
    # ------------------------------------------------------------------ #
    def record_prefill_step(
        self,
        step_gpu_time: float,
        step_tokens: int,
        start_pos: Optional[int] = None,
    ) -> None:
        """Feed one measured prefill (extend) step.

        Args:
            step_gpu_time: measured GPU seconds for this prefill step.
            step_tokens: number of tokens processed in this step.
            start_pos: absolute position of the first token in this step (i.e.
                the cached prefix length before this step). Enables the
                position-aware slope fit; if None, only the flat model updates.
        """
        if step_tokens <= 0 or step_gpu_time <= 0:
            return
        sample = float(step_gpu_time) / float(step_tokens)  # seconds/token

        # Flat EMA (used as fallback when the slope is not yet identifiable).
        if self._flat_per_token is None:
            self._flat_per_token = sample
        else:
            a = self.config.ema_alpha
            self._flat_per_token = a * sample + (1.0 - a) * self._flat_per_token

        # Position-aware least-squares fit y = alpha*x + beta, with EMA decay on
        # the running sums so the fit tracks drift instead of accumulating forever.
        if start_pos is None or start_pos < 0:
            return
        x = float(start_pos) + float(step_tokens) / 2.0  # step midpoint position
        y = sample
        decay = 1.0 - self.config.ema_alpha
        self._fit_n = decay * self._fit_n + 1.0
        self._fit_sx = decay * self._fit_sx + x
        self._fit_sy = decay * self._fit_sy + y
        self._fit_sxx = decay * self._fit_sxx + x * x
        self._fit_sxy = decay * self._fit_sxy + x * y

    def record_transfer_rate(self, tokens: int, seconds: float) -> None:
        """Feed one measured L3->host transfer observation (tokens over seconds)."""
        if tokens <= 0 or seconds <= 0:
            return
        rate = float(tokens) / float(seconds)  # tokens/s
        if self._transfer_rate is None:
            self._transfer_rate = rate
        else:
            a = self.config.ema_alpha
            self._transfer_rate = a * rate + (1.0 - a) * self._transfer_rate

    # ------------------------------------------------------------------ #
    # Model readouts
    # ------------------------------------------------------------------ #
    def fit_alpha_beta(self) -> Optional[Tuple[float, float]]:
        """Solve the 2x2 normal equations for ``y = alpha*x + beta``.

        Returns (alpha, beta) or None when the slope is not yet identifiable
        (too few samples or too little spread in x). A negative slope is
        nonsensical for attention recompute cost and is clamped to flat.
        """
        n = self._fit_n
        if n < float(self.config.min_fit_samples):
            return None
        denom = n * self._fit_sxx - self._fit_sx * self._fit_sx
        if self._fit_sxx <= 0 or denom <= 1e-9 * n * self._fit_sxx:
            # near-zero variance in x -> slope unidentifiable
            return None
        alpha = (n * self._fit_sxy - self._fit_sx * self._fit_sy) / denom
        beta = (self._fit_sy - alpha * self._fit_sx) / n
        if alpha < 0:
            alpha = 0.0
            beta = self._fit_sy / n
        return alpha, beta

    def transfer_seconds_per_token(self) -> Optional[float]:
        """tau = 1 / transfer_rate (seconds/token), or None if uncalibrated."""
        if self._transfer_rate is None or self._transfer_rate <= 0:
            return None
        return 1.0 / self._transfer_rate

    # ------------------------------------------------------------------ #
    # Split-point
    # ------------------------------------------------------------------ #
    def compute_split_point(
        self, prefix_len: int, page_size: int = 1
    ) -> Optional[int]:
        """Compute the recompute/prefetch split ``x*`` for a reusable prefix.

        Returns the number of prefix tokens to RECOMPUTE (``[0, x*)``); the
        remaining suffix ``[x*, prefix_len)`` should be prefetched from L3.

        Returns:
            x* in [0, prefix_len], page-aligned, or None if not yet calibrated
            (caller should fall back to an existing policy).

        Semantics of the endpoints:
            x* == 0            -> recompute nothing, prefetch the whole prefix
                                  (fetch is cheaper everywhere; == wait_complete).
            x* == prefix_len   -> recompute everything, prefetch nothing
                                  (recompute is cheaper everywhere; == best_effort).
            0 < x* < prefix_len -> genuine interior split (the useful case).
        """
        self._split_calls += 1
        if prefix_len <= 0:
            return 0

        tau = self.transfer_seconds_per_token()
        if tau is None:
            return None  # transfer rate not calibrated yet
        tau *= self.config.gamma

        fit = self.fit_alpha_beta()
        if fit is not None:
            alpha, beta = fit
        elif self._flat_per_token is not None:
            alpha, beta = 0.0, self._flat_per_token  # flat fallback
        else:
            return None  # recompute cost not calibrated yet

        if alpha <= 0.0:
            # Flat recompute cost: no crossover. Pick the globally cheaper side.
            # recompute cost per token = beta (constant) vs fetch cost tau.
            x_star = 0 if beta > tau else prefix_len
        else:
            # alpha * x + beta = tau  =>  x* = (tau - beta) / alpha
            x_raw = (tau - beta) / alpha
            x_star = int(math.floor(x_raw))

        # Clamp to [0, prefix_len].
        x_star = max(0, min(prefix_len, x_star))

        # Page-align the recompute length DOWN so the prefetched suffix starts on
        # a page boundary (storage is addressed per page).
        if page_size > 1:
            x_star = (x_star // page_size) * page_size

        return x_star

    def describe(self, prefix_len: int, page_size: int = 1) -> str:
        """Human-readable one-line summary for logging (no state mutation of
        note beyond the split-call counter inside compute_split_point)."""
        fit = self.fit_alpha_beta()
        tau = self.transfer_seconds_per_token()
        x = self.compute_split_point(prefix_len, page_size)
        if fit is not None:
            alpha, beta = fit
            model = f"alpha={alpha:.3e} beta={beta:.3e}(fit)"
        elif self._flat_per_token is not None:
            model = f"alpha=0 beta={self._flat_per_token:.3e}(flat)"
        else:
            model = "recompute=uncalibrated"
        tau_s = f"{tau:.3e}" if tau is not None else "uncalibrated"
        if x is None:
            split = "x*=None(fallback)"
        elif x == 0:
            split = "x*=0(prefetch-all)"
        elif x == prefix_len:
            split = f"x*={x}(recompute-all)"
        else:
            split = f"x*={x}(interior,recompute[0,{x}) prefetch[{x},{prefix_len}))"
        return f"N={prefix_len} {model} tau={tau_s} -> {split}"
