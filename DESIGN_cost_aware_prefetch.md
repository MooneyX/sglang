# Cost-Aware L3→L2 预取终止策略设计

## 目标
在"请求排队期间从 L3 预取可复用 KV cache 到 L2(host)"的机制中，当调度轮到该请求、预取尚未完成时，决定**立即终止（只用已取部分）还是继续等待**。

现有策略（`hicache_storage_prefetch_policy`）：
- `best_effort`：立即终止，只用已完成部分。
- `wait_complete`：必须全部取完才终止。
- `timeout`：完成或线性超时（base + per_ki_token·n/1024，封顶 max）才终止。

新策略 `cost_aware`：**比较"继续等待预取完成所需的传输时间" vs "这些未取 KV 若不复用、需要现场重算所省下的计算时间"**，据此决定何时停止。

## 决策模型

记（单个 prefetch operation）：
- `total = len(operation.hash_value) * page_size`  —— L3 命中、计划预取的总 token 数
- `done  = operation.completed_tokens`             —— 已取回 token 数
- `R = total - done`                                —— 剩余未取 token 数
- `elapsed = now - operation.start_time`            —— 该 operation 已运行时长

若 `R <= 0`：已完成 → 可终止（True）。

**等待传输时间**（继续等把剩余取完的代价）：
```
prefetch_rate = done / elapsed            # tokens/s，在线实测，自适应后端带宽波动
T_wait = R / prefetch_rate                # s
```

**省下的计算时间**（这 R 个 token 若复用预取结果，就不必现场 prefill 计算；反之终止则要现场算）：
```
T_save = R * t_prefill_per_token          # s
```
其中 `t_prefill_per_token` 用**在线标定**：调度器每个 prefill step 实测 GPU 时间 ÷ 该 step 处理的 token 数，做指数滑动平均（EMA）。天然随 batch 规模、序列长度、位置的平均负载自适应，且不依赖 model_config。

**判定**：
```
if T_save > gamma * T_wait:   # 值得等：省的算力时间 > 等待传输时间
    can_terminate = False     # 继续预取，不终止
else:
    can_terminate = True      # 终止，用已取部分
```
`gamma`（默认 1.0）：可配的保守系数。gamma>1 更倾向终止（少等），gamma<1 更倾向等待。

## 回退（无法估计时）
- `done == 0`（还没取到任何 token，速率无法估）→ 回退到 `timeout` 逻辑。
- `t_prefill_per_token` 尚未标定出（服务刚启动、还没跑过 prefill step）→ 回退到 `timeout` 逻辑。

## 可配参数（走 hicache_storage_backend_extra_config）
- `cost_aware_gamma`（默认 1.0）
- `cost_aware_perf_ema_alpha`（默认 0.1，标定 EMA 系数）
- 复用现有 timeout 三参数作为回退：prefetch_timeout_base / per_ki_token / max

## 改动点
1. `mem_cache/hicache_storage.py`：`PrefetchTimeoutConfig` 旁新增/扩展配置（或复用 extra_config 解析）。
2. `mem_cache/hiradix_cache.py`：
   - 注册策略名 `cost_aware` 到 allowed 列表（attach 校验、_parse 配置）。
   - `HiRadixCache` 增加在线标定字段 `self._prefill_time_per_token`（EMA）与更新方法 `update_prefill_perf(step_gpu_time, step_tokens)`。
   - `can_terminate_prefetch` 增加 `cost_aware` 分支实现上面的判定。
3. `managers/scheduler.py`：在 prefill step 实测耗时处调用 `tree_cache.update_prefill_perf(...)` 喂入标定数据（若已有 DeviceTimer/step_time_dict 更佳，否则用 step 墙钟时间近似）。

## TP 一致性
沿用现有机制：`can_terminate_prefetch` 末尾已有 `_all_reduce_attn_groups` 对 `can_terminate`/`terminated` 做 MAX 归约，新分支只需产出本地布尔值即可，无需额外处理。
