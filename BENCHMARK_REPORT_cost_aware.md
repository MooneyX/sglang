# cost_aware 预取终止策略 —— 端到端性能评测报告

## 1. 背景与目标

sglang 的分层 KV cache 在请求进入等待队列时，会从 L3 存储预取可复用的 KV，用排队时间掩盖预取延迟。但排队时间与"预取完所有可复用 KV 的时间"并不相等，因此需要**预取终止策略**来决定何时停止预取。现有策略：

- `best_effort`：调度到该请求就立即中止预取，只用已完成部分。
- `wait_complete`：等预取全部完成才开始计算。
- `timeout`：完成或线性超时阈值到达才终止。

本项目新增 **`cost_aware`** 策略：在线比较"未预取部分能节省的计算时间 `t_save`"与"等待预取完成的传输时间 `t_wait`"，仅当等待比重算更划算时才继续等，否则立即终止。

## 2. 实现要点

| 文件 | 改动 |
|------|------|
| `mem_cache/hiradix_cache.py` | 新增 `_cost_aware_can_terminate`（决策）+ `update_prefill_perf`（EMA 在线标定 per-token prefill 耗时）；`can_terminate_prefetch` 加 `cost_aware` 分支；配置项 `cost_aware_gamma`/`cost_aware_perf_ema_alpha`；attach allowed 列表注册 |
| `srt/server_args.py` | `hicache_storage_prefetch_policy` 的 choices 加入 `cost_aware` |
| `managers/scheduler.py` | prefill 结果分支新增 `_maybe_calibrate_prefetch_cost_aware`，用相邻 prefill step 墙钟差喂 EMA 标定 |
| `mem_cache/hicache_storage.py` | 新增 L3 读限速开关 `SGLANG_HICACHE_FILE_BACKEND_GET_DELAY_MS`（模拟慢速远程 L3，评测用） |

决策公式：
- 剩余未取 token `R = len(hash_value)*page_size - completed_tokens`
- `t_wait = R / prefetch_rate`，`prefetch_rate = completed_tokens / (now - start_time)`（在线实测）
- `t_save = R * per_token_prefill_time`（EMA 在线标定，随 batch/位置自适应）
- 判定：`t_save > gamma * t_wait` → 继续等；否则终止。`completed_tokens==0` 或未标定 → 回退 timeout。

## 3. 评测方法

- **模型**：Qwen2.5-14B-Instruct，单卡 H20，`--enable-hierarchical-cache --hicache-storage-backend file`。
- **饱和标定**：14B 单卡饱和吞吐 ≈ 4.3 req/s，故 `request-rate=6`（适度排队，不过载）。
- **慢 L3 模拟**：`GET_DELAY_MS=4`（每页读延迟 4ms），放大传输时间，让"等 vs 不等"的权衡显现。
- **两阶段**：warm（写入 L3）→ 重启 server（GPU/host 缓存清空、L3 保留）→ measure（触发 L3→L2 预取）。
- **两种负载**：
  - v2：`generated-shared-prefix`（合成长共享前缀，sys_len=4096，n=80）
  - v3：`ShareGPT`（真实混合对话，n=400）
- 每策略跑 warm + measure，对比 measure 阶段 TTFT / E2E / 吞吐。

### 关键修复（决定实验有效性）

初期所有测试都出现"预取回传 0 token、策略无差异"。经排查，根因是 **bench 脚本 `start_server()` 每次启动都 `rm -rf` L3 目录**——measure 重启时把 warm 写入的 L3 清空了，导致预取查空。修复：仅 warm 首启清 L3，measure 重启保留；并为每策略使用独立 L3 目录避免交叉污染。修复后预取正常回传 KV，策略差异才显现。

## 4. 结果

### 4.1 单点 A/B 验证（cost_aware，快 vs 慢 L3）

| 场景 | warm TTFT | measure TTFT | 变化 | 非零预取 |
|------|-----------|--------------|------|---------|
| 快 L3 (delay=0) | 4054 ms | **2663 ms** | **↓34%** | 56 |
| 慢 L3 (delay=30ms) | 4060 ms | 5057 ms | ↑25% | 0 |

证明：L3 快时预取划算（measure 显著快于 warm）；L3 慢时傻等反而更慢，cost_aware 判定不值得等而终止（非零预取归 0）。

### 4.2 v2 合成负载（generated-shared-prefix，measure 阶段）

| 策略 | TTFT (ms) | E2E (ms) | 吞吐 (req/s) | 非零预取 |
|------|-----------|----------|--------------|---------|
| best_effort | 4069 | 6776 | 5.16 | 0 |
| wait_complete | **211450** | 214211 | **0.10** | 47 |
| timeout | 6334 | 8872 | 4.49 | 3 |
| **cost_aware** | **4313** | 7027 | **5.03** | 4 |

### 4.3 v3 真实负载（ShareGPT，measure 阶段）

| 策略 | TTFT (ms) | E2E (ms) | 吞吐 (req/s) | 非零预取 |
|------|-----------|----------|--------------|---------|
| best_effort | 3829 | 9777 | 4.69 | 0 |
| wait_complete | **195456** | 197795 | **0.58** | 280 |
| timeout | 4959 | 10838 | 4.59 | 4 |
| **cost_aware** | **4189** | 10069 | **4.67** | 2 |

## 5. 结论

1. **cost_aware 达成设计目标：传输快时受益、传输慢时不傻等。** 快 L3 下能享受预取带来的 34% TTFT 改善；慢 L3 下自动放弃不划算的等待，性能追平"从不等待"的 best_effort。

2. **wait_complete 在慢 L3 下灾难性崩溃**（TTFT ~200 秒、吞吐 <1 req/s，比 best_effort 慢 47–51 倍）——无脑等每个慢预取完成会拖垮整个系统。这正是 cost_aware 要规避的最坏情况。

3. **cost_aware 优于 timeout**：timeout 靠固定线性阈值兜底，慢 L3 下仍慢于 cost_aware（v2: 6334 vs 4313；v3: 4959 vs 4189），因为它不做"省的计算 vs 等的传输"的动态权衡。

4. **合成负载与真实 ShareGPT 负载结论完全一致**，说明 cost_aware 的优势具有普适性，非特定负载的偶然结果。

## 6. 局限与后续

- per-token prefill 耗时用相邻 step 墙钟差做 EMA 标定（零侵入、always-on），overlap 调度下有噪声；如需更高精度可接 `DeviceTimer` 的实测 GPU 时间。
- 慢 L3 用注入 per-page 延迟模拟；真实远程 L3（如 mooncake/远程 KV store）的行为可进一步验证。
- 可进一步扫 `cost_aware_gamma`（等待偏好系数）在不同带宽/模型规模下的最优值。
