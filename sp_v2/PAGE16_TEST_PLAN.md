# 大 page_size 三策略端到端测试方案（PAGE16_TEST_PLAN）

> 目标：在 `--page-size 16`（备选 64）下重跑 wait_complete / suffix_race / best_effort 三策略矩阵，
> 验证带宽提升预估（file 1.5→~3.5GB/s、mooncake 4.4→~12GB/s），并观察大页粒度下 race 策略行为。
> 前提事实（已代码确认）：非 MUSA 平台默认 page_size=1（server_args.py:4965）；`--page-size` 为现成 server 参数；
> prefetch 限流按 tokens 计（`prefetch_capacity_limit = 0.5 × host_pool.size`），不随 page 变。

---

## 1. 测试目标与判定标准

| # | 目标 | 判定标准 |
|---|---|---|
| G1 | 验证带宽上限达成 | file0 64K wc 的 `prefetch_dur` 3.13s → ≤1.4s（≈3.5GB/s）；rdma 64K wc 1.08s → ≤0.45s（≈12GB/s） |
| G2 | 三策略排序是否变化 | best_effort 纯重算 TTFT **必须不变**（控制组）；race 是否仍 ≤ min(全取, 全算) |
| G3 | race 大页会合正常 | 无 FULL_FETCH/长期不收敛退化；日志无 RaceStep 异常 |
| G4 | 装置自洽 | be 行不受 page 影响 → 证明测试装置未被破坏 |

**理论假设（测试要验证）**：带宽接近上限后，race 相对 wc 的优势应收窄——race 的价值在带宽不足时用重算补位，带宽不稀缺时两者都收敛到"取数时间"。

## 2. 参数选择（含理由）

| 参数 | 取值 | 理由 |
|---|---|---|
| `--page-size` | **16（主测）**；64（备选） | 16：单批 128 页 = 143MB > 探针 71MB 门槛（mc 12.5GB/s 阈值）；会合粒度误差 ≤16 tokens；host 单批 143MB 无压力。64：FlashMLA 强制值，若 attention 后端改写直接用 |
| backend | file0（delay=0）+ rdma | 纯带宽上限验证；中速档见 §4 轮2 |
| policy | wait_complete / suffix_race / best_effort | 三策略全矩阵 |
| 长度 | probe 8K；sweep rdma {16,32,48,64}K、file {16,32,64}K | 沿用 bigrun.sh 结构；**全部长度 %16/%32/%64==0 已核对** |
| MEMFRAC | 0.90 | 64K 需 host pool ≥ 131072 tokens 完整预取，0.90 已验证可行 |
| CHUNK | 8192 | 是 16/32/64 的整倍数，chunk 边界消费钩子不变 |
| prefetch_threshold | 64（保持） | page=64 时 1 页=64 tokens 恰过线；page=16 时 4 页，无问题 |
| MC_MASTER / MC_SEG | 127.0.0.1:50052 / 64gb | 不变 |
| delay 旋钮 | **0（本轮）** | 见 §3 坑②，中速档语义需按 page 缩放，放轮2 |

## 3. 必须注意的坑（大 page 特有）

1. **换 page 档必须清空存储**：hash 链按页分组（`RadixKey.page_aligned`），page=16 的 key 与 page=1 的旧 `.bin` 文件不兼容 → 每档 `rm -rf /data1/hicache_v2/*` 再起 server。
2. **delay 旋钮是"每页 sleep"**：`HiCacheFile.get()` 内 `time.sleep(delay)` 每页一次。page=16 时 delay=100us 的等效带宽 = 1.12MB/148μs ≈ **7.6GB/s**（不再是 0.34GB/s 的中速语义）。若保持等效 0.34GB/s 需 `delay × page_size`（=1600us）→ 放轮2。
3. **attention 后端强制改写 page_size**：FlashMLA=64 / Cutlass MLA=128 / TRT-LLM MLA=32|64（server_args.py:4660-4705）。起 server 后 **grep 日志确认实际 page_size**，若被改写为 64 直接按 64 跑。
4. **prefetch 限流按 tokens**：`prefetch_rate_limited()` 在 `prefetch_tokens_occupied >= 0.5×host_pool` 时拒发。64K 前缀 65536 tokens 接近临界，观察 sweep 里 `l3_loaded` 是否 = 前缀长度（不完整=被限流截断）。
5. **race 会合粒度变粗**：claimed_upto / fetched_from 按页推进，会合误差 ≤ page_size tokens（16-64 tokens，代价 ≤1 页 IO 时间，可忽略）；chunk 8192 是页的整倍数，边界对齐无虞。
6. **host 大块连续内存**：page=16 单批 143MB、page=64 单批 573MB。分配失败会写 server 日志（`Failed to allocate`），届时降 batch 或改 page=32。
7. **客户端长度整除**：probe/sweep/fluct/realdata 全部长度（8192/16384/32768/49152/65536）均被 16/32/64 整除 ✓（fluct_test.py 的 PAGE=64 断言对 16/32 也成立）。

## 4. 测试矩阵与执行顺序

### 轮1（核心验证，PAGE=16）

| 顺序 | server 组合 | 客户端 | 备注 |
|---|---|---|---|
| 1a | file0 × wait_complete | probe 8K n=24 + sweep {16,32,64}K ×6 | 带宽上限基准 |
| 1b | file0 × suffix_race | 同上 | race 大页行为 |
| 1c | file0 × best_effort | 同上 | **控制组：必须与 page=1 基线一致** |
| 1d | rdma × wait_complete | probe 8K n=32 + sweep {16,32,48,64}K ×8 | mooncake 上限 |
| 1e | rdma × suffix_race | 同上 | race 大页行为 |
| 1f | rdma × best_effort | 同上 | 控制组 |

每个组合 = 1 个 server（`start_server clean`）+ probe + sweep，沿用 `bigrun.sh` 骨架，注入 `PAGE=16`。约 6 轮 server 起停。

### 轮2（可选，确认边际与中速语义）

| 组合 | 目的 |
|---|---|
| PAGE=64 × {file0, rdma} × {wc, race}，仅扫 64K | 确认 16→64 边际收益是否封顶（预期无增益，验证"已到盘/NIC 上限"） |
| PAGE=16 × file100（delay=1600us）× {wc, race} × {32K, 64K} | 中速档等效 0.34GB/s 语义保持，验证慢盘+大页下 race 策略排序不变 |

## 5. 脚本改动清单（最小，不碰核心逻辑）

1. **`start_dsv3.sh`**：加 `PAGE` 环境变量（默认 1），在 launch_server 参数中注入 `--page-size ${PAGE}`。
2. **`probe_v2.py`**：`PAGE` 常量仅用于前缀长度断言，8192 对 16/32/64 均整除，无需改（若常量=1 断言恒真也无需改）。
3. **`sweep_v2.py` / `realdata_test.py` / `fluct_test.py`**：长度均已整除，无需改。
4. **`bigrun.sh`**：外层循环注入 `PAGE=16` 环境变量；输出文件名加 page 标注（如 `sweep_p16_file0_wait_complete.json`）避免与旧数据混淆。

## 6. 预期结果对照表

| 后端 | page | 64K wc prefetch_dur | 64K wc TTFT(估) | 依据 |
|---|---|---|---|---|
| file0 | 1 | 3.13s（实测基线） | ~3.3s | sweep_file0_wait_complete |
| file0 | 16 | ~1.31s | ~1.5s | 3.5GB/s 盘顶 + loadback |
| rdma | 1 | 1.08s（实测基线） | ~1.4s | sweep_rdma_wait_complete |
| rdma | 16 | ~0.38s | ~0.55s | ~12GB/s loopback 顶 |
| be（任意） | — | 不读 L3 | 15.2s（不变） | 控制组 |

> 注意：64K wc TTFT 里包含 host→device loadback 时间（约 0.1-0.3s），prefetch_dur 只含 L3→host 段，**带宽判定用 prefetch_dur**。

## 7. 数据归集与分析方法

1. 复用现有聚合脚本（`aggregate_bigrun_A.py`），backend 标注加 page 维度（p16_file0 等）。
2. 带宽指标：`l3bw = l3_loaded × 71680 / prefetch_dur`，与 §1 表格逐行对照。
3. race 行为：`prefetch_ttft` vs `recompute_ttft` vs wc 行对比；若 race ≈ wc（差距 <5%）→ 支持"带宽充裕后 race 优势收窄"假设。
4. 控制组校验：be 行的 TTFT 与 page=1 基线差 <2% 才算装置自洽，否则排查 server 配置差异。
5. 产出：更新 RESULTS_SUMMARY.md 加"page=16"区块。

## 8. 执行入口（沿用 DEV_TEST_WORKFLOW）

```bash
# 本地改 start_dsv3.sh + bigrun.sh（加 PAGE 注入）→ commit → push → 远端 pull → 容器 py_compile 校验
# 容器内：
PAGE=16 bash bigrun.sh          # 轮1（先查 GPU：nvidia-smi 最空卡 ≥60GB）
# 轮2：
PAGE=64 bash bigrun_short.sh    # 只扫 64K
bash bigrun_mid.sh              # file100 delay=1600us
```
