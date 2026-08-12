# SSD 带宽与单机 Mooncake 带宽分析报告

> 对象：SuffixPrefetchV2（sglang fork，hicache file backend + mooncake 单机后端，DeepSeek-V3 MLA 70KB/token，page_size=1）
> 依据：`fio_bench/`（8-06 硬件基准）、`bigrun/sweep_*.json`（端到端 wait_complete 行）、`mc_rdma_bw.py`（mooncake 探针）、最终代码 f58af3d3c

---

## 1. 现状量化：硬件可达 vs 实际达到

### 1.1 SSD（file 后端，/data1/hicache_v2）

| 场景 | 带宽 | 来源 |
|---|---|---|
| fio 顺序 1MB，depth32×4jobs | **3.47 GB/s** | `fio_bench/seq_1m_d32_j4.json` |
| fio 随机 64K，depth32×4jobs | 3.41 GB/s | `rand_64k_d32_j4.json` |
| fio 缓冲 70K 顺序，depth1/8（模拟页大小） | **3.13~3.18 GB/s** | `buf_seq_70k_*.json` |
| fio 随机 64K，depth8 单流 | 2.35 GB/s | `rand_64k_d8_j1.json` |
| fio 随机 64K，depth1 单流（≈当前代码 IO 模式） | **0.31~0.33 GB/s** | `rand_64k_d1_j1.json` |
| fio 顺序 64K，depth1（direct） | 0.71 GB/s | `seq_64k_d1_j1.json` |
| **sglang 端到端预取（file0）** | **1.38~1.50 GB/s**（45~48μs/token） | `sweep_file0_wait_complete.json` 反推 |

结论：
- 该盘经 fio 测得的**实际可达上限 ≈ 3.5 GB/s**（深度队列），sglang 只用到 **1.5 GB/s（约 43%）**。
- 若硬件标称更高（如 PCIe4 NVMe 7GB/s），则 fio 本身也只摸到 3.5——需先确认设备（见 §4 验证），本文以 fio 上限 3.5 为“可达到”基准。
- 关键对照：**同样的盘，fio 单流随机 depth1 只有 0.33GB/s，深度队列才有 2.35~3.4GB/s** —— IO 模式（队列深度）是决定性变量。

### 1.2 单机 Mooncake（master 127.0.0.1:50052，RDMA loopback）

| 场景 | 带宽 | 来源 |
|---|---|---|
| `mc_rdma_bw.py` 探针：70KB×8255 页，batch=1024 | **put/get ≈ 12.5 GB/s** | 探针实测（注释记录） |
| 同上，单流/小批量 | **~4.7 GB/s**（单流） | 同上 |
| **sglang 端到端预取（rdma）** | **4.31~4.37 GB/s**（~15μs/token） | `sweep_rdma_wait_complete.json` 反推 |
| mooncake TCP 单机 | 0.56 GB/s | 探针实测 |

结论：
- 单机 loopback RDMA 探针最高 **12.5GB/s**，sglang 端到端只到 **4.36GB/s（约 35%）**，正好等于“单流”水平。
- 探针里 batch=1024 与单流之间 4.7→12.5 的差距说明：**批量大小直接决定吞吐**，sglang 用的是 128。
- TCP 只有 0.56GB/s —— 传输引擎在 TCP 下几乎无流水线，属引擎自身问题，不是物理带宽。

### 1.3 带宽缺口汇总

| 后端 | 硬件/探针可达 | sglang 实际 | 缺口 |
|---|---|---|---|
| SSD file | ~3.5 GB/s（fio 深队列） | ~1.5 GB/s | **2.3×** |
| mooncake 单机 | ~12.5 GB/s（探针 B=1024） | ~4.36 GB/s | **2.9×** |

---

## 2. 根因分析（代码级）

### 2.1 file 后端（SSD）—— 四个叠加瓶颈

**① 物理随机访问：每页一个独立文件（根因，最大）**
- `hicache_storage.py` 每页 = 一个 `<sha256链>.bin` 文件（`set()` 写 tmp+rename，`get()` 按名打开）。
- 文件名是内容哈希链，**页的磁盘物理位置无任何顺序保证**——逻辑顺序前缀在盘上是随机散布的。
- 后果：page-cache 热时读 70KB 只花内存拷贝时间（fio 缓冲顺序 3.1GB/s），**冷缓存时退化成随机读，内核 readahead 完全失效**（fio 随机 depth1 仅 0.33GB/s，与顺序差 10×）。
- 65K 页 = 同一目录 65K 个文件：inode/dentry 缓存压力、`exists`/`scandir`（`_collect_existing_component_keys`）整目录扫描。

**② 队列深度 = 1：同步串行单页读**
- `_batch_io_v2`（hicache_storage.py:639）对每个 key 串行调 `_read_page`→`get()`→`os.preadv`，**同一时刻只有一个 outstanding IO**。
- 单页 70KB，NVMe 随机读延迟 ~200μs → 理论上限 70KB/200μs = **0.35GB/s**（与 fio depth1 吻合）；当前 1.5GB/s 是靠 page-cache 热 + FD 缓存撑出来的。
- fio 证明：depth 8 → 2.35GB/s，depth 32 → 3.4GB/s，**必须多 IO 在飞才能喂满盘**。

**③ 每页 Python/系统调用开销（解释 1.5 vs fio 3.1 的差额）**
- `get()`（hicache_storage.py:434-455）每页执行：
  1. `_open_cached()`：dict 锁 + `move_to_end`；
  2. `_current_read_delay()`：**每页一次 `os.stat`**（即使 delay 文件不存在也发起一次失败的 stat 系统调用）；
  3. `os.preadv` + memoryview/numpy 转换；
  4. `_evictor.touch()`（本轮未配 eviction 是 no-op，可忽略）。
- 每页固定开销 ~20-30μs，而 70KB@3.1GB/s 只需 22μs/页 → **一半以上的页时间花在 Python 层**。

**④ 双拷贝**
- `_read_page`（hicache_storage.py:602）先读到 `get_dummy_flat_data_page()`，再 `set_from_flat_data_page()` 拷进真实 slot —— 每页多一次 70KB 内存拷贝。

**⑤ 单 IO 线程全局串行（并发场景）**
- `cache_controller.py` 只有 1 个 `prefetch_io_aux_thread`，`prefetch_buffer` 一次取一个 op 完整跑完才处理下一个 → **多个并发请求的预取严格排队**（设计文档实测：8 并发阶梯 0.87→5.11s，Δ≈0.6s），无法靠多请求并行流叠带宽。

### 2.2 mooncake 单机 —— 批量与并发不足

**① 单机 = RDMA loopback，本身有物理上限**
- 单机部署（master 127.0.0.1）下，put/get 都是进程内自注册内存上的 **loopback RDMA**，数据实际不离开本机。
- loopback 吞吐通常只有线速 ~50%（例如 200G 网卡 ~12-13GB/s 常见）——**12.5GB/s 很可能已接近该卡 loopback 物理上限**（需 ibstat/ib_write_bw 确认线速，见 §4）。
- 所以 mooncake 的“硬件带宽”应理解为**探针可达 12.5GB/s**，而不是网卡线速。

**② 批量大小敏感：sglang 用 128 页/批**
- `cache_controller._page_transfer` 按 `STORAGE_BATCH_SIZE=128` 分批（每批 8.75MB），批次之间串行等待。
- 探针 B=1024（70MB/批）才到 12.5GB/s；128 页/批落到 ~4.4GB/s。每批固定开销（元数据查询、WR 提交、CQ 轮询、跨 TP allreduce 等）在 128 页下摊不掉。
- 端到端 4.36GB/s 恰好等于“单流”水平，说明当前调用模式没有把引擎的并发能力用起来。

**③ 无多流并发**
- 单请求只有一个预取流，多请求又被单 IO 线程串行化；没有把 128 页/批再拆成多个并发子批并行下发。

### 2.3 共性：预取与 loadback 不重叠（wait_complete 路径）
- wait_complete 是“全量取完 → 再 host→device 拷入 GPU”，取数与拷贝严格串行，PCIe 与 NIC/磁盘空档交替，TTFT 里两段时间相加。suffix_race 的会合机制已缓解，但 L3→host 段的带宽瓶颈依旧。

---

## 3. 改进方案

> 优先级排序：P0 改动小、立刻见效；P1 结构改造、收益最大；P2 系统级/验证。

### P0 —— 不改布局，先把 IO 流水线喂满（预计 file 1.5→2.3~3.4GB/s，mooncake 4.4→12GB/s）

1. **file：批内并发读（队列深度 8~32）**
   - 把 `_batch_io_v2` 的串行循环改成**线程池并发**（4~8 个 worker 各 `preadv` 一个页，`ThreadPoolExecutor` + `asyncio`/`concurrent.futures` 皆可），或改用 `io_uring`/`libaio`。
   - 批内页数从 128 提到 512~1024（`STORAGE_BATCH_SIZE`），让每批有足够在飞 IO。
   - 预期：直接命中 fio depth8-32 的 2.35~3.4GB/s 区间。
   - 注意 reverse（从尾取）语义：批内并发读完成后按原序计数 `completed_from_end`，顺序无关不影响正确性（页落位后统一 increment）。

2. **file：去掉每页 `os.stat`（delay 旋钮检查改到 batch 粒度）**
   - `get()` 里的 `_current_read_delay()` 每页一次 stat；把它移到 `batch_get`/`_batch_io_v2` 层，每批只查一次。纯节省 ~1-2μs×页数 + 一批 syscall 风暴。
   - 同理 FD cache 的 dict 锁与 `move_to_end` 可批量换取 FD 列表后统一归还。

3. **mooncake：批量 128→1024 + 多流并发**
   - 把 `STORAGE_BATCH_SIZE` 参数化（仅 mooncake 路径用 512~1024），与探针 B=1024 对齐，预期端到端从 4.36 走向 ~12GB/s。
   - 若单批并发仍受限，把一批拆成 2~4 个子批**并行下发**（并发线程各调 `batch_get_into`，全部完成后统一 increment）；探针的多 QP/大批量已证明引擎支持 12.5GB/s。

4. **预取线程隔离 + 多 worker**
   - `prefetch_io_aux_thread` 是单 Python 线程，与 scheduler/GPU 调度线程抢 GIL 和 CPU（实测 race 模式下预取速率 111μs/tok vs 空闲 48μs/tok，2.3× 劣化）。
   - 用 `taskset`/`affinity` 把预取线程钉到专用核，并把 IO 执行体从 1 个扩到 2~4 个（每请求一流，全局限流）。

### P1 —— 结构改造（收益最大，file 冷读也有 3GB/s+）

5. **file 布局：单文件 append-only 日志 + 内存偏移索引（推荐）**
   - 所有页写进**一个大文件**（或每 N 页一个 slab 文件），页间按写入顺序连续排列；内存维护 `hash → (fd, offset, len)` 索引（进程内 dict，启动时扫一次重建）。
   - 收益：
     - 顺序前缀的页在磁盘上**物理连续** → 冷缓存也触发内核 readahead（fio 缓冲顺序 = 3.1GB/s，随机 = 0.33，**布局本身 10× 差距**）；
     - 消除 65K 文件的 inode/dentry 开销与 scandir 全目录扫描；
     - 一个 FD 覆盖全部页，FD cache/打开开销归零。
   - 配套：写路径 `O_APPEND` 追加（当前 tmp+rename 也去掉）；可选 `posix_fadvise(WILLNEED)` 在预取指针前 1MB 预读。
   - 内容寻址语义不变（hash 仍是 key），只改物理布局——`radix_cache.py`、存储接口零改动。

6. **零拷贝读：直接读入 host slot**
   - file 后端仿照 mooncake v1 的 `batch_get_v1`（直接给 buffer ptr），去掉 `get_dummy_flat_data_page` + `set_from_flat_data_page` 双拷贝。

7. **loadback 流水线化**
   - 每完成一批预取立即把该批 host→device（不等全前缀），与下一批 L3 取数重叠，消除“取完再拷”的空档（wait_complete 路径 TTFT 直接受益；suffix_race 会合机制天然兼容）。

### P2 —— 系统验证与确认

8. **先确认硬件真实上限（一次性诊断）**：
   - SSD：`nvme list` / `lspci | grep -i nvme` / `iostat -x 1` 确认设备与队列；若标称 7GB/s 但 fio 只有 3.5，检查是否 RAID/网络盘、直连 vs HBA、`/sys/block/*/queue/nr_requests`。
   - mooncake：`ibstat` 看 HCA 型号与线速；`ib_write_bw -d mlx5_bond_1 --loopback` 测 loopback 单流/多流上限，判定 12.5GB/s 是否已是物理天花板。
   - 之后把“探针/硬件可达”当作改进目标，逐项对齐。
9. **回归验证**：每步改动后用 `sweep_*.json` 同法复测，指标 `l3bw = l3_loaded × 70KB / prefetch_dur`（现成字段），与 §1 表格逐行对照。

### 预期收益总表

| 动作 | 后端 | 当前 | 预期 | 依据 |
|---|---|---|---|---|
| P0-1 批内并发 depth 8-32 | file | 1.5 | 2.4~3.4 GB/s | fio depth8/32 |
| P0-2/4 开销削减+线程隔离 | file | 1.5 | +0.3~0.5（随上叠加） | 每页 20-30μs 开销 |
| P1-5 单文件连续布局 | file | 1.5（热）/0.3（冷） | 3.1+ GB/s（冷热同） | fio 缓冲顺序 |
| P0-3 批量 1024+多流 | mooncake | 4.4 | ~12 GB/s | 探针 B=1024 |
| P1-7 loadback 流水线 | 两者 | — | TTFT 再降（取/拷重叠） | — |

---

## 4. 一句话结论

- **SSD 1.5GB/s 的根因不是盘慢，而是“每页一个文件导致的物理随机读 + 队列深度=1 的同步单页读 + 每页 20-30μs Python 开销”三者叠加**；fio 同盘深队列 3.4GB/s 是现成的证据。
- **单机 mooncake 4.36GB/s 的根因是批量太小（128 页/批）且无多流并发**，探针 B=1024 已证明引擎能到 12.5GB/s（且那大概率就是 loopback RDMA 的物理上限）。
- 两个后端的改进方向完全一致：**把单流同步 IO 换成“深队列 + 大批量 + 多流并发”，并让物理布局连续化**。P0 即可拿到 2~3×，P1 布局改造后 file 冷读也能站上 3GB/s。
