# HiCache SSD Variability Benchmark

This directory contains the first-stage benchmark for observing natural SSD
variability while SSD-backed HiCache is on the inference critical path. It does
not inject artificial I/O interference.

## Workload

The experiment runs three deterministic phases:

1. `fill`: submit one reusable prefix per client.
2. `evict`: run unrelated decode work to finish write-through offload.
3. Flush GPU and host cache while preserving the file backend.
4. `measure`: resubmit the original prefixes and record whether tokens came
   from device, host, or storage.

Every request is written to `requests.jsonl` with monotonic timestamps for
submission, first token, completion, TTFT, E2E latency, ITL, and cache-source
breakdown.

## Start SGLang

The exact memory sizes depend on the model and test machine. A representative
configuration is:

```bash
mkdir -p /root/data/sglang-hicache
SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/root/data/sglang-hicache \
SGLANG_HICACHE_TRANSFER_TRACE_PATH=/tmp/hicache-transfers.{pid}.jsonl \
sglang serve \
  --model-path <model> \
  --enable-hierarchical-cache \
  --hicache-size 1 \
  --hicache-storage-backend file \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config '{"hicache_storage_pass_prefix_keys": true}' \
  --hicache-write-policy write_through \
  --enable-cache-report
```

The built-in file backend currently reads its directory from
`SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR`; `--file-storage-path` is not wired to
this backend in the current source revision. A small host HiCache makes storage
promotion easier to trigger during the
functional test. Keep `prefix_tokens` above the configured storage prefetch
threshold; the current default threshold is 256 tokens. Tune `--hicache-size` and workload pressure on the formal test
machine rather than treating this example as a final experiment setting.

## Configure

Copy `config.example.json` and update:

- `server.model`: model or tokenizer path.
- `storage.path`: the HiCache directory.
- `storage.device`: block device such as `/dev/nvme1n1` on the formal test
  machine. Keep it `null` on Lustre or when no block device maps uniquely to the
  path.
- `storage.transfer_trace_glob`: glob matching scheduler trace files, for
  example `/tmp/hicache-transfers.*.jsonl`. The experiment filters them to the current run
  time window and concatenates matching events into `hicache_transfers.jsonl`.
- `workload.eviction_prompts` and `eviction_tokens`: pressure needed to move the
  original prefixes out of GPU and host cache.

Verify the formal SSD mapping separately:

```bash
findmnt -T /mnt/local-nvme
lsblk -o NAME,MAJ:MIN,TYPE,SIZE,ROTA,MODEL,SERIAL,MOUNTPOINTS
```

## Run

```bash
python3 benchmark/hicache/ssd/experiment.py \
  --config benchmark/hicache/ssd/config.example.json
```

Each run creates:

```text
results/<UTC-run-id>/
├── config.json
├── manifest.json
├── requests.jsonl
├── device.csv       # only when storage.device is configured
├── hicache_transfers.jsonl  # when transfer_trace_glob matches files
├── summary.json
└── request_storage_windows.csv  # created by analyze.py
```

On the independent SSD machine, correlate each request's submission-to-first-token
window with device samples:

```bash
python3 benchmark/hicache/ssd/analyze.py <results/run-id>
```

The run fails intentionally when the measurement phase contains no storage
cache hits. In that case, increase eviction pressure or verify the SGLang
HiCache arguments.

## Current limitations

- `/proc/diskstats` reports device-wide interval averages, not per-I/O tail
  latency. Add an eBPF block trace collector on the formal SSD machine for
  request-level latency distributions.
- The block device must be dedicated to the experiment or device metrics can
  include unrelated traffic.
- The development machine's `/root/data` is Lustre, so `storage.device` should
  remain `null`; it validates request and result plumbing only.

## HiCache transfer events

Set `SGLANG_HICACHE_TRANSFER_TRACE_PATH` before starting SGLang. The value may
contain `{pid}`, `{rank}`, and `{local_rank}` placeholders, which avoids
multiple scheduler processes writing the same file. Events also include PID,
rank, and local rank. Each JSONL event represents
one logical storage prefetch or backup operation, rather than one event per
page. Fields include queue time, storage I/O time, requested/completed tokens,
bytes, effective MiB/s, status, backend type, and request ID for prefetches.

Example:

```json
{"operation":"prefetch","request_id":"measure-client-0000","io_ms":42.1,"completed_bytes":67108864,"throughput_mib_s":1520.2,"status":"completed"}
```

Backup operations currently have no request ID because write-through backups
are attached to radix-cache nodes after request execution. They remain useful
for separating foreground prefetch reads from background storage writes.

## Stable versus fluctuating bandwidth experiment

The paired experiment tests whether storage-rate variability causes additional
inference degradation when the configured time-average read capacity is held
constant. Bandwidth control is enabled only for the `measure` phase. The
`fill` and `evict` phases run without a read limit, so the controlled factor is
the storage service-rate trace seen by foreground HiCache prefetches.

The implementation uses cgroup v2 `io.max`. It must run on the independent SSD
machine; it is not valid for the development machine's Lustre mount. First
identify the dedicated SSD and create a cgroup with the I/O controller enabled:

```bash
findmnt -T /mnt/local-nvme -o TARGET,SOURCE,FSTYPE,MAJ:MIN
lsblk -o NAME,MAJ:MIN,TYPE,SIZE,ROTA,MODEL,MOUNTPOINTS
sudo benchmark/hicache/ssd/setup_cgroup.sh /sys/fs/cgroup/sglang-hicache
```

Start SGLang inside that cgroup. Environment variables and server arguments are
otherwise the same as in the normal HiCache example:

```bash
sudo benchmark/hicache/ssd/run_in_cgroup.sh \
  /sys/fs/cgroup/sglang-hicache \
  env \
    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/mnt/local-nvme/hicache \
    SGLANG_HICACHE_TRANSFER_TRACE_PATH='/tmp/hicache-transfers.{pid}.jsonl' \
  sglang serve \
    --model-path <model> \
    --enable-hierarchical-cache \
    --hicache-size 1 \
    --hicache-storage-backend file \
    --hicache-storage-prefetch-policy wait_complete \
    --hicache-storage-backend-extra-config \
      '{"hicache_storage_pass_prefix_keys": true}' \
    --hicache-write-policy write_through \
    --enable-cache-report
```

Verify that the SGLang PID is a member of the cgroup before running:

```bash
cat /sys/fs/cgroup/sglang-hicache/cgroup.procs
cat /proc/<sglang-pid>/cgroup
```

Copy both example configurations. In the base configuration:

- Set `storage.path` to the HiCache directory on the dedicated SSD.
- Set `storage.device` to the SSD block device for diskstats monitoring.
- Set `storage.transfer_trace_glob` to the scheduler trace path.
- Keep `workload.minimum_storage_hit_fraction` at `1.0` for a strict
  comparison. A run fails if not every successful measurement request hits L3.

In `paired_config.example.json`, set the same `target_read_mib_s` for every
condition. Each schedule uses a duration-weighted mean, so the example stable
schedule is constantly 500 MiB/s while the fluctuating schedule alternates
between 100 and 900 MiB/s for equal durations. Both have a configured mean of
500 MiB/s. `randomize_start_segment` prevents every repetition from aligning
request submission with the low-rate segment.

Run the paired experiment as a user that can write the cgroup's `io.max` file:

```bash
sudo -E python3 benchmark/hicache/ssd/paired_experiment.py \
  --config benchmark/hicache/ssd/paired_config.example.json
```

Each condition starts by clearing GPU/host cache and the HiCache storage
backend, then rebuilds the same deterministic prefixes. Condition order is
randomized independently in every repetition. The output includes one
`configured_bandwidth.csv` per run, recording every applied rate transition,
and `configured_bandwidth.summary.json`, containing the duration-weighted mean
over the actual measurement window. A run fails when that observed configured
mean differs from the target by more than
`measurement_mean_relative_tolerance`. The controller restores the original
`io.max` setting after measurement, including when the workload raises an
exception.

After all repetitions finish, generate paired statistics and figures:

```bash
python3 benchmark/hicache/ssd/compare_paired.py \
  benchmark/hicache/ssd/paired-results/<UTC-run-id> \
  --baseline stable \
  --experiment fluctuating
```

The comparison produces:

```text
paired-results/<UTC-run-id>/
├── paired_config.json
├── paired_runs.json
├── paired_summary.json
├── run_metrics.csv
├── paired_differences.csv
├── comparison_summary.json
├── paired_comparison.png
├── paired_comparison.svg
└── runs/rep-<n>-<order>-<condition>/
    ├── configured_bandwidth.csv
    ├── configured_bandwidth.summary.json
    ├── device.csv
    ├── hicache_transfers.jsonl
    ├── requests.jsonl
    └── summary.json
```

The primary causal comparison is the within-repetition difference
`fluctuating - stable`, especially for TTFT P95/P99 and prefetch queue P95.
`comparison_summary.json` reports the mean paired difference and a paired
bootstrap 95% confidence interval. The configured mean capacity being equal
does not imply the observed transfer throughput will be equal: idle periods,
queueing, and application demand affect achieved throughput, so report both the
configured trace and observed HiCache/device metrics.
