import sys
import tempfile
import unittest
from pathlib import Path
from pathlib import Path

SSD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SSD_DIR))

from analyze import DEVICE_METRICS, correlate_requests
from bandwidth import (
    CgroupReadBandwidthController,
    BandwidthSegment,
    format_io_max_line,
    parse_io_max_line,
    parse_schedule,
    schedule_mean_mib_s,
    validate_schedule_mean,
)
from monitor import calculate_sample, parse_diskstats
from workload import RequestRecord, WorkloadConfig, build_workload


class WorkloadTest(unittest.TestCase):
    def test_measurement_reuses_fill_prefixes(self):
        config = WorkloadConfig(
            num_clients=3,
            prefix_tokens=8,
            eviction_prompts=4,
            eviction_tokens=6,
            output_tokens=2,
            offload_output_tokens=4,
            max_concurrency=2,
            seed=7,
        )
        phases = build_workload(config, [10, 20, 30, 40])

        self.assertEqual(len(phases["fill"]), 3)
        self.assertEqual(len(phases["evict"]), 4)
        self.assertEqual(
            [request.input_ids for request in phases["fill"]],
            [request.input_ids for request in phases["measure"]],
        )
        self.assertNotEqual(
            phases["fill"][0].input_ids,
            phases["evict"][0].input_ids,
        )

    def test_workload_is_deterministic(self):
        config = WorkloadConfig(2, 4, 2, 4, 1, 2, 1, seed=9)
        first = build_workload(config, [1, 3, 7])
        second = build_workload(config, [1, 3, 7])
        self.assertEqual(first, second)

    def test_request_record_derived_times(self):
        record = RequestRecord(
            request_id="r",
            phase="measure",
            client_id=0,
            scheduled_ns=900,
            submitted_ns=1000,
            first_token_ns=2_001_000,
            completed_ns=5_001_000,
            itl_ns=[1_000_000],
            success=True,
        )
        data = record.to_dict()
        self.assertEqual(data["ttft_ms"], 2.0)
        self.assertEqual(data["e2e_ms"], 5.0)
        self.assertEqual(data["itl_ms"], [1.0])


class DiskstatsTest(unittest.TestCase):
    def test_parse_and_calculate(self):
        line = "259 0 nvme1n1 100 0 200 300 400 0 800 900 2 1000 1100 0 0 0 0 5 6"
        previous = parse_diskstats([line], "/dev/nvme1n1")
        current = dict(previous)
        current.update(
            reads_completed=110,
            sectors_read=2248,
            read_ms=350,
            writes_completed=420,
            sectors_written=4896,
            write_ms=980,
            io_in_progress=3,
            io_ms=1500,
        )
        sample = calculate_sample(previous, current, 2.0, 100, 123)
        self.assertEqual(sample.read_iops, 5.0)
        self.assertEqual(sample.write_iops, 10.0)
        self.assertEqual(sample.read_mib_s, 0.5)
        self.assertEqual(sample.write_mib_s, 1.0)
        self.assertEqual(sample.read_await_ms, 5.0)
        self.assertEqual(sample.write_await_ms, 4.0)
        self.assertEqual(sample.queue_depth, 3)
        self.assertEqual(sample.util_pct, 25.0)

    def test_missing_device(self):
        with self.assertRaises(ValueError):
            parse_diskstats([], "nvme9n9")


class AnalysisTest(unittest.TestCase):
    def test_correlate_measurement_window(self):
        requests = [
            {
                "request_id": "measure-client-0000",
                "phase": "measure",
                "client_id": 0,
                "submitted_ns": 100,
                "first_token_ns": 300,
                "ttft_ms": 0.0002,
                "prompt_tokens": 8,
                "cached_tokens": 8,
                "cached_tokens_details": {"storage": 8},
                "success": True,
            },
            {"request_id": "fill", "phase": "fill", "success": True},
        ]
        samples = [
            {
                "interval_start_ns": 0,
                "interval_end_ns": 50,
                **{metric: 1.0 for metric in DEVICE_METRICS},
            },
            {
                "interval_start_ns": 50,
                "interval_end_ns": 150,
                **{metric: 2.0 for metric in DEVICE_METRICS},
            },
            {
                "interval_start_ns": 150,
                "interval_end_ns": 250,
                **{metric: 4.0 for metric in DEVICE_METRICS},
            },
        ]
        transfers = [
            {
                "event": "hicache_storage_transfer",
                "operation": "prefetch",
                "request_id": "measure-client-0000",
                "io_ms": 0.15,
                "completed_bytes": 8192,
                "throughput_mib_s": 52.0833333333,
            }
        ]
        rows = correlate_requests(requests, samples, transfers)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["storage_cached_tokens"], 8)
        self.assertEqual(rows[0]["device_samples"], 2)
        self.assertEqual(rows[0]["mean_read_iops"], 3.0)
        self.assertEqual(rows[0]["max_read_iops"], 4.0)
        self.assertEqual(rows[0]["hicache_prefetch_count"], 1)
        self.assertEqual(rows[0]["hicache_prefetch_io_ms"], 0.15)
        self.assertEqual(rows[0]["hicache_prefetch_bytes"], 8192)


class BandwidthControlTest(unittest.TestCase):
    def test_weighted_schedule_mean(self):
        segments = parse_schedule(
            [
                {"duration_seconds": 1, "read_mib_s": 100},
                {"duration_seconds": 3, "read_mib_s": 700},
            ]
        )
        self.assertEqual(schedule_mean_mib_s(segments), 550)
        self.assertEqual(validate_schedule_mean(segments, 550), 550)
        with self.assertRaises(ValueError):
            validate_schedule_mean(segments, 500)

    def test_io_max_round_trip(self):
        number, values = parse_io_max_line(
            "259:0 rbps=524288000 wbps=max riops=max wiops=max"
        )
        self.assertEqual(number, "259:0")
        self.assertEqual(values["rbps"], "524288000")
        self.assertEqual(
            format_io_max_line(number, values),
            "259:0 rbps=524288000 wbps=max riops=max wiops=max",
        )

    def test_controller_applies_and_restores_limit(self):
        writes = []
        with tempfile.TemporaryDirectory() as directory:
            controller = CgroupReadBandwidthController(
                cgroup_path=Path(directory),
                device_number="259:0",
                segments=[BandwidthSegment(60, 500)],
                output_path=Path(directory) / "configured.csv",
                write_line=writes.append,
                read_lines=lambda: ["259:0 rbps=max wbps=1048576 riops=max wiops=max"],
            )
            controller.start()
            controller.stop()
        self.assertEqual(
            writes[0],
            "259:0 rbps=524288000 wbps=1048576 riops=max wiops=max",
        )
        self.assertEqual(
            writes[-1],
            "259:0 rbps=max wbps=1048576 riops=max wiops=max",
        )


if __name__ == "__main__":
    unittest.main()
