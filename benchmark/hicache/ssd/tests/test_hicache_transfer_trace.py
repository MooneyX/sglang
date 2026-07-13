import json
import sys
import tempfile
import unittest
from pathlib import Path

SRT_DIR = Path(__file__).resolve().parents[3] / "python"
sys.path.insert(0, str(SRT_DIR))

from sglang.srt.observability.hicache_transfer_trace import (
    HiCacheTransferTracer,
    transfer_event,
)


class TransferEventTest(unittest.TestCase):
    def test_transfer_metrics(self):
        event = transfer_event(
            operation="prefetch",
            operation_id=7,
            request_id="req-1",
            status="completed",
            enqueue_ns=1_000_000,
            io_start_ns=2_000_000,
            io_end_ns=6_000_000,
            requested_tokens=8,
            completed_tokens=6,
            page_size=2,
            bytes_per_token=1024,
            storage_backend="HiCacheFile",
        )
        self.assertEqual(event["queue_ms"], 1.0)
        self.assertEqual(event["io_ms"], 4.0)
        self.assertEqual(event["total_ms"], 5.0)
        self.assertEqual(event["completed_pages"], 3)
        self.assertEqual(event["completed_bytes"], 6144)
        self.assertAlmostEqual(event["throughput_mib_s"], 1.46484375)

    def test_jsonl_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            tracer = HiCacheTransferTracer(str(path))
            tracer.emit({"event": "test", "value": 3})
            tracer.close()
            data = json.loads(path.read_text())
            self.assertEqual(data["schema_version"], 1)
            self.assertEqual(data["event"], "test")
            self.assertEqual(data["value"], 3)
            self.assertIn("timestamp_ns", data)


if __name__ == "__main__":
    unittest.main()
