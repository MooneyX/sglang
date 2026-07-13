import csv
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

DISKSTAT_FIELDS = (
    "reads_completed",
    "reads_merged",
    "sectors_read",
    "read_ms",
    "writes_completed",
    "writes_merged",
    "sectors_written",
    "write_ms",
    "io_in_progress",
    "io_ms",
    "weighted_io_ms",
    "discards_completed",
    "discards_merged",
    "sectors_discarded",
    "discard_ms",
    "flushes_completed",
    "flush_ms",
)


def parse_diskstats(lines: Iterable[str], device: str) -> Dict[str, int]:
    target = os.path.basename(device)
    for line in lines:
        parts = line.split()
        if len(parts) < 14 or parts[2] != target:
            continue
        values = [int(value) for value in parts[3:]]
        return {
            field: values[index] if index < len(values) else 0
            for index, field in enumerate(DISKSTAT_FIELDS)
        }
    raise ValueError(f"Device {target!r} not found in /proc/diskstats")


def read_diskstats(device: str, path: str = "/proc/diskstats") -> Dict[str, int]:
    with open(path) as file:
        return parse_diskstats(file, device)


@dataclass(frozen=True)
class DiskstatsSample:
    interval_start_ns: int
    interval_end_ns: int
    elapsed_seconds: float
    read_iops: float
    write_iops: float
    read_mib_s: float
    write_mib_s: float
    read_await_ms: Optional[float]
    write_await_ms: Optional[float]
    queue_depth: int
    util_pct: float


def calculate_sample(
    previous: Dict[str, int],
    current: Dict[str, int],
    elapsed_seconds: float,
    interval_start_ns: int,
    interval_end_ns: int,
) -> DiskstatsSample:
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")

    reads = current["reads_completed"] - previous["reads_completed"]
    writes = current["writes_completed"] - previous["writes_completed"]
    sectors_read = current["sectors_read"] - previous["sectors_read"]
    sectors_written = current["sectors_written"] - previous["sectors_written"]
    read_ms = current["read_ms"] - previous["read_ms"]
    write_ms = current["write_ms"] - previous["write_ms"]
    io_ms = current["io_ms"] - previous["io_ms"]

    return DiskstatsSample(
        interval_start_ns=interval_start_ns,
        interval_end_ns=interval_end_ns,
        elapsed_seconds=elapsed_seconds,
        read_iops=reads / elapsed_seconds,
        write_iops=writes / elapsed_seconds,
        read_mib_s=sectors_read * 512 / 2**20 / elapsed_seconds,
        write_mib_s=sectors_written * 512 / 2**20 / elapsed_seconds,
        read_await_ms=read_ms / reads if reads else None,
        write_await_ms=write_ms / writes if writes else None,
        queue_depth=current["io_in_progress"],
        util_pct=min(100.0, io_ms / (elapsed_seconds * 1000) * 100),
    )


class DiskstatsMonitor:
    def __init__(self, device: str, output_path: Path, interval_seconds: float):
        self.device = device
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        read_diskstats(self.device)
        self._thread = threading.Thread(target=self._run, name="diskstats-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(DiskstatsSample.__dataclass_fields__)
        with self.output_path.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            previous = read_diskstats(self.device)
            previous_ns = time.monotonic_ns()
            while not self._stop_event.wait(self.interval_seconds):
                current_ns = time.monotonic_ns()
                current = read_diskstats(self.device)
                sample = calculate_sample(
                    previous,
                    current,
                    (current_ns - previous_ns) / 1e9,
                    previous_ns,
                    current_ns,
                )
                writer.writerow(sample.__dict__)
                output_file.flush()
                previous = current
                previous_ns = current_ns
