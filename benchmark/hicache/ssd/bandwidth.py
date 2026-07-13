import csv
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class BandwidthSegment:
    duration_seconds: float
    read_mib_s: float

    @property
    def read_bytes_s(self) -> int:
        return round(self.read_mib_s * 2**20)


def parse_schedule(values: Iterable[Dict[str, Any]]) -> List[BandwidthSegment]:
    segments = [
        BandwidthSegment(
            duration_seconds=float(value["duration_seconds"]),
            read_mib_s=float(value["read_mib_s"]),
        )
        for value in values
    ]
    if not segments:
        raise ValueError("bandwidth schedule must contain at least one segment")
    for segment in segments:
        if segment.duration_seconds <= 0:
            raise ValueError("bandwidth segment duration_seconds must be positive")
        if segment.read_mib_s <= 0:
            raise ValueError("bandwidth segment read_mib_s must be positive")
    return segments


def schedule_mean_mib_s(segments: Iterable[BandwidthSegment]) -> float:
    segments = list(segments)
    total_seconds = sum(segment.duration_seconds for segment in segments)
    if total_seconds <= 0:
        raise ValueError("bandwidth schedule duration must be positive")
    return (
        sum(segment.duration_seconds * segment.read_mib_s for segment in segments)
        / total_seconds
    )


def validate_schedule_mean(
    segments: Iterable[BandwidthSegment],
    target_mib_s: float,
    relative_tolerance: float = 1e-6,
) -> float:
    actual = schedule_mean_mib_s(segments)
    allowed_error = max(abs(target_mib_s) * relative_tolerance, 1e-9)
    if abs(actual - target_mib_s) > allowed_error:
        raise ValueError(
            f"schedule mean {actual:.6f} MiB/s does not match target "
            f"{target_mib_s:.6f} MiB/s"
        )
    return actual


def resolve_device_number(device: str) -> str:
    stat_result = os.stat(device)
    if not stat_result.st_rdev:
        raise ValueError(f"Not a block or character device: {device}")
    return f"{os.major(stat_result.st_rdev)}:{os.minor(stat_result.st_rdev)}"


def parse_io_max_line(line: str) -> tuple[str, Dict[str, str]]:
    fields = line.split()
    if not fields:
        raise ValueError("empty io.max line")
    values = {}
    for field in fields[1:]:
        key, value = field.split("=", 1)
        values[key] = value
    return fields[0], values


def format_io_max_line(device_number: str, values: Dict[str, str]) -> str:
    order = ("rbps", "wbps", "riops", "wiops")
    fields = [f"{key}={values[key]}" for key in order if key in values]
    fields.extend(f"{key}={value}" for key, value in values.items() if key not in order)
    return " ".join([device_number, *fields])


class CgroupReadBandwidthController:
    def __init__(
        self,
        cgroup_path: Path,
        device_number: str,
        segments: List[BandwidthSegment],
        output_path: Path,
        start_segment_index: int = 0,
        target_read_mib_s: Optional[float] = None,
        measurement_mean_relative_tolerance: Optional[float] = None,
        write_line: Optional[Callable[[str], None]] = None,
        read_lines: Optional[Callable[[], List[str]]] = None,
    ):
        self.cgroup_path = cgroup_path
        self.device_number = device_number
        self.segments = segments
        self.output_path = output_path
        if not 0 <= start_segment_index < len(segments):
            raise ValueError("start_segment_index is outside the bandwidth schedule")
        self.start_segment_index = start_segment_index
        self.target_read_mib_s = target_read_mib_s
        self.measurement_mean_relative_tolerance = measurement_mean_relative_tolerance
        self.io_max_path = cgroup_path / "io.max"
        self._write_line = write_line or self._write_io_max
        self._read_lines = read_lines or self._read_io_max
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._original_line: Optional[str] = None
        self._error: Optional[BaseException] = None
        self._transitions: List[tuple[int, float]] = []

    def _read_io_max(self) -> List[str]:
        return self.io_max_path.read_text().splitlines()

    def _write_io_max(self, line: str) -> None:
        self.io_max_path.write_text(line + "\n")

    def _current_values(self) -> Dict[str, str]:
        for line in self._read_lines():
            number, values = parse_io_max_line(line)
            if number == self.device_number:
                self._original_line = line
                return values
        self._original_line = None
        return {"rbps": "max", "wbps": "max", "riops": "max", "wiops": "max"}

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("bandwidth controller has already started")
        values = self._current_values()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._transitions = []
        self._apply(self.segments[self.start_segment_index], values)
        self._transitions.append(
            (time.monotonic_ns(), self.segments[self.start_segment_index].read_mib_s)
        )
        self._thread = threading.Thread(
            target=self._run,
            args=(values,),
            name="hicache-bandwidth-controller",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        restore_line = self._original_line or format_io_max_line(
            self.device_number,
            {"rbps": "max", "wbps": "max", "riops": "max", "wiops": "max"},
        )
        stop_ns = time.monotonic_ns()
        self._write_line(restore_line)
        if self._error is not None:
            raise RuntimeError("bandwidth controller failed") from self._error
        summary = self._write_measurement_summary(stop_ns)
        if (
            self.measurement_mean_relative_tolerance is not None
            and self.target_read_mib_s is not None
            and summary["relative_error"] > self.measurement_mean_relative_tolerance
        ):
            raise RuntimeError(
                "measurement-window configured mean "
                f"{summary['measurement_mean_read_mib_s']:.3f} MiB/s differs from "
                f"target {self.target_read_mib_s:.3f} MiB/s by "
                f"{summary['relative_error'] * 100:.2f}%"
            )

    def _write_measurement_summary(self, stop_ns: int) -> Dict[str, Any]:
        if not self._transitions:
            raise RuntimeError("bandwidth controller recorded no transitions")
        weighted_mib_ns = 0.0
        for index, (start_ns, read_mib_s) in enumerate(self._transitions):
            end_ns = (
                self._transitions[index + 1][0]
                if index + 1 < len(self._transitions)
                else stop_ns
            )
            weighted_mib_ns += max(0, end_ns - start_ns) * read_mib_s
        start_ns = self._transitions[0][0]
        duration_ns = max(1, stop_ns - start_ns)
        actual_mean = weighted_mib_ns / duration_ns
        relative_error = (
            abs(actual_mean - self.target_read_mib_s) / self.target_read_mib_s
            if self.target_read_mib_s
            else None
        )
        summary = {
            "measurement_start_ns": start_ns,
            "measurement_end_ns": stop_ns,
            "measurement_duration_seconds": duration_ns / 1e9,
            "measurement_mean_read_mib_s": actual_mean,
            "target_read_mib_s": self.target_read_mib_s,
            "relative_error": relative_error,
            "transitions": len(self._transitions),
            "start_segment_index": self.start_segment_index,
        }
        self.output_path.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        return summary

    def _apply(self, segment: BandwidthSegment, base_values: Dict[str, str]) -> None:
        values = dict(base_values)
        values["rbps"] = str(segment.read_bytes_s)
        self._write_line(format_io_max_line(self.device_number, values))

    def _run(self, base_values: Dict[str, str]) -> None:
        fieldnames = (
            "applied_ns",
            "segment_index",
            "cycle_index",
            "duration_seconds",
            "configured_read_mib_s",
            "configured_read_bytes_s",
        )
        try:
            with self.output_path.open("w", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                segment_index = self.start_segment_index
                cycle_index = 0
                segment_start = time.monotonic()
                while True:
                    segment = self.segments[segment_index]
                    writer.writerow(
                        {
                            "applied_ns": time.monotonic_ns(),
                            "segment_index": segment_index,
                            "cycle_index": cycle_index,
                            "duration_seconds": segment.duration_seconds,
                            "configured_read_mib_s": segment.read_mib_s,
                            "configured_read_bytes_s": segment.read_bytes_s,
                        }
                    )
                    output_file.flush()
                    deadline = segment_start + segment.duration_seconds
                    if self._stop_event.wait(max(0.0, deadline - time.monotonic())):
                        return
                    segment_index += 1
                    if segment_index == len(self.segments):
                        segment_index = 0
                        cycle_index += 1
                    segment_start = deadline
                    self._apply(self.segments[segment_index], base_values)
                    self._transitions.append(
                        (time.monotonic_ns(), self.segments[segment_index].read_mib_s)
                    )
        except BaseException as exc:
            self._error = exc
            self._stop_event.set()


def create_bandwidth_controller(
    config: Optional[Dict[str, Any]], output_path: Path
) -> Optional[CgroupReadBandwidthController]:
    if not config or not config.get("enabled", True):
        return None
    if config.get("backend", "cgroup_v2") != "cgroup_v2":
        raise ValueError("Only the cgroup_v2 bandwidth backend is supported")
    segments = parse_schedule(config["schedule"])
    target = float(config["target_read_mib_s"])
    validate_schedule_mean(
        segments,
        target,
        float(config.get("mean_relative_tolerance", 1e-6)),
    )
    device_number = config.get("device_number")
    if not device_number:
        device = config.get("device")
        if not device:
            raise ValueError("bandwidth control requires device or device_number")
        device_number = resolve_device_number(device)
    return CgroupReadBandwidthController(
        cgroup_path=Path(config["cgroup_path"]),
        device_number=device_number,
        segments=segments,
        output_path=output_path,
        start_segment_index=int(config.get("start_segment_index", 0)),
        target_read_mib_s=target,
        measurement_mean_relative_tolerance=(
            float(config["measurement_mean_relative_tolerance"])
            if "measurement_mean_relative_tolerance" in config
            else None
        ),
    )
