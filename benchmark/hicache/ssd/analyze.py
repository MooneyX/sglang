import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

DEVICE_METRICS = (
    "read_iops",
    "write_iops",
    "read_mib_s",
    "write_mib_s",
    "read_await_ms",
    "write_await_ms",
    "queue_depth",
    "util_pct",
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def load_transfer_events(path: Path) -> List[Dict[str, Any]]:
    return [
        event
        for event in load_jsonl(path)
        if event.get("event") == "hicache_storage_transfer"
    ]


def load_device_samples(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["interval_start_ns"] = int(row["interval_start_ns"])
        row["interval_end_ns"] = int(row["interval_end_ns"])
        for metric in DEVICE_METRICS:
            value = row.get(metric)
            row[metric] = float(value) if value not in (None, "", "None") else None
    return rows


def samples_in_window(
    samples: Iterable[Dict[str, Any]], start_ns: int, end_ns: int
) -> List[Dict[str, Any]]:
    return [
        sample
        for sample in samples
        if sample["interval_end_ns"] >= start_ns
        and sample["interval_start_ns"] <= end_ns
    ]


def aggregate_metric(
    samples: List[Dict[str, Any]], metric: str
) -> tuple[Optional[float], Optional[float]]:
    values = [sample[metric] for sample in samples if sample.get(metric) is not None]
    if not values:
        return None, None
    return mean(values), max(values)


def correlate_requests(
    requests: Iterable[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    transfers: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    rows = []
    transfers = transfers or []
    transfers_by_request: Dict[str, List[Dict[str, Any]]] = {}
    for transfer in transfers:
        request_id = transfer.get("request_id")
        if request_id:
            transfers_by_request.setdefault(request_id, []).append(transfer)
    for request in requests:
        if request.get("phase") != "measure" or not request.get("success"):
            continue
        start_ns = request["submitted_ns"]
        end_ns = request.get("first_token_ns")
        if end_ns is None:
            continue
        window = samples_in_window(samples, start_ns, end_ns)
        details = request.get("cached_tokens_details") or {}
        row = {
            "request_id": request["request_id"],
            "client_id": request.get("client_id"),
            "submitted_ns": start_ns,
            "first_token_ns": end_ns,
            "ttft_ms": request.get("ttft_ms"),
            "prompt_tokens": request.get("prompt_tokens"),
            "cached_tokens": request.get("cached_tokens", 0),
            "device_cached_tokens": details.get("device", 0),
            "host_cached_tokens": details.get("host", 0),
            "storage_cached_tokens": details.get("storage", 0),
            "device_samples": len(window),
        }
        for metric in DEVICE_METRICS:
            average, maximum = aggregate_metric(window, metric)
            row[f"mean_{metric}"] = average
            row[f"max_{metric}"] = maximum
        request_transfers = transfers_by_request.get(request["request_id"], [])
        prefetches = [
            transfer
            for transfer in request_transfers
            if transfer.get("operation") == "prefetch"
        ]
        row["hicache_prefetch_count"] = len(prefetches)
        row["hicache_prefetch_io_ms"] = sum(
            transfer.get("io_ms") or 0 for transfer in prefetches
        )
        row["hicache_prefetch_bytes"] = sum(
            transfer.get("completed_bytes") or 0 for transfer in prefetches
        )
        row["hicache_prefetch_mean_mib_s"] = (
            mean(
                transfer["throughput_mib_s"]
                for transfer in prefetches
                if transfer.get("throughput_mib_s") is not None
            )
            if any(
                transfer.get("throughput_mib_s") is not None for transfer in prefetches
            )
            else None
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No successful measurement requests found")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlate request TTFT windows with block-device samples"
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    requests_path = args.run_dir / "requests.jsonl"
    device_path = args.run_dir / "device.csv"
    if not device_path.exists():
        raise RuntimeError(
            "device.csv is missing. Configure storage.device on the independent SSD machine."
        )
    requests = load_jsonl(requests_path)
    samples = load_device_samples(device_path)
    transfer_path = args.run_dir / "hicache_transfers.jsonl"
    transfers = load_transfer_events(transfer_path) if transfer_path.exists() else []
    rows = correlate_requests(requests, samples, transfers)
    output_path = args.run_dir / "request_storage_windows.csv"
    write_csv(output_path, rows)
    print(f"Wrote {len(rows)} request windows to {output_path}")


if __name__ == "__main__":
    main()
