import argparse
import asyncio
import json
import platform
import glob
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

try:
    from .bandwidth import create_bandwidth_controller
    from .monitor import DiskstatsMonitor
    from .workload import RequestRecord, WorkloadConfig, build_workload, run_phase
except ImportError:
    from bandwidth import create_bandwidth_controller
    from monitor import DiskstatsMonitor
    from workload import RequestRecord, WorkloadConfig, build_workload, run_phase


def load_config(path: Path) -> Dict[str, Any]:
    with path.open() as file:
        config = json.load(file)
    for section in ("server", "storage", "workload", "output"):
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    return config


def run_command(command: List[str]) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=30,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "output": result.stdout,
        }
    except Exception as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}


def collect_manifest(config: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    storage_path = config["storage"]["path"]
    commands = [
        ["git", "rev-parse", "HEAD"],
        ["git", "status", "--short", "--branch"],
        ["findmnt", "-T", storage_path, "-o", "TARGET,SOURCE,FSTYPE,OPTIONS,MAJ:MIN"],
        ["df", "-hT", storage_path],
        [
            "lsblk",
            "-o",
            "NAME,MAJ:MIN,TYPE,SIZE,ROTA,TRAN,MODEL,SERIAL,FSTYPE,MOUNTPOINTS",
        ],
        ["nvidia-smi", "-L"],
    ]
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_start_ns": time.monotonic_ns(),
        "realtime_start_ns": time.time_ns(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "config": config,
        "commands": [run_command(command) for command in commands],
    }


def wait_for_server(base_url: str, timeout_seconds: int) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base_url}/server_info", timeout=5)
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(1)
    raise RuntimeError(f"SGLang server did not become ready: {last_error}")


def flush_cache(base_url: str, timeout_seconds: int = 30) -> None:
    response = requests.post(
        f"{base_url}/flush_cache",
        params={"timeout": timeout_seconds},
        timeout=timeout_seconds + 10,
    )
    response.raise_for_status()


def clear_hicache_storage(base_url: str, timeout_seconds: int = 300) -> None:
    response = requests.post(
        f"{base_url}/clear_hicache_storage_backend",
        timeout=timeout_seconds,
    )
    response.raise_for_status()


def load_token_ids(model: str, tokenizer_path: Optional[str]) -> List[int]:
    from sglang.benchmark.utils import get_tokenizer

    tokenizer = get_tokenizer(tokenizer_path or model)
    special_ids = set(tokenizer.all_special_ids)
    token_ids = sorted(set(tokenizer.get_vocab().values()) - special_ids)
    if len(token_ids) < 2:
        raise RuntimeError("Tokenizer has fewer than two non-special token ids")
    return token_ids


def append_records(path: Path, records: List[RequestRecord]) -> None:
    with path.open("a") as file:
        for record in records:
            file.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def summarize(records: List[RequestRecord]) -> Dict[str, Any]:
    successful = [record for record in records if record.success]
    storage_hits = [
        record
        for record in successful
        if (record.cached_tokens_details or {}).get("storage", 0) > 0
    ]
    ttfts = [
        (record.first_token_ns - record.submitted_ns) / 1e6
        for record in successful
        if record.first_token_ns is not None
    ]
    return {
        "requests": len(records),
        "successful": len(successful),
        "failed": len(records) - len(successful),
        "storage_hit_requests": len(storage_hits),
        "storage_hit_tokens": sum(
            (record.cached_tokens_details or {}).get("storage", 0)
            for record in storage_hits
        ),
        "mean_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else None,
        "max_ttft_ms": max(ttfts) if ttfts else None,
    }


async def execute(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    experiment_start_ns = time.monotonic_ns()
    server = config["server"]
    workload_values = config["workload"]
    storage = config["storage"]
    base_url = server.get("base_url", "http://127.0.0.1:30000").rstrip("/")
    storage_path = Path(storage["path"])
    if not storage_path.is_dir():
        raise ValueError(
            f"Storage path does not exist or is not a directory: {storage_path}"
        )

    server_info = wait_for_server(base_url, server.get("ready_timeout_seconds", 300))
    if server.get("flush_before_run", True):
        flush_cache(base_url, workload_values.get("flush_timeout_seconds", 30))
    if server.get("clear_storage_before_run", False):
        clear_hicache_storage(
            base_url, workload_values.get("clear_storage_timeout_seconds", 300)
        )

    workload_config = WorkloadConfig(
        **{
            field: workload_values[field]
            for field in WorkloadConfig.__dataclass_fields__
            if field in workload_values
        }
    )
    token_ids = load_token_ids(server["model"], server.get("tokenizer"))
    phases = build_workload(workload_config, token_ids)
    requests_path = run_dir / "requests.jsonl"
    monitor = None
    device = storage.get("device")
    if device:
        monitor = DiskstatsMonitor(
            device=device,
            output_path=run_dir / "device.csv",
            interval_seconds=storage.get("monitor_interval_ms", 100) / 1000,
        )
        monitor.start()

    summaries: Dict[str, Any] = {}
    all_records: List[RequestRecord] = []
    try:
        for phase_name in ("fill", "evict", "measure"):
            phase_start_ns = time.monotonic_ns()
            phase_output_tokens = (
                workload_config.offload_output_tokens
                if phase_name == "evict"
                else workload_config.output_tokens
            )
            bandwidth_controller = None
            if phase_name == "measure":
                bandwidth_controller = create_bandwidth_controller(
                    config.get("bandwidth_control"),
                    run_dir / "configured_bandwidth.csv",
                )
                if bandwidth_controller is not None:
                    bandwidth_controller.start()
            try:
                records = await run_phase(
                    url=f"{base_url}/generate",
                    specs=phases[phase_name],
                    output_tokens=phase_output_tokens,
                    max_concurrency=workload_config.max_concurrency,
                    timeout_seconds=workload_config.request_timeout_seconds,
                )
            finally:
                if bandwidth_controller is not None:
                    bandwidth_controller.stop()
            append_records(requests_path, records)
            all_records.extend(records)
            summaries[phase_name] = {
                **summarize(records),
                "start_ns": phase_start_ns,
                "end_ns": time.monotonic_ns(),
            }
            failed = [record for record in records if not record.success]
            if failed:
                raise RuntimeError(
                    f"Phase {phase_name} had {len(failed)} failed requests; see requests.jsonl"
                )
            pause_seconds = workload_values.get("phase_pause_seconds", 0)
            if pause_seconds:
                await asyncio.sleep(pause_seconds)
            if phase_name == "evict" and workload_values.get(
                "flush_before_measure", True
            ):
                flush_cache(base_url, workload_values.get("flush_timeout_seconds", 30))
                flush_settle_seconds = workload_values.get("flush_settle_seconds", 1)
                if flush_settle_seconds:
                    await asyncio.sleep(flush_settle_seconds)
    finally:
        if monitor is not None:
            monitor.stop()

    transfer_trace_glob = storage.get("transfer_trace_glob")
    if transfer_trace_glob:
        trace_files = [Path(path) for path in sorted(glob.glob(transfer_trace_glob))]
        if trace_files:
            experiment_end_ns = time.monotonic_ns()
            with (run_dir / "hicache_transfers.jsonl").open("w") as output_file:
                for trace_file in trace_files:
                    with trace_file.open() as input_file:
                        for line in input_file:
                            event = json.loads(line)
                            timestamp_ns = event.get("timestamp_ns", 0)
                            if experiment_start_ns <= timestamp_ns <= experiment_end_ns:
                                output_file.write(json.dumps(event) + "\n")

    measure_summary = summaries["measure"]
    require_storage_hits = workload_values.get("require_storage_hits", True)
    if require_storage_hits and measure_summary["storage_hit_requests"] == 0:
        raise RuntimeError(
            "Measurement phase had no storage cache hits. Increase eviction pressure or "
            "verify that HiCache file storage and --enable-cache-report are enabled."
        )
    minimum_storage_hit_fraction = float(
        workload_values.get("minimum_storage_hit_fraction", 0.0)
    )
    actual_storage_hit_fraction = (
        measure_summary["storage_hit_requests"] / measure_summary["successful"]
        if measure_summary["successful"]
        else 0.0
    )
    measure_summary["storage_hit_fraction"] = actual_storage_hit_fraction
    if actual_storage_hit_fraction < minimum_storage_hit_fraction:
        raise RuntimeError(
            f"Measurement storage-hit fraction {actual_storage_hit_fraction:.3f} is below "
            f"the required {minimum_storage_hit_fraction:.3f}. Increase eviction pressure "
            "or reduce the host HiCache size."
        )
    return {
        "server_info": server_info,
        "phases": summaries,
        "overall": summarize(all_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Observe SSD variability during HiCache inference"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(config["output"]["root_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    manifest = collect_manifest(config, run_id)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    try:
        summary = asyncio.run(execute(config, run_dir))
        summary["status"] = "completed"
    except Exception as exc:
        summary = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"Experiment failed; results kept in {run_dir}: {summary['error']}",
            file=sys.stderr,
        )
        raise

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Experiment completed: {run_dir}")


if __name__ == "__main__":
    main()
