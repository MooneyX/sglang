import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def percentile(values: List[float], value: float) -> float:
    return float(np.percentile(values, value))


def coefficient_of_variation(values: List[float]) -> float:
    average = mean(values)
    return float(np.std(values) / average) if average else math.nan


def run_metrics(run_dir: Path) -> Dict[str, float]:
    requests = [
        row
        for row in load_jsonl(run_dir / "requests.jsonl")
        if row.get("phase") == "measure" and row.get("success")
    ]
    storage_requests = [
        row
        for row in requests
        if (row.get("cached_tokens_details") or {}).get("storage", 0) > 0
    ]
    if not storage_requests:
        raise RuntimeError(f"No measurement storage hits in {run_dir}")
    ttft = [float(row["ttft_ms"]) for row in storage_requests]
    transfers = []
    transfer_path = run_dir / "hicache_transfers.jsonl"
    if transfer_path.exists():
        request_ids = {row["request_id"] for row in storage_requests}
        transfers = [
            row
            for row in load_jsonl(transfer_path)
            if row.get("operation") == "prefetch"
            and row.get("status") == "completed"
            and row.get("request_id") in request_ids
        ]
    transfer_request_ids = {row.get("request_id") for row in transfers}
    missing_transfer_ids = {
        row["request_id"] for row in storage_requests
    } - transfer_request_ids
    if missing_transfer_ids:
        raise RuntimeError(
            f"Missing completed prefetch traces for {sorted(missing_transfer_ids)} in {run_dir}"
        )
    io_ms = [float(row["io_ms"]) for row in transfers if row.get("io_ms") is not None]
    queue_ms = [
        float(row["queue_ms"]) for row in transfers if row.get("queue_ms") is not None
    ]
    throughput = [
        float(row["throughput_mib_s"])
        for row in transfers
        if row.get("throughput_mib_s") is not None
    ]
    metrics = {
        "measure_requests": len(requests),
        "storage_hit_requests": len(storage_requests),
        "storage_hit_fraction": len(storage_requests) / len(requests),
        "ttft_mean_ms": mean(ttft),
        "ttft_p50_ms": median(ttft),
        "ttft_p95_ms": percentile(ttft, 95),
        "ttft_p99_ms": percentile(ttft, 99),
        "ttft_max_ms": max(ttft),
        "ttft_cv": coefficient_of_variation(ttft),
        "prefetch_count": len(transfers),
        "prefetch_io_mean_ms": mean(io_ms) if io_ms else math.nan,
        "prefetch_queue_mean_ms": mean(queue_ms) if queue_ms else math.nan,
        "prefetch_queue_p95_ms": percentile(queue_ms, 95) if queue_ms else math.nan,
        "prefetch_throughput_mean_mib_s": mean(throughput) if throughput else math.nan,
        "prefetch_throughput_cv": (
            coefficient_of_variation(throughput) if throughput else math.nan
        ),
    }
    return metrics


def paired_bootstrap_ci(
    differences: List[float],
    confidence: float = 0.95,
    samples: int = 10000,
    seed: int = 1,
) -> tuple[float, float]:
    if not differences:
        raise ValueError("differences must not be empty")
    if len(differences) == 1:
        return differences[0], differences[0]
    rng = random.Random(seed)
    estimates = [
        mean(rng.choice(differences) for _ in differences) for _ in range(samples)
    ]
    alpha = (1 - confidence) / 2
    return percentile(estimates, alpha * 100), percentile(estimates, (1 - alpha) * 100)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare paired stable and fluctuating HiCache experiments"
    )
    parser.add_argument("paired_dir", type=Path)
    parser.add_argument("--baseline", default="stable")
    parser.add_argument("--experiment", default="fluctuating")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    run_records = json.loads((args.paired_dir / "paired_runs.json").read_text())
    rows = []
    for record in run_records:
        if record.get("status") != "completed":
            continue
        metrics = run_metrics(Path(record["run_dir"]))
        rows.append({**record, **metrics})
    if not rows:
        raise RuntimeError("No completed paired runs found")
    write_csv(args.paired_dir / "run_metrics.csv", rows)

    by_pair = {
        (row["repetition"], row["condition"]): row
        for row in rows
        if row["condition"] in (args.baseline, args.experiment)
    }
    metrics = (
        "ttft_mean_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "ttft_cv",
        "prefetch_io_mean_ms",
        "prefetch_queue_mean_ms",
        "prefetch_queue_p95_ms",
        "prefetch_throughput_mean_mib_s",
        "prefetch_throughput_cv",
    )
    paired_rows = []
    repetitions = sorted({row["repetition"] for row in rows})
    for repetition in repetitions:
        baseline = by_pair.get((repetition, args.baseline))
        experiment = by_pair.get((repetition, args.experiment))
        if baseline is None or experiment is None:
            continue
        paired = {"repetition": repetition}
        for metric in metrics:
            paired[f"{args.baseline}_{metric}"] = baseline[metric]
            paired[f"{args.experiment}_{metric}"] = experiment[metric]
            paired[f"delta_{metric}"] = experiment[metric] - baseline[metric]
        paired_rows.append(paired)
    if not paired_rows:
        raise RuntimeError("No complete baseline/experiment repetition pairs found")
    write_csv(args.paired_dir / "paired_differences.csv", paired_rows)

    summary = {
        "baseline": args.baseline,
        "experiment": args.experiment,
        "complete_pairs": len(paired_rows),
        "metrics": {},
    }
    for metric in metrics:
        differences = [row[f"delta_{metric}"] for row in paired_rows]
        low, high = paired_bootstrap_ci(differences, samples=args.bootstrap_samples)
        baseline_values = [row[f"{args.baseline}_{metric}"] for row in paired_rows]
        experiment_values = [row[f"{args.experiment}_{metric}"] for row in paired_rows]
        summary["metrics"][metric] = {
            "baseline_mean": mean(baseline_values),
            "experiment_mean": mean(experiment_values),
            "mean_paired_delta": mean(differences),
            "mean_paired_delta_pct": (
                mean(differences) / mean(baseline_values) * 100
                if mean(baseline_values)
                else None
            ),
            "bootstrap_95_ci": [low, high],
        }
    (args.paired_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    figure_metrics = (
        ("ttft_mean_ms", "Mean TTFT (ms)"),
        ("ttft_p95_ms", "P95 TTFT (ms)"),
        ("ttft_p99_ms", "P99 TTFT (ms)"),
        ("prefetch_queue_p95_ms", "P95 prefetch queue (ms)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    x = np.arange(len(paired_rows))
    for axis, (metric, title) in zip(axes.flat, figure_metrics):
        baseline_values = [row[f"{args.baseline}_{metric}"] for row in paired_rows]
        experiment_values = [row[f"{args.experiment}_{metric}"] for row in paired_rows]
        for index, (baseline_value, experiment_value) in enumerate(
            zip(baseline_values, experiment_values)
        ):
            axis.plot(
                [index - 0.12, index + 0.12],
                [baseline_value, experiment_value],
                color="0.65",
                linewidth=1,
            )
        axis.scatter(x - 0.12, baseline_values, label=args.baseline, s=45)
        axis.scatter(x + 0.12, experiment_values, label=args.experiment, s=45)
        axis.set_xticks(x, [str(row["repetition"]) for row in paired_rows])
        axis.set_xlabel("Repetition")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Stable vs fluctuating SSD service rate at equal mean bandwidth")
    fig.tight_layout()
    fig.savefig(args.paired_dir / "paired_comparison.png", dpi=180)
    fig.savefig(args.paired_dir / "paired_comparison.svg")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
