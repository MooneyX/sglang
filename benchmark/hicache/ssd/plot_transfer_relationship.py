import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


def load_jsonl(path: Path):
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def correlation(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {
            "pearson_r": None,
            "pearson_p": None,
            "spearman_r": None,
            "spearman_p": None,
        }
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir

    requests = {
        row["request_id"]: row
        for row in load_jsonl(run_dir / "requests.jsonl")
        if row.get("phase") == "measure" and row.get("success")
    }
    transfers = [
        row
        for row in load_jsonl(run_dir / "hicache_transfers.jsonl")
        if row.get("operation") == "prefetch"
        and row.get("status") == "completed"
        and row.get("request_id") in requests
    ]

    rows = []
    for transfer in transfers:
        request = requests[transfer["request_id"]]
        rows.append(
            {
                "request_id": transfer["request_id"],
                "ttft_ms": request["ttft_ms"],
                "storage_cached_tokens": (
                    request.get("cached_tokens_details") or {}
                ).get("storage", 0),
                "queue_ms": transfer["queue_ms"],
                "io_ms": transfer["io_ms"],
                "total_transfer_ms": transfer["total_ms"],
                "completed_bytes": transfer["completed_bytes"],
                "throughput_mib_s": transfer["throughput_mib_s"],
            }
        )
    if not rows:
        raise RuntimeError("No completed measurement prefetch events found")

    csv_path = run_dir / "transfer_request_relationship.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    io_ms = [row["io_ms"] for row in rows]
    throughput = [row["throughput_mib_s"] for row in rows]
    ttft = [row["ttft_ms"] for row in rows]
    queue_ms = [row["queue_ms"] for row in rows]
    all_measure_ttft = [row["ttft_ms"] for row in requests.values()]
    non_storage_ttft = [
        row["ttft_ms"]
        for row in requests.values()
        if (row.get("cached_tokens_details") or {}).get("storage", 0) == 0
    ]

    summary = {
        "measure_requests": len(requests),
        "completed_storage_prefetches": len(rows),
        "prefetch_io_ms": {
            "mean": mean(io_ms),
            "median": median(io_ms),
            "p95": percentile(io_ms, 95),
            "min": min(io_ms),
            "max": max(io_ms),
            "cv": float(np.std(io_ms) / np.mean(io_ms)),
        },
        "prefetch_throughput_mib_s": {
            "mean": mean(throughput),
            "median": median(throughput),
            "p5": percentile(throughput, 5),
            "p95": percentile(throughput, 95),
            "min": min(throughput),
            "max": max(throughput),
            "cv": float(np.std(throughput) / np.mean(throughput)),
        },
        "prefetch_queue_ms": {
            "mean": mean(queue_ms),
            "min": min(queue_ms),
            "max": max(queue_ms),
        },
        "storage_hit_ttft_ms": {
            "mean": mean(ttft),
            "median": median(ttft),
            "min": min(ttft),
            "max": max(ttft),
        },
        "non_storage_ttft_ms": (
            {"mean": mean(non_storage_ttft), "median": median(non_storage_ttft)}
            if non_storage_ttft
            else None
        ),
        "io_ms_vs_ttft": correlation(io_ms, ttft),
        "throughput_vs_ttft": correlation(throughput, ttft),
        "queue_ms_vs_ttft": correlation(queue_ms, ttft),
    }
    (run_dir / "relationship_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    order = np.argsort([row["request_id"] for row in rows])
    labels = [rows[index]["request_id"].split("-")[-1] for index in order]
    ordered_io = [io_ms[index] for index in order]
    ordered_tp = [throughput[index] for index in order]
    ordered_ttft = [ttft[index] for index in order]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(len(rows))
    axis = axes[0, 0]
    axis.plot(x, ordered_io, "o-", label="HiCache prefetch I/O latency (ms)")
    axis.set_xticks(x, labels, rotation=45)
    axis.set_ylabel("I/O latency (ms)")
    twin = axis.twinx()
    twin.plot(
        x, ordered_tp, "s--", color="tab:orange", label="Effective throughput (MiB/s)"
    )
    twin.set_ylabel("Throughput (MiB/s)")
    axis.set_title("Storage transfer variability by request")
    lines = axis.lines + twin.lines
    axis.legend(lines, [line.get_label() for line in lines], loc="best")

    axis = axes[0, 1]
    axis.scatter(io_ms, ttft, s=55)
    if len(rows) >= 2:
        fit = np.polyfit(io_ms, ttft, 1)
        fit_x = np.linspace(min(io_ms), max(io_ms), 100)
        axis.plot(fit_x, np.polyval(fit, fit_x), color="tab:red", alpha=0.8)
    axis.set_xlabel("HiCache prefetch I/O latency (ms)")
    axis.set_ylabel("TTFT (ms)")
    axis.set_title(
        f"I/O latency vs TTFT (Pearson r={summary['io_ms_vs_ttft']['pearson_r']:.2f})"
    )

    axis = axes[1, 0]
    axis.scatter(throughput, ttft, s=55, color="tab:orange")
    if len(rows) >= 2:
        fit = np.polyfit(throughput, ttft, 1)
        fit_x = np.linspace(min(throughput), max(throughput), 100)
        axis.plot(fit_x, np.polyval(fit, fit_x), color="tab:red", alpha=0.8)
    axis.set_xlabel("Effective prefetch throughput (MiB/s)")
    axis.set_ylabel("TTFT (ms)")
    axis.set_title(
        f"Throughput vs TTFT (Pearson r={summary['throughput_vs_ttft']['pearson_r']:.2f})"
    )

    axis = axes[1, 1]
    axis.boxplot(
        [ttft, non_storage_ttft],
        tick_labels=[
            f"L3 hit (n={len(ttft)})",
            f"No L3 hit (n={len(non_storage_ttft)})",
        ],
    )
    axis.set_ylabel("TTFT (ms)")
    axis.set_title("Measurement TTFT by storage-hit status")

    fig.suptitle(
        "HiCache storage variability and inference TTFT\nQwen2.5-1.5B, 4096-token prefix, concurrency=4"
    )
    fig.tight_layout()
    fig.savefig(run_dir / "ssd_inference_relationship.png", dpi=180)
    fig.savefig(run_dir / "ssd_inference_relationship.svg")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
