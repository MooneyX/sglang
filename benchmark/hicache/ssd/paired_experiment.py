import argparse
import asyncio
import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

try:
    from .bandwidth import parse_schedule, validate_schedule_mean
    from .experiment import collect_manifest, execute, load_config
except ImportError:
    from bandwidth import parse_schedule, validate_schedule_mean
    from experiment import collect_manifest, execute, load_config


def validate_conditions(conditions: Dict[str, Dict[str, Any]]) -> float:
    if len(conditions) < 2:
        raise ValueError("paired experiment requires at least two conditions")
    targets = []
    for name, condition in conditions.items():
        control = condition["bandwidth_control"]
        target = float(control["target_read_mib_s"])
        validate_schedule_mean(
            parse_schedule(control["schedule"]),
            target,
            float(control.get("mean_relative_tolerance", 1e-6)),
        )
        targets.append(target)
    reference = targets[0]
    if any(abs(target - reference) > max(reference * 1e-6, 1e-9) for target in targets):
        raise ValueError(
            f"condition target throughputs must match, got {dict(zip(conditions, targets))}"
        )
    return reference


def create_run_config(
    base_config: Dict[str, Any],
    condition: Dict[str, Any],
    run_root: Path,
    rng: random.Random,
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["bandwidth_control"] = copy.deepcopy(condition["bandwidth_control"])
    control = config["bandwidth_control"]
    if control.pop("randomize_start_segment", False):
        control["start_segment_index"] = rng.randrange(len(control["schedule"]))
    config["server"]["clear_storage_before_run"] = True
    config["output"]["root_dir"] = str(run_root)
    return config


async def run_paired(config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    base_config_path = Path(config["base_config"])
    base_config = load_config(base_config_path)
    conditions = config["conditions"]
    target_mib_s = validate_conditions(conditions)
    repetitions = int(config.get("repetitions", 5))
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    rng = random.Random(config.get("random_seed", 1))
    run_records = []

    for repetition in range(repetitions):
        order = list(conditions)
        rng.shuffle(order)
        for order_index, condition_name in enumerate(order):
            condition = conditions[condition_name]
            run_dir = (
                output_dir
                / "runs"
                / f"rep-{repetition:03d}-{order_index}-{condition_name}"
            )
            run_dir.mkdir(parents=True, exist_ok=False)
            run_config = create_run_config(base_config, condition, run_dir.parent, rng)
            run_config["paired_run"] = {
                "repetition": repetition,
                "order_index": order_index,
                "condition": condition_name,
                "description": condition.get("description"),
            }
            (run_dir / "config.json").write_text(
                json.dumps(run_config, indent=2, sort_keys=True) + "\n"
            )
            manifest = collect_manifest(
                run_config, f"rep-{repetition:03d}-{condition_name}"
            )
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            record = {
                "repetition": repetition,
                "order_index": order_index,
                "condition": condition_name,
                "run_dir": str(run_dir),
            }
            try:
                summary = await execute(run_config, run_dir)
                summary["status"] = "completed"
                record["status"] = "completed"
            except Exception as exc:
                summary = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                record.update(summary)
                (run_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
                run_records.append(record)
                (output_dir / "paired_runs.json").write_text(
                    json.dumps(run_records, indent=2) + "\n"
                )
                if config.get("stop_on_failure", True):
                    raise
                continue
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            run_records.append(record)
            (output_dir / "paired_runs.json").write_text(
                json.dumps(run_records, indent=2) + "\n"
            )

    return {
        "status": "completed",
        "target_read_mib_s": target_mib_s,
        "repetitions": repetitions,
        "conditions": list(conditions),
        "runs": run_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run stable-vs-fluctuating HiCache bandwidth experiments"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(config["output_root"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "paired_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    try:
        summary = asyncio.run(run_paired(config, output_dir))
    except Exception as exc:
        print(
            f"Paired experiment failed; results kept in {output_dir}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
    (output_dir / "paired_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Paired experiment completed: {output_dir}")


if __name__ == "__main__":
    main()
