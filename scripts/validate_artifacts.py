#!/usr/bin/env python3
"""Validate publication comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


MODES = ("hc", "mhc", "harm")
SEEDS = (0, 1, 2)
REQUIRED_FILES = (
    "config.json",
    "command.txt",
    "summary.json",
    "metrics.csv",
    "manifest.json",
    "stdout.log",
    "stderr.log",
)
REQUIRED_SUMMARY_FIELDS = (
    "mode",
    "seed",
    "steps",
    "final_accuracy",
    "final_loss",
    "max_raw_gain",
    "max_applied_gain",
    "mean_applied_gain",
    "std_applied_gain",
    "min_scale",
    "mean_scale",
    "floor_hits",
    "runtime_seconds",
    "device",
    "commit_sha",
)
REQUIRED_METRIC_COLUMNS = (
    "step",
    "loss",
    "accuracy",
    "raw_gain",
    "applied_gain",
    "scale",
    "mode",
    "seed",
)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def validate_run(run_dir: Path, mode: str, seed: int, steps: int) -> list[str]:
    errors: list[str] = []
    if not run_dir.exists():
        return [f"missing run directory: {run_dir}"]

    for name in REQUIRED_FILES:
        if not (run_dir / name).exists():
            errors.append(f"{run_dir}: missing {name}")

    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.csv"

    if config_path.exists():
        config = load_json(config_path)
        for key, expected in (("mode", mode), ("seed", seed), ("steps", steps)):
            if config.get(key) != expected:
                errors.append(f"{config_path}: {key}={config.get(key)!r}, expected {expected!r}")

    if summary_path.exists():
        summary = load_json(summary_path)
        for field in REQUIRED_SUMMARY_FIELDS:
            if field not in summary:
                errors.append(f"{summary_path}: missing summary field {field}")
        for key, expected in (("mode", mode), ("seed", seed), ("steps", steps)):
            if summary.get(key) != expected:
                errors.append(f"{summary_path}: {key}={summary.get(key)!r}, expected {expected!r}")
        if summary.get("status") == "failed":
            if not summary.get("error_type") or not summary.get("error_message"):
                errors.append(f"{summary_path}: failed run lacks error_type/error_message")
            return errors

    if metrics_path.exists():
        with open(metrics_path, newline="") as f:
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames or [])
            for col in REQUIRED_METRIC_COLUMNS:
                if col not in columns:
                    errors.append(f"{metrics_path}: missing metrics column {col}")
            rows = list(reader)
        if not rows:
            errors.append(f"{metrics_path}: no metric rows")
        else:
            last = rows[-1]
            if int(float(last["step"])) != steps - 1:
                errors.append(f"{metrics_path}: final step {last['step']} != {steps - 1}")
            bad_modes = {row.get("mode") for row in rows if row.get("mode") != mode}
            bad_seeds = {row.get("seed") for row in rows if int(float(row.get("seed", -1))) != seed}
            if bad_modes:
                errors.append(f"{metrics_path}: unexpected mode values {sorted(bad_modes)}")
            if bad_seeds:
                errors.append(f"{metrics_path}: unexpected seed values {sorted(bad_seeds)}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", default="pub15k")
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument(
        "--allow-failed",
        action="store_true",
        help="Accept documented failed runs with failure summaries.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    failed = 0
    for mode in MODES:
        for seed in SEEDS:
            run_dir = args.runs_dir / f"{args.prefix}_{mode}_seed{seed}"
            run_errors = validate_run(run_dir, mode, seed, args.steps)
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                summary = load_json(summary_path)
                if summary.get("status") == "failed":
                    failed += 1
                    if not args.allow_failed:
                        run_errors.append(f"{summary_path}: run failed ({summary.get('error_type')})")
            errors.extend(run_errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if failed:
        print(f"All artifacts validated with {failed} documented failed run(s).")
    else:
        print("All publication artifacts validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
