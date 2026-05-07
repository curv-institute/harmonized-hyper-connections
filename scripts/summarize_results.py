#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Generate seed-level and aggregate result tables from run summaries."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from validate_artifacts import MODES, SEEDS, REQUIRED_SUMMARY_FIELDS


SEED_COLUMNS = [
    "mode",
    "seed",
    "steps",
    "status",
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
    "hostname",
    "commit_sha",
    "error_type",
    "error_message",
    "run_dir",
]
AGG_METRICS = [
    "final_accuracy",
    "final_loss",
    "max_raw_gain",
    "max_applied_gain",
    "mean_applied_gain",
    "min_scale",
    "mean_scale",
    "floor_hits",
    "runtime_seconds",
]


def load_rows(runs_dir: Path, prefix: str, modes: list[str], allow_failed: bool) -> list[dict[str, object]]:
    rows = []
    for mode in modes:
        for seed in SEEDS:
            run_dir = runs_dir / f"{prefix}_{mode}_seed{seed}"
            with open(run_dir / "summary.json") as f:
                row = json.load(f)
            for field in REQUIRED_SUMMARY_FIELDS:
                if field not in row:
                    raise KeyError(f"{run_dir / 'summary.json'} missing {field}")
            if row.get("status") == "failed" and not allow_failed:
                raise RuntimeError(f"{run_dir} is a documented failed run; use --allow-failed to summarize it")
            row["run_dir"] = str(run_dir)
            rows.append(row)
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SEED_COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_csv(rows: list[dict[str, object]], path: Path, modes: list[str]) -> None:
    aggs = aggregate_rows(rows, modes)
    fieldnames = ["mode", "n"]
    for metric in AGG_METRICS:
        for suffix in ("mean", "std", "min", "max"):
            fieldnames.append(f"{metric}_{suffix}")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(aggs)


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(out)


def aggregate_rows(rows: list[dict[str, object]], modes: list[str]) -> list[dict[str, object]]:
    aggs = []
    for mode in modes:
        mode_rows = [r for r in rows if r["mode"] == mode and r.get("status") != "failed"]
        agg: dict[str, object] = {"mode": mode, "n": len(mode_rows)}
        if not mode_rows:
            continue
        for metric in AGG_METRICS:
            values = [float(r[metric]) for r in mode_rows]
            agg[f"{metric}_mean"] = statistics.fmean(values)
            agg[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            agg[f"{metric}_min"] = min(values)
            agg[f"{metric}_max"] = max(values)
        aggs.append(agg)
    return aggs


def write_markdown(rows: list[dict[str, object]], path: Path, modes: list[str]) -> None:
    seed_headers = [
        "mode",
        "seed",
        "status",
        "final_accuracy",
        "final_loss",
        "max_raw_gain",
        "max_applied_gain",
        "min_scale",
        "runtime_seconds",
        "run_dir",
    ]
    seed_table = markdown_table(seed_headers, [[r.get(h) for h in seed_headers] for r in rows])

    aggs = aggregate_rows(rows, modes)
    task_headers = [
        "mode",
        "n",
        "final_accuracy_mean",
        "final_accuracy_std",
        "final_accuracy_min",
        "final_accuracy_max",
        "final_loss_mean",
        "final_loss_std",
    ]
    task_table = markdown_table(task_headers, [[r[h] for h in task_headers] for r in aggs])

    gain_headers = [
        "mode",
        "n",
        "max_raw_gain_mean",
        "max_raw_gain_std",
        "max_raw_gain_min",
        "max_raw_gain_max",
        "max_applied_gain_mean",
        "max_applied_gain_std",
        "max_applied_gain_min",
        "max_applied_gain_max",
    ]
    gain_table = markdown_table(gain_headers, [[r[h] for h in gain_headers] for r in aggs])

    scale_headers = [
        "mode",
        "n",
        "min_scale_mean",
        "min_scale_std",
        "min_scale_min",
        "min_scale_max",
        "mean_scale_mean",
        "floor_hits_mean",
        "floor_hits_max",
        "runtime_seconds_mean",
    ]
    scale_table = markdown_table(scale_headers, [[r[h] for h in scale_headers] for r in aggs])

    path.write_text(
        "# Results Summary\n\n"
        "## Seed-Level Results\n\n"
        f"{seed_table}\n\n"
        "## Aggregate Task Metrics\n\n"
        f"{task_table}\n\n"
        "## Aggregate Gain Metrics\n\n"
        f"{gain_table}\n\n"
        "## Aggregate Scale and Runtime Metrics\n\n"
        f"{scale_table}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", default="pub15k")
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--csv-out", type=Path, default=Path("results_summary.csv"))
    parser.add_argument("--agg-csv-out", type=Path, default=Path("results_aggregate.csv"))
    parser.add_argument("--md-out", type=Path, default=Path("results_summary.md"))
    parser.add_argument("--allow-failed", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.runs_dir, args.prefix, list(args.modes), args.allow_failed)
    write_csv(rows, args.csv_out)
    write_aggregate_csv(rows, args.agg_csv_out, list(args.modes))
    write_markdown(rows, args.md_out, list(args.modes))
    print(f"Wrote {args.csv_out}")
    print(f"Wrote {args.agg_csv_out}")
    print(f"Wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
