#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib>=3.10.0",
#   "pandas>=2.3.0",
# ]
# ///
"""Generate publication figures from committed run artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODES = ("hc", "mhc", "harm")
SEEDS = (0, 1, 2)
MODE_LABELS = {"hc": "HC", "mhc": "mHC", "harm": "Harmonizer", "res_scale": "Residual scale"}
MODE_COLORS = {"hc": "#4C78A8", "mhc": "#F58518", "harm": "#54A24B", "res_scale": "#B279A2"}


def load_metrics(runs_dir: Path, prefix: str) -> pd.DataFrame:
    frames = []
    for mode in MODES:
        for seed in SEEDS:
            run_dir = runs_dir / f"{prefix}_{mode}_seed{seed}"
            path = run_dir / "metrics.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing metrics file: {path}")
            df = pd.read_csv(path)
            df["mode"] = mode
            df["seed"] = seed
            df["run_dir"] = str(run_dir)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, title: str, out: Path, log_y: bool = False) -> None:
    plt.figure(figsize=(9, 5.5))
    ax = plt.gca()
    for mode in MODES:
        mode_df = df[df["mode"] == mode]
        color = MODE_COLORS[mode]
        for seed in SEEDS:
            seed_df = mode_df[mode_df["seed"] == seed].sort_values("step")
            ax.plot(seed_df["step"], seed_df[metric], color=color, alpha=0.25, linewidth=1)
        mean_df = mode_df.groupby("step", as_index=False)[metric].mean()
        ax.plot(mean_df["step"], mean_df[metric], label=MODE_LABELS[mode], color=color, linewidth=2.2)

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def plot_harmonizer_scale(df: pd.DataFrame, out: Path) -> None:
    harm = df[df["mode"] == "harm"]
    plt.figure(figsize=(9, 5.5))
    ax = plt.gca()
    for seed in SEEDS:
        seed_df = harm[harm["seed"] == seed].sort_values("step")
        ax.plot(seed_df["step"], seed_df["scale"], color=MODE_COLORS["harm"], alpha=0.35, linewidth=1)
    mean_df = harm.groupby("step", as_index=False)["scale"].mean()
    ax.plot(mean_df["step"], mean_df["scale"], label="Harmonizer mean", color=MODE_COLORS["harm"], linewidth=2.2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Applied transport scale")
    ax.set_title("Harmonizer Scale Over Training")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def plot_seed_robustness(runs_dir: Path, prefix: str, out: Path) -> None:
    rows = []
    for mode in MODES:
        for seed in SEEDS:
            path = runs_dir / f"{prefix}_{mode}_seed{seed}" / "summary.json"
            summary = pd.read_json(path, typ="series")
            rows.append({
                "mode": mode,
                "seed": seed,
                "final_accuracy": float(summary["final_accuracy"]),
            })
    summary_df = pd.DataFrame(rows)

    plt.figure(figsize=(7.5, 5.2))
    ax = plt.gca()
    positions = range(len(MODES))
    data = [summary_df[summary_df["mode"] == mode]["final_accuracy"].to_numpy() for mode in MODES]
    ax.boxplot(data, positions=list(positions), widths=0.45, showmeans=True)
    for pos, mode in zip(positions, MODES):
        y = summary_df[summary_df["mode"] == mode]["final_accuracy"].to_numpy()
        x = [pos - 0.08, pos, pos + 0.08]
        ax.scatter(x[: len(y)], y, color=MODE_COLORS[mode], zorder=3)
    ax.set_xticks(list(positions), [MODE_LABELS[m] for m in MODES])
    ax.set_ylabel("Final accuracy")
    ax.set_title("Seed Robustness")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def main() -> int:
    global MODES
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", default="pub15k")
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--outdir", type=Path, default=Path("paper_figs"))
    args = parser.parse_args()

    MODES = tuple(args.modes)
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_metrics(args.runs_dir, args.prefix)

    outputs = [
        ("accuracy", "Accuracy", "Accuracy Over Training", args.outdir / "accuracy_over_steps.png", False),
        ("applied_gain", "Applied composite gain", "Applied Gain Over Training", args.outdir / "applied_gain_over_steps.png", True),
        ("raw_gain", "Raw composite gain", "Raw Gain Over Training", args.outdir / "raw_gain_over_steps.png", True),
    ]
    for metric, ylabel, title, out, log_y in outputs:
        plot_metric(df, metric, ylabel, title, out, log_y=log_y)
        print(f"Wrote {out}")

    scale_out = args.outdir / "harmonizer_scale_over_steps.png"
    plot_harmonizer_scale(df, scale_out)
    print(f"Wrote {scale_out}")

    robustness_out = args.outdir / "seed_robustness.png"
    plot_seed_robustness(args.runs_dir, args.prefix, robustness_out)
    print(f"Wrote {robustness_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
