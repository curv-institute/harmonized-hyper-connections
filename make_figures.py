#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "pandas",
#   "matplotlib",
# ]
# ///

"""
Generate paper figures from logged CSVs.

Assumptions:
- Each run directory contains a CSV named one of:
    metrics.csv, log.csv, logs.csv, history.csv
- CSV contains at least:
    step, acc
- For gain plots, CSV contains:
    gain_raw_max, gain_applied_max
  (or compatible columns; see COLUMN_ALIASES below)

Outputs:
- fig_gain_raw_vs_applied.(png|pdf)
- fig_acc_vs_step.(png|pdf)   (if enough runs provided)

Usage examples (your paths):
  uv run make_figures.py \
    --harm15k ~/mhc/runs/20260107_113333_harm15k \
    --hc15k   ~/mhc/runs/A2_hc_15k \
    --outdir  ~/mhc/paper_figs

If you have mHC runs with metrics too:
  uv run make_figures.py \
    --harm15k ~/mhc/runs/20260107_113333_harm15k \
    --hc15k   ~/mhc/runs/A2_hc_15k \
    --mhc3k   ~/mhc/runs/A1_mhc_s0 \
    --outdir  ~/mhc/paper_figs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import pandas as pd
import matplotlib.pyplot as plt


CSV_CANDIDATES = ["metrics.csv", "log.csv", "logs.csv", "history.csv"]

# Allow some flexibility if your column names differ slightly.
COLUMN_ALIASES: Dict[str, List[str]] = {
    "step": ["step", "steps", "global_step", "iter", "iteration"],
    "acc": ["acc", "accuracy", "val_acc"],
    "loss": ["loss", "train_loss"],
    "gain_raw": ["gain_raw_max", "G_raw", "gain_raw", "max_G_raw", "gain_raw_maximum"],
    "gain_applied": ["gain_applied_max", "G_applied", "gain_applied", "max_G_app", "gain_app_max"],
}


def find_csv(run_dir: Path) -> Path:
    if run_dir.is_file() and run_dir.suffix.lower() == ".csv":
        return run_dir
    for name in CSV_CANDIDATES:
        p = run_dir / name
        if p.exists():
            return p
    # fallback: first csv in dir
    csvs = sorted(run_dir.glob("*.csv"))
    if csvs:
        return csvs[0]
    raise FileNotFoundError(f"No CSV found in {run_dir}. Looked for {CSV_CANDIDATES} and *.csv")


def resolve_col(df: pd.DataFrame, key: str) -> str:
    for cand in COLUMN_ALIASES[key]:
        if cand in df.columns:
            return cand
    raise KeyError(f"Missing required column for '{key}'. Tried {COLUMN_ALIASES[key]}. Found: {list(df.columns)}")


def load_run(run_dir: Path) -> pd.DataFrame:
    csv_path = find_csv(run_dir)
    df = pd.read_csv(csv_path)
    # Normalize step to int-ish and sort
    step_col = resolve_col(df, "step")
    df = df.sort_values(step_col).reset_index(drop=True)
    return df


def plot_gain_raw_vs_applied(
    harm_df: pd.DataFrame,
    outdir: Path,
    title: str = "Composite Gain: Raw vs Applied (Harmonizer)",
) -> Tuple[Path, Path]:
    step_col = resolve_col(harm_df, "step")
    raw_col = resolve_col(harm_df, "gain_raw")
    app_col = resolve_col(harm_df, "gain_applied")

    plt.figure(figsize=(10, 6))
    plt.plot(harm_df[step_col], harm_df[raw_col], label="Raw composite gain", linewidth=2)
    plt.plot(harm_df[step_col], harm_df[app_col], label="Applied composite gain", linewidth=2)
    plt.yscale("log")
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Composite gain (log scale)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    out_png = outdir / "fig_gain_raw_vs_applied.png"
    out_pdf = outdir / "fig_gain_raw_vs_applied.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    return out_png, out_pdf


def plot_acc_vs_step(
    runs: List[Tuple[str, pd.DataFrame]],
    outdir: Path,
    title: str = "KV Retrieval Accuracy vs Step",
) -> Optional[Tuple[Path, Path]]:
    if len(runs) < 1:
        return None

    plt.figure(figsize=(10, 6))
    for label, df in runs:
        step_col = resolve_col(df, "step")
        acc_col = resolve_col(df, "acc")
        plt.plot(df[step_col], df[acc_col], label=label, linewidth=2)

    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    out_png = outdir / "fig_acc_vs_step.png"
    out_pdf = outdir / "fig_acc_vs_step.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    return out_png, out_pdf


def plot_harm_scale_vs_step(
    harm_df: pd.DataFrame,
    outdir: Path,
    title: str = "Harmonizer Scale Adaptation",
) -> Tuple[Path, Path]:
    step_col = resolve_col(harm_df, "step")

    if "harm_scale" not in harm_df.columns:
        raise KeyError("harm_scale column not found")

    plt.figure(figsize=(10, 6))
    plt.plot(harm_df[step_col], harm_df["harm_scale"], label="harm_scale", linewidth=2, color="green")
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Scale factor", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    out_png = outdir / "fig_harm_scale.png"
    out_pdf = outdir / "fig_harm_scale.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    return out_png, out_pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harm15k", required=True, help="Path to Harmonizer 15k run dir (or CSV).")
    ap.add_argument("--hc15k", required=False, help="Path to HC 15k run dir (or CSV).")
    ap.add_argument("--mhc3k", required=False, help="Optional: path to an mHC 3k run dir (or CSV).")
    ap.add_argument("--outdir", required=True, help="Output directory for figures.")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    harm_df = load_run(Path(args.harm15k).expanduser().resolve())

    # Plot 1: Gain raw vs applied
    gain_png, gain_pdf = plot_gain_raw_vs_applied(harm_df, outdir)
    print(f"Wrote: {gain_png}")
    print(f"Wrote: {gain_pdf}")

    # Plot 2: harm_scale over time
    try:
        scale_png, scale_pdf = plot_harm_scale_vs_step(harm_df, outdir)
        print(f"Wrote: {scale_png}")
        print(f"Wrote: {scale_pdf}")
    except KeyError as e:
        print(f"Skipping harm_scale plot: {e}")

    # Plot 3: Accuracy plot - include whatever runs were provided.
    acc_runs: List[Tuple[str, pd.DataFrame]] = [("Harmonizer (15k)", harm_df)]
    if args.hc15k:
        try:
            hc_df = load_run(Path(args.hc15k).expanduser().resolve())
            acc_runs.insert(0, ("HC (15k)", hc_df))
        except Exception as e:
            print(f"Warning: Could not load HC run: {e}")
    if args.mhc3k:
        try:
            mhc_df = load_run(Path(args.mhc3k).expanduser().resolve())
            acc_runs.append(("mHC (3k)", mhc_df))
        except Exception as e:
            print(f"Warning: Could not load mHC run: {e}")

    acc_out = plot_acc_vs_step(acc_runs, outdir)
    if acc_out:
        print(f"Wrote: {acc_out[0]}")
        print(f"Wrote: {acc_out[1]}")

    print(f"\nAll figures saved to: {outdir}")


if __name__ == "__main__":
    main()
