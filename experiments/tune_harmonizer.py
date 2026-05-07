#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.3.0",
#   "torch>=2.11.0",
# ]
# ///
"""
Staged Tuning Sweep for Harmonizer

Stage 0 (screen): seed=0, steps=1200, keep top 6
Stage 1 (confirm): seeds=[0,1], steps=3000, keep top 2
Stage 2 (final): seeds=[0,1,2], steps=3000, top 1 config

Sweep parameters:
- gain_target: [3.0, 5.0, 8.0, 12.0]
- min_scale: [0.02, 0.05, 0.10]
- harm_k: [0.5, 1.0]
"""

import argparse
import csv
import itertools
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from harmonizer_applied_gain import RunConfig, train


@dataclass
class SweepConfig:
    """Configuration for a single sweep point."""
    gain_target: float
    min_scale: float
    harm_k: float
    beta: float = 0.95


def compute_score(result: Dict) -> float:
    """
    Score function:
    score = final_acc
            - 0.05 * max(0, max_gain_applied - 20)^2
            - 0.01 * floor_hits
    """
    acc = result.get("final_acc", 0.0)
    max_gain = result.get("max_gain_applied_max", 0.0)
    floor_hits = result.get("floor_hits", 0)

    gain_penalty = 0.05 * max(0, max_gain - 20) ** 2
    floor_penalty = 0.01 * floor_hits

    return acc - gain_penalty - floor_penalty


def run_config_with_seeds(
    sweep_cfg: SweepConfig,
    seeds: List[int],
    steps: int,
    out_dir: Path,
    log_every: int = 100,
) -> Tuple[Dict, List[Dict]]:
    """Run a config with multiple seeds and aggregate results."""
    results = []

    for seed in seeds:
        name = f"gt{sweep_cfg.gain_target}_ms{sweep_cfg.min_scale}_k{sweep_cfg.harm_k}_s{seed}"
        run_dir = out_dir / name

        cfg = RunConfig(
            name=name,
            mode="harm",
            seed=seed,
            steps=steps,
            gain_target=sweep_cfg.gain_target,
            min_scale=sweep_cfg.min_scale,
            harm_k=sweep_cfg.harm_k,
            beta=sweep_cfg.beta,
            log_every=log_every,
            device="cuda",
        )

        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        result = train(cfg, run_dir)
        result["score"] = compute_score(result)
        results.append(result)

    # Aggregate across seeds
    agg = {
        "gain_target": sweep_cfg.gain_target,
        "min_scale": sweep_cfg.min_scale,
        "harm_k": sweep_cfg.harm_k,
        "beta": sweep_cfg.beta,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "acc_mean": sum(r["final_acc"] for r in results) / len(results),
        "acc_std": (sum((r["final_acc"] - sum(r2["final_acc"] for r2 in results)/len(results))**2 for r in results) / len(results)) ** 0.5,
        "max_gain_raw_mean": sum(r["max_gain_raw_max"] for r in results) / len(results),
        "max_gain_applied_mean": sum(r["max_gain_applied_max"] for r in results) / len(results),
        "harm_scale_mean": sum(r["harm_scale_mean"] for r in results) / len(results),
        "harm_scale_min_mean": sum(r["harm_scale_min"] for r in results) / len(results),
        "floor_hits_mean": sum(r["floor_hits"] for r in results) / len(results),
        "score_mean": sum(r["score"] for r in results) / len(results),
        "time_sec_total": sum(r["time_sec"] for r in results),
    }

    return agg, results


def main():
    parser = argparse.ArgumentParser(description="Staged Harmonizer Tuning Sweep")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--stage", type=int, default=None,
                        help="Run only specific stage (0, 1, or 2)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tuning sweep output: {out_dir}")

    # Define sweep grid
    gain_targets = [3.0, 5.0, 8.0, 12.0]
    min_scales = [0.02, 0.05, 0.10]
    harm_ks = [0.5, 1.0]

    all_configs = [
        SweepConfig(gain_target=gt, min_scale=ms, harm_k=k)
        for gt, ms, k in itertools.product(gain_targets, min_scales, harm_ks)
    ]

    print(f"Total configs in grid: {len(all_configs)}")

    all_results = []
    stage_results = {}

    # =========================================================================
    # Stage 0: Screen (seed=0, steps=1200)
    # =========================================================================
    if args.stage is None or args.stage == 0:
        print(f"\n{'#'*60}")
        print("STAGE 0: Screen (seed=0, steps=1200)")
        print(f"{'#'*60}")

        stage0_dir = out_dir / "stage0"
        stage0_results = []

        for i, sweep_cfg in enumerate(all_configs):
            print(f"\n[Stage 0] Config {i+1}/{len(all_configs)}: "
                  f"gt={sweep_cfg.gain_target}, ms={sweep_cfg.min_scale}, k={sweep_cfg.harm_k}")

            agg, runs = run_config_with_seeds(
                sweep_cfg,
                seeds=[0],
                steps=1200,
                out_dir=stage0_dir,
                log_every=200,
            )
            stage0_results.append(agg)
            all_results.extend(runs)

        # Sort by score and keep top 6
        stage0_results.sort(key=lambda x: x["score_mean"], reverse=True)
        top6 = stage0_results[:6]

        stage_results["stage0"] = {
            "all": stage0_results,
            "top": top6,
        }

        print(f"\n{'='*60}")
        print("Stage 0 Top 6 Configs:")
        print(f"{'='*60}")
        for i, cfg in enumerate(top6):
            print(f"  {i+1}. gt={cfg['gain_target']}, ms={cfg['min_scale']}, k={cfg['harm_k']} "
                  f"| acc={cfg['acc_mean']:.4f}, G_app={cfg['max_gain_applied_mean']:.2f}, "
                  f"score={cfg['score_mean']:.4f}")

        # Save stage 0 results
        with open(stage0_dir / "summary.json", "w") as f:
            json.dump(stage_results["stage0"], f, indent=2)
    else:
        # Load stage 0 results if skipping
        stage0_file = out_dir / "stage0" / "summary.json"
        if stage0_file.exists():
            with open(stage0_file) as f:
                stage_results["stage0"] = json.load(f)
            top6 = stage_results["stage0"]["top"]
        else:
            print("ERROR: Stage 0 results not found. Run stage 0 first.")
            return

    # =========================================================================
    # Stage 1: Confirm (seeds=[0,1], steps=3000)
    # =========================================================================
    if args.stage is None or args.stage == 1:
        print(f"\n{'#'*60}")
        print("STAGE 1: Confirm (seeds=[0,1], steps=3000)")
        print(f"{'#'*60}")

        stage1_dir = out_dir / "stage1"
        stage1_results = []

        for i, cfg_agg in enumerate(top6):
            sweep_cfg = SweepConfig(
                gain_target=cfg_agg["gain_target"],
                min_scale=cfg_agg["min_scale"],
                harm_k=cfg_agg["harm_k"],
            )
            print(f"\n[Stage 1] Config {i+1}/{len(top6)}: "
                  f"gt={sweep_cfg.gain_target}, ms={sweep_cfg.min_scale}, k={sweep_cfg.harm_k}")

            agg, runs = run_config_with_seeds(
                sweep_cfg,
                seeds=[0, 1],
                steps=3000,
                out_dir=stage1_dir,
                log_every=100,
            )
            stage1_results.append(agg)
            all_results.extend(runs)

        # Sort by score and keep top 2
        stage1_results.sort(key=lambda x: x["score_mean"], reverse=True)
        top2 = stage1_results[:2]

        stage_results["stage1"] = {
            "all": stage1_results,
            "top": top2,
        }

        print(f"\n{'='*60}")
        print("Stage 1 Top 2 Configs:")
        print(f"{'='*60}")
        for i, cfg in enumerate(top2):
            print(f"  {i+1}. gt={cfg['gain_target']}, ms={cfg['min_scale']}, k={cfg['harm_k']} "
                  f"| acc={cfg['acc_mean']:.4f} (std={cfg['acc_std']:.4f}), "
                  f"G_app={cfg['max_gain_applied_mean']:.2f}, score={cfg['score_mean']:.4f}")

        # Save stage 1 results
        with open(stage1_dir / "summary.json", "w") as f:
            json.dump(stage_results["stage1"], f, indent=2)
    else:
        # Load stage 1 results if skipping
        stage1_file = out_dir / "stage1" / "summary.json"
        if stage1_file.exists():
            with open(stage1_file) as f:
                stage_results["stage1"] = json.load(f)
            top2 = stage_results["stage1"]["top"]
        else:
            print("ERROR: Stage 1 results not found. Run stage 1 first.")
            return

    # =========================================================================
    # Stage 2: Final (seeds=[0,1,2], steps=3000)
    # =========================================================================
    if args.stage is None or args.stage == 2:
        print(f"\n{'#'*60}")
        print("STAGE 2: Final (seeds=[0,1,2], steps=3000)")
        print(f"{'#'*60}")

        stage2_dir = out_dir / "stage2"
        stage2_results = []

        # Only run top 1 config from stage 1
        best_cfg_agg = top2[0]
        sweep_cfg = SweepConfig(
            gain_target=best_cfg_agg["gain_target"],
            min_scale=best_cfg_agg["min_scale"],
            harm_k=best_cfg_agg["harm_k"],
        )

        print(f"\n[Stage 2] Best config: "
              f"gt={sweep_cfg.gain_target}, ms={sweep_cfg.min_scale}, k={sweep_cfg.harm_k}")

        agg, runs = run_config_with_seeds(
            sweep_cfg,
            seeds=[0, 1, 2],
            steps=3000,
            out_dir=stage2_dir,
            log_every=100,
        )
        stage2_results.append(agg)
        all_results.extend(runs)

        stage_results["stage2"] = {
            "best": agg,
            "runs": [r for r in runs],
        }

        # Save stage 2 results
        with open(stage2_dir / "summary.json", "w") as f:
            # Convert runs to serializable format
            save_data = {
                "best": agg,
                "runs": [{k: v for k, v in r.items()} for r in runs]
            }
            json.dump(save_data, f, indent=2)

    # =========================================================================
    # Final Summary
    # =========================================================================
    print(f"\n{'#'*60}")
    print("FINAL SUMMARY")
    print(f"{'#'*60}")

    if "stage2" in stage_results:
        best = stage_results["stage2"]["best"]
        print(f"\nBest Configuration:")
        print(f"  gain_target = {best['gain_target']}")
        print(f"  min_scale   = {best['min_scale']}")
        print(f"  harm_k      = {best['harm_k']}")
        print(f"  beta        = {best['beta']}")

        print(f"\nStage 2 Results (3 seeds):")
        print(f"  acc_mean          = {best['acc_mean']:.4f} (std={best['acc_std']:.4f})")
        print(f"  max_gain_raw      = {best['max_gain_raw_mean']:.2f}")
        print(f"  max_gain_applied  = {best['max_gain_applied_mean']:.2f}")
        print(f"  harm_scale_mean   = {best['harm_scale_mean']:.4f}")
        print(f"  harm_scale_min    = {best['harm_scale_min_mean']:.4f}")
        print(f"  floor_hits_mean   = {best['floor_hits_mean']:.1f}")
        print(f"  score_mean        = {best['score_mean']:.4f}")

        # Individual seed results
        if "runs" in stage_results["stage2"]:
            print(f"\nPer-seed results:")
            print(f"| seed | final_acc | final_loss | max_G_raw | max_G_app | scale_mean | scale_min | floor_hits | time_sec |")
            print(f"|------|-----------|------------|-----------|-----------|------------|-----------|------------|----------|")
            for r in stage_results["stage2"]["runs"]:
                print(f"| {r['seed']} | {r['final_acc']:.4f} | {r['final_loss']:.4f} | "
                      f"{r['max_gain_raw_max']:.2f} | {r['max_gain_applied_max']:.2f} | "
                      f"{r['harm_scale_mean']:.4f} | {r['harm_scale_min']:.4f} | "
                      f"{r['floor_hits']} | {r['time_sec']:.1f} |")

    # Save overall summary CSV
    summary_csv = out_dir / "summary.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
        print(f"\nAll results saved to: {summary_csv}")

    # Save full stage results
    with open(out_dir / "all_stages.json", "w") as f:
        json.dump(stage_results, f, indent=2, default=str)

    print(f"\nLogs saved to: {out_dir}")


if __name__ == "__main__":
    main()
