from pathlib import Path
import csv

from harmonizer_applied_gain import RunConfig, train


def tiny_cfg(seed: int) -> RunConfig:
    return RunConfig(
        name=f"determinism_s{seed}",
        mode="harm",
        seed=seed,
        vocab=32,
        d=8,
        layers=2,
        n=3,
        steps=4,
        batch=2,
        seq=16,
        lr=1e-3,
        num_kv_pairs=2,
        log_every=1,
        device="cpu",
    )


def test_seeded_tiny_training_is_deterministic(tmp_path: Path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    summary_a = train(tiny_cfg(seed=7), out_a)
    summary_b = train(tiny_cfg(seed=7), out_b)

    assert summary_a["final_loss"] == summary_b["final_loss"]
    assert summary_a["final_accuracy"] == summary_b["final_accuracy"]
    with open(out_a / "metrics.csv", newline="") as f:
        rows_a = list(csv.DictReader(f))
    with open(out_b / "metrics.csv", newline="") as f:
        rows_b = list(csv.DictReader(f))
    for row in rows_a:
        row.pop("wall_time_sec", None)
    for row in rows_b:
        row.pop("wall_time_sec", None)
    assert rows_a == rows_b
