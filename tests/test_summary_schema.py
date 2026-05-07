import json
from pathlib import Path

from jsonschema import Draft202012Validator

from harmonizer_applied_gain import RunConfig, train
from scripts.validate_artifacts import REQUIRED_SUMMARY_FIELDS


SUMMARY_SCHEMA = {
    "type": "object",
    "required": list(REQUIRED_SUMMARY_FIELDS),
    "properties": {
        "mode": {"enum": ["hc", "mhc", "harm"]},
        "seed": {"type": "integer"},
        "steps": {"type": "integer", "minimum": 1},
        "final_accuracy": {"type": "number"},
        "final_loss": {"type": "number"},
        "max_raw_gain": {"type": "number"},
        "max_applied_gain": {"type": "number"},
        "mean_applied_gain": {"type": "number"},
        "std_applied_gain": {"type": "number"},
        "min_scale": {"type": "number"},
        "mean_scale": {"type": "number"},
        "floor_hits": {"type": "integer"},
        "runtime_seconds": {"type": "number"},
        "device": {"type": "string"},
        "commit_sha": {"type": ["string", "null"]},
    },
}


def test_summary_json_matches_required_schema(tmp_path: Path):
    cfg = RunConfig(
        name="schema_smoke",
        mode="hc",
        seed=0,
        vocab=32,
        d=8,
        layers=2,
        n=3,
        steps=2,
        batch=2,
        seq=16,
        num_kv_pairs=2,
        log_every=1,
        device="cpu",
    )

    train(cfg, tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())

    Draft202012Validator(SUMMARY_SCHEMA).validate(summary)
