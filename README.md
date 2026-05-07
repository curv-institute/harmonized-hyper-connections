# Harmonized Hyper-Connections

Evidence package for:

**Harmonized Hyper-Connections: Stabilizing Residual Transport Through Feedback on Geometry**  
J. W. Miller, CURV Institute, January 2026

Paper: https://curv.institute/publications/harmonized-hyper-connections/

## Scope

This repository is an experimental package for synthetic key-value retrieval runs over Hyper-Connection routing variants. The strongest supported result is the gain-control claim: Harmonizer keeps applied composite gain bounded even when raw composite gain grows sharply.

Routing-preservation and performance-recovery claims are treated as matched-run measurements, not general conclusions. Do not read these artifacts as LLM-scale validation or proof of broader residual-stream theory.

## Modes

- `hc`: unconstrained residual transport baseline.
- `mhc`: hard manifold projection through Sinkhorn-normalized transport.
- `harm`: applied-gain feedback controller over residual transport geometry.

## Main Artifacts

- `runs/pub15k_<mode>_seed<seed>/`: matched 15k-step runs for `hc`, `mhc`, and `harm` over seeds `0`, `1`, and `2`.
- `results_summary.csv`: seed-level publication-run summary.
- `results_aggregate.csv`: aggregate means, standard deviations, minima, and maxima by mode.
- `results_summary.md`: generated seed-level and aggregate Markdown tables.
- `paper_figs/accuracy_over_steps.png`
- `paper_figs/applied_gain_over_steps.png`
- `paper_figs/raw_gain_over_steps.png`
- `paper_figs/harmonizer_scale_over_steps.png`
- `paper_figs/seed_robustness.png`
- `MANIFEST.md`: claim-to-artifact mapping and validation notes.

## Reproduction

Use `uv` from a clean checkout:

```bash
uv sync
uv run pytest
```

Validate the matched run directories:

```bash
uv run python scripts/validate_artifacts.py --prefix pub15k --steps 15000
```

Regenerate the derived tables and figures:

```bash
uv run python scripts/summarize_results.py
uv run python make_figures.py
uv run python scripts/validate_artifacts.py --prefix pub15k --steps 15000 --check-derived
```

Run or resume the full publication matrix:

```bash
EXECUTION_MODE=cluster ./scripts/run_pub15k_matrix.sh
```

The matrix script skips runs that already have `summary.json`, `metrics.csv`, and `manifest.json`.

## Files

- `harmonizer_applied_gain.py`: implementation and training entry point.
- `scripts/run_pub15k_matrix.sh`: matched comparison launcher.
- `scripts/validate_artifacts.py`: run and derived-artifact validator.
- `scripts/summarize_results.py`: generated result tables.
- `make_figures.py`: generated figures from committed run metrics.
- `tests/`: deterministic smoke and invariant tests.
- `evidence/env/`: hardware, cluster, Python, CUDA, and runtime evidence.

## Requirements

- `uv`
- Python `>=3.12`
- PyTorch with CPU support for tests
- CUDA-capable PyTorch for reproducing the 15k GPU runs efficiently

The current evidence artifacts were generated with Python `3.14.3`, PyTorch `2.11.0+cu130`, and NVIDIA GB10 GPUs.

## License

MIT
