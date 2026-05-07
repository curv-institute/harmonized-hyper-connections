# Evidence Manifest

This manifest maps the current repository claims to exact artifacts for the
matched 15k-step Hyper-Connection comparison pack and fixed residual-scaling
baseline.

Generated and validated on 2026-05-07 UTC.

## Scope

The publication comparison pack covers synthetic key-value retrieval only. It
uses matched task, model, optimizer, logging schema, and seeds across the
central comparison modes:

- `hc`
- `mhc`
- `harm`

It also includes a fixed residual-scaling stabilization baseline:

- `res_scale` with `residual_scale = 0.25`

Seeds: `0`, `1`, `2`  
Steps per run: `15000`

The strongest supported claim is gain control. Routing preservation and
performance recovery should be treated as limited matched-run measurements, not
general conclusions.

## Claim Map

### Claim A: Gain Control

Status: supported for this synthetic 15k-step setting.

Harmonizer bounded applied composite gain near the configured target while raw
composite gain grew sharply:

- `runs/pub15k_harm_seed0`: max raw gain `653.6419`, max applied gain `12.4039`, floor hits `0`.
- `runs/pub15k_harm_seed1`: max raw gain `12985592.0`, max applied gain `12.6605`, floor hits `0`.
- `runs/pub15k_harm_seed2`: max raw gain `123775856.0`, max applied gain `12.3913`, floor hits `0`.

The fixed residual-scaling baseline attenuated transport but did not provide
target-bounded control in all matched seeds:

- `runs/pub15k_res_scale_seed0`: max raw gain `292508.9375`, max applied gain `27.1986`, floor hits `0`.
- `runs/pub15k_res_scale_seed1`: max raw gain `3614366.5`, max applied gain `43.2408`, floor hits `0`.
- `runs/pub15k_res_scale_seed2`: max raw gain `6019.4346`, max applied gain `9.3065`, floor hits `0`.

Primary artifacts:

- `results_summary.csv`
- `results_aggregate.csv`
- `results_summary.md`
- `paper_figs/applied_gain_over_steps.png`
- `paper_figs/raw_gain_over_steps.png`
- `paper_figs/harmonizer_scale_over_steps.png`
- `runs/pub15k_harm_seed0/summary.json`
- `runs/pub15k_harm_seed1/summary.json`
- `runs/pub15k_harm_seed2/summary.json`
- `runs/pub15k_harm_seed0/metrics.csv`
- `runs/pub15k_harm_seed1/metrics.csv`
- `runs/pub15k_harm_seed2/metrics.csv`
- `runs/pub15k_res_scale_seed0/summary.json`
- `runs/pub15k_res_scale_seed1/summary.json`
- `runs/pub15k_res_scale_seed2/summary.json`

Interpretation boundary: this supports bounded applied transport under the
measured synthetic setup. It also shows that fixed scalar attenuation is not the
same mechanism as feedback on applied transport geometry. It does not establish
LLM-scale behavior.

### Claim B: Routing Preservation

Status: not established as a central claim by the current artifacts.

The matched pack includes `hc`, `mhc`, and `harm` task metrics, but it does not
include a separate routing-capacity or routing-preservation measurement beyond
final synthetic retrieval accuracy. As a result, routing preservation should be
described as future work or as an open measurement target.

Relevant artifacts:

- `results_summary.csv`
- `results_aggregate.csv`
- `paper_figs/accuracy_over_steps.png`
- `paper_figs/seed_robustness.png`
- `runs/pub15k_mhc_seed0/`
- `runs/pub15k_mhc_seed1/`
- `runs/pub15k_mhc_seed2/`
- `runs/pub15k_harm_seed0/`
- `runs/pub15k_harm_seed1/`
- `runs/pub15k_harm_seed2/`

Current result: `harm` and `mhc` have the same mean final accuracy in this
3-seed pack (`0.0729167`), with different seed-level behavior.

### Claim C: Performance Recovery

Status: mixed and not supported as a broad claim by this pack.

Seed-level final accuracy:

| mode | seed 0 | seed 1 | seed 2 | mean |
| --- | ---: | ---: | ---: | ---: |
| `hc` | 0.078125 | 0.125 | 0.125 | 0.109375 |
| `mhc` | 0.109375 | 0.0625 | 0.046875 | 0.0729167 |
| `harm` | 0.046875 | 0.125 | 0.046875 | 0.0729167 |
| `res_scale` | 0.0625 | 0.046875 | 0.015625 | 0.0416667 |

`harm` recovered `hc`-level final accuracy on seed 1, but not on seeds 0 or 2.
The mean final accuracy matches `mhc` and trails `hc` in the current matched
pack. `res_scale` trails the central comparison modes on mean final accuracy.
Any public statement should reflect this mixed evidence.

Primary artifacts:

- `results_summary.csv`
- `results_aggregate.csv`
- `results_summary.md`
- `paper_figs/accuracy_over_steps.png`
- `paper_figs/seed_robustness.png`

## Run Matrix

Each run directory contains `config.json`, `command.txt`, `summary.json`,
`metrics.csv`, `manifest.json`, `stdout.log`, and `stderr.log`.

| mode | seed | status | final accuracy | max raw gain | max applied gain | min scale | floor hits | commit sha | run dir |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `hc` | 0 | completed | 0.078125 | 62.8828 | 62.8828 | 1.0 | 0 | `f6bdf6155b2b683c4193c698db2f70f7db849918` | `runs/pub15k_hc_seed0` |
| `hc` | 1 | completed | 0.125 | 3764.6646 | 3764.6646 | 1.0 | 0 | `71be6567d492762cd995ca8927cb27e637cf6645` | `runs/pub15k_hc_seed1` |
| `hc` | 2 | completed | 0.125 | 740.9824 | 740.9824 | 1.0 | 0 | `71be6567d492762cd995ca8927cb27e637cf6645` | `runs/pub15k_hc_seed2` |
| `mhc` | 0 | completed | 0.109375 | 605.2688 | 1.0307 | 1.0 | 0 | `71be6567d492762cd995ca8927cb27e637cf6645` | `runs/pub15k_mhc_seed0` |
| `mhc` | 1 | completed | 0.0625 | 37.2023 | 1.0300 | 1.0 | 0 | `71be6567d492762cd995ca8927cb27e637cf6645` | `runs/pub15k_mhc_seed1` |
| `mhc` | 2 | completed | 0.046875 | 415.6855 | 1.0308 | 1.0 | 0 | `71be6567d492762cd995ca8927cb27e637cf6645` | `runs/pub15k_mhc_seed2` |
| `harm` | 0 | completed | 0.046875 | 653.6419 | 12.4039 | 0.3300 | 0 | `71be6567d492762cd995ca8927cb27e637cf6645` | `runs/pub15k_harm_seed0` |
| `harm` | 1 | completed | 0.125 | 12985592.0 | 12.6605 | 0.1055 | 0 | `71be6567d492762cd995ca8927cb27e637cf6645` | `runs/pub15k_harm_seed1` |
| `harm` | 2 | completed | 0.046875 | 123775856.0 | 12.3913 | 0.1071 | 0 | `71be6567d492762cd995ca8927cb27e637cf6645` | `runs/pub15k_harm_seed2` |
| `res_scale` | 0 | completed | 0.0625 | 292508.9375 | 27.1986 | 1.0 | 0 | `db9c517a62c4b07a0556514f8d343d5d797442a6` | `runs/pub15k_res_scale_seed0` |
| `res_scale` | 1 | completed | 0.046875 | 3614366.5 | 43.2408 | 1.0 | 0 | `db9c517a62c4b07a0556514f8d343d5d797442a6` | `runs/pub15k_res_scale_seed1` |
| `res_scale` | 2 | completed | 0.015625 | 6019.4346 | 9.3065 | 1.0 | 0 | `db9c517a62c4b07a0556514f8d343d5d797442a6` | `runs/pub15k_res_scale_seed2` |

Commit note: `runs/pub15k_hc_seed0` was generated under earlier commit
`f6bdf6155b2b683c4193c698db2f70f7db849918` with the same matched run config and
logging schema before tooling commit `71be6567d492762cd995ca8927cb27e637cf6645`.
It is preserved as generated rather than edited. Rerun `hc` seed 0 if a strict
single-commit evidence pack is required.

Aborted-run note: `runs/aborted_pub15k_res_scale_seed0_scheduler_restart_20260507/`
preserves a partial scheduler launch for `res_scale` seed 0. It was stopped
after the first logged row because the skipped existing modes left the local GPU
idle; the publication baseline seed 0 run is `runs/pub15k_res_scale_seed0/`.

## Derived Outputs

Generated outputs:

- `results_summary.csv`
- `results_aggregate.csv`
- `results_summary.md`
- `paper_figs/accuracy_over_steps.png`
- `paper_figs/applied_gain_over_steps.png`
- `paper_figs/raw_gain_over_steps.png`
- `paper_figs/harmonizer_scale_over_steps.png`
- `paper_figs/seed_robustness.png`

Generation commands:

```bash
uv run python scripts/summarize_results.py --modes hc mhc harm res_scale
uv run python make_figures.py --modes hc mhc harm res_scale
```

## Validation

Run artifact validation:

```bash
uv run python scripts/validate_artifacts.py --prefix pub15k --steps 15000 --modes hc mhc harm res_scale
```

Result on 2026-05-07 UTC: passed.

Run and derived artifact validation:

```bash
uv run python scripts/validate_artifacts.py --prefix pub15k --steps 15000 --modes hc mhc harm res_scale --check-derived
```

Result on 2026-05-07 UTC: passed.

Test command:

```bash
uv run pytest
```

Latest result on 2026-05-07 UTC after the residual-scaling baseline update:
`8 passed`.

## Environment Evidence

Environment and cluster evidence:

- `evidence/env/python-version.txt`
- `evidence/env/uv-version.txt`
- `evidence/env/uv-python-version.txt`
- `evidence/env/nvidia-smi.txt`
- `evidence/env/nvidia-topology.txt`
- `evidence/env/cluster/cluster_inventory.json`
- `evidence/env/cluster/local-hostname.txt`
- `evidence/env/cluster/local-fqdn.txt`
- `evidence/env/cluster/local-ip-brief.txt`
- `evidence/env/cluster/local-ibstat.txt`

The current artifacts were generated with Python `3.14.3`, PyTorch
`2.11.0+cu130`, CUDA-enabled execution, and NVIDIA GB10 GPUs.

## Paper Source Note

No local paper source file is present in this checkout. The repository-facing
README and this manifest have been updated to avoid unsupported claims. Public
paper text still needs a separate source update if that source lives outside the
repository.
