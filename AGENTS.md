# AGENTS.md

This repository is an experimental evidence package for **Harmonized Hyper-Connections: Stabilizing Residual Transport Through Feedback on Geometry**.

Agents working here must prioritize reproducibility, claim discipline, and artifact-backed results over new theoretical expansion.

## Mission

Turn the current prototype into a publication-grade, reproducible evidence package.

The immediate objective is to complete `TASKS.md` in priority order:

1. Claim discipline.
2. Matched HC / mHC / Harmonizer comparison runs.
3. Artifact-derived tables and figures.
4. Baselines and ablations.
5. Tests and invariant checks.
6. Tooling cleanup.
7. Paper update.
8. Optional transfer experiments only after the above are clean.

Do not skip ahead to broader claims or new architectures until the matched evidence pack is complete.

## Core claims and evidence standard

Keep three claims separate:

- **Claim A — gain control:** Harmonizer bounds applied composite gain even when raw composite gain drifts or explodes.
- **Claim B — routing preservation:** Harmonizer preserves more useful routing capacity than hard mHC projection on key-value retrieval.
- **Claim C — performance recovery:** Harmonizer recovers performance comparable to unconstrained HC under matched settings.

Current working interpretation:

- Claim A is the strongest supported claim.
- Claims B and C require matched multi-seed HC, mHC, and Harmonizer comparison artifacts before they should be treated as public central claims.

Every claim must map to committed artifacts. If a claim cannot be mapped to a run directory, config, seed, metric file, table, and figure, weaken the claim or mark it as future work.

## Non-negotiable methodology rules

- Determinism first.
- One experimental axis per cycle.
- Negative results are first-class outputs.
- Do not silently drop failed runs.
- Do not cherry-pick best seeds.
- Do not replace seed-level results with averages only.
- Prefer distributions and tails over means alone.
- Keep mechanism, measurement, and interpretation separate.
- Record exact commands and environment details for every non-trivial run.
- Preserve all non-trivial outputs needed to audit conclusions.

## Repository workflow

Before making changes:

1. Read `README.md`.
2. Read `TASKS.md`.
3. Inspect current `runs/`, `paper_figs/`, and experiment scripts.
4. Identify the smallest priority item that can be completed without mixing unrelated changes.

During changes:

- Make small, reviewable commits.
- Keep docs-only, experiment-code, generated-artifact, and paper edits separated when practical.
- Do not perform silent refactors.
- If a refactor is required, explain why and keep it separate from result interpretation.
- Update `TASKS.md` only when there is committed evidence that the item is complete.

After changes:

- Run the relevant validation command.
- Record what was run and what passed or failed.
- Update manifests or summaries with exact artifact references.

## Python and tooling rules

Preferred execution path:

```bash
uv run pytest
```

For directly executable standalone scripts, prefer PEP 723 metadata and a `uv` shebang, for example:

```python
#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "numpy",
# ]
# ///
```

For package-style work, use `pyproject.toml` with explicit dependencies.

Do not add hidden dependency assumptions. A clean checkout should either run through documented `uv` commands or fail with a clear missing capability explanation.

## Required artifact shape

Publication runs should be stored under a deterministic, auditable directory layout such as:

```text
runs/<run_id>_<mode>_<seed>/
  config.json
  command.txt
  summary.json
  metrics.csv
  manifest.json
```

Each run summary should include at minimum:

- `mode`
- `seed`
- `steps`
- `final_accuracy`
- `final_loss`
- `max_raw_gain`
- `max_applied_gain`
- `mean_applied_gain`
- `std_applied_gain`
- `min_scale`
- `mean_scale`
- `floor_hits`
- `runtime_seconds`
- `device`
- `commit_sha`

If a metric is unavailable, do not fake it. Add the missing instrumentation as a task or explicitly mark the metric as unavailable.

## Matched comparison requirement

The central comparison must use the same task, model shape, optimizer settings, logging schema, and seeds across:

- `hc`
- `mhc`
- `harm`

Minimum publication comparison:

```text
hc   15k steps x 3 seeds
mhc  15k steps x 3 seeds
harm 15k steps x 3 seeds
```

Until that pack exists, avoid language claiming definitive performance superiority or routing preservation.

## Figure and table rules

Figures and tables must be generated from committed run artifacts.

Do not hand-enter final numbers into the paper, README, or summary tables unless the source artifact is cited in the same change.

Required outputs:

- `results_summary.csv`
- seed-level Markdown table
- aggregate Markdown table
- accuracy-over-steps figure
- applied-gain-over-steps figure
- raw-gain-over-steps figure
- Harmonizer-scale-over-steps figure
- seed robustness plot or table

`make_figures.py` should regenerate the committed files in `paper_figs/` from committed data.

## Baseline discipline

At least one stronger stabilization baseline is required before broad claims are made.

Candidate baselines:

- residual scaling
- LayerScale-style learned residual gain
- spectral normalization
- norm clipping
- gradient clipping

If Harmonizer does not beat a simpler baseline, report that directly and narrow the claim.

## Test requirements

Add tests before expanding the architecture.

Required tests:

- Sinkhorn projection behavior.
- Composite gain calculation.
- Applied-gain scaling.
- Harmonizer bounds applied gain under synthetic exploding raw gain.
- Deterministic seed smoke behavior.
- JSON schema validation for summaries.

Validation target:

```bash
uv run pytest
```

If a full training run is too expensive for routine tests, create small deterministic unit or smoke tests that validate the invariant directly.

## Paper update rules

The paper should not rely on narrative-only results.

Before updating the public claims:

- Add exact result tables.
- Add generated figures.
- Replace approximate values with artifact-derived values.
- Add limitations.
- Add related work against mHC, residual scaling, LayerScale, sparse routing, and residual-stream stabilization methods.

Limitations must explicitly cover:

- synthetic key-value retrieval
- small seed count
- short horizon
- lack of LLM-scale validation
- hardware/runtime constraints

## What not to do

- Do not claim MLR/RIFT proof from this repo.
- Do not claim LLM-scale validation from synthetic KV retrieval.
- Do not claim Harmonizer is generally superior without matched baselines.
- Do not hide failed seeds.
- Do not delete inconvenient run artifacts unless there is a documented storage or corruption reason.
- Do not mix paper rewriting with new experiment implementation in the same change unless unavoidable.
- Do not expand into new tasks before closing the matched comparison pack.

## Completion definition

The work is complete when a reviewer can:

1. Clone the repo.
2. Run the documented validation command.
3. Inspect `MANIFEST.md`.
4. Regenerate or validate all key figures and tables.
5. Trace every central paper claim to exact committed artifacts.
6. See matched HC / mHC / Harmonizer results across the same seeds and settings.

Until then, treat the repo as a strong prototype and gain-control demonstration, not a finished publication-grade proof package.
