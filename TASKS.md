# TASKS

This backlog is scoped to turning the current Harmonized Hyper-Connections prototype into a publication-grade, reproducible evidence package.

## Priority 0 — Claim discipline

- [ ] Split the paper claims into three explicit classes:
  - [ ] **Claim A: gain control** — Harmonizer bounds applied composite gain even when raw composite gain drifts or explodes.
  - [ ] **Claim B: routing preservation** — Harmonizer preserves more useful routing capacity than hard mHC projection on key-value retrieval.
  - [ ] **Claim C: performance recovery** — Harmonizer recovers performance comparable to unconstrained HC under matched settings.
- [ ] Add a `MANIFEST.md` mapping every paper claim to exact run directories, figure files, configs, seeds, and commit SHAs.
- [ ] Downgrade or qualify any paper sentence that is not backed by committed artifacts.

Acceptance:

- Every central claim has a specific artifact reference.
- No result is described only narratively when a table or figure should exist.

## Priority 1 — Reproducible comparison pack

Run the same task, model shape, optimizer settings, logging schema, and seed set across all three modes.

- [ ] Add 15k-step runs for `hc` across at least 3 seeds.
- [ ] Add 15k-step runs for `mhc` across at least 3 seeds.
- [ ] Add 15k-step runs for `harm` across at least 3 seeds.
- [ ] Record exact commands in each run directory.
- [ ] Store each run under `runs/<timestamp>_<mode>_<seed>/` or another deterministic naming scheme.
- [ ] Include `config.json`, `summary.json`, and per-step metrics for every run.

Minimum metric schema:

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

Acceptance:

- One command regenerates all comparison runs or validates that all expected artifacts are present.
- The repo contains matched HC, mHC, and Harmonizer evidence, not only the strongest single-mode run.

## Priority 2 — Summary tables and figures

- [ ] Add `results_summary.csv` aggregating all publication runs.
- [ ] Add a generated Markdown table for the paper or README.
- [ ] Add seed-level rows, not only averages.
- [ ] Add aggregate rows with mean, standard deviation, minimum, and maximum.
- [ ] Generate figures from `results_summary.csv` or committed per-run metrics, not hand-entered values.

Required figures:

- [ ] Accuracy over training steps by mode.
- [ ] Applied composite gain over training steps by mode.
- [ ] Raw composite gain over training steps by mode.
- [ ] Harmonizer scale over training steps.
- [ ] Seed robustness plot or table.

Acceptance:

- `make_figures.py` can regenerate every file in `paper_figs/` from committed run artifacts.
- Figures include enough labels to interpret without reading the code.

## Priority 3 — Baselines and ablations

Add at least one stronger stabilization baseline so the comparison is not only HC vs mHC vs Harmonizer.

Candidate baselines:

- [ ] Residual scaling.
- [ ] LayerScale-style learned residual gain.
- [ ] Spectral normalization.
- [ ] Norm clipping.
- [ ] Gradient clipping.

Harmonizer ablations:

- [ ] Gain target sweep.
- [ ] Controller strength sweep.
- [ ] Floor value sweep.
- [ ] Update frequency sweep.
- [ ] Number of residual streams sweep.

Acceptance:

- The paper can say why feedback on applied transport geometry is better than a simpler stabilization trick, or it clearly narrows the claim if it is not better.

## Priority 4 — Tests and invariants

- [ ] Add `pyproject.toml` with project metadata and test/lint dependencies.
- [ ] Add `pytest` tests for Sinkhorn projection behavior.
- [ ] Add tests for composite gain calculation.
- [ ] Add tests for applied-gain scaling.
- [ ] Add tests that Harmonizer bounds applied gain under a synthetic exploding raw-gain case.
- [ ] Add deterministic seed smoke tests.
- [ ] Add JSON schema validation for run summaries.

Acceptance:

- `uv run pytest` passes from a clean checkout.
- At least the core math/control invariants are tested independent of a long training run.

## Priority 5 — Script and tooling cleanup

- [ ] Convert executable Python scripts to `uv`-runnable scripts or provide a proper `pyproject.toml` path.
- [ ] Prefer PEP 723 standalone metadata for single-file experiment scripts that are intended to be run directly.
- [ ] Add exact reproduction commands to the README.
- [ ] Add a `scripts/validate_artifacts.py` command that checks expected run directories, summaries, tables, and figures.
- [ ] Add `.gitignore` rules that keep generated caches out while preserving intentional result artifacts.

Acceptance:

- A new reviewer can clone the repo, run the documented command, and reproduce or validate the committed evidence package.

## Priority 6 — Paper update

- [ ] Insert the results table directly into the PDF/source paper.
- [ ] Insert the key figures directly into the PDF/source paper.
- [ ] Replace approximate narrative values with exact artifact-derived values.
- [ ] Add a limitations section covering synthetic KV retrieval, small seed count, short horizon, and lack of LLM-scale validation.
- [ ] Add a related-work comparison against mHC, residual scaling, LayerScale, sparse routing, and residual-stream stabilization methods.

Acceptance:

- The public paper is self-contained enough that readers can understand the evidence without browsing the repo first.

## Priority 7 — Optional next experiment

After the publication pack is clean, test whether the same control idea transfers beyond synthetic key-value retrieval.

Candidate tasks:

- [ ] Longer sequence key-value retrieval.
- [ ] Multi-query retrieval.
- [ ] Noisy key/value retrieval.
- [ ] Routing contradiction or interference task.
- [ ] Small language-model pretraining smoke test.

Acceptance:

- Do not expand the paper claim until at least one non-synthetic or less synthetic task supports the same pattern.

## Current working interpretation

The repo currently supports the gain-control claim most strongly: Harmonizer can bound applied residual transport while raw transport parameters drift. The performance-recovery and routing-preservation claims need matched multi-seed comparison artifacts before they should be used as central public claims.
