# Aborted Run Note

This was a partial `res_scale` seed 0 attempt started by
`scripts/run_pub15k_matrix.sh` on 2026-05-07 UTC.

It was intentionally stopped after the first logged row because the launcher had
started only the remote baseline job after skipping existing HC/mHC/Harmonizer
runs, leaving the local GPU idle. The attempt is preserved here so the scheduler
restart is not silent.

The publication baseline run for `res_scale` seed 0 was restarted in
`runs/pub15k_res_scale_seed0/`.
