# Agent Kickoff: Harmonized Hyper-Connections Evidence Pack on 2-Node DGX Cluster

You are operating as an execution agent on the CURV 2-node DGX Spark cluster. Your job is to turn `curv-institute/harmonized-hyper-connections` into a publication-grade, reproducible evidence package by completing the highest-priority items in `TASKS.md` and following `AGENTS.md` exactly.

This kickoff has been updated using the operating conventions from `curv-sophia`:

- use `/gfs/git/...` for local repository checkouts
- use `/gfs/shared/...` for durable evidence outputs
- use `/gfs/tmp/...` for scratch worktrees and temporary execution artifacts
- prefer local reference repositories under `/gfs/git/` before remote mirrors
- use `uv` and PEP 723 / `pyproject.toml` discipline
- prefer Python 3.12 operational posture when compatible with CUDA/PyTorch
- preserve negative results and failed seeds
- keep claim boundaries explicit

## Repository

Expected checkout location:

```bash
/gfs/git/harmonized-hyper-connections
```

If the repo is missing, clone it under `/gfs/git`:

```bash
mkdir -p /gfs/git
cd /gfs/git
gh repo clone curv-institute/harmonized-hyper-connections
cd /gfs/git/harmonized-hyper-connections
```

Use the existing default branch unless directed otherwise. Create a work branch for changes:

```bash
cd /gfs/git/harmonized-hyper-connections
git status
git checkout main
git pull --ff-only
git checkout -b evidence/matched-15k-pack
```

## Cluster conventions from curv-sophia

The concrete two-node cluster topology from the `curv-sophia` work is:

```text
dgx        = controller / NFS host / shared `/gfs` exporter / primary orchestration node
gx10-e313  = worker node / `/gfs` NFS client / secondary execution node
```

Known cluster posture:

```text
SSH: passwordless SSH is configured between nodes
sudo: passwordless sudo is configured
storage: `/gfs` is shared from `dgx` to `gx10-e313`
local storage: both nodes have local NVMe for hot/cache/scratch work
network: nodes are direct-connected by InfiniBand / RDMA-capable link
```

Use `dgx` and `gx10-e313` as the default SSH targets unless live probes prove aliases differ. Before launching jobs, still verify and record hostnames, SSH reachability, RDMA/IP addresses, sudo posture, and shared filesystem state.

Use these filesystem roles:

```text
/gfs/git/harmonized-hyper-connections          # canonical repo checkout
/gfs/shared/harmonized-hyper-connections       # durable non-repo evidence, large artifacts, cross-node outputs
/gfs/tmp/harmonized-hyper-connections          # scratch worktrees, temporary execution dirs, disposable staging
```

Do not use `/tmp` for new scratch artifacts unless the machine environment forces it. Use `/gfs/tmp` instead.

Use local reference repos first when available:

```text
/gfs/git/curv-sophia
/gfs/git/curv-embedding
/gfs/git/universal_tokenizer
/gfs/git/route-bench
/gfs/git/harmonized-hyper-connections
/gfs/git/curv-jepa
/gfs/git/curv-cot-harness
/gfs/git/curv-wiki
```

Reference repos are lineage and implementation guidance only:

- inspect them for workflow, evidence, validation, and artifact patterns
- do not copy runtime modules wholesale
- do not create imports across repos
- do not create hidden runtime dependencies on sibling repositories

## Hardware goal

Use the 2-node DGX Spark cluster to produce the matched evidence pack quickly and reproducibly.

Do not use the second node just for show. Use it only if it lets you run independent seeds/modes concurrently while preserving deterministic, isolated run directories.

Known cluster posture from CURV / `curv-sophia` work:

- 2 DGX Spark nodes are available.
- `dgx` is the controller and NFS host.
- `gx10-e313` is the worker node.
- `/gfs` is the shared working filesystem exported from `dgx` to `gx10-e313`.
- local NVMe is available on both nodes for hot/cache/scratch work.
- the nodes are direct-connected by InfiniBand / RDMA-capable link.
- passwordless SSH is configured.
- passwordless sudo is configured.
- GPUs should be visible through CUDA/NVIDIA tooling when configured correctly.

Before relying on the RDMA/IP layer, measure it and record it. The concrete RDMA IPs, interface names, SSH reachability, and sudo state must be written into the evidence pack before running the 15k matrix.

## Primary objective

Complete the matched publication comparison pack:

```text
hc   15k steps x 3 seeds
mhc  15k steps x 3 seeds
harm 15k steps x 3 seeds
```

All runs must use the same:

- task
- model shape
- optimizer settings
- step count
- logging schema
- seed set
- hardware/environment recording format

Use seeds:

```text
0, 1, 2
```

If any run fails, keep the failed artifact directory and record the failure. Do not hide failed seeds.

## Claim discipline

Keep these claims separate:

1. **Gain control** — Harmonizer bounds applied composite gain even when raw composite gain drifts or explodes.
2. **Routing preservation** — Harmonizer preserves more useful routing capacity than hard mHC projection on key-value retrieval.
3. **Performance recovery** — Harmonizer recovers performance comparable to unconstrained HC under matched settings.

The repo currently supports claim 1 most strongly. Claims 2 and 3 require this matched multi-seed evidence pack before becoming central public claims.

Do not claim proof of MLR/RIFT from this repo. Do not claim LLM-scale validation from synthetic KV retrieval.

## First-pass inspection

Run:

```bash
cd /gfs/git/harmonized-hyper-connections
pwd
git status
find . -maxdepth 2 -type f | sort
find runs -maxdepth 2 -type f | sort || true
find paper_figs -maxdepth 2 -type f | sort || true
sed -n '1,220p' README.md
sed -n '1,260p' TASKS.md
sed -n '1,360p' AGENTS.md
sed -n '1,260p' harmonizer_applied_gain.py
sed -n '1,260p' make_figures.py
```

Record what already exists before changing anything.

## Environment setup

Prefer `uv`.

CURV/Sophia operational posture uses:

```text
UV_PYTHON=3.12.13
uv run python --version
uv python list
uv pip list --outdated --format columns
```

For this repo, use Python 3.12 if CUDA/PyTorch compatibility is clean. If PyTorch/CUDA requires another runtime, record the exact reason in `MANIFEST.md`, `evidence/env/`, and the final report.

If `pyproject.toml` does not exist, either add it or use PEP 723 standalone metadata for directly executable scripts. Do not leave hidden dependencies.

Minimum validation setup:

```bash
cd /gfs/git/harmonized-hyper-connections
python --version
UV_PYTHON=3.12.13 uv run python --version || true
uv --version || true
uv python list || true
uv pip list --outdated --format columns || true
nvidia-smi || true
```

Capture hardware and runtime info:

```bash
mkdir -p evidence/env
hostname > evidence/env/hostname.txt
uname -a > evidence/env/uname.txt
python --version > evidence/env/python-version.txt
UV_PYTHON=3.12.13 uv run python --version > evidence/env/uv-python-version.txt 2>&1 || true
uv --version > evidence/env/uv-version.txt 2>&1 || true
uv python list > evidence/env/uv-python-list.txt 2>&1 || true
uv pip list --outdated --format columns > evidence/env/uv-outdated.txt 2>&1 || true
nvidia-smi > evidence/env/nvidia-smi.txt 2>&1 || true
nvidia-smi topo -m > evidence/env/nvidia-topology.txt 2>&1 || true
lscpu > evidence/env/lscpu.txt 2>&1 || true
free -h > evidence/env/memory.txt 2>&1 || true
df -h /gfs > evidence/env/gfs-df.txt 2>&1 || true
git rev-parse HEAD > evidence/env/base-commit.txt
```

Verify the concrete cluster targets before running jobs:

```bash
mkdir -p evidence/env/cluster
cat > evidence/env/cluster/expected_cluster_topology.json <<'EOF'
{
  "controller": {
    "hostname": "dgx",
    "ssh_target": "dgx",
    "roles": ["controller", "nfs_host", "gfs_exporter", "primary_orchestrator"]
  },
  "worker": {
    "hostname": "gx10-e313",
    "ssh_target": "gx10-e313",
    "roles": ["worker", "gfs_client", "secondary_executor"]
  },
  "storage": {
    "shared_mount": "/gfs",
    "exporter": "dgx",
    "client": "gx10-e313",
    "local_nvme_on_both_nodes": true
  },
  "access": {
    "passwordless_ssh_expected": true,
    "passwordless_sudo_expected": true
  },
  "network": {
    "direct_infiniband_or_rdma_link_expected": true,
    "rdma_ips": "discover_and_fill"
  }
}
EOF

# Local node identity.
hostname > evidence/env/cluster/local-hostname.txt
hostname -f > evidence/env/cluster/local-fqdn.txt 2>&1 || true
ip -br addr > evidence/env/cluster/local-ip-brief.txt 2>&1 || true
ip addr > evidence/env/cluster/local-ip-addr.txt 2>&1 || true
ip route > evidence/env/cluster/local-ip-route.txt 2>&1 || true

# Look for existing Sophia cluster notes or evidence with hostnames/RDMA/SSH/sudo details.
if [ -d /gfs/git/curv-sophia ]; then
  find /gfs/git/curv-sophia -maxdepth 4 -type f \
    \( -iname '*cluster*' -o -iname '*dgx*' -o -iname '*spark*' -o -iname '*rdma*' -o -iname '*infiniband*' -o -iname '*ssh*' -o -iname '*sudo*' -o -iname '*env*' \) \
    | sort > evidence/env/cluster/curv-sophia-cluster-candidate-files.txt

  grep -RInE 'rdma|infiniband|ib[0-9]|ssh|sudo|hostname|hostnames|/gfs|node[0-9]|DGX|Spark|ip addr|192\.|10\.|172\.' \
    /gfs/git/curv-sophia \
    --exclude-dir=.git \
    --exclude-dir=.venv \
    --exclude-dir=runs \
    --exclude-dir=.pytest_cache \
    > evidence/env/cluster/curv-sophia-cluster-grep.txt 2>&1 || true
fi

# Discover likely RDMA interfaces locally.
ibstat > evidence/env/cluster/local-ibstat.txt 2>&1 || true
ibv_devinfo > evidence/env/cluster/local-ibv-devinfo.txt 2>&1 || true
rdma link > evidence/env/cluster/local-rdma-link.txt 2>&1 || true
ls -R /sys/class/infiniband > evidence/env/cluster/local-sys-class-infiniband.txt 2>&1 || true

# Sudo posture. This must be non-interactive; do not block on password prompts.
sudo -n true > evidence/env/cluster/local-sudo-noninteractive.txt 2>&1 && echo ok >> evidence/env/cluster/local-sudo-noninteractive.txt || true
sudo -n -l > evidence/env/cluster/local-sudo-list.txt 2>&1 || true
```

Build a concrete inventory file before launching jobs:

```bash
cat > evidence/env/cluster/cluster_inventory.template.json <<'EOF'
{
  "nodes": [
    {
      "name": "REPLACE_WITH_REAL_HOSTNAME",
      "ssh_target": "REPLACE_WITH_REAL_SSH_TARGET",
      "management_ip": "REPLACE_WITH_REAL_IP_OR_NULL",
      "rdma_ip": "REPLACE_WITH_REAL_RDMA_IP_OR_NULL",
      "gfs_mounted": null,
      "sudo_noninteractive": null,
      "gpu_count": null,
      "notes": "populate from local probes and curv-sophia evidence"
    }
  ]
}
EOF
```

Probe both known nodes explicitly:

```bash
for node in dgx gx10-e313; do
  mkdir -p "evidence/env/cluster/${node}"
  ssh "$node" 'hostname' > "evidence/env/cluster/${node}/hostname.txt" 2>&1 || true
  ssh "$node" 'hostname -f' > "evidence/env/cluster/${node}/fqdn.txt" 2>&1 || true
  ssh "$node" 'ip -br addr' > "evidence/env/cluster/${node}/ip-brief.txt" 2>&1 || true
  ssh "$node" 'ip addr' > "evidence/env/cluster/${node}/ip-addr.txt" 2>&1 || true
  ssh "$node" 'ip route' > "evidence/env/cluster/${node}/ip-route.txt" 2>&1 || true
  ssh "$node" 'nvidia-smi' > "evidence/env/cluster/${node}/nvidia-smi.txt" 2>&1 || true
  ssh "$node" 'nvidia-smi topo -m' > "evidence/env/cluster/${node}/nvidia-topology.txt" 2>&1 || true
  ssh "$node" 'df -h /gfs' > "evidence/env/cluster/${node}/gfs-df.txt" 2>&1 || true
  ssh "$node" 'ibstat' > "evidence/env/cluster/${node}/ibstat.txt" 2>&1 || true
  ssh "$node" 'ibv_devinfo' > "evidence/env/cluster/${node}/ibv-devinfo.txt" 2>&1 || true
  ssh "$node" 'rdma link' > "evidence/env/cluster/${node}/rdma-link.txt" 2>&1 || true
  ssh "$node" 'sudo -n true' > "evidence/env/cluster/${node}/sudo-noninteractive.txt" 2>&1 && echo ok >> "evidence/env/cluster/${node}/sudo-noninteractive.txt" || true
  ssh "$node" 'sudo -n -l' > "evidence/env/cluster/${node}/sudo-list.txt" 2>&1 || true
done
```

Populate a final inventory file:

```text
evidence/env/cluster/cluster_inventory.json
```

The final inventory must include:

- actual hostname
- SSH target / alias
- node role: `controller` for `dgx`, `worker` for `gx10-e313`
- management IP if identifiable
- RDMA IP if identifiable
- RDMA interface name
- GPU count
- `/gfs` mount status
- whether `/gfs` is exporter or client on that node
- sudo non-interactive status
- evidence file paths used to confirm each fact

If the cluster has a scheduler, use it. If there is no scheduler, use `tmux` or carefully controlled background jobs. Do not oversubscribe GPUs blindly.

## Output placement rules

Small, publication-grade artifacts should be committed when reasonable:

```text
runs/pub15k_<mode>_seed<seed>/summary.json
runs/pub15k_<mode>_seed<seed>/config.json
runs/pub15k_<mode>_seed<seed>/command.txt
runs/pub15k_<mode>_seed<seed>/metrics.csv
runs/pub15k_<mode>_seed<seed>/manifest.json
results_summary.csv
results_summary.md
paper_figs/*.png
MANIFEST.md
```

Large raw logs, checkpoints, or heavyweight traces should go to:

```text
/gfs/shared/harmonized-hyper-connections/<run_id>/
```

Then commit manifest entries with:

- path
- byte size
- SHA256 checksum
- producing command
- producing commit SHA
- hostname/device

Scratch worktrees and temporary staging belong under:

```text
/gfs/tmp/harmonized-hyper-connections/<run_id>/
```

## Execution strategy

### Step 1 — single-node smoke first

Before launching all 15k runs, run a short smoke test for each mode on one GPU:

```bash
python harmonizer_applied_gain.py --mode hc --steps 100 --seed 0 --out-dir runs/smoke_hc_seed0
python harmonizer_applied_gain.py --mode mhc --steps 100 --seed 0 --out-dir runs/smoke_mhc_seed0
python harmonizer_applied_gain.py --mode harm --steps 100 --seed 0 --out-dir runs/smoke_harm_seed0
```

If the script does not support `--seed`, `--out-dir`, or structured output paths, add those first.

### Step 2 — distributed independent run pack

Run independent mode/seed combinations concurrently across available GPUs/nodes.

Target matrix:

```text
mode=hc   seed=0 steps=15000
mode=hc   seed=1 steps=15000
mode=hc   seed=2 steps=15000
mode=mhc  seed=0 steps=15000
mode=mhc  seed=1 steps=15000
mode=mhc  seed=2 steps=15000
mode=harm seed=0 steps=15000
mode=harm seed=1 steps=15000
mode=harm seed=2 steps=15000
```

Each run must write to a unique directory:

```text
runs/pub15k_<mode>_seed<seed>/
```

Required files per run:

```text
config.json
command.txt
summary.json
metrics.csv
manifest.json
stdout.log
stderr.log
```

Minimum summary fields:

```json
{
  "mode": "harm",
  "seed": 0,
  "steps": 15000,
  "final_accuracy": null,
  "final_loss": null,
  "max_raw_gain": null,
  "max_applied_gain": null,
  "mean_applied_gain": null,
  "std_applied_gain": null,
  "min_scale": null,
  "mean_scale": null,
  "floor_hits": null,
  "runtime_seconds": null,
  "device": null,
  "hostname": null,
  "commit_sha": null
}
```

Do not fake unavailable metrics. Add instrumentation instead.

## Suggested launcher

Create:

```bash
scripts/run_pub15k_matrix.sh
```

It should:

- fail fast on missing repo state problems
- create deterministic run directories
- write `command.txt` before execution
- capture stdout/stderr
- record hostname, GPU, commit SHA, and timing
- allow per-run resume/skip only when the expected summary exists and validates
- support node-local execution and SSH-launched remote execution
- avoid running multiple processes on the same GPU unless explicitly configured

Example shape:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODES=(hc mhc harm)
SEEDS=(0 1 2)
STEPS=15000

for mode in "${MODES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="runs/pub15k_${mode}_seed${seed}"
    mkdir -p "$out"
    cmd="python harmonizer_applied_gain.py --mode ${mode} --steps ${STEPS} --seed ${seed} --out-dir ${out}"
    echo "$cmd" > "$out/command.txt"
    echo "Running $cmd"
    bash -lc "$cmd" > "$out/stdout.log" 2> "$out/stderr.log"
  done
done
```

If the current Python script does not support this interface, implement it cleanly.

## Parallelization guidance for 2-node DGX

Use independent seeds/modes as embarrassingly parallel jobs. Avoid distributed training unless the code already supports it.

Use the concrete hostnames from:

```text
evidence/env/cluster/cluster_inventory.json
```

Default mapping:

```text
dgx:       hc seed0, mhc seed0, harm seed0, plus summaries/figures
gx10-e313: hc seed1, mhc seed1, harm seed1, then seed2 spillover
```

Do not use `node1` / `node2` in launcher scripts.

If each node has multiple GPUs, assign one process per GPU with `CUDA_VISIBLE_DEVICES`.

Example local pattern:

```bash
CUDA_VISIBLE_DEVICES=0 python harmonizer_applied_gain.py --mode hc   --seed 0 --steps 15000 --out-dir runs/pub15k_hc_seed0 &
CUDA_VISIBLE_DEVICES=1 python harmonizer_applied_gain.py --mode mhc  --seed 0 --steps 15000 --out-dir runs/pub15k_mhc_seed0 &
CUDA_VISIBLE_DEVICES=2 python harmonizer_applied_gain.py --mode harm --seed 0 --steps 15000 --out-dir runs/pub15k_harm_seed0 &
wait
```

Use SSH only for independent jobs after `/gfs` consistency is verified:

```bash
ssh gx10-e313 'cd /gfs/git/harmonized-hyper-connections && CUDA_VISIBLE_DEVICES=0 python harmonizer_applied_gain.py --mode hc --seed 1 --steps 15000 --out-dir runs/pub15k_hc_seed1'
```

Capture remote stdout/stderr into the correct run directory. Do not rely on terminal scrollback as evidence.

## Required code improvements if missing

Add or fix CLI arguments:

- `--mode {hc,mhc,harm}`
- `--steps N`
- `--seed N`
- `--out-dir PATH`
- `--device cuda|cpu|auto`
- `--log-every N`

Add deterministic seeding for:

- Python random
- NumPy, if used
- PyTorch CPU
- PyTorch CUDA

Add structured metric logging:

```text
step,loss,accuracy,raw_gain,applied_gain,scale,mode,seed
```

Add summary generation from the metric stream rather than ad hoc print parsing.

## Validation scripts

Add:

```text
scripts/validate_artifacts.py
scripts/summarize_results.py
```

`validate_artifacts.py` must verify:

- all 9 run directories exist
- every required file exists
- each `summary.json` contains the required schema
- each `metrics.csv` has expected columns
- `steps` match expected value
- mode/seed in config and summary match the directory name

`summarize_results.py` must produce:

```text
results_summary.csv
results_summary.md
```

Both scripts should be runnable with `uv run` or via the project environment.

## Figures

Update or extend `make_figures.py` so it regenerates:

```text
paper_figs/accuracy_over_steps.png
paper_figs/applied_gain_over_steps.png
paper_figs/raw_gain_over_steps.png
paper_figs/harmonizer_scale_over_steps.png
paper_figs/seed_robustness.png
```

Figures must be generated from committed run artifacts, not hard-coded numbers.

## Tests

Add tests before claiming closure:

```text
tests/test_sinkhorn.py
tests/test_composite_gain.py
tests/test_harmonizer_scaling.py
tests/test_summary_schema.py
tests/test_seed_determinism.py
```

Minimum command:

```bash
uv run pytest
```

If `uv run pytest` is not possible yet, make it possible.

## Manifest

Create `MANIFEST.md` mapping:

- claim
- artifact path
- run IDs
- seeds
- exact commands
- commit SHA
- validation command
- result summary table path
- figure path
- hardware node / GPU allocation
- concrete hostnames and SSH targets: `dgx`, `gx10-e313`
- node roles: `dgx` controller/NFS host, `gx10-e313` worker/NFS client
- management IPs and RDMA IPs if identifiable
- sudo non-interactive status
- `/gfs/shared` paths and checksums for large artifacts

Minimum sections:

```markdown
# MANIFEST

## Repository state

## Hardware / environment

## Claims

### Claim A — Gain control

### Claim B — Routing preservation

### Claim C — Performance recovery

## Runs

## Tables

## Figures

## Validation

## Shared-storage artifacts

## Known limitations
```

## Paper/README update

After artifacts exist:

- Update `README.md` with reproduction commands.
- Add a concise results table.
- Add limitations.
- Link `MANIFEST.md`.
- Do not overclaim.

Do not update the public paper text unless the exact evidence is already committed or clearly referenced.

## Commit discipline

Suggested commit sequence:

```text
1. tooling: add structured run output and validation scripts
2. tests: add invariant and summary schema coverage
3. evidence: add matched 15k comparison run artifacts
4. figures: regenerate publication figures from run artifacts
5. docs: add manifest and update README claim boundaries
```

Keep generated artifacts separate from code changes where practical.

## Final report format

When done, produce a concise final report:

```markdown
# Harmonized Hyper-Connections Evidence Pack Report

## Branch / commits

## Hardware used

## Cluster/node allocation

## Runs completed

## Validation commands

## Result summary

## Figures generated

## Claims supported

## Claims not yet supported

## Failures / negative results

## Shared-storage artifacts

## Next required work
```

Include exact commit SHAs and paths.

## Stop conditions

Stop and report instead of improvising if:

- the repo script cannot run at all after a minimal repair attempt
- GPU/CUDA is unavailable on both nodes
- `/gfs` is unavailable or inconsistent across nodes
- result metrics are missing and require major model surgery
- runs are unstable or produce contradictory evidence
- artifact size becomes too large to commit safely

Negative or weak results are acceptable. Hidden or undocumented results are not.
