#!/usr/bin/env -S uv run
#/// script
# dependencies = [
#   "torch",
# ]
#///

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sinkhorn_doubly_stochastic(logits: torch.Tensor, iters: int = 6, eps: float = 1e-6) -> torch.Tensor:
    """Sinkhorn-Knopp projection onto approximate doubly-stochastic matrices."""
    m = torch.exp(logits).clamp_min(eps).to(torch.bfloat16)
    for _ in range(iters):
        m = m / m.sum(dim=1, keepdim=True).clamp_min(eps)
        m = m / m.sum(dim=0, keepdim=True).clamp_min(eps)
    return m.to(logits.dtype)


@torch.no_grad()
def projection_quality(hres_list: List[torch.Tensor]) -> Tuple[float, float]:
    """Measure how close Hres matrices are to doubly-stochastic."""
    max_row_err = 0.0
    max_col_err = 0.0
    for h in hres_list:
        row_err = (h.sum(dim=1) - 1.0).abs().max().item()
        col_err = (h.sum(dim=0) - 1.0).abs().max().item()
        max_row_err = max(max_row_err, row_err)
        max_col_err = max(max_col_err, col_err)
    return max_row_err, max_col_err


@torch.no_grad()
def composite_gain_metrics(hres_list: List[torch.Tensor]) -> Tuple[float, float]:
    """Composite mapping C = H_L-1 ... H_0. Returns max row/col sums."""
    if not hres_list:
        return 1.0, 1.0
    n = hres_list[0].shape[0]
    comp = torch.eye(n, dtype=torch.float32, device=hres_list[0].device)
    for h in hres_list:
        comp = h.float() @ comp
    row_gain = float(comp.abs().sum(dim=1).max().item())
    col_gain = float(comp.abs().sum(dim=0).max().item())
    return row_gain, col_gain


@torch.no_grad()
def grad_norm(model: nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        s += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(s)


# ----------------------------
# Model components
# ----------------------------

class CoreBlock(nn.Module):
    """Single-head causal attention + small MLP."""
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.ln_attn = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)
        self.ln_mlp = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 2 * d)
        self.fc2 = nn.Linear(2 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln_attn(x)
        qkv = self.qkv(h).reshape(B, T, 3, D).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = 1.0 / math.sqrt(D)
        attn = (q @ k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn_out = attn @ v
        y = self.out_proj(attn_out)
        h2 = self.ln_mlp(y)
        y = self.fc2(F.gelu(self.fc1(h2)))
        return y


class StreamResidualBlock(nn.Module):
    """Residual-stream transport with n parallel streams."""
    def __init__(self, d: int, n: int, mode: str):
        super().__init__()
        assert mode in ("hc", "mhc", "harm")
        self.n = n
        self.mode = mode
        self.hpre_logits = nn.Parameter(torch.zeros(n))
        self.hpost_logits = nn.Parameter(torch.zeros(n))
        self.hres_logits = nn.Parameter(torch.zeros(n, n))
        self.core = CoreBlock(d)
        nn.init.normal_(self.hres_logits, mean=0.0, std=0.05)

    @torch.no_grad()
    def hres_raw(self) -> torch.Tensor:
        I = torch.eye(self.n, device=self.hres_logits.device, dtype=self.hres_logits.dtype)
        return I + 0.05 * self.hres_logits

    def mappings(self, harm_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hpre = F.softmax(self.hpre_logits, dim=0)
        hpost = F.softmax(self.hpost_logits, dim=0)
        I = torch.eye(self.n, device=self.hres_logits.device)
        if self.mode == "mhc":
            eps = 0.2
            hres = (1 - eps) * I + eps * sinkhorn_doubly_stochastic(self.hres_logits)
        elif self.mode == "harm":
            raw = I + 0.05 * self.hres_logits
            s = torch.tensor(harm_scale, device=raw.device, dtype=raw.dtype)
            hres = I + s * (raw - I)
        else:
            hres = I + 0.05 * self.hres_logits
        return hpre, hpost, hres

    def forward(self, stream: torch.Tensor, harm_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        hpre, hpost, hres = self.mappings(harm_scale)
        x = (stream * hpre.view(1, 1, self.n, 1)).sum(dim=2)
        y = self.core(x)
        y_stream = y.unsqueeze(2) * hpost.view(1, 1, self.n, 1)
        mixed = torch.einsum("ij,btjd->btid", hres, stream)
        return mixed + y_stream, hres


class TinyLM(nn.Module):
    """Model with configurable Harmonizer parameters."""
    def __init__(self, vocab: int, d: int, layers: int, n: int, mode: str,
                 gain_target: float = 2.5, harm_k: float = 1.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([StreamResidualBlock(d, n, mode) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.n = n
        self.mode = mode
        self.harm_scale = 1.0
        self.gain_ema = 1.0
        self.gain_target = gain_target
        self.harm_k = harm_k

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.emb(idx)
        stream = x.unsqueeze(2).repeat(1, 1, self.n, 1)
        current_scale = 1.0
        if self.mode == "harm":
            with torch.no_grad():
                raw_list = [blk.hres_raw() for blk in self.blocks]
                G_row, G_col = composite_gain_metrics(raw_list)
                G_raw = max(G_row, G_col)
                self.gain_ema = 0.95 * self.gain_ema + 0.05 * G_raw
                ratio = self.gain_target / max(1e-6, self.gain_ema)
                scale = ratio ** self.harm_k
                self.harm_scale = float(max(0.02, min(1.0, scale)))
            current_scale = self.harm_scale
        hres_used: List[torch.Tensor] = []
        for blk in self.blocks:
            stream, hres = blk(stream, current_scale)
            hres_used.append(hres)
        out = self.ln_f(stream.mean(dim=2))
        logits = self.head(out)
        return logits, hres_used


# ----------------------------
# Experiment configuration
# ----------------------------

@dataclass
class RunCfg:
    name: str
    mode: str
    n: int = 16
    layers: int = 96
    steps: int = 3000
    batch: int = 8  # Reduced from 32 for 96-layer models on 3090
    seq: int = 256
    vocab: int = 256
    d: int = 128
    lr: float = 2e-3
    device: str = "cuda"
    log_every: int = 200
    task: str = "kv_retrieval"
    num_kv_pairs: int = 8
    seed: int = 0
    gain_target: float = 2.5
    harm_k: float = 1.0
    grad_clip: Optional[float] = None
    weight_decay: float = 0.01


def make_batch(cfg: RunCfg) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = cfg.device
    B, T = cfg.batch, cfg.seq
    num_kv = cfg.num_kv_pairs
    keys = torch.randint(0, 128, (B, num_kv), device=device)
    values = torch.randint(128, 256, (B, num_kv), device=device)
    x = torch.zeros(B, T, dtype=torch.long, device=device)
    y = torch.zeros(B, T, dtype=torch.long, device=device)
    mask = torch.zeros(B, T, dtype=torch.float32, device=device)
    for i in range(num_kv):
        x[:, 2*i] = keys[:, i]
        x[:, 2*i + 1] = values[:, i]
    query_start = T - num_kv
    perm = torch.stack([torch.randperm(num_kv, device=device) for _ in range(B)])
    for i in range(num_kv):
        x[:, query_start + i] = keys.gather(1, perm[:, i:i+1]).squeeze(1)
        y[:, query_start + i] = values.gather(1, perm[:, i:i+1]).squeeze(1)
        mask[:, query_start + i] = 1.0
    return x, y, mask


def train(cfg: RunCfg) -> Dict[str, List[float]]:
    set_seed(cfg.seed)
    model = TinyLM(cfg.vocab, cfg.d, cfg.layers, cfg.n, cfg.mode,
                   cfg.gain_target, cfg.harm_k).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    logs: Dict[str, List[float]] = {
        "step": [], "loss": [], "accuracy": [], "grad_norm": [],
        "gain_row": [], "gain_col": [], "proj_row_err": [], "proj_col_err": []
    }

    t0 = time.perf_counter()
    for step in range(cfg.steps):
        x, y, mask = make_batch(cfg)
        logits, hres_used = model(x)
        logits_flat = logits.view(-1, cfg.vocab)
        y_flat = y.view(-1)
        mask_flat = mask.view(-1)
        xent = (F.cross_entropy(logits_flat, y_flat, reduction='none') * mask_flat).sum() / mask_flat.sum()
        loss = xent

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = grad_norm(model)

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        opt.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                correct = ((preds == y).float() * mask).sum()
                acc = correct / mask.sum()
                row_gain, col_gain = composite_gain_metrics([h.detach() for h in hres_used])
                proj_row_err, proj_col_err = projection_quality([h.detach() for h in hres_used])

            logs["step"].append(float(step))
            logs["loss"].append(float(loss.item()))
            logs["accuracy"].append(float(acc.item()))
            logs["grad_norm"].append(float(gnorm))
            logs["gain_row"].append(float(row_gain))
            logs["gain_col"].append(float(col_gain))
            logs["proj_row_err"].append(float(proj_row_err))
            logs["proj_col_err"].append(float(proj_col_err))

        if not math.isfinite(float(loss.item())):
            print(f"  [WARN] {cfg.name} diverged at step {step}")
            break

    elapsed = time.perf_counter() - t0
    logs["elapsed_sec"] = [elapsed]
    return logs


def run_experiment(cfg: RunCfg, log_dir: Path) -> Dict:
    print(f"  Running: {cfg.name} (mode={cfg.mode}, seed={cfg.seed}, steps={cfg.steps})")
    logs = train(cfg)

    # Extract summary metrics
    result = {
        "run_name": cfg.name,
        "mode": cfg.mode,
        "seed": cfg.seed,
        "steps": cfg.steps,
        "kv_pairs": cfg.num_kv_pairs,
        "final_loss": logs["loss"][-1] if logs["loss"] else float('nan'),
        "final_acc": logs["accuracy"][-1] if logs["accuracy"] else float('nan'),
        "max_gain_row": max(logs["gain_row"]) if logs["gain_row"] else float('nan'),
        "max_gain_col": max(logs["gain_col"]) if logs["gain_col"] else float('nan'),
        "max_grad_norm": max(logs["grad_norm"]) if logs["grad_norm"] else float('nan'),
        "time_sec": logs["elapsed_sec"][0] if logs["elapsed_sec"] else float('nan'),
        "trajectory": {k: v for k, v in logs.items() if k != "elapsed_sec"}
    }

    # Save detailed log
    log_file = log_dir / f"{cfg.name}.jsonl"
    with open(log_file, 'w') as f:
        f.write(json.dumps({"config": asdict(cfg)}) + "\n")
        for i, step in enumerate(logs["step"]):
            entry = {k: logs[k][i] for k in logs if k != "elapsed_sec" and i < len(logs[k])}
            f.write(json.dumps(entry) + "\n")

    return result


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path.home() / "mhc" / "runs" / f"tier2_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Log directory: {log_dir}")

    # Common settings
    LAYERS = 96
    N = 16
    D = 128
    GAIN_TARGET = 2.5
    HARM_K = 1.0

    results = []

    # --- Matrix A1: Seed Robustness ---
    print("\n=== Matrix A1: Seed Robustness ===")
    for mode in ["hc", "harm", "mhc"]:
        for seed in [0, 1, 2]:
            cfg = RunCfg(
                name=f"A1_{mode}_s{seed}",
                mode=mode, seed=seed, steps=3000, log_every=200,
                layers=LAYERS, n=N, d=D, num_kv_pairs=8,
                gain_target=GAIN_TARGET, harm_k=HARM_K, device=device
            )
            results.append(run_experiment(cfg, log_dir))

    # --- Matrix A2: Long-Horizon Stability ---
    print("\n=== Matrix A2: Long-Horizon Stability ===")
    for mode in ["hc", "harm"]:
        cfg = RunCfg(
            name=f"A2_{mode}_15k",
            mode=mode, seed=0, steps=15000, log_every=500,
            layers=LAYERS, n=N, d=D, num_kv_pairs=8,
            gain_target=GAIN_TARGET, harm_k=HARM_K, device=device
        )
        results.append(run_experiment(cfg, log_dir))

    # --- Matrix A3: Increased Difficulty ---
    print("\n=== Matrix A3: Increased Difficulty ===")
    for mode in ["hc", "harm"]:
        cfg = RunCfg(
            name=f"A3_{mode}_kv16",
            mode=mode, seed=0, steps=3000, log_every=200,
            layers=LAYERS, n=N, d=D, num_kv_pairs=16,
            gain_target=GAIN_TARGET, harm_k=HARM_K, device=device
        )
        results.append(run_experiment(cfg, log_dir))

    # --- Matrix B: Controls ---
    print("\n=== Matrix B: Controls ===")
    # B1: HC + Gradient Clipping
    cfg = RunCfg(
        name="B1_hc_gradclip",
        mode="hc", seed=0, steps=3000, log_every=200,
        layers=LAYERS, n=N, d=D, num_kv_pairs=8,
        gain_target=GAIN_TARGET, harm_k=HARM_K, device=device,
        grad_clip=1.0
    )
    results.append(run_experiment(cfg, log_dir))

    # B2: HC + Strong Weight Decay
    cfg = RunCfg(
        name="B2_hc_wd030",
        mode="hc", seed=0, steps=3000, log_every=200,
        layers=LAYERS, n=N, d=D, num_kv_pairs=8,
        gain_target=GAIN_TARGET, harm_k=HARM_K, device=device,
        weight_decay=0.30
    )
    results.append(run_experiment(cfg, log_dir))

    # Save summary
    summary_file = log_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print results
    print("\n" + "="*120)
    print("TIER-2 RESULTS")
    print("="*120)

    # Section 1: Metadata
    git_commit = "dirty"
    try:
        import subprocess
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(Path.home()/"mhc"), text=True).strip()[:8]
    except:
        pass

    print(f"""
ENV:
  host: gigaplex
  user: jwm
  gpu: RTX 3090
  cuda: {torch.cuda.is_available()}
  repo_dir: ~/mhc
  script: rift_mhc_sandbox.py
  git_commit: {git_commit}
  timestamp: {timestamp}

GLOBAL_CONFIG:
  task: kv_retrieval
  seq: 256
  layers: {LAYERS}
  n: {N}
  d: {D}
  gain_target: {GAIN_TARGET}
  harm_k: {HARM_K}
""")

    # Section 2: Per-Run Summary Table
    print("\n| run_name | mode | seed | steps | kv_pairs | final_loss | final_acc | max_gain_row | max_gain_col | max_grad_norm | time_sec |")
    print("|----------|------|------|-------|----------|------------|-----------|--------------|--------------|---------------|----------|")
    for r in results:
        print(f"| {r['run_name']} | {r['mode']} | {r['seed']} | {r['steps']} | {r['kv_pairs']} | {r['final_loss']:.4f} | {r['final_acc']:.3f} | {r['max_gain_row']:.2f} | {r['max_gain_col']:.2f} | {r['max_grad_norm']:.2f} | {r['time_sec']:.1f} |")

    # Section 3: A1 Aggregates
    print("\nA1_AGGREGATES:")
    for mode in ["hc", "harm", "mhc"]:
        mode_results = [r for r in results if r['run_name'].startswith('A1_') and r['mode'] == mode]
        accs = [r['final_acc'] for r in mode_results]
        gains = [max(r['max_gain_row'], r['max_gain_col']) for r in mode_results]
        import statistics
        acc_mean = statistics.mean(accs) if accs else 0
        acc_std = statistics.stdev(accs) if len(accs) > 1 else 0
        gain_mean = statistics.mean(gains) if gains else 0
        gain_std = statistics.stdev(gains) if len(gains) > 1 else 0
        print(f"  {mode}:")
        print(f"    acc_mean: {acc_mean:.4f}")
        print(f"    acc_std: {acc_std:.4f}")
        print(f"    gain_mean: {gain_mean:.2f}")
        print(f"    gain_std: {gain_std:.2f}")

    # Section 4: A2 Long-Run Time Series
    print("\nA2_LONGRUN_HC:")
    print("| step | loss | acc | gain_row | gain_col | grad_norm |")
    print("|------|------|-----|----------|----------|-----------|")
    a2_hc = [r for r in results if r['run_name'] == 'A2_hc_15k'][0] if any(r['run_name'] == 'A2_hc_15k' for r in results) else None
    if a2_hc:
        traj = a2_hc['trajectory']
        for target_step in [3000, 6000, 9000, 12000, 15000]:
            idx = None
            for i, s in enumerate(traj['step']):
                if s >= target_step - 250:
                    idx = i
                    break
            if idx is not None and idx < len(traj['step']):
                print(f"| {int(traj['step'][idx])} | {traj['loss'][idx]:.4f} | {traj['accuracy'][idx]:.3f} | {traj['gain_row'][idx]:.2f} | {traj['gain_col'][idx]:.2f} | {traj['grad_norm'][idx]:.2f} |")

    print("\nA2_LONGRUN_HARM:")
    print("| step | loss | acc | gain_row | gain_col | grad_norm |")
    print("|------|------|-----|----------|----------|-----------|")
    a2_harm = [r for r in results if r['run_name'] == 'A2_harm_15k'][0] if any(r['run_name'] == 'A2_harm_15k' for r in results) else None
    if a2_harm:
        traj = a2_harm['trajectory']
        for target_step in [3000, 6000, 9000, 12000, 15000]:
            idx = None
            for i, s in enumerate(traj['step']):
                if s >= target_step - 250:
                    idx = i
                    break
            if idx is not None and idx < len(traj['step']):
                print(f"| {int(traj['step'][idx])} | {traj['loss'][idx]:.4f} | {traj['accuracy'][idx]:.3f} | {traj['gain_row'][idx]:.2f} | {traj['gain_col'][idx]:.2f} | {traj['grad_norm'][idx]:.2f} |")

    # Section 5: Matrix B Controls
    print("\nMatrix B Controls:")
    print("| run_name | final_loss | final_acc | max_gain |")
    print("|----------|------------|-----------|----------|")
    for r in results:
        if r['run_name'].startswith('B'):
            max_gain = max(r['max_gain_row'], r['max_gain_col'])
            print(f"| {r['run_name']} | {r['final_loss']:.4f} | {r['final_acc']:.3f} | {max_gain:.2f} |")

    print("\n" + "="*120)
    print(f"Logs saved to: {log_dir}")


if __name__ == "__main__":
    main()
