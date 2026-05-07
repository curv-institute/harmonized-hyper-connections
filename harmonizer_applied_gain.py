#!/usr/bin/env python3
"""
Harmonizer with Applied-Gain Control

Key fix: Controller measures G_applied (composite gain of scaled transport)
instead of G_raw (which drifts unbounded like HC).

Modes:
- hc: Unconstrained residual transport
- mhc: Hard manifold projection via Sinkhorn
- harm: Soft equilibrium via applied-gain feedback control
- res_scale: Fixed residual-transport scaling baseline
"""

import argparse
import csv
import json
import math
import os
import platform
import random
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is declared in pyproject for normal use.
    np = None


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_commit_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    return requested


def command_line() -> str:
    return shlex.join([sys.executable, *sys.argv])


def device_description(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        return f"cuda:{idx} {torch.cuda.get_device_name(idx)}"
    return device


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_manifest(out_dir: Path, cfg: "RunConfig", summary: Dict[str, object]) -> None:
    files = []
    for name in ["config.json", "command.txt", "metrics.csv", "summary.json", "stdout.log", "stderr.log"]:
        path = out_dir / name
        if path.exists():
            files.append({
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })

    manifest = {
        "run_name": cfg.name,
        "mode": cfg.mode,
        "seed": cfg.seed,
        "steps": cfg.steps,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": (out_dir / "command.txt").read_text().strip()
        if (out_dir / "command.txt").exists()
        else command_line(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": summary.get("device"),
        "commit_sha": summary.get("commit_sha"),
        "files": files,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def write_failure_summary(out_dir: Path, cfg: "RunConfig", exc: BaseException, runtime_seconds: float) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    command_file = out_dir / "command.txt"
    if not command_file.exists():
        command_file.write_text(command_line() + "\n")
    config_file = out_dir / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(asdict(cfg), f, indent=2)

    summary: Dict[str, object] = {
        "run_name": cfg.name,
        "mode": cfg.mode,
        "seed": cfg.seed,
        "steps": cfg.steps,
        "completed_steps": 0,
        "gain_target": cfg.gain_target,
        "scale_floor": cfg.min_scale,
        "harm_k": cfg.harm_k,
        "residual_scale": cfg.residual_scale,
        "final_accuracy": None,
        "final_loss": None,
        "max_raw_gain": None,
        "max_applied_gain": None,
        "mean_applied_gain": None,
        "std_applied_gain": None,
        "min_scale": None,
        "mean_scale": None,
        "floor_hits": None,
        "runtime_seconds": runtime_seconds,
        "device": cfg.device,
        "hostname": platform.node(),
        "commit_sha": git_commit_sha(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_run_manifest(out_dir, cfg, summary)
    return summary


def sinkhorn_doubly_stochastic(logits: torch.Tensor, iters: int = 6, eps: float = 1e-6) -> torch.Tensor:
    """Sinkhorn-Knopp iteration for doubly-stochastic projection."""
    m = torch.exp(logits).clamp_min(eps).to(torch.bfloat16)
    for _ in range(iters):
        m = m / m.sum(dim=1, keepdim=True).clamp_min(eps)
        m = m / m.sum(dim=0, keepdim=True).clamp_min(eps)
    return m.to(logits.dtype)


@torch.no_grad()
def composite_gain_metrics(hres_list: List[torch.Tensor]) -> Tuple[float, float]:
    """Compute row and column gains from composite of all Hres matrices."""
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
    """Multi-stream residual block with configurable transport constraint."""
    def __init__(self, d: int, n: int, mode: str, residual_scale: float = 1.0):
        super().__init__()
        assert mode in ("hc", "mhc", "harm", "res_scale")
        self.n = n
        self.mode = mode
        self.residual_scale = residual_scale
        self.hpre_logits = nn.Parameter(torch.zeros(n))
        self.hpost_logits = nn.Parameter(torch.zeros(n))
        self.hres_logits = nn.Parameter(torch.zeros(n, n))
        self.core = CoreBlock(d)
        nn.init.normal_(self.hres_logits, mean=0.0, std=0.05)

    @torch.no_grad()
    def hres_raw(self) -> torch.Tensor:
        """Get raw (unscaled) Hres matrix."""
        I = torch.eye(self.n, device=self.hres_logits.device, dtype=self.hres_logits.dtype)
        return I + 0.05 * self.hres_logits

    def hres_applied(self, harm_scale: float) -> torch.Tensor:
        """Get applied (scaled) Hres matrix for harm mode."""
        I = torch.eye(self.n, device=self.hres_logits.device, dtype=self.hres_logits.dtype)
        raw = I + 0.05 * self.hres_logits
        s = torch.tensor(harm_scale, device=raw.device, dtype=raw.dtype)
        return I + s * (raw - I)

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
        elif self.mode == "res_scale":
            raw = I + 0.05 * self.hres_logits
            s = torch.tensor(self.residual_scale, device=raw.device, dtype=raw.dtype)
            hres = I + s * (raw - I)
        else:  # hc
            hres = I + 0.05 * self.hres_logits
        return hpre, hpost, hres

    def forward(self, stream: torch.Tensor, harm_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        hpre, hpost, hres = self.mappings(harm_scale)
        x = (stream * hpre.view(1, 1, self.n, 1)).sum(dim=2)
        y = self.core(x)
        y_stream = y.unsqueeze(2) * hpost.view(1, 1, self.n, 1)
        mixed = torch.einsum("ij,btjd->btid", hres, stream)
        return mixed + y_stream, hres


@dataclass
class HarmonizerConfig:
    """Controller configuration for Harmonizer mode."""
    gain_target: float = 5.0
    min_scale: float = 0.05
    harm_k: float = 1.0
    beta: float = 0.95  # EMA decay


class TinyLM(nn.Module):
    """Language model with multi-stream residual transport."""
    def __init__(self, vocab: int, d: int, layers: int, n: int, mode: str,
                 harm_cfg: Optional[HarmonizerConfig] = None, residual_scale: float = 1.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([StreamResidualBlock(d, n, mode, residual_scale) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.n = n
        self.mode = mode
        self.residual_scale = residual_scale

        # Harmonizer state
        self.harm_cfg = harm_cfg or HarmonizerConfig()
        self.harm_scale = 1.0
        self.gain_ema_applied = 1.0
        self.floor_hits = 0

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass with gain metrics.

        Returns:
            logits: Output logits
            metrics: Dict with G_raw, G_applied, harm_scale, floor_hits
        """
        x = self.emb(idx)
        stream = x.unsqueeze(2).repeat(1, 1, self.n, 1)

        # Compute current scale for this step
        current_scale = 1.0

        with torch.no_grad():
            # Always compute G_raw from unscaled matrices
            raw_list = [blk.hres_raw() for blk in self.blocks]
            G_row_raw, G_col_raw = composite_gain_metrics(raw_list)
            G_raw = max(G_row_raw, G_col_raw)

        if self.mode == "harm":
            with torch.no_grad():
                # Compute G_applied from scaled matrices using CURRENT scale
                applied_list = [blk.hres_applied(self.harm_scale) for blk in self.blocks]
                G_row_app, G_col_app = composite_gain_metrics(applied_list)
                G_applied = max(G_row_app, G_col_app)

                # Update EMA of applied gain
                self.gain_ema_applied = self.harm_cfg.beta * self.gain_ema_applied + \
                                        (1 - self.harm_cfg.beta) * G_applied

                # Compute new scale using applied gain
                ratio = self.harm_cfg.gain_target / max(1e-6, self.gain_ema_applied)
                new_scale = self.harm_scale * (ratio ** self.harm_cfg.harm_k)
                new_scale = max(self.harm_cfg.min_scale, min(1.0, new_scale))

                # Track floor hits
                if new_scale <= self.harm_cfg.min_scale + 1e-6:
                    self.floor_hits += 1

                self.harm_scale = float(new_scale)

            current_scale = self.harm_scale

        # Forward through blocks with current scale
        hres_used: List[torch.Tensor] = []
        for blk in self.blocks:
            stream, hres = blk(stream, current_scale)
            hres_used.append(hres)

        out = self.ln_f(stream.mean(dim=2))
        logits = self.head(out)

        # Compute final applied gain metrics
        with torch.no_grad():
            G_row_applied, G_col_applied = composite_gain_metrics([h.detach() for h in hres_used])

        metrics = {
            "G_raw_row": G_row_raw,
            "G_raw_col": G_col_raw,
            "G_raw_max": G_raw,
            "G_applied_row": G_row_applied,
            "G_applied_col": G_col_applied,
            "G_applied_max": max(G_row_applied, G_col_applied),
            "harm_scale": self.harm_scale if self.mode == "harm" else 1.0,
            "floor_hits": self.floor_hits,
        }

        return logits, metrics


@dataclass
class RunConfig:
    """Run configuration."""
    name: str = "run"
    mode: str = "hc"
    seed: int = 0

    # Model
    vocab: int = 256
    d: int = 128
    layers: int = 96
    n: int = 16

    # Training
    steps: int = 3000
    batch: int = 8
    seq: int = 256
    lr: float = 2e-3
    weight_decay: float = 0.01

    # Task
    task: str = "kv_retrieval"
    num_kv_pairs: int = 8

    # Harmonizer
    gain_target: float = 5.0
    min_scale: float = 0.05
    harm_k: float = 1.0
    beta: float = 0.95
    residual_scale: float = 0.25

    # Logging
    log_every: int = 100
    device: str = "auto"


def make_kv_batch(cfg: RunConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create KV retrieval batch."""
    device = cfg.device
    B, T = cfg.batch, cfg.seq
    num_kv = cfg.num_kv_pairs
    key_vocab = max(1, cfg.vocab // 2)
    value_vocab = max(1, cfg.vocab - key_vocab)

    keys = torch.randint(0, key_vocab, (B, num_kv), device=device)
    values = key_vocab + torch.randint(0, value_vocab, (B, num_kv), device=device)

    x = torch.zeros(B, T, dtype=torch.long, device=device)
    y = torch.zeros(B, T, dtype=torch.long, device=device)
    mask = torch.zeros(B, T, dtype=torch.float32, device=device)

    # Fill KV pairs at start
    for i in range(num_kv):
        x[:, 2*i] = keys[:, i]
        x[:, 2*i + 1] = values[:, i]

    # Queries at end (shuffled)
    query_start = T - num_kv
    perm = torch.stack([torch.randperm(num_kv, device=device) for _ in range(B)])
    for i in range(num_kv):
        x[:, query_start + i] = keys.gather(1, perm[:, i:i+1]).squeeze(1)
        y[:, query_start + i] = values.gather(1, perm[:, i:i+1]).squeeze(1)
        mask[:, query_start + i] = 1.0

    return x, y, mask


def train(cfg: RunConfig, out_dir: Path) -> Dict:
    """Train model and return results."""
    cfg.device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    command_file = out_dir / "command.txt"
    if not command_file.exists():
        command_file.write_text(command_line() + "\n")

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Create model
    harm_cfg = HarmonizerConfig(
        gain_target=cfg.gain_target,
        min_scale=cfg.min_scale,
        harm_k=cfg.harm_k,
        beta=cfg.beta,
    )
    model = TinyLM(
        cfg.vocab,
        cfg.d,
        cfg.layers,
        cfg.n,
        cfg.mode,
        harm_cfg,
        residual_scale=cfg.residual_scale,
    ).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Metrics CSV
    metrics_file = out_dir / "metrics.csv"
    fieldnames = [
        "step", "loss", "accuracy", "raw_gain", "applied_gain", "scale", "mode", "seed",
        "acc",
        "gain_raw_row", "gain_raw_col", "gain_raw_max",
        "gain_applied_row", "gain_applied_col", "gain_applied_max",
        "harm_scale", "floor_hits", "wall_time_sec"
    ]

    logs = []
    t0 = time.perf_counter()
    harm_scale_values = []

    with open(metrics_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(cfg.steps):
            x, y, mask = make_kv_batch(cfg)
            logits, metrics = model(x)

            logits_flat = logits.view(-1, cfg.vocab)
            y_flat = y.view(-1)
            mask_flat = mask.view(-1)

            loss = (F.cross_entropy(logits_flat, y_flat, reduction='none') * mask_flat).sum() / mask_flat.sum()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % cfg.log_every == 0 or step == cfg.steps - 1:
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    correct = ((preds == y).float() * mask).sum()
                    acc = float((correct / mask.sum()).item())

                wall_time = time.perf_counter() - t0

                entry = {
                    "step": step,
                    "loss": float(loss.item()),
                    "accuracy": acc,
                    "raw_gain": metrics["G_raw_max"],
                    "applied_gain": metrics["G_applied_max"],
                    "scale": metrics["harm_scale"],
                    "mode": cfg.mode,
                    "seed": cfg.seed,
                    "acc": acc,
                    "gain_raw_row": metrics["G_raw_row"],
                    "gain_raw_col": metrics["G_raw_col"],
                    "gain_raw_max": metrics["G_raw_max"],
                    "gain_applied_row": metrics["G_applied_row"],
                    "gain_applied_col": metrics["G_applied_col"],
                    "gain_applied_max": metrics["G_applied_max"],
                    "harm_scale": metrics["harm_scale"],
                    "floor_hits": metrics["floor_hits"],
                    "wall_time_sec": wall_time,
                }
                logs.append(entry)
                writer.writerow(entry)
                f.flush()

                harm_scale_values.append(metrics["harm_scale"])

                print(f"  step={step:5d} loss={loss.item():.4f} acc={acc:.4f} "
                      f"G_raw={metrics['G_raw_max']:.2f} G_app={metrics['G_applied_max']:.2f} "
                      f"scale={metrics['harm_scale']:.4f} floor={metrics['floor_hits']}")

            if not math.isfinite(float(loss.item())):
                print(f"  [WARN] {cfg.name} diverged at step {step}")
                break

    elapsed = time.perf_counter() - t0

    # Compute summary stats
    applied_gain_values = [float(e["applied_gain"]) for e in logs]
    raw_gain_values = [float(e["raw_gain"]) for e in logs]
    scale_values = [float(e["scale"]) for e in logs]
    final_entry = logs[-1] if logs else {}
    device_text = device_description(cfg.device)
    commit_sha = git_commit_sha()
    result = {
        "run_name": cfg.name,
        "mode": cfg.mode,
        "seed": cfg.seed,
        "steps": cfg.steps,
        "completed_steps": int(final_entry["step"]) + 1 if final_entry else 0,
        "gain_target": cfg.gain_target,
        "scale_floor": cfg.min_scale,
        "harm_k": cfg.harm_k,
        "residual_scale": cfg.residual_scale,
        "final_accuracy": final_entry.get("accuracy", float("nan")),
        "final_loss": final_entry.get("loss", float("nan")),
        "max_raw_gain": max(raw_gain_values) if raw_gain_values else float("nan"),
        "max_applied_gain": max(applied_gain_values) if applied_gain_values else float("nan"),
        "mean_applied_gain": statistics.fmean(applied_gain_values) if applied_gain_values else float("nan"),
        "std_applied_gain": statistics.pstdev(applied_gain_values) if len(applied_gain_values) > 1 else 0.0,
        "min_scale": min(scale_values) if scale_values else float("nan"),
        "mean_scale": statistics.fmean(scale_values) if scale_values else float("nan"),
        "floor_hits": final_entry.get("floor_hits", 0),
        "runtime_seconds": elapsed,
        "device": device_text,
        "hostname": platform.node(),
        "commit_sha": commit_sha,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "status": "completed" if final_entry and int(final_entry["step"]) == cfg.steps - 1 else "incomplete",
        # Backward-compatible keys used by older analysis scripts.
        "final_acc": final_entry.get("accuracy", float("nan")),
        "max_gain_raw_max": max(raw_gain_values) if raw_gain_values else float("nan"),
        "max_gain_applied_max": max(applied_gain_values) if applied_gain_values else float("nan"),
        "harm_scale_mean": statistics.fmean(scale_values) if scale_values else float("nan"),
        "harm_scale_min": min(scale_values) if scale_values else float("nan"),
        "time_sec": elapsed,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"Run complete: {cfg.name}")
    print(f"  final_loss={result['final_loss']:.4f}, final_acc={result['final_acc']:.4f}")
    print(f"  max_gain_raw={result['max_gain_raw_max']:.2f}, max_gain_applied={result['max_gain_applied_max']:.2f}")
    print(f"  harm_scale_mean={result['harm_scale_mean']:.4f}, harm_scale_min={result['harm_scale_min']:.4f}")
    print(f"  floor_hits={result['floor_hits']}, time={result['time_sec']:.1f}s")
    print(f"{'='*60}\n")

    # Save summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump(result, f, indent=2)

    write_run_manifest(out_dir, cfg, result)

    return result


def main():
    parser = argparse.ArgumentParser(description="Harmonizer with Applied-Gain Control")
    parser.add_argument("--mode", type=str, default="harm", choices=["hc", "mhc", "harm", "res_scale"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--layers", type=int, default=96)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=0.01)
    parser.add_argument("--num-kv-pairs", "--num_kv_pairs", dest="num_kv_pairs", type=int, default=8)
    parser.add_argument("--gain_target", type=float, default=5.0)
    parser.add_argument("--min_scale", type=float, default=0.05)
    parser.add_argument("--harm_k", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--residual-scale", "--residual_scale", dest="residual_scale", type=float, default=0.25)
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--name", type=str, default=None, help="Run name")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path.home() / "mhc" / "runs" / f"{args.mode}_{timestamp}"

    name = args.name or f"{args.mode}_s{args.seed}"

    cfg = RunConfig(
        name=name,
        mode=args.mode,
        seed=args.seed,
        vocab=args.vocab,
        d=args.d,
        layers=args.layers,
        n=args.n,
        steps=args.steps,
        batch=args.batch,
        seq=args.seq,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_kv_pairs=args.num_kv_pairs,
        gain_target=args.gain_target,
        min_scale=args.min_scale,
        harm_k=args.harm_k,
        beta=args.beta,
        residual_scale=args.residual_scale,
        log_every=args.log_every,
        device=device,
    )

    print(f"Config: {asdict(cfg)}")
    print(f"Output: {out_dir}")

    t0 = time.perf_counter()
    try:
        result = train(cfg, out_dir)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        write_failure_summary(out_dir, cfg, exc, elapsed)
        print(f"\nRun failed: {name}")
        print(f"  error={type(exc).__name__}: {exc}")
        print(f"  failure artifacts saved to: {out_dir}")
        raise

    print(f"\nLogs saved to: {out_dir}")
    return result


if __name__ == "__main__":
    main()
