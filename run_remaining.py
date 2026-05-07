#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch>=2.11.0",
# ]
# ///
"""
Focused experiments for Harmonizer paper:
1. A2_hc_15k: Long-horizon HC drift test
2. Harmonizer recovery: gain_target=5.0 and 8.0
"""

import json
import math
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sinkhorn_doubly_stochastic(logits: torch.Tensor, iters: int = 6, eps: float = 1e-6) -> torch.Tensor:
    m = torch.exp(logits).clamp_min(eps).to(torch.bfloat16)
    for _ in range(iters):
        m = m / m.sum(dim=1, keepdim=True).clamp_min(eps)
        m = m / m.sum(dim=0, keepdim=True).clamp_min(eps)
    return m.to(logits.dtype)


@torch.no_grad()
def composite_gain_metrics(hres_list: List[torch.Tensor]) -> Tuple[float, float]:
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
    def __init__(self, d: int, n: int, mode: str):
        super().__init__()
        assert mode in ("hc", "harm")
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
        if self.mode == "harm":
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

    def forward(self, idx: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], float]:
        x = self.emb(idx)
        stream = x.unsqueeze(2).repeat(1, 1, self.n, 1)
        current_scale = 1.0

        # Compute G_raw before any scaling
        with torch.no_grad():
            raw_list = [blk.hres_raw() for blk in self.blocks]
            G_row_raw, G_col_raw = composite_gain_metrics(raw_list)
            G_raw = max(G_row_raw, G_col_raw)

        if self.mode == "harm":
            with torch.no_grad():
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
        return logits, hres_used, G_raw


@dataclass
class RunCfg:
    name: str
    mode: str
    n: int = 16
    layers: int = 96
    steps: int = 3000
    batch: int = 8
    seq: int = 256
    vocab: int = 256
    d: int = 128
    lr: float = 2e-3
    device: str = "cuda"
    log_every: int = 200
    num_kv_pairs: int = 8
    seed: int = 0
    gain_target: float = 2.5
    harm_k: float = 1.0


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


def train(cfg: RunCfg, log_file: Path) -> Dict:
    set_seed(cfg.seed)
    model = TinyLM(cfg.vocab, cfg.d, cfg.layers, cfg.n, cfg.mode,
                   cfg.gain_target, cfg.harm_k).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    logs = []
    t0 = time.perf_counter()

    with open(log_file, 'w') as f:
        f.write(json.dumps({"config": asdict(cfg)}) + "\n")

        for step in range(cfg.steps):
            x, y, mask = make_batch(cfg)
            logits, hres_used, G_raw = model(x)
            logits_flat = logits.view(-1, cfg.vocab)
            y_flat = y.view(-1)
            mask_flat = mask.view(-1)
            xent = (F.cross_entropy(logits_flat, y_flat, reduction='none') * mask_flat).sum() / mask_flat.sum()
            loss = xent

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = grad_norm(model)
            opt.step()

            if step % cfg.log_every == 0 or step == cfg.steps - 1:
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    correct = ((preds == y).float() * mask).sum()
                    acc = float((correct / mask.sum()).item())
                    row_gain, col_gain = composite_gain_metrics([h.detach() for h in hres_used])

                entry = {
                    "step": step,
                    "loss": float(loss.item()),
                    "accuracy": acc,
                    "grad_norm": gnorm,
                    "G_raw": G_raw,
                    "gain_row": row_gain,
                    "gain_col": col_gain,
                    "max_gain": max(row_gain, col_gain),
                    "harm_scale": model.harm_scale if cfg.mode == "harm" else 1.0
                }
                logs.append(entry)
                f.write(json.dumps(entry) + "\n")
                f.flush()

                print(f"  step={step:5d} loss={loss.item():.4f} acc={acc:.4f} "
                      f"max_gain={max(row_gain, col_gain):.2f} G_raw={G_raw:.2f} "
                      f"harm_scale={entry['harm_scale']:.3f}")

            if not math.isfinite(float(loss.item())):
                print(f"  [WARN] {cfg.name} diverged at step {step}")
                break

    elapsed = time.perf_counter() - t0

    return {
        "run_name": cfg.name,
        "mode": cfg.mode,
        "seed": cfg.seed,
        "gain_target": cfg.gain_target,
        "final_loss": logs[-1]["loss"] if logs else float('nan'),
        "final_acc": logs[-1]["accuracy"] if logs else float('nan'),
        "max_gain": max(e["max_gain"] for e in logs) if logs else float('nan'),
        "time_sec": elapsed,
        "logs": logs
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path.home() / "mhc" / "runs" / f"recovery_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Log directory: {log_dir}")

    results = []

    # === A2: Long-horizon HC drift test ===
    print("\n" + "="*60)
    print("A2: Long-horizon HC drift test (15k steps)")
    print("="*60)

    cfg = RunCfg(
        name="A2_hc_15k",
        mode="hc",
        seed=0,
        steps=15000,
        log_every=500,
        device=device
    )
    result = train(cfg, log_dir / f"{cfg.name}.jsonl")
    results.append(result)
    print(f"\nA2_hc_15k complete: acc={result['final_acc']:.4f}, max_gain={result['max_gain']:.2f}")

    # === Harmonizer Recovery: gain_target=5.0 ===
    print("\n" + "="*60)
    print("Harmonizer Recovery Config A: gain_target=5.0")
    print("="*60)

    for seed in [0, 1, 2]:
        cfg = RunCfg(
            name=f"harm_t5_s{seed}",
            mode="harm",
            seed=seed,
            steps=3000,
            log_every=100,
            gain_target=5.0,
            device=device
        )
        result = train(cfg, log_dir / f"{cfg.name}.jsonl")
        results.append(result)
        print(f"\nharm_t5_s{seed} complete: acc={result['final_acc']:.4f}, max_gain={result['max_gain']:.2f}")

    # === Harmonizer Recovery: gain_target=8.0 ===
    print("\n" + "="*60)
    print("Harmonizer Recovery Config B: gain_target=8.0")
    print("="*60)

    for seed in [0, 1, 2]:
        cfg = RunCfg(
            name=f"harm_t8_s{seed}",
            mode="harm",
            seed=seed,
            steps=3000,
            log_every=100,
            gain_target=8.0,
            device=device
        )
        result = train(cfg, log_dir / f"{cfg.name}.jsonl")
        results.append(result)
        print(f"\nharm_t8_s{seed} complete: acc={result['final_acc']:.4f}, max_gain={result['max_gain']:.2f}")

    # === Summary ===
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    print("\n| run_name | mode | seed | gain_target | final_acc | max_gain | time_sec |")
    print("|----------|------|------|-------------|-----------|----------|----------|")
    for r in results:
        print(f"| {r['run_name']} | {r['mode']} | {r['seed']} | {r.get('gain_target', 'N/A')} | "
              f"{r['final_acc']:.4f} | {r['max_gain']:.2f} | {r['time_sec']:.1f} |")

    # Save summary
    summary_file = log_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump([{k: v for k, v in r.items() if k != 'logs'} for r in results], f, indent=2)

    print(f"\nLogs saved to: {log_dir}")


if __name__ == "__main__":
    main()
