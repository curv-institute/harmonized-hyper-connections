import torch

from harmonizer_applied_gain import HarmonizerConfig, StreamResidualBlock, TinyLM


def test_applied_gain_scaling_moves_transport_toward_identity():
    block = StreamResidualBlock(d=4, n=3, mode="harm")
    with torch.no_grad():
        block.hres_logits.fill_(4.0)

    raw = block.hres_applied(1.0)
    scaled = block.hres_applied(0.25)
    identity = torch.eye(3)

    assert torch.linalg.vector_norm(scaled - identity) < torch.linalg.vector_norm(raw - identity)


def test_fixed_residual_scale_baseline_moves_transport_toward_identity():
    block = StreamResidualBlock(d=4, n=3, mode="res_scale", residual_scale=0.25)
    with torch.no_grad():
        block.hres_logits.fill_(4.0)

    raw = block.hres_raw()
    _, _, scaled = block.mappings()
    identity = torch.eye(3)

    assert torch.linalg.vector_norm(scaled - identity) < torch.linalg.vector_norm(raw - identity)


def test_harmonizer_reduces_scale_under_large_raw_transport():
    cfg = HarmonizerConfig(gain_target=2.0, min_scale=1e-4, harm_k=1.0, beta=0.5)
    model = TinyLM(vocab=16, d=8, layers=3, n=3, mode="harm", harm_cfg=cfg)
    with torch.no_grad():
        for block in model.blocks:
            block.hres_logits.fill_(20.0)

    idx = torch.zeros(2, 8, dtype=torch.long)
    metrics = None
    for _ in range(25):
        _, metrics = model(idx)

    assert metrics is not None
    assert metrics["G_raw_max"] > metrics["G_applied_max"]
    assert metrics["G_applied_max"] < 3.0
    assert metrics["harm_scale"] < 1.0
