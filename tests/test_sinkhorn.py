import torch

from harmonizer_applied_gain import sinkhorn_doubly_stochastic


def test_sinkhorn_returns_nonnegative_approximately_doubly_stochastic_matrix():
    logits = torch.tensor(
        [[0.2, -0.4, 1.0], [1.2, 0.1, -0.8], [-0.3, 0.7, 0.5]],
        dtype=torch.float32,
    )

    projected = sinkhorn_doubly_stochastic(logits, iters=20)

    assert torch.all(projected >= 0)
    assert torch.allclose(projected.sum(dim=0), torch.ones(3), atol=2e-2)
    assert torch.allclose(projected.sum(dim=1), torch.ones(3), atol=2e-2)
