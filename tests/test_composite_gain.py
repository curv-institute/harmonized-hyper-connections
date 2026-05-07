import torch

from harmonizer_applied_gain import composite_gain_metrics


def test_composite_gain_identity_is_one():
    row_gain, col_gain = composite_gain_metrics([torch.eye(4), torch.eye(4)])

    assert row_gain == 1.0
    assert col_gain == 1.0


def test_composite_gain_uses_composed_transport():
    a = torch.tensor([[2.0, 0.0], [0.0, 0.5]])
    b = torch.tensor([[1.0, 1.0], [0.0, 1.0]])

    row_gain, col_gain = composite_gain_metrics([a, b])

    comp = b @ a
    assert row_gain == float(comp.abs().sum(dim=1).max())
    assert col_gain == float(comp.abs().sum(dim=0).max())
