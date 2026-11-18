import pytest
import torch

from torchist.metrics import wasserstein_distance_1d


@pytest.mark.parametrize("sparse_p", [False, True])
@pytest.mark.parametrize("sparse_q", [False, True])
def test_wasserstein_distance_1d(sparse_p, sparse_q):
    p = torch.tensor([0.5, 0, 0.5, 0])
    q = torch.tensor([0, 0.25, 0, 0.75])
    if sparse_p:
        p = p.to_sparse()
    if sparse_q:
        q = q.to_sparse()
    positions = torch.tensor([0.0, 1.0, 3.0, 7.0])

    assert wasserstein_distance_1d(p, q, positions).item() == 4
    assert wasserstein_distance_1d(p, q).item() == 1.5
    assert wasserstein_distance_1d(p, p).item() == 0


def test_sparse_wasserstein_skips_empty_bins():
    size = (100_000_000,)
    p = torch.sparse_coo_tensor([[2, 2]], [0.25, 0.75], size)
    q = torch.sparse_coo_tensor([[12]], [1.0], size)
    assert wasserstein_distance_1d(p, q).item() == 10
