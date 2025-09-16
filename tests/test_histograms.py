r"""Tests for the torchist.histograms module."""

import numpy as np
import pytest
import torch

from torchist.histograms import (
    histogram,
    histogramdd,
)


@pytest.mark.parametrize("size", [10000])
@pytest.mark.parametrize("bins", [10, 100])
@pytest.mark.parametrize("bound", [False, True])
def test_histogram(
    size: int,
    bins: int,
    bound: bool,
):
    x = torch.randn(size)

    if bound:
        low, upp = None, None
    else:
        low, upp = x.min().item() + 1, x.max().item() - 1

    h = histogram(x, bins=bins, low=low, upp=upp)
    h_np, _ = np.histogram(
        x.numpy(force=True),
        bins=bins,
        range=None if bound else (low, upp),
    )

    assert np.all(h.numpy(force=True) == h_np)


@pytest.mark.parametrize("size", [10000])
@pytest.mark.parametrize("bins", [10, 100])
@pytest.mark.parametrize("bound", [False, True])
def test_histogram_edges(
    size: int,
    bins: int,
    bound: bool,
):
    x = torch.randn(size)

    if bound:
        low, upp = x.min(), x.max()
    else:
        low, upp = x.min() + 1, x.max() - 1

    edges = torch.linspace(0, 1, bins + 1) ** 2
    edges = (upp - low) * edges + low

    h = histogram(x, edges=edges)
    h_np, _ = np.histogram(x.numpy(force=True), bins=edges.numpy(force=True))

    assert np.all(h.numpy(force=True) == h_np)


@pytest.mark.parametrize("size", [10000])
@pytest.mark.parametrize("bins", [(3, 5, 7)])
@pytest.mark.parametrize("bound", [False, True])
def test_histogramdd_edges(
    size: int,
    bins: int,
    bound: bool,
):
    x = torch.randn(size, len(bins))

    if bound:
        low, upp = None, None
    else:
        low, upp = x.min(dim=0).values + 1, x.max(dim=0).values - 1

    h = histogramdd(x, bins=bins, low=low, upp=upp)
    h_np, _ = np.histogramdd(
        x.numpy(force=True),
        bins=bins,
        range=None if bound else list(zip(low.tolist(), upp.tolist())),
    )

    assert np.all(h.numpy(force=True) == h_np)


@pytest.mark.parametrize("size", [10000])
@pytest.mark.parametrize("bins", [(3, 5, 7)])
@pytest.mark.parametrize("bound", [False, True])
def test_histogramdd(
    size: int,
    bins: int,
    bound: bool,
):
    x = torch.randn(size, len(bins))

    if bound:
        low, upp = x.min(dim=0).values, x.max(dim=0).values
    else:
        low, upp = x.min(dim=0).values + 1, x.max(dim=0).values - 1

    edges = [None] * len(bins)

    for i in range(len(bins)):
        edges[i] = torch.linspace(0, 1, bins[i] + 1) ** 2
        edges[i] = (upp[i] - low[i]) * edges[i] + low[i]

    h = histogramdd(x, edges=edges)
    h_np, _ = np.histogramdd(
        x.numpy(force=True),
        bins=[e.numpy(force=True) for e in edges],
    )

    assert np.all(h.numpy(force=True) == h_np)
