"""NumPy-style histograms in PyTorch"""

__version__ = '0.0.1'


import torch

from typing import List, Union


def ravel_multi_index(
    coords: torch.Tensor,
    shape: torch.Size,
) -> torch.Tensor:
    r"""Converts a tensor of coordinate vectors into a tensor of flat indices.

    This is a `torch` implementation of `numpy.ravel_multi_index`.

    Args:
        coords: A tensor of coordinate vectors, (*, D).
        shape: The source shape.

    Returns:
        The raveled indices, (*,).
    """

    index = torch.zeros_like(coords[..., 0])

    for i, dim in enumerate(shape):
        index = dim * index + coords[..., i]

    return index


def unravel_index(
    indices: torch.Tensor,
    shape: torch.Size,
) -> torch.Tensor:
    r"""Converts a tensor of flat indices into a tensor of coordinate vectors.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of flat indices, (*,).
        shape: The target shape.

    Returns:
        The unraveled coordinates, (*, D).
    """

    coords = []

    for dim in reversed(shape):
        coords.append(indices % dim)
        indices = indices // dim

    coords = torch.stack(coords[::-1], dim=-1)

    return coords


def histogramdd(
    x: torch.Tensor,
    bins: Union[int, List[int]] = 10,
    low: Union[float, torch.Tensor] = 0.,
    upp: Union[float, torch.Tensor] = 0.,
    bounded: bool = False,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    r"""Computes the multidimensional histogram of a tensor.

    This is a `torch` implementation of `numpy.histogramdd`.

    Args:
        x: A tensor, (*, D).
        bins: The number of bins in each dimension. If list, len(bins) is D.
        low: The lower bound in each dimension. If tensor, (D,).
        upp: The upper bound in each dimension. If tensor, (D,).
        bounded: Whether `x` is bounded by `low` and `upp`, included.
            When set to `False`, out-of-bounds values are filtered out.
            If `low` is equal to `upp`, the min and max of `x` are used instead
            and the value of `bounded` is ignored.
        weights: A tensor of weights, (*,). Each sample of `x` contributes
            its associated weight towards the bin count (instead of 1).

    Returns:
        The histogram, (*bins,).
    """

    # Preprocess
    D = x.size(-1)
    x = x.view(-1, D)

    bins = torch.tensor(bins)
    shape = torch.Size(bins.expand(D))
    bins = bins.to(x)

    low, upp = torch.tensor(low), torch.tensor(upp)

    if torch.all(low == upp):
        low, upp = x.min(dim=0)[0], x.max(dim=0)[0]
        bounded = True
    else:
        low, upp = low.to(x), upp.to(x)

    if weights is None:
        weights = torch.ones_like(x[:, 0])
    else:
        weights = weights.view(-1)

    # Filter out-of-bound values
    if not bounded:
        mask = torch.logical_and(
            torch.all(low <= x, dim=-1),
            torch.all(x <= upp, dim=-1)
        )

        x = x[mask]
        weights = weights[mask]

    # Discretize
    span = torch.where(upp > low, upp - low, bins)  # prevents null span

    x = (x - low) / span  # in [0., 1.]
    x = torch.where(x < 1., x, 1. - .5 / bins)  # in [0., 1.)
    x = torch.floor(x * bins)  # in [0., bins)

    if D > 1:
        x = ravel_multi_index(x, shape)
    else:
        x = x.view(-1)

    # Count
    x = x.long()
    hist = x.bincount(weights, minlength=shape.numel()).view(shape)

    return hist


def histogram(
    x: torch.Tensor,
    bins: int = 10,
    low: float = 0.,
    upp: float = 0.,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the histogram of a tensor.

    This is a `torch` implementation of `numpy.histogram`.

    Args:
        x: A tensor, (*,).
        bins: The number of bins.
        low: The lower bound.
        upp: The upper bound.

        **kwargs are passed on to `histogramdd`.

    Returns:
        The histogram, (bins,).
    """

    return histogramdd(x.unsqueeze(-1), bins, low, upp, **kwargs)


if __name__ == '__main__':
    import numpy as np
    import timeit

    x = torch.rand(100000, 5)
    x_np = x.numpy()

    for key, f in {
        'np.histogram': np.histogram,
        'torchist.histogram': histogram,
        'np.histogramdd': np.histogramdd,
        'torchist.histogramdd': histogramdd,
    }.items():
        if 'torch' in key:
            y = x
        else:
            y = x_np

        if 'dd' not in key:
            y = y[:, 0]

        g = lambda: f(y)

        print(key, ':', timeit.timeit(g, number=100), 's')
