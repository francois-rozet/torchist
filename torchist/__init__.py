"""NumPy-style histograms in PyTorch"""

__version__ = '0.0.3'


import torch

from typing import Iterable, List, Union


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

    index = coords[..., 0]

    for i, dim in enumerate(shape[1:], 1):
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


def out_of_bounds(
    x: torch.Tensor,
    low: torch.Tensor,
    upp: torch.Tensor,
) -> torch.Tensor:
    r"""Returns a mask of out-of-bounds values in `x`.

    Args:
        x: A tensor, (*, D).
        low: The lower bound in each dimension, (,) or (D,).
        upp: The upper bound in each dimension, (,) or (D,).

    Returns:
        The mask tensor, (*,).
    """

    return torch.logical_or(
        torch.any(x < low, dim=-1),
        torch.any(x > upp, dim=-1)
    )


def discretize(
    x: torch.Tensor,
    bins: torch.Tensor,
    low: torch.Tensor,
    upp: torch.Tensor,
) -> torch.Tensor:
    r"""Maps the values of `x` to integers in [0, bins).

    Inverse of `torch.linspace`.

    Args:
        x: A tensor, (*, D).
        bins: The number of bins in each dimension, (,) or (D,).
        low: The lower bound in each dimension, (,) or (D,).
        upp: The upper bound in each dimension, (,) or (D,).

    Returns:
        The discretized tensor, (*, D).
    """

    span = torch.where(upp > low, upp - low, bins)  # > 0.

    x = (x - low) / span  # in [0., 1.]
    x = torch.where(x < 1., x, 1. - 1. / bins)  # in [0., 1.)
    x = torch.floor(x * bins)  # in [0, bins)

    return x


def histogramdd(
    x: torch.Tensor,
    bins: Union[int, List[int], torch.Tensor] = 10,
    low: Union[float, torch.Tensor] = 0.,
    upp: Union[float, torch.Tensor] = 0.,
    bounded: bool = False,
    weights: torch.Tensor = None,
    sparse: bool = False,
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
        sparse: Whether the histogram is returned as a sparse tensor or not.

    Returns:
        The histogram, (*bins,).
    """

    # Preprocess
    D = x.size(-1)
    x = x.view(-1, D)

    bins = torch.as_tensor(bins)
    shape = torch.Size(bins.int().expand(D))
    bins = bins.to(x)

    low, upp = torch.as_tensor(low), torch.as_tensor(upp)

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
        mask = ~out_of_bounds(x, low, upp)

        x = x[mask]
        weights = weights[mask]

    # Discretize
    idx = discretize(x, bins, low, upp).long()

    # Count
    if sparse:
        hist = torch.sparse_coo_tensor(idx.t(), weights, shape).coalesce()
    else:
        idx = ravel_multi_index(idx, shape)
        hist = idx.bincount(weights, minlength=shape.numel()).view(shape)

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

        `**kwargs` are passed on to `histogramdd`.

    Returns:
        The histogram, (bins,).
    """

    return histogramdd(x.unsqueeze(-1), bins, low, upp, **kwargs)


def reduce_histogramdd(
    seq: Iterable[torch.Tensor],
    *args,
    device: torch.device = None,
    hist: torch.Tensor = None,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the multidimensional histogram of a sequence of tensors.

    This is useful for large datasets that don't fit on CUDA memory or
    distributed datasets.

    Args:
        seq: A sequence of tensors, each (*, D).
        device: The device of the output histogram. If `None`,
            use the device of the first element of `seq`.
        hist: A histogram to aggregate the data of `seq` to.
            If provided, `device`, `bins` and `sparse` are ignored.
            Otherwise, a new (empty) histogram is used.

        `*args` and `**kwargs` are passed on to `histogramdd`.

    Returns:
        The histogram, (*bins,).
    """

    if hist is not None:
        kwargs['bins'] = torch.tensor(hist.shape)
        kwargs['sparse'] = hist.is_sparse

    for x in seq:
        temp = histogramdd(x, *args, **kwargs)

        if hist is None:
            hist = temp

            if device is not None:
                hist = hist.to(device)
        else:
            hist = hist + temp.to(hist)

    if hist.is_sparse:
        hist = hist.coalesce()

    return hist


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
