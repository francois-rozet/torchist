"""NumPy-style histograms in PyTorch"""

__version__ = '0.1.3'


import torch

from typing import Iterable, List, Tuple, Union


Scalar = Union[int, float]
Tensor = torch.Tensor
Vector = Union[Scalar, List[Scalar], Tuple[Scalar, ...], Tensor]  # anything working with `torch.as_tensor`
Shape = Union[List[int], Tuple[int, ...], torch.Size]
Device = torch.device


def ravel_multi_index(coords: Tensor, shape: Shape) -> Tensor:
    r"""Converts a tensor of coordinate vectors into a tensor of flat indices.

    This is a `torch` implementation of `numpy.ravel_multi_index`.

    Args:
        coords: A tensor of coordinate vectors, (*, D).
        shape: The source shape.

    Returns:
        The raveled indices, (*,).
    """

    coef = coords.new_tensor(shape[1:] + (1,))
    coef = coef.flipud().cumprod(0).flipud()

    if coords.is_cuda and not coords.is_floating_point():
        return (coords * coef).sum(dim=-1)
    else:
        return coords @ coef


def unravel_index(indices: Tensor, shape: Shape) -> Tensor:
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


def out_of_bounds(x: Tensor, low: Tensor, upp: Tensor) -> Tensor:
    r"""Returns a mask of out-of-bounds values in `x`.

    Args:
        x: A tensor, (*, D).
        low: The lower bound in each dimension, scalar or (D,).
        upp: The upper bound in each dimension, scalar or (D,).

    Returns:
        The mask tensor, (*,).
    """

    a, b = x < low, x > upp

    if x.dim() > 1:
        a, b = torch.any(a, dim=-1), torch.any(b, dim=-1)

    return torch.logical_or(a, b)


def quantize(x: Tensor, bins: Tensor, low: Tensor, upp: Tensor) -> Tensor:
    r"""Maps the values of `x` to integers in [0, bins).

    Args:
        x: A tensor, (*, D).
        bins: The number of bins in each dimension, scalar or (D,).
        low: The lower bound in each dimension, scalar or (D,).
        upp: The upper bound in each dimension, scalar or (D,).

    Returns:
        The quantized tensor, (*, D).
    """

    span = torch.where(upp > low, upp - low, bins)  # > 0.

    x = (x - low) * (bins / span)  # in [0., bins]
    x = torch.where(x < bins, x, bins - 1.)  # in [0., bins)
    x = x.long()  # in [0, bins)

    return x


def pack_edges(edges: List[Tensor]) -> Tensor:
    r"""Packs a list of edges vector as a single tensor.

    Shorther vectors are padded with the `inf` value.

    Args:
        edges: A list of D edges vectors, each (bins + 1,).

    Returns:
        The edges tensor, (D, max(bins) + 1).
    """

    maxlen = max(e.numel() for e in edges)

    pack = edges[0].new_full((len(edges), maxlen), float('inf'))
    for i, e in enumerate(edges):
        pack[i, :e.numel()] = e.view(-1)

    return pack


def len_packed_edges(edges: Tensor) -> Tensor:
    r"""Computes the length of each vector in a packed edges tensor.

    Args:
        edges: A edges tensor, (D, max(bins) + 1).

    Returns:
        The lengths, (D,).
    """

    return torch.count_nonzero(edges.isfinite(), dim=-1)


def histogramdd(
    x: Tensor,
    bins: Vector = 10,
    low: Vector = 0.,
    upp: Vector = 0.,
    bounded: bool = False,
    weights: Tensor = None,
    sparse: bool = False,
    edges: Union[Tensor, List[Tensor]] = None,
) -> Tensor:
    r"""Computes the multidimensional histogram of a tensor.

    This is a `torch` implementation of `numpy.histogramdd`.

    Args:
        x: A tensor, (*, D).
        bins: The number of bins in each dimension, scalar or (D,).
        low: The lower bound in each dimension, scalar or (D,).
        upp: The upper bound in each dimension, scalar or (D,).
            If `upp` is equal to `low`, the min and max of `x` are used instead
            and `bounded` is ignored.
        bounded: Whether `x` is bounded by `low` and `upp`, included.
            If `False`, out-of-bounds values are filtered out.
        weights: A tensor of weights, (*,). Each sample of `x` contributes
            its associated weight towards the bin count (instead of 1).
        sparse: Whether the histogram is returned as a sparse tensor or not.
        edges: The edges of the histogram. Either a vector, list of vectors or
            packed tensor of bin edges, (bins + 1,) or (D, max(bins) + 1).
            If provided, `bins`, `low` and `upp` are inferred from `edges`.

    Returns:
        The histogram, (*bins,).
    """

    # Preprocess
    D = x.size(-1)
    x = x.view(-1, D).squeeze(-1)

    if edges is None:
        bins = torch.as_tensor(bins).squeeze().long()
        low = torch.as_tensor(low).squeeze()
        upp = torch.as_tensor(upp).squeeze()

        if torch.all(low == upp):
            low, upp = x.min(dim=0)[0], x.max(dim=0)[0]
            bounded = True
        else:
            low, upp = low.to(x), upp.to(x)
    else:  # non-uniform binning
        if type(edges) is list:
            edges = pack_edges(edges)

        edges = edges.squeeze(0).to(x)

        if edges.dim() > 1:  # (D, max(bins) + 1)
            bins = len_packed_edges(edges) - 1
            low, upp = edges[:, 0], edges[torch.arange(D), bins]
        else:  # (bins + 1,)
            bins = torch.tensor(len(edges) - 1)
            low, upp = edges[0], edges[-1]

    # Weights
    if weights is None:
        weights = x.new_ones(x.size(0))
    else:
        weights = weights.view(-1)

    # Filter out-of-bound values
    if not bounded:
        mask = ~out_of_bounds(x, low, upp)

        x = x[mask]
        weights = weights[mask]

    # Indexing
    if edges is None:
        idx = quantize(x, bins.to(x), low, upp)
    else:  # non-uniform binning
        if edges.dim() > 1:
            idx = torch.searchsorted(edges, x.t().contiguous(), right=True).t() - 1
        else:
            idx = torch.bucketize(x, edges, right=True) - 1

    # Histogram
    shape = torch.Size(bins.expand(D))

    if sparse:
        hist = torch.sparse_coo_tensor(idx.t(), weights, shape).coalesce()
    else:
        if D > 1:
            idx = ravel_multi_index(idx, shape)
        hist = idx.bincount(weights, minlength=shape.numel()).view(shape)

    return hist


def histogramdd_edges(
    x: Tensor,
    bins: Vector = 10,
    low: Vector = 0.,
    upp: Vector = 0.,
) -> List[Tensor]:
    r"""Computes the edges of the uniform bins used by `histogramdd`.

    This is useful when plotting an histogram along with axes.

    Args:
        x: A tensor, (*, D).
        bins: The number of bins in each dimension, scalar or (D,).
        low: The lower bound in each dimension, scalar or (D,).
        upp: The upper bound in each dimension, scalar or (D,).
            If `upp` is equal to `low`, the min and max of `x` are used instead.

    Returns:
        The list of D bin edges, each (bins + 1,).
    """

    D = x.size(-1)
    x = x.view(-1, D)

    bins = torch.as_tensor(bins).int().expand(D)
    low, upp = torch.as_tensor(low), torch.as_tensor(upp)

    if torch.all(low == upp):
        low, upp = x.min(dim=0)[0], x.max(dim=0)[0]
    else:
        low, upp = low.expand(D), upp.expand(D)

    return [
        torch.linspace(l, u, b + 1)
        for (l, u, b) in zip(low, upp, bins)
    ]


def histogram(
    x: Tensor,
    bins: int = 10,
    low: float = 0.,
    upp: float = 0.,
    **kwargs,
) -> Tensor:
    r"""Computes the histogram of a tensor.

    This is a `torch` implementation of `numpy.histogram`.

    Args:
        x: A tensor, (*,).
        bins: The number of bins.
        low: The lower bound.
        upp: The upper bound.
            If `upp` is equal to `low`, the min and max of `x` are used instead
            and `bounded` is ignored.

        `**kwargs` are passed on to `histogramdd`.

    Returns:
        The histogram, (bins,).
    """

    return histogramdd(x.view(-1, 1), bins, low, upp, **kwargs)


def histogram_edges(
    x: Tensor,
    bins: int = 10,
    low: float = 0.,
    upp: float = 0.,
) -> Tensor:
    r"""Computes the edges of the uniform bins used by `histogramdd`.

    This is a `torch` implementation of `numpy.histogram_bin_edges`.

    Args:
        x: A tensor, (*,).
        bins: The number of bins.
        low: The lower bound.
        upp: The upper bound.
            If `upp` is equal to `low`, the min and max of `x` are used instead.

    Returns:
        The bin edges, (bins + 1,).
    """

    return histogramdd_bin_edges(x.view(-1, 1), bins, low, upp)[0]


def reduce_histogramdd(
    seq: Iterable[Tensor],
    *args,
    device: Device = None,
    **kwargs,
) -> Tensor:
    r"""Computes the multidimensional histogram of a sequence of tensors.

    Each element of the sequence is processed on its own device before
    being transferred to `device`. This is useful for distributed datasets
    or large datasets that don't fit on CUDA memory at once.

    Args:
        seq: A sequence of tensors, each (*, D).
        device: The device of the output histogram. If `None`,
            use the device of the first element of `seq`.

        `*args` and `**kwargs` are passed on to `histogramdd`.

    Returns:
        The histogram, (*bins,).

    Warning:
        In this function, `histogramdd` is called on each element of `seq`.
        If `low` and `high` are not set, the computed histograms will have
        different sets of bin edges and their sum will be incoherent.
    """

    hist = None

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


def normalize(hist: Tensor) -> Tensor:
    r"""Normalizes a histogram, that is, the sum of its elements is equal to one.

    Args:
        hist: A dense or sparse histogram, (*,).

    Returns:
        The normalized histogram, (*,).
    """

    if hist.is_sparse:
        norm = torch.sparse.sum(hist)
    else:
        norm = hist.sum()

    return hist / norm


def marginalize(hist: Tensor, dim: Union[int, Shape], keep: bool = False) -> Tensor:
    r"""Marginalizes (reduces by sum) a histogram over given dimensions.

    Args:
        hist: A dense or sparse histogram, (*,).
        dim: The dimension or set of dimensions to marginalize over.
        keep: Whether the dimensions in `dim` are the ones that are kept
            or the ones that are reduced.

    Returns:
        The marginalized histogram, (*,) or smaller.
    """

    if type(dim) is int:
        dim = [dim]

    if keep:
        dim = [i for i in range(hist.dim()) if i not in dim]

    if dim:
        if hist.is_sparse:
            hist = torch.sparse.sum(hist, dim=dim)
        else:
            hist = hist.sum(dim=dim)

    return hist


if __name__ == '__main__':  # bad practice
    import numpy as np
    import timeit

    print('CPU')
    print('---')

    x = np.random.rand(1000000)
    xdd = np.random.rand(1000000, 5)
    edges10 = np.linspace(0., 1., 11)
    edges100 = np.linspace(0., 1., 101)

    x_t = torch.from_numpy(x)
    xdd_t = torch.from_numpy(xdd)
    edges10_t = torch.from_numpy(edges10)
    edges100_t = torch.from_numpy(edges100)

    for key, f in {
        'np.histogram': lambda: np.histogram(x, bins=100),
        'np.histogramdd': lambda: np.histogramdd(xdd, bins=10),
        'np.histogram (non-uniform)': lambda: np.histogram(x, bins=edges100),
        'np.histogramdd (non-uniform)': lambda: np.histogramdd(xdd, bins=[edges10] * 5),
        'torchist.histogram': lambda: histogram(x_t, bins=100),
        'torchist.histogramdd': lambda: histogramdd(xdd_t, bins=10),
        'torchist.histogram (non-uniform)': lambda: histogram(x_t, edges=edges100_t),
        'torchist.histogramdd (non-uniform)': lambda: histogramdd(xdd_t, edges=[edges10_t] * 5),
    }.items():
        time = timeit.timeit(f, number=100)
        print(key, ':', '{:.04f}'.format(time), 's')

    if not torch.cuda.is_available():
        exit()

    print()
    print('CUDA')
    print('----')

    x_t = x_t.cuda()
    xdd_t = xdd_t.cuda()
    edges10_t = edges10_t.cuda()
    edges100_t = edges100_t.cuda()

    for key, f in {
        'torchist.histogram': lambda: histogram(x_t, bins=100),
        'torchist.histogramdd': lambda: histogramdd(xdd_t, bins=10),
        'torchist.histogram (non-uniform)': lambda: histogram(x_t, edges=edges100_t),
        'torchist.histogramdd (non-uniform)': lambda: histogramdd(xdd_t, edges=[edges10_t] * 5),
    }.items():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(100):
            f()
        end.record()

        torch.cuda.synchronize()
        time = start.elapsed_time(end) / 1000  # ms -> s

        print(key, ':', '{:.04f}'.format(time), 's')
