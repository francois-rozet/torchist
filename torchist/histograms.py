r"""NumPy-style histograms"""

import math
import torch

from torch import BoolTensor, LongTensor, Tensor
from typing import Iterable, List, Sequence, Tuple, Union


def ravel_multi_index(coords: LongTensor, shape: Sequence[int]) -> LongTensor:
    r"""Converts a tensor of coordinate vectors into a tensor of flat indices.

    This is a `torch` implementation of `numpy.ravel_multi_index`.

    Args:
        coords: A tensor of coordinate vectors, (*, D).
        shape: The source shape.

    Returns:
        The raveled indices, (*,).
    """

    shape = coords.new_tensor((*shape, 1))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return (coords * coefs).sum(dim=-1)


def unravel_index(indices: LongTensor, shape: Sequence[int]) -> LongTensor:
    r"""Converts a tensor of flat indices into a tensor of coordinate vectors.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of flat indices, (*,).
        shape: The target shape.

    Returns:
        The unraveled coordinates, (*, D).
    """

    shape = indices.new_tensor((*shape, 1))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return torch.div(indices[..., None], coefs, rounding_mode="trunc") % shape[:-1]


def out_of_bounds(x: Tensor, low: Tensor, upp: Tensor) -> BoolTensor:
    r"""Returns a mask of out-of-bounds values in `x`.

    Args:
        x: A tensor, (*, D).
        low: The lower bound in each dimension, scalar or (D,).
        upp: The upper bound in each dimension, scalar or (D,).

    Returns:
        The mask tensor, (*,).
    """

    a, b = x < low, x > upp

    if x.ndim > 1:
        a, b = torch.any(a, dim=-1), torch.any(b, dim=-1)

    return torch.logical_or(a, b)


@torch.no_grad()
def quantize(x: Tensor, bins: Tensor, low: Tensor, upp: Tensor) -> LongTensor:
    r"""Maps the values of `x` to integers.

    Args:
        x: A tensor, (*, D).
        bins: The number of bins in each dimension, scalar or (D,).
        low: The lower bound in each dimension, scalar or (D,).
        upp: The upper bound in each dimension, scalar or (D,).

    Returns:
        The quantized tensor, (*, D).
    """

    x = (x - low) / (upp - low)  # in [0.0, 1.0]
    x = (bins * x).long()  # in [0, bins]
    x = x.clip(max=bins - 1)  # in [0, bins - 1]

    return x


def histogramdd(
    x: Tensor,
    bins: Union[int, Sequence[int]] = 10,
    low: Union[float, Sequence[float]] = None,
    upp: Union[float, Sequence[float]] = None,
    edges: Union[Tensor, Sequence[Tensor]] = None,
    bound: bool = False,
    weights: Tensor = None,
    sparse: bool = False,
) -> Tensor:
    r"""Computes the multidimensional histogram of a tensor.

    This is a `torch` implementation of `numpy.histogramdd`. This function is
    differentiable with respect to `weights`, but not with respect to `x` and `edges`.

    Note:
        Similar to `numpy.histogram`, all bins are half-open except the last bin which
        also includes the upper bound.

    Args:
        x: A tensor, (*, D).
        bins: The number of bins in each dimension, scalar or (D,).
        low: The lower bound in each dimension, scalar or (D,). If `low` is `None`,
            the min of `x` is used instead.
        upp: The upper bound in each dimension, scalar or (D,). If `upp` is `None`,
            the max of `x` is used instead.
        edges: The edges of the histogram. Either a vector or a list of vectors.
            If provided, `bins`, `low` and `upp` are inferred from `edges`.
        bound: Whether `x` is bound by `low` and `upp`, included.
            If `False`, out-of-bounds values are filtered out.
        weights: A tensor of weights, (*,). Each sample of `x` contributes
            its associated weight towards the bin count (instead of 1).
        sparse: Whether the histogram is returned as a sparse tensor or not.

    Returns:
        The histogram, (*bins,).
    """

    # Preprocess
    D = x.shape[-1]
    x = x.reshape(-1, D).detach()

    if edges is None:
        bound = bound or (low is None and upp is None)

        if low is None:
            low = x.min(dim=0).values

        if upp is None:
            upp = x.max(dim=0).values
    elif torch.is_tensor(edges):
        edges = edges.flatten().detach().to(x)
        bins = edges.numel() - 1
        low = edges[0]
        upp = edges[-1]
    else:
        edges = [e.flatten().detach().to(x) for e in edges]
        bins = [e.numel() - 1 for e in edges]
        low = [e[0] for e in edges]
        upp = [e[-1] for e in edges]

    bins = torch.as_tensor(bins, dtype=torch.long).squeeze()
    low = torch.as_tensor(low, dtype=x.dtype, device=x.device).squeeze()
    upp = torch.as_tensor(upp, dtype=x.dtype, device=x.device).squeeze()

    assert torch.all(upp > low), "The upper bound must be strictly larger than the lower bound"

    if weights is not None:
        weights = weights.flatten()

    # Filter out-of-bound values
    if not bound:
        mask = ~out_of_bounds(x, low, upp)

        x = x[mask]

        if weights is not None:
            weights = weights[mask]

    # Indexing
    if edges is None:
        idx = quantize(x, bins.to(device=x.device), low, upp)
    elif torch.is_tensor(edges):
        idx = torch.bucketize(x, edges[1:-1], right=True)
    else:
        x = x.T.contiguous()
        idx = torch.stack(
            [torch.bucketize(x[i], edges[i][1:-1], right=True) for i in range(D)],
            dim=-1,
        )

    # Histogram
    shape = bins.expand(D).tolist()

    if sparse:
        if weights is None:
            idx, values = torch.unique(idx, dim=0, return_counts=True)
        else:
            idx, inverse = torch.unique(idx, dim=0, return_inverse=True)
            values = weights.new_zeros(len(idx))
            values = values.scatter_add(dim=0, index=inverse, src=weights)

        hist = torch.sparse_coo_tensor(idx.t(), values, shape)
        hist._coalesced_(True)
    else:
        idx = ravel_multi_index(idx, shape)
        hist = idx.bincount(weights, minlength=math.prod(shape)).reshape(shape)

    return hist


def histogramdd_edges(
    x: Tensor,
    bins: Union[int, Sequence[int]] = 10,
    low: Union[float, Sequence[float]] = None,
    upp: Union[float, Sequence[float]] = None,
) -> List[Tensor]:
    r"""Computes the edges of the uniform bins used by `histogramdd`.

    This is useful when plotting an histogram along with axes.

    Args:
        x: A tensor, (*, D).
        bins: The number of bins in each dimension, scalar or (D,).
        low: The lower bound in each dimension, scalar or (D,). If `low` is `None`,
            the min of `x` is used instead.
        upp: The upper bound in each dimension, scalar or (D,). If `upp` is `None`,
            the max of `x` is used instead.

    Returns:
        The list of D bin edges, each (bins + 1,).
    """

    D = x.shape[-1]
    x = x.reshape(-1, D)

    bins = torch.as_tensor(bins).long().expand(D)

    if low is None:
        low = x.min(dim=0)[0]
    else:
        low = torch.as_tensor(low).squeeze().expand(D)

    if upp is None:
        upp = x.max(dim=0)[0]
    else:
        upp = torch.as_tensor(upp).squeeze().expand(D)

    return [torch.linspace(l, u, b + 1) for (l, u, b) in zip(low, upp, bins)]


def histogram(
    x: Tensor,
    bins: int = 10,
    low: float = None,
    upp: float = None,
    **kwargs,
) -> Tensor:
    r"""Computes the histogram of a tensor.

    This is a `torch` implementation of `numpy.histogram`.

    Args:
        x: A tensor, (*,).
        bins: The number of bins.
        low: The lower bound. If `low` is `None` the min of `x` is used instead.
        upp: The upper bound. If `upp` is `None` the max of `x` is used instead.
        kwargs: Keyword arguments passed to `histogramdd`.

    Returns:
        The histogram, (bins,).
    """

    return histogramdd(x.unsqueeze(-1), bins, low, upp, **kwargs)


def histogram_edges(
    x: Tensor,
    bins: int = 10,
    low: float = None,
    upp: float = None,
) -> Tensor:
    r"""Computes the edges of the uniform bins used by `histogramdd`.

    This is a `torch` implementation of `numpy.histogram_bin_edges`.

    Args:
        x: A tensor, (*,).
        bins: The number of bins.
        low: The lower bound. If `low` is `None` the min of `x` is used instead.
        upp: The upper bound. If `upp` is `None` the max of `x` is used instead.

    Returns:
        The bin edges, (bins + 1,).
    """

    return histogramdd_edges(x.unsqueeze(-1), bins, low, upp)[0]


def reduce_histogramdd(
    seq: Iterable[Tensor],
    *args,
    device: torch.device = None,
    **kwargs,
) -> Tensor:
    r"""Computes the multidimensional histogram of a sequence of tensors.

    Each element of the sequence is processed on its own device before
    being transferred to `device`. This is useful for distributed datasets
    or large datasets that don't fit on CUDA memory at once.

    Args:
        seq: A sequence of tensors, each (*, D).
        args: Positional arguments passed to `histogramdd`.
        device: The device of the output histogram. If `None`,
            use the device of the first element of `seq`.
        kwargs: Keyword arguments passed to `histogramdd`.

    Returns:
        The histogram, (*bins,).
    """

    assert "edges" in kwargs or ("low" in kwargs and "upp" in kwargs), (
        "Either the bin edges or the bin bounds must be set."
    )

    hist = None

    for x in seq:
        temp = histogramdd(x, *args, **kwargs)

        if hist is None:
            hist = temp.to(device)
        else:
            hist = hist + temp.to(hist)

    return hist


def normalize(hist: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Normalizes a histogram, that is, the sum of its elements is equal to one.

    Args:
        hist: A dense or sparse histogram, (*,).

    Returns:
        The normalized histogram and the normalization constant, (*,) and ().
    """

    if hist.is_sparse:
        norm = torch.sparse.sum(hist)
    else:
        norm = hist.sum()

    return hist / norm, norm


def marginalize(
    hist: Tensor,
    dim: Union[int, Sequence[int]],
    keep: bool = False,
) -> Tensor:
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
        dim = [i for i in range(hist.ndim) if i not in dim]

    if dim:
        if hist.is_sparse:
            hist = torch.sparse.sum(hist, dim=dim)
        else:
            hist = hist.sum(dim=dim)

    return hist


def benchmark():
    import numpy as np
    import timeit
    import torch

    print("CPU")
    print("---")

    x = np.random.rand(1000000)
    xdd = np.random.rand(1000000, 5)
    edges10 = np.linspace(0.0, 1.0, 11) ** 2
    edges100 = np.linspace(0.0, 1.0, 101) ** 2

    x_t = torch.from_numpy(x)
    xdd_t = torch.from_numpy(xdd)
    edges10_t = torch.from_numpy(edges10)
    edges100_t = torch.from_numpy(edges100)

    # Speed
    for key, f in {
        "np.histogram": lambda: np.histogram(x, bins=100),
        "np.histogram (edges)": lambda: np.histogram(x, bins=edges100),
        "np.histogramdd": lambda: np.histogramdd(xdd, bins=10),
        "np.histogramdd (edges)": lambda: np.histogramdd(xdd, bins=[edges10] * 5),
        "torch.histogram": lambda: torch.histogram(x_t, bins=100),
        "torch.histogram (edges)": lambda: torch.histogram(x_t, bins=edges100_t),
        "torch.histogramdd": lambda: torch.histogramdd(xdd_t, bins=10),
        "torch.histogramdd (edges)": lambda: torch.histogramdd(xdd_t, bins=[edges10_t] * 5),
        "torchist.histogram": lambda: histogram(x_t, bins=100),
        "torchist.histogram (edges)": lambda: histogram(x_t, edges=edges100_t),
        "torchist.histogramdd": lambda: histogramdd(xdd_t, bins=10),
        "torchist.histogramdd (edges)": lambda: histogramdd(xdd_t, edges=[edges10_t] * 5),
    }.items():
        time = timeit.timeit(f, number=100)
        print(key.ljust(32), "{:7.04f}".format(time), "s")

    if torch.cuda.is_available():
        print()
        print("CUDA")
        print("----")

        x_t = x_t.cuda()
        xdd_t = xdd_t.cuda()
        edges10_t = edges10_t.cuda()
        edges100_t = edges100_t.cuda()

        ## Speed
        for key, f in {
            "torchist.histogram": lambda: histogram(x_t, bins=100),
            "torchist.histogram (edges)": lambda: histogram(x_t, edges=edges100_t),
            "torchist.histogramdd": lambda: histogramdd(xdd_t, bins=10),
            "torchist.histogramdd (edges)": lambda: histogramdd(xdd_t, edges=[edges10_t] * 5),
        }.items():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(100):
                f()
            end.record()

            torch.cuda.synchronize()
            time = start.elapsed_time(end) / 1000  # ms -> s

            print(key.ljust(32), "{:7.04f}".format(time), "s")
