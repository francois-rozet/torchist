"""NumPy-style histograms in PyTorch"""

__version__ = '0.0.5'


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
    low: Union[float, List[float], torch.Tensor] = 0.,
    upp: Union[float, List[float], torch.Tensor] = 0.,
    bounded: bool = False,
    weights: torch.Tensor = None,
    sparse: bool = False,
) -> torch.Tensor:
    r"""Computes the multidimensional histogram of a tensor.

    This is a `torch` implementation of `numpy.histogramdd`.

    Args:
        x: A tensor, (*, D).
        bins: The number of bins in each dimension.
        low: The lower bound in each dimension.
        upp: The upper bound in each dimension.
            If `upp` is equal to `low`, the min and max of `x` are used instead
            and `bounded` is ignored.
        bounded: Whether `x` is bounded by `low` and `upp`, included.
            If `False`, out-of-bounds values are filtered out.
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


def histogramdd_edges(
    x: torch.Tensor,
    bins: Union[int, List[int], torch.Tensor] = 10,
    low: Union[float, List[float], torch.Tensor] = 0.,
    upp: Union[float, List[float], torch.Tensor] = 0.,
) -> List[torch.Tensor]:
    r"""Computes the edges of the uniform bins used by `histogramdd`.

    This is useful when plotting an histogram along with axes.

    Args:
        x: A tensor, (*, D).
        bins: The number of bins in each dimension.
        low: The lower bound in each dimension.
        upp: The upper bound in each dimension.
            If `upp` is equal to `low`, the min and max of `x` are used instead.

    Returns:
        The list of D bin edges, each (bins + 1,).
    """

    D = x.size(-1)
    x = x.view(-1, D)

    bins = torch.as_tensor(bins).expand(D)
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
            If `upp` is equal to `low`, the min and max of `x` are used instead
            and `bounded` is ignored.

        `**kwargs` are passed on to `histogramdd`.

    Returns:
        The histogram, (bins,).
    """

    return histogramdd(x.view(-1, 1), bins, low, upp, **kwargs)


def histogram_edges(
    x: torch.Tensor,
    bins: int = 10,
    low: float = 0.,
    upp: float = 0.,
) -> torch.Tensor:
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

    Warning:
        In this function, `histogramdd` is called on each element of `seq`.
        If `low` and `high` are not set, the computed histograms will have
        different sets of bin edges and their reduction (sum) will be incoherent.
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


def normalize(hist: torch.Tensor) -> torch.Tensor:
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


def marginalize(
    hist: torch.Tensor,
    dim: Union[int, List[int]],
    keep: bool = False,
) -> torch.Tensor:
    r"""Marginalizes (reduces by sum) a histogram over given dimensions.

    Args:
        hist: A dense or sparse histogram, (*,).
        dim: The list of dimensions to marginalize over.
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


def sinkhorn_transport(
    r: torch.Tensor,
    c: torch.Tensor,
    M: torch.Tensor,
    gamma: float = 100.,
    max_iter: int = 1000,
    threshold: float = 1e-8,
    step: int = 100,
) -> torch.Tensor:
    r"""Computes the entropic regularized optimal transport between
    a source and a target distribution with respect to a cost matrix.

    This function implements the Sinkhorn-Knopp algorithm from [1].

    Args:
        r: A source dense histogram, (N,).
        c: A target dense histogram, (M,).
        M: A cost matrix, (N, M).
        gamma: The regularization term.
        max_iter: The maximum number of iterations.
        threshold: The stopping threshold on the error.
        step: The number of iterations between two checks of the error.

    Returns:
        The transport, (N, M).

    References:
        [1] Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances
        (Cuturi, 2013)
        https://arxiv.org/pdf/1306.0895.pdf
    """

    K = (-gamma * M).exp()
    Kt = K.t().contiguous()

    u = torch.full_like(r, 1. / len(r))

    for i in range(max_iter):
        v = c / (Kt @ u)
        u = r / (K @ v)

        if i % step == 0:
            marginal = (Kt @ u) * v

            err = torch.linalg.norm(marginal - c)
            if err < threshold:
                break

    return u.view(-1, 1) * K * v


def rw_distance(
    p: torch.Tensor,
    q: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the regularized Wasserstein distance between two distributions,
    assuming an Euclidean distance matrix.

    Args:
        p: A dense or sparse histogram, (*,).
        q: A dense or sparse histogram, (*,).

        `**kwargs` are passed on to `sinkhorn_transport`.

    Returns:
        The distance, (,).
    """

    # Sparsify
    p = p.coalesce() if p.is_sparse else p.to_sparse()
    q = q.coalesce() if q.is_sparse else q.to_sparse()

    # Euclidean distance matrix
    scale = 1. / torch.tensor(p.shape).to(p)

    x_p = p.indices().t() * scale
    x_q = q.indices().t() * scale

    M = torch.cdist(x_p[None], x_q[None])[0]

    # Regularized optimal transport
    T = sinkhorn_transport(p.values(), q.values(), M, **kwargs)

    return (T * M).sum()


def kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    r"""Computes the Kullback-Leibler divergence between two distributions.

    Args:
        p: A dense histogram, (*,).
        q: A dense histogram, (*,).

    Returns:
        The divergence, (,).
    """

    mask = p > 0.
    p, q = p[mask], q[mask]

    return (p * (p.log() - q.log())).sum()


if __name__ == '__main__':
    import numpy as np
    import timeit

    x = torch.rand(100000, 5)
    x0 = x[:, 0].clone()

    print('CPU')
    print('---')

    for key, f in {
        'np.histogram': np.histogram,
        'torchist.histogram': histogram,
        'np.histogramdd': np.histogramdd,
        'torchist.histogramdd': histogramdd,
    }.items():
        y = x if 'dd' in key else x0
        y = y if 'torch' in key else y.numpy()

        time = timeit.timeit(lambda: f(y), number=100)

        print(key, ':', '{:.04f}'.format(time), 's')

    if torch.cuda.is_available():
        print()
        print('CUDA')
        print('----')

        x, x0 = x.cuda(), x0.cuda()

        for key, f in {
            'torchist.histogram': histogram,
            'torchist.histogramdd': histogramdd,
        }.items():
            y = x if 'dd' in key else x0

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            for _ in range(100):
                f(y)

            end.record()

            torch.cuda.synchronize()

            time = start.elapsed_time(end) / 1000 # ms -> s

            print(key, ':', '{:.04f}'.format(time), 's')
