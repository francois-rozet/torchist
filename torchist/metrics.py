"""Miscellaneous metrics over distributions"""


from . import *


def entropy(p: Tensor) -> Tensor:
    r"""Computes the entropy of a distribution.

    Args:
        p: A dense histogram, (*,).

    Returns:
        The entropy, (,).
    """

    zero = p.new_tensor(0.)

    h = p * p.log()
    h = torch.where(p > 0., h, zero)

    return -h.sum()


def kl_divergence(p: Tensor, q: Tensor) -> Tensor:
    r"""Computes the Kullback-Leibler divergence between two distributions.

    Args:
        p: A dense histogram, (*,).
        q: A dense histogram, (*,).

    Returns:
        The divergence, (,).
    """

    zero = p.new_tensor(0.)

    kl = p * (p.log() - q.log())
    kl = torch.where(p > 0., kl, zero)

    return kl.sum()


def sinkhorn_transport(
    r: Tensor,
    c: Tensor,
    M: Tensor,
    gamma: float = 100.,
    max_iter: int = 1000,
    threshold: float = 1e-8,
    step: int = 100,
) -> Tensor:
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


def w_distance(p: Tensor, q: Tensor, **kwargs) -> Tensor:
    r"""Computes the regularized Wasserstein distance between two distributions,
    assuming a unit Euclidean distance matrix.

    Args:
        p: A dense histogram, (*,).
        q: A dense histogram, (*,).

        `**kwargs` are passed on to `sinkhorn_transport`.

    Returns:
        The distance, (,).
    """

    # Positions
    idx = torch.arange(p.numel(), device=p.device)
    idx = unravel_index(idx, p.shape)

    x = idx.to(p) / p.new_tensor(p.shape)

    # Euclidean distance matrix
    M = torch.cdist(x[None], x[None])[0]

    # Regularized optimal transport
    T = sinkhorn_transport(p.view(-1), q.view(-1), M, **kwargs)

    return (T * M).sum()
