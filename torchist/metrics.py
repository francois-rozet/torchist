r"""Miscellaneous metrics over distributions"""

from . import *


def entropy(p: Tensor) -> Tensor:
    r"""Computes the entropy of a distribution.

    Args:
        p: A dense or sparse histogram, (*,).

    Returns:
        The entropy, (,).
    """

    if p.is_sparse:
        p = p.coalesce().values()

    zero = p.new_tensor(0.)

    h = p * p.log()
    h = torch.where(p > 0., h, zero)

    return -h.sum()


def kl_divergence(p: Tensor, q: Tensor, eps: float = 1e-42) -> Tensor:
    r"""Computes the Kullback-Leibler divergence between two distributions.

    Args:
        p: A dense or sparse histogram, (*,).
        q: A dense or sparse histogram, (*,).
        eps: A threshold value.

    Returns:
        The divergence, (,).
    """

    if p.is_sparse:
        p, q = p.coalesce(), q.coalesce()

        log_p = torch.sparse_coo_tensor(p.indices(), p.values().log(), p.shape)

        log_q = q + 0 * p
        log_q._values().clip_(min=eps)
        log_q._values().log_()

        kl = p * (log_p - log_q)
        kl = kl._values()
    else:
        zero = p.new_tensor(0.)

        kl = p * (p.log() - q.clip(min=eps).log())
        kl = torch.where(p > 0., kl, zero)

    return kl.sum()


def js_divergence(p: Tensor, q: Tensor, gamma: float = 0.5, **kwargs) -> Tensor:
    r"""Computes the Jensen-Shannon divergence between two distributions.

    Args:
        p: A dense or sparse histogram, (*,).
        q: A dense or sparse histogram, (*,).
        gamma: The mixing rate.

        `**kwargs` are passed on to `kl_divergence`.

    Returns:
        The divergence, (,).
    """

    m = gamma * p + (1 - gamma) * q
    left = kl_divergence(p, m, **kwargs)
    right = kl_divergence(q, m, **kwargs)

    return gamma * left + (1 - gamma) * right


def sinkhorn_transport(
    r: Tensor,
    c: Tensor,
    M: Tensor,
    gamma: float = 100.0,
    max_iter: int = 1000,
    threshold: float = 1e-8,
    step: int = 100,
) -> Tensor:
    r"""Computes the entropic regularized optimal transport between
    a source and a target distribution with respect to a cost matrix.

    This function implements the Sinkhorn-Knopp algorithm from [1].

    Args:
        r: A source dense histogram, (A,).
        c: A target dense histogram, (B,).
        M: A cost matrix, (A, B).
        gamma: The regularization term.
        max_iter: The maximum number of iterations.
        threshold: The stopping threshold on the error.
        step: The number of iterations between two checks of the error.

    Returns:
        The transport, (A, B).

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


def em_distance(p: Tensor, q: Tensor, **kwargs) -> Tensor:
    r"""Computes the regularized earth mover's distance between two distributions,
    assuming a unit Euclidean distance matrix.

    Args:
        p: A dense or sparse histogram, (*,).
        q: A dense or sparse histogram, (*,).

        `**kwargs` are passed on to `sinkhorn_transport`.

    Returns:
        The distance, (,).
    """

    shape = p.new_tensor(p.shape)

    # Positions
    if p.is_sparse:
        p, q = p.coalesce(), q.coalesce()
        p, idx_p = p.values(), p.indices()
        q, idx_q = p.values(), p.indices()

        x_p, x_q = idx_p / shape, idx_q / shape
    else:
        idx = torch.arange(p.numel(), device=p.device)
        idx = unravel_index(idx, p.shape)

        x_p = idx.to(p) / shape
        x_q = x_p

    # Euclidean distance matrix
    M = torch.cdist(x_p[None], x_q[None])[0]

    # Regularized optimal transport
    T = sinkhorn_transport(p.view(-1), q.view(-1), M, **kwargs)

    return (T * M).sum()
