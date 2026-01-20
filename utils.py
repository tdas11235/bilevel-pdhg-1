import torch


def grp_grad(
        x: torch.Tensor,
        p: float,
        q: float,
        eps: float
) -> torch.Tensor:
    """
    Gradient of ||x||_p^q with epsilon smoothing.
    Assumes p > 1, q >= 1
    """
    norm_pp = torch.sum(torch.abs(x) ** p)
    norm_p = (norm_pp + eps) ** (1.0 / p)
    c = q * (norm_p ** (q - p))
    grad = c * (torch.abs(x) ** (p - 2)) * x
    return grad