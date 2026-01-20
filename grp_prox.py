import torch
from typing import List
from base import BaseProx
from utils import grp_grad


class GroupProx(BaseProx):
    """
    Generates the proximal operator for sum_G ||x_G||_p^q
    Solved using damped fixed point iteration
    """
    def __init__(
            self,
            groups: List[torch.Tensor],
            tau: float,
            p: float,
            q: float,
            l: torch.Tensor,
            u: torch.Tensor,
            eps: float = 1e-8,
            eta: float = 0.5,
            max_iter: int = 5,
            tol: float = 1e-4,
            detach_internal: bool = True
    ):
        super().__init__()
        self.groups = groups
        self.tau = tau
        self.p = p
        self.q = q
        self.eps = eps
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.detach_internal = detach_internal
        self.register_buffer("l", l)
        self.register_buffer("u", u)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute prox operator at v
        """
        x_out = v.clone()
        for idx in self.groups:
            v_g = v[idx]
            x_g = v_g.clone()
            if self.detach_internal: x_g = x_g.detach()
            for _ in range(self.max_iter):
                grad = grp_grad(
                    x_g, self.p, self.q, self.eps
                )
                x_next = (1.0 - self.eta) * x_g + self.eta * (v_g - self.tau * grad)
                if torch.max(torch.abs(x_next - x_g)) < self.tol:
                    x_g = x_next
                    break
                x_g = x_next
            x_g = torch.max(torch.min(x_g, self.u[idx]), self.l[idx])
            x_out[idx] = x_g
        return x_out