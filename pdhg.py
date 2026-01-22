import torch
import torch.nn as nn
from typing import Tuple, Optional, Union
from base import BaseConstraintMap, BaseProx


class PDHGSolver(nn.Module):
    def __init__(
            self,
            prox: BaseProx,
            ineq_map: Optional[BaseConstraintMap],
            eq_map: Optional[BaseConstraintMap],
            c: torch.Tensor,
            tau: float,
            sigma: float,
            theta: float,
            max_iter: int
    ):
        super().__init__()
        self.prox = prox
        self.ineq_map = ineq_map
        self.eq_map = eq_map
        self.tau = tau
        self.sigma = sigma
        self.theta = theta
        self.max_iter = max_iter
        self.register_buffer("c", c)
    
    def forward(
            self,
            x0: torch.Tensor,
            y1_0: Union[torch.Tensor, None],
            y2_0: Union[torch.Tensor, None],
            z: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x = x0
        x_bar = x0
        y1 = y1_0
        y2 = y2_0
        for _ in range(self.max_iter):
            # dual updates
            if self.ineq_map is not None:
                y1 = y1 + self.sigma * self.ineq_map(x_bar, z)
                y1 = torch.clamp(y1, min=0.0)
            if self.eq_map is not None:
                y2 = y2 + self.sigma * self.eq_map(x_bar, z)
            # primal update
            grad = self.c.clone()
            if self.ineq_map is not None:
                grad = grad + self.ineq_map.adjoint(y1, z)
            if self.eq_map is not None:
                grad = grad + self.eq_map.adjoint(y2, z)
            v = x - self.tau * grad
            # prox operator
            x_next = self.prox(v)
            # extrapolation
            x_bar = x_next + self.theta * (x_next - x)
            x = x_next
        return x, y1, y2
