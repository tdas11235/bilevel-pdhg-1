import torch
import torch.nn as nn
from typing import Union
from pdhg import PDHGSolver
from grp_prox import GroupProx


class Bilevel(nn.Module):
    def __init__(
            self,
            solver: PDHGSolver,
            prox: GroupProx,
            z_init: torch.Tensor,
            z_l: torch.Tensor,
            z_u: torch.Tensor,
            z_lr: float,
            c: torch.Tensor
    ):
        super().__init__()
        self.solver = solver
        self.prox = prox
        self.c = c
        self.z = nn.Parameter(z_init.clone())
        self.register_buffer("z_l", z_l)
        self.register_buffer("z_u", z_u)
        self.z_lr = z_lr
    
    def value_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes c.T * x + ||x||_{q, p}
        """
        val = torch.dot(self.c, x)
        # mixed norm
        for idx in self.prox.groups:
            xg = x[idx]
            norm = torch.sum(torch.abs(xg) ** self.prox.p)
            norm = (norm + self.prox.eps) ** (1.0 / self.prox.p)
            val = val + self.prox.mu * (norm ** self.prox.q)
        return val
    
    def step(
            self,
            x0: torch.Tensor,
            y1_0: Union[torch.Tensor, None],
            y2_0: Union[torch.Tensor, None],
    ):
        # approximate solve (inner problem)
        x, y1, y2 = self.solver(
            x0=x0,
            y1_0=y1_0,
            y2_0=y2_0,
            z=self.z
        )
        # value function (outer objective)
        f_val = self.value_function(x)
        # backprop
        f_val.backward()
        # step on z
        with torch.no_grad():
            self.z -= self.z_lr * self.z.grad
            self.z[:] = torch.max(torch.min(self.z, self.z_u), self.z_l)
        print(self.z.grad)
        self.z.grad.zero_()
        return f_val.item(), x, y1, y2
    
    def solve(
            self,
            x0: torch.Tensor,
            y1_0: torch.Tensor,
            y2_0: torch.Tensor,
            num_steps: int,
            verbose: bool = True
    ):
        history = []
        for k in range(num_steps):
            fval, x, y1, y2 = self.step(x0, y1_0, y2_0)
            history.append(fval)
            if verbose:
                print(f"[Outer {k:03d}] value = {fval:.6e}")
            x0 = x.detach()
            y1_0 = y1.detach() if y1 is not None else None
            y2_0 = y2.detach() if y2 is not None else None
        return history, x0, y1_0, y2_0