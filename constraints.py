import torch
from base import BaseConstraintMap


class Inequality(BaseConstraintMap):
    def __init__(self, A_fn, b_fn):
        """
        :param A_fn: matrix or linear operator in z
        :param b_fn: vector in z
        """
        super().__init__()
        self.A_fn = A_fn
        self.b_fn = b_fn
    
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        A = self.A_fn(z)
        b = self.b_fn(z)
        return A @ x - b
    
    def adjoint(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        A = self.A_fn(z)
        return A.T @ y
    

class Equality(BaseConstraintMap):
    def __init__(self, P_fn, r_fn):
        """
        :param P_fn: matrix or linear operator in z
        :param r_fn: vector in z
        """
        super().__init__()
        self.P_fn = P_fn
        self.r_fn = r_fn

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        P = self.P_fn(z)
        r = self.r_fn(z)
        return P @ x - r

    def adjoint(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        P = self.P_fn(z)
        return P.T @ y
