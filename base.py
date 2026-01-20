import torch
import torch.nn as nn


class BaseConstraintMap(nn.Module):
    """
    Constraint operator of the form K(z) x - d(z)
    """
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def adjoint(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    

class BaseProx(nn.Module):
    """
    Base class for proximal operators
    """
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError