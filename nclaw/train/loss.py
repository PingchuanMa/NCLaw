import torch
import torch.nn as nn
from torch import Tensor

class EulerDistanceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: Tensor, b: Tensor):
        return torch.linalg.norm(a - b, dim=1).mean()
