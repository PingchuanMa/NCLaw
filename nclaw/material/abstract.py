import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig
from einops.layers.torch import Rearrange
from nclaw.warp import SVD


class Material(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.dim = 3

        self.svd = SVD()

        self.transpose = Rearrange('b d1 d2 -> b d2 d1', d1=self.dim, d2=self.dim)

    def forward(self, F: Tensor) -> Tensor:
        raise NotImplementedError


class Elasticity(Material):
    def forward(self, F: Tensor) -> Tensor:
        # F -> P
        raise NotImplementedError


class Plasticity(Material):
    def forward(self, F: Tensor) -> Tensor:
        # F -> F
        raise NotImplementedError
