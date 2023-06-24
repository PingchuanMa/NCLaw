from typing import Optional

from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange

from .abstract import Elasticity, Plasticity
from .utils import get_nonlinearity, get_norm, init_weight


class MLPBlock(nn.Module):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            no_bias: bool,
            norm: Optional[str],
            nonlinearity: Optional[str]) -> None:

        super().__init__()
        if norm == 'wn':
            self.fc = nn.utils.weight_norm(nn.Linear(in_planes, out_planes, not no_bias))
        else:
            self.fc = nn.Linear(in_planes, out_planes, bias=not no_bias and norm is None)
        self.norm = get_norm(norm, out_planes, dim=1, affine=not no_bias)
        self.nonlinearity = get_nonlinearity(nonlinearity)

    def forward(self, x: Tensor) -> Tensor:

        x = self.fc(x)
        x = self.norm(x)
        x = self.nonlinearity(x)
        return x


class MetaElasticity(Elasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=self.dim, d2=self.dim)
        self.unflatten = Rearrange('b (d1 d2) -> b d1 d2', d1=self.dim, d2=self.dim)

        self.normalize_input: bool = cfg.normalize_input

    def forward(self, F: Tensor) -> Tensor:
        raise NotImplementedError


class PlainMetaElasticity(MetaElasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = self.dim * self.dim
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)

        if self.normalize_input:
            x = self.flatten(F - I)
        else:
            x = self.flatten(F)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        P = self.unflatten(x)
        cauchy = torch.matmul(P, self.transpose(F))
        return cauchy


class PolarMetaElasticity(MetaElasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = self.dim * self.dim
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)

        U, sigma, Vh = self.svd(F)

        R = torch.matmul(U, Vh)
        S = torch.matmul(torch.matmul(self.transpose(Vh), torch.diag_embed(sigma)), Vh)

        if self.normalize_input:
            x = self.flatten(S - I)
        else:
            x = self.flatten(S)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        x = self.unflatten(x)
        x = 0.5 * (self.transpose(x) + x)
        P = torch.matmul(R, x)
        cauchy = torch.matmul(P, self.transpose(F))
        return cauchy


class InvariantMetaElasticity(MetaElasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = 3
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor) -> Tensor:
        U, sigma, Vh = self.svd(F)

        R = torch.matmul(U, Vh)

        Ft = self.transpose(F)

        if self.normalize_input:
            I1 = sigma.sum(dim=1) - 3.0
            I2 = torch.diagonal(torch.matmul(Ft, F), dim1=1, dim2=2).sum(dim=1) - 1.0
            I3 = torch.linalg.det(F) - 1.0
        else:
            I1 = sigma.sum(dim=1)
            I2 = torch.diagonal(torch.matmul(Ft, F), dim1=1, dim2=2).sum(dim=1)
            I3 = torch.linalg.det(F)

        x = torch.stack([I1, I2, I3], dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        x = self.unflatten(x)
        x = 0.5 * (self.transpose(x) + x)
        P = torch.matmul(R, x)
        cauchy = torch.matmul(P, self.transpose(F))
        return cauchy


class InvariantFullMetaElasticity(MetaElasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = self.dim + self.dim * self.dim + 1
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)
        U, sigma, Vh = self.svd(F)
        R = torch.matmul(U, Vh)

        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)

        if self.normalize_input:
            I1 = sigma - 1.0
            I2 = self.flatten(FtF - I)
            I3 = torch.linalg.det(F).unsqueeze(dim=1) - 1.0
        else:
            I1 = sigma
            I2 = self.flatten(FtF)
            I3 = torch.linalg.det(F).unsqueeze(dim=1)

        x = torch.cat([I1, I2, I3], dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        x = self.unflatten(x)
        x = 0.5 * (self.transpose(x) + x)
        P = torch.matmul(R, x)
        cauchy = torch.matmul(P, Ft)
        return cauchy


class SVDMetaElasticity(MetaElasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = self.dim
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor) -> Tensor:
        Ft = self.transpose(F)

        U, sigma, Vh = self.svd(F)

        if self.normalize_input:
            x = sigma - 1.0
        else:
            x = sigma
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)

        P = torch.matmul(torch.matmul(U, torch.diag_embed(x)), Vh)

        cauchy = torch.matmul(P, Ft)
        return cauchy


# http://viterbi-web.usc.edu/~jbarbic/isotropicMaterialEditor/XuSinZhuBarbic-Siggraph2015.pdf
class SplineMetaElasticity(MetaElasticity):

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.num_side_points: int = cfg.num_side_points
        self.xk_min: float = 0.0
        self.xk_max: float = cfg.xk_max
        self.yk_min: float = -cfg.yk_max
        self.yk_max: float = cfg.yk_max

        self.npoints = 2 * self.num_side_points + 1
        left_points = np.linspace(self.xk_min, 1.0, cfg.num_side_points + 1)
        right_points = np.linspace(1.0, self.xk_max , cfg.num_side_points + 1)
        xk = torch.tensor(left_points.tolist()[:-1] + [1.0] + right_points.tolist()[1:])
        self.register_buffer('xk', xk)

        w = torch.tensor([
            [-1.0, 3.0, -3.0, 1.0],
            [3.0, -6.0, 3.0, 0.0],
            [-3.0, 3.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ]).view(1, 4, 4)
        self.register_buffer('w', w)

        # if cfg.E is not None and cfg.nu is not None:
        #     E = cfg.E
        #     nu = cfg.nu
        #     mu = E / (2 * (1 + nu))
        #     la = E * nu / ((1 + nu) * (1 - 2 * nu))

        #     self.yk_f = nn.Parameter(la * xk - 3 * la + 2 * mu * (xk - 1))
        #     self.yk_g = nn.Parameter(torch.ones_like(xk) * la)
        #     self.yk_h = nn.Parameter(torch.zeros_like(xk))
        # else:

        self.yk_f = nn.Parameter(torch.linspace(self.yk_min, self.yk_max, xk.size(0)))
        self.yk_g = nn.Parameter(torch.linspace(self.yk_min, self.yk_max, xk.size(0)))
        self.yk_h = nn.Parameter(torch.linspace(self.yk_min, self.yk_max, xk.size(0)))

    def get_ak(self, yk):
        ak_1 = 2 / 3 * yk[0] + 2 / 3 * yk[1] - 1 / 3 * yk[2]
        ak_else = yk[1:-1] - 1 / 6 * yk[:-2] + 1 / 6 * yk[2:]
        return torch.cat([ak_1.unsqueeze(0), ak_else], dim=0)

    def get_bk(self, yk):
        bk_else = yk[1:-1] + 1 / 6 * yk[:-2] - 1 / 6 * yk[2:]
        bk_m = 2 / 3 * yk[-1] + 2 / 3 * yk[-2] - 1 / 3 * yk[-3]
        return torch.cat([bk_else, bk_m.unsqueeze(0)], dim=0)

    def get_func(self, yk, lambd):
        indices = torch.searchsorted(self.xk, lambd, right=False).view(-1)
        indices[indices < 0] = 0
        indices[indices > self.num_side_points - 1] = self.num_side_points - 1

        ak = self.get_ak(yk)
        bk = self.get_bk(yk)

        y_left = yk[indices].view_as(lambd)
        y_right = yk[indices + 1].view_as(lambd)
        a = ak[indices].view_as(lambd)
        b = bk[indices].view_as(lambd)
        temp_right = torch.stack([y_left, a, b, y_right], dim=2)

        xi = (lambd - self.xk[indices].view_as(lambd)) / (self.xk[indices + 1].view_as(lambd) - self.xk[indices].view_as(lambd))
        xi_vector = torch.stack([xi**3, xi**2, xi, torch.ones_like(xi)], dim=2) # batch, #lambda, 4

        temp_left = torch.matmul(xi_vector, self.w) # batch, #lambda, 4
        func = (temp_left * temp_right).sum(dim=2) # batch, #lambda

        return func

    def forward(self, F: Tensor) -> Tensor:
        U, sigma, Vh = self.svd(F)

        f = self.get_func(self.yk_f, sigma)

        areas = torch.stack([
            sigma[:, 0] * sigma[:, 1],
            sigma[:, 1] * sigma[:, 2],
            sigma[:, 0] * sigma[:, 2]], dim=1)
        g = self.get_func(self.yk_g, areas)

        g1 = g[:, [0, 0, 2]] * sigma[:, [1, 0, 0]]
        g2 = g[:, [2, 1, 1]] * sigma[:, [2, 2, 1]]

        volume = (sigma[:, 0] * sigma[:, 1] * sigma[:, 2]).unsqueeze(1)
        h = self.get_func(self.yk_h, volume) * sigma[:, [1, 0, 0]] * sigma[:, [2, 2, 1]]

        new_sigma = f + g1 + g2 + h
        P = torch.matmul(torch.matmul(U, torch.diag_embed(new_sigma)), Vh)

        Ft = self.transpose(F)

        cauchy = torch.matmul(P, Ft)
        return cauchy


class MetaPlasticity(Plasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.alpha: float = cfg.alpha

        self.flatten = Rearrange('b d1 d2 -> b (d1 d2)', d1=self.dim, d2=self.dim)
        self.unflatten = Rearrange('b (d1 d2) -> b d1 d2', d1=self.dim, d2=self.dim)

        self.normalize_input: bool = cfg.normalize_input

    def forward(self, F: Tensor) -> Tensor:
        raise NotImplementedError


class PlainMetaPlasticity(MetaPlasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = self.dim * self.dim
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)

        if self.normalize_input:
            x = self.flatten(F - I)
        else:
            x = self.flatten(F)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        delta_Fp = self.alpha * self.unflatten(x)
        Fp = delta_Fp + F
        return Fp


class PolarMetaPlasticity(MetaPlasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = self.dim * self.dim
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)

        U, sigma, Vh = self.svd(F)

        R = torch.matmul(U, Vh)
        S = torch.matmul(torch.matmul(self.transpose(Vh), torch.diag_embed(sigma)), Vh)

        if self.normalize_input:
            x = self.flatten(S - I)
        else:
            x = self.flatten(S)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        x = self.unflatten(x)
        x = 0.5 * (self.transpose(x) + x)
        delta_Fp = self.alpha * torch.matmul(R, x)
        Fp = delta_Fp + F
        return Fp


class InvariantFullMetaPlasticity(MetaPlasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = 3 + 9 + 1
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim * self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor) -> Tensor:
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device, requires_grad=False)
        U, sigma, Vh = self.svd(F)
        R = torch.matmul(U, Vh)

        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)

        I1 = sigma - 1.0
        I2 = self.flatten(FtF - I)
        I3 = torch.linalg.det(F).unsqueeze(dim=1) - 1.0

        invariants = torch.cat([I1, I2, I3], dim=1)
        x = invariants
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        x = self.unflatten(x)
        x = 0.5 * (self.transpose(x) + x)
        delta_Fp = self.alpha * torch.matmul(R, x)
        Fp = delta_Fp + F
        return Fp


class SplineMetaPlasticity(MetaPlasticity):

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.num_side_points: int = cfg.num_side_points
        self.xk_min: float = 0.0
        self.xk_max: float = cfg.xk_max
        self.yk_min: float = -cfg.yk_max
        self.yk_max: float = cfg.yk_max

        self.npoints = 2 * self.num_side_points + 1
        left_points = np.linspace(self.xk_min, 1.0, cfg.num_side_points + 1)
        right_points = np.linspace(1.0, self.xk_max , cfg.num_side_points + 1)
        xk = torch.tensor(left_points.tolist()[:-1] + [1.0] + right_points.tolist()[1:])
        self.register_buffer('xk', xk)

        w = torch.tensor([
            [-1.0, 3.0, -3.0, 1.0],
            [3.0, -6.0, 3.0, 0.0],
            [-3.0, 3.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ]).view(1, 4, 4)
        self.register_buffer('w', w)

        self.yk_f = nn.Parameter(torch.zeros_like(xk))
        self.yk_g = nn.Parameter(torch.zeros_like(xk))
        self.yk_h = nn.Parameter(torch.zeros_like(xk))

    def get_ak(self, yk):
        ak_1 = 2 / 3 * yk[0] + 2 / 3 * yk[1] - 1 / 3 * yk[2]
        ak_else = yk[1:-1] - 1 / 6 * yk[:-2] + 1 / 6 * yk[2:]
        return torch.cat([ak_1.unsqueeze(0), ak_else], dim=0)

    def get_bk(self, yk):
        bk_else = yk[1:-1] + 1 / 6 * yk[:-2] - 1 / 6 * yk[2:]
        bk_m = 2 / 3 * yk[-1] + 2 / 3 * yk[-2] - 1 / 3 * yk[-3]
        return torch.cat([bk_else, bk_m.unsqueeze(0)], dim=0)

    def get_func(self, yk, lambd):
        indices = torch.searchsorted(self.xk, lambd, right=False).view(-1)
        indices[indices < 0] = 0
        indices[indices > self.num_side_points - 1] = self.num_side_points - 1

        ak = self.get_ak(yk)
        bk = self.get_bk(yk)

        y_left = yk[indices].view_as(lambd)
        y_right = yk[indices + 1].view_as(lambd)
        a = ak[indices].view_as(lambd)
        b = bk[indices].view_as(lambd)
        temp_right = torch.stack([y_left, a, b, y_right], dim=2)

        xi = (lambd - self.xk[indices].view_as(lambd)) / (self.xk[indices + 1].view_as(lambd) - self.xk[indices].view_as(lambd))
        xi_vector = torch.stack([xi**3, xi**2, xi, torch.ones_like(xi)], dim=2) # batch, #lambda, 4

        temp_left = torch.matmul(xi_vector, self.w) # batch, #lambda, 4
        func = (temp_left * temp_right).sum(dim=2) # batch, #lambda

        return func

    def forward(self, F: Tensor) -> Tensor:
        U, sigma, Vh = self.svd(F)

        f = self.get_func(self.yk_f, sigma)

        areas = torch.stack([
            sigma[:, 0] * sigma[:, 1],
            sigma[:, 1] * sigma[:, 2],
            sigma[:, 0] * sigma[:, 2]], dim=1)
        g = self.get_func(self.yk_g, areas)

        g1 = g[:, [0, 0, 2]] * sigma[:, [1, 0, 0]]
        g2 = g[:, [2, 1, 1]] * sigma[:, [2, 2, 1]]

        volume = (sigma[:, 0] * sigma[:, 1] * sigma[:, 2]).unsqueeze(1)
        h = self.get_func(self.yk_h, volume) * sigma[:, [1, 0, 0]] * sigma[:, [2, 2, 1]]

        new_sigma = f + g1 + g2 + h
        delta_Fp = self.alpha * torch.matmul(torch.matmul(U, torch.diag_embed(new_sigma)), Vh)

        Fp = delta_Fp + F
        return Fp


class SVDMetaPlasticity(MetaPlasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.layers = nn.ModuleList()

        width = self.dim
        for next_width in cfg.layer_widths:
            self.layers.append(MLPBlock(width, next_width, cfg.no_bias, cfg.norm, cfg.nonlinearity))
            width = next_width

        self.final_layer = MLPBlock(width, self.dim, cfg.no_bias, None, None)

        for m in self.modules():
            init_weight(m)

    def forward(self, F: Tensor) -> Tensor:
        U, sigma, Vh = self.svd(F)

        if self.normalize_input:
            x = sigma - 1.0
        else:
            x = sigma
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)

        delta_Fp = self.alpha * torch.matmul(torch.matmul(U, torch.diag_embed(x)), Vh)
        Fp = delta_Fp + F

        return Fp
