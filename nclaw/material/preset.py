from typing import Sequence
import math

import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from .abstract import Elasticity, Plasticity, Material


class ComposeMaterial(Material):
    def __init__(self, materials: list[Material], sections: Sequence[int]) -> None:
        super().__init__(None)
        self.materials = nn.ModuleList(materials)
        self.sections = sections

    def update_sections(self, sections: Sequence[int]) -> None:
        self.sections = sections

    def forward(self, F: Tensor) -> Tensor:
        outs = []
        for m, f in zip(self.materials, torch.split(F, self.sections, dim=0)):
            if f.numel() == 0:
                continue
            outs.append(m(f))
        return torch.cat(outs, dim=0)


class CorotatedElasticity(Elasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.log_E = nn.Parameter(torch.Tensor([cfg.E]).log())
        self.register_buffer('nu', torch.Tensor([cfg.nu]))

        if cfg.random:
            self.log_E.data.mul_(0.8)

    def forward(self, F: Tensor) -> Tensor:
        E = self.log_E.exp()
        nu = self.nu

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        # warp svd
        U, sigma, Vh = self.svd(F)

        corotated_stress = 2 * mu * torch.matmul(F - torch.matmul(U, Vh), F.transpose(1, 2))

        J = torch.prod(sigma, dim=1).view(-1, 1, 1)
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)
        volume_stress = la * J * (J - 1) * I

        stress = corotated_stress + volume_stress

        return stress


class StVKElasticity(Elasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.log_E = nn.Parameter(torch.Tensor([cfg.E]).log())
        self.register_buffer('nu', torch.Tensor([cfg.nu]))

        if cfg.random:
            self.log_E.data.mul_(0.8)

    def forward(self, F: Tensor) -> Tensor:
        E = self.log_E.exp()
        nu = self.nu

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        # warp svd
        U, sigma, Vh = self.svd(F)

        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)
        Ft = self.transpose(F)
        FtF = torch.matmul(Ft, F)

        E = 0.5 * (FtF - I)

        stvk_stress = 2 * mu * torch.matmul(F, E)

        J = torch.prod(sigma, dim=1).view(-1, 1, 1)
        volume_stress = la * J * (J - 1) * I

        stress = stvk_stress + volume_stress

        return stress


class VolumeElasticity(Elasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.log_E = nn.Parameter(torch.Tensor([cfg.E]).log())
        self.register_buffer('nu', torch.Tensor([cfg.nu]))

        if cfg.random:
            self.log_E.data.mul_(0.8)

        self.mode = cfg.mode

    def forward(self, F: Tensor) -> Tensor:
        E = self.log_E.exp()
        nu = self.nu

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        J = torch.det(F).view(-1, 1, 1)
        I = torch.eye(self.dim, dtype=F.dtype, device=F.device).unsqueeze(0)

        if self.mode.casefold() == 'ziran':

            #  https://en.wikipedia.org/wiki/Bulk_modulus
            kappa = 2 / 3 * mu + la

            # https://github.com/penn-graphics-research/ziran2020/blob/master/Lib/Ziran/Physics/ConstitutiveModel/EquationOfState.h
            # using gamma = 7 would have gradient issue, fix later
            gamma = 2

            stress = kappa * (J - 1 / torch.pow(J, gamma-1)) * I

        elif self.mode.casefold() == 'taichi':

            stress = la * J * (J - 1) * I

        else:
            raise ValueError('invalid mode for volume plasticity: {}'.format(self.mode))

        return stress


class SigmaElasticity(Elasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.log_E = nn.Parameter(torch.Tensor([cfg.E]).log())
        self.register_buffer('nu', torch.Tensor([cfg.nu]))

        if cfg.random:
            self.log_E.data.mul_(0.8)

    def forward(self, F: Tensor) -> Tensor:
        E = self.log_E.exp()
        nu = self.nu

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        # warp svd
        U, sigma, Vh = self.svd(F)

        epsilon = sigma.log()
        trace = epsilon.sum(dim=1, keepdim=True)
        tau = 2 * mu * epsilon + la * trace

        stress = torch.matmul(torch.matmul(U, torch.diag_embed(tau)), self.transpose(U))

        return stress



class IdentityPlasticity(Plasticity):
    def forward(self, F: Tensor) -> Tensor:
        return F


class SigmaPlasticity(Plasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def forward(self, F: Tensor) -> Tensor:
        J = torch.det(F)

        # unilateral incompressibility: https://github.com/penn-graphics-research/ziran2020/blob/master/Lib/Ziran/Physics/PlasticityApplier.cpp#L1084
        # J = torch.clamp(J, min=0.05, max=1.2)

        Je_1_3 = torch.pow(J, 1.0 / 3.0).view(-1, 1).expand(-1, 3)
        F = torch.diag_embed(Je_1_3)
        return F


class VonMisesPlasticity(Plasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.log_E = nn.Parameter(torch.Tensor([cfg.E]).log())
        self.register_buffer('nu', torch.Tensor([cfg.nu]))
        self.sigma_y = nn.Parameter(torch.Tensor([cfg.sigma_y]))

        if cfg.random:
            self.log_E.data.mul_(0.8)
            self.sigma_y.data.mul_(0.8)

    def forward(self, F: Tensor) -> Tensor:

        E = self.log_E.exp()
        nu = self.nu
        sigma_y = self.sigma_y

        mu = E / (2 * (1 + nu))

        # warp svd
        U, sigma, Vh = self.svd(F)

        # prevent NaN
        thredhold = 0.05
        sigma = torch.clamp_min(sigma, thredhold)

        epsilon = torch.log(sigma)
        trace = epsilon.sum(dim=1, keepdim=True)
        epsilon_hat = epsilon - trace / self.dim
        epsilon_hat_norm = torch.linalg.norm(epsilon_hat, dim=1, keepdim=True)

        delta_gamma = epsilon_hat_norm - sigma_y / (2 * mu)
        cond_yield = (delta_gamma > 0).view(-1, 1, 1)

        yield_epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        yield_F = torch.matmul(torch.matmul(U, torch.diag_embed(yield_epsilon.exp())), Vh)

        F = torch.where(cond_yield, yield_F, F)

        return F


class DruckerPragerPlasticity(Plasticity):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.log_E = nn.Parameter(torch.Tensor([cfg.E]).log())
        self.register_buffer('nu', torch.Tensor([cfg.nu]))
        self.friction_angle = nn.Parameter(torch.Tensor([cfg.friction_angle]))
        self.register_buffer('cohesion', torch.Tensor([cfg.cohesion]))

        if cfg.random:
            self.log_E.data.mul_(0.8)
            self.friction_angle.data.mul_(0.8)

    def forward(self, F: Tensor) -> Tensor:

        E = self.log_E.exp()
        nu = self.nu
        friction_angle = self.friction_angle
        sin_phi = torch.sin(torch.deg2rad(friction_angle))
        alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)
        cohesion = self.cohesion

        mu = E / (2 * (1 + nu))
        la = E * nu / ((1 + nu) * (1 - 2 * nu))

        # warp svd
        U, sigma, Vh = self.svd(F)

        # prevent NaN
        thredhold = 0.05
        sigma = torch.clamp_min(sigma, thredhold)

        epsilon = torch.log(sigma)
        trace = epsilon.sum(dim=1, keepdim=True)
        epsilon_hat = epsilon - trace / self.dim
        epsilon_hat_norm = torch.linalg.norm(epsilon_hat, dim=1, keepdim=True)

        expand_epsilon = torch.ones_like(epsilon) * cohesion

        shifted_trace = trace - cohesion * self.dim
        cond_yield = (shifted_trace < 0).view(-1, 1)

        delta_gamma = epsilon_hat_norm + (self.dim * la + 2 * mu) / (2 * mu) * shifted_trace * alpha
        compress_epsilon = epsilon - (torch.clamp_min(delta_gamma, 0.0) / epsilon_hat_norm) * epsilon_hat

        epsilon = torch.where(cond_yield, compress_epsilon, expand_epsilon)

        F = torch.matmul(torch.matmul(U, torch.diag_embed(epsilon.exp())), Vh)

        return F
