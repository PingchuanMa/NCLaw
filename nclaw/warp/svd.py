import torch
import torch.nn as nn
import torch.autograd as autograd
from torch import Tensor
import warp as wp

from .tape import Tape


class SVDFunction(autograd.Function):

    @staticmethod
    def forward(ctx: autograd.function.FunctionCtx, F: Tensor) -> tuple[Tensor, Tensor, Tensor]:

        tape: Tape = Tape()
        shape = F.size(0)

        device = wp.device_from_torch(F.device)
        requires_grad = F.requires_grad

        A = wp.from_torch(F.contiguous(), dtype=wp.mat33)
        U = wp.zeros(shape, dtype=wp.mat33, device=device, requires_grad=requires_grad)
        sigma = wp.zeros(shape, dtype=wp.vec3, device=device, requires_grad=requires_grad)
        Vh = wp.zeros(shape, dtype=wp.mat33, device=device, requires_grad=requires_grad)

        with tape:
            wp.launch(SVDFunction.batch_svd, dim=shape, inputs=[A, U, sigma, Vh], device=device)

        U_torch = wp.to_torch(U).requires_grad_(requires_grad)
        sigma_torch = wp.to_torch(sigma).requires_grad_(requires_grad)
        Vh_torch = wp.to_torch(Vh).requires_grad_(requires_grad)

        ctx.tape = tape
        ctx.A = A
        ctx.U = U
        ctx.sigma = sigma
        ctx.Vh = Vh

        return U_torch, sigma_torch, Vh_torch

    @staticmethod
    def backward(ctx: autograd.function.FunctionCtx, grad_U: Tensor, grad_sigma: Tensor, grad_Vh: Tensor) -> tuple[Tensor]:

        tape = ctx.tape
        A = ctx.A
        U = ctx.U
        sigma = ctx.sigma
        Vh = ctx.Vh

        U.grad = wp.zeros_like(U) if grad_U is None else wp.from_torch(grad_U.contiguous(), dtype=wp.mat33)
        sigma.grad = wp.zeros_like(sigma) if grad_sigma is None else wp.from_torch(grad_sigma.contiguous(), dtype=wp.vec3)
        Vh.grad = wp.zeros_like(Vh) if grad_Vh is None else wp.from_torch(grad_Vh.contiguous(), dtype=wp.mat33)

        tape.backward()

        grad_A = wp.to_torch(A.grad)

        return grad_A

    @staticmethod
    @wp.kernel
    def batch_svd(
            A: wp.array(dtype=wp.mat33),
            U: wp.array(dtype=wp.mat33),
            sigma: wp.array(dtype=wp.vec3),
            Vh: wp.array(dtype=wp.mat33)) -> None:

        p = wp.tid()

        zero3 = wp.vec3(0.0)
        U_p = wp.mat33(zero3, zero3, zero3)
        sigma_p = wp.vec3(0.0)
        V_p = wp.mat33(zero3, zero3, zero3)

        wp.svd3(A[p], U_p, sigma_p, V_p)

        U_p_det = wp.determinant(U_p)
        V_p_det = wp.determinant(V_p)

        if U_p_det < 0.0:
            U_p = wp.mat33(
                U_p[0, 0], U_p[0, 1], -U_p[0, 2],
                U_p[1, 0], U_p[1, 1], -U_p[1, 2],
                U_p[2, 0], U_p[2, 1], -U_p[2, 2]
            )
            sigma_p = wp.vec3(sigma_p[0], sigma_p[1], -sigma_p[2])
        if V_p_det < 0.0:
            V_p = wp.mat33(
                V_p[0, 0], V_p[0, 1], -V_p[0, 2],
                V_p[1, 0], V_p[1, 1], -V_p[1, 2],
                V_p[2, 0], V_p[2, 1], -V_p[2, 2]
            )
            sigma_p = wp.vec3(sigma_p[0], sigma_p[1], -sigma_p[2])

        U[p] = U_p
        sigma[p] = sigma_p
        Vh[p] = wp.transpose(V_p)


class SVD(nn.Module):
    def forward(self, F: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return SVDFunction.apply(F)
