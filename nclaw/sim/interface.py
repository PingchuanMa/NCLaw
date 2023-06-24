from typing import Optional

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import Tensor
import warp as wp

from .mpm import MPMModel, MPMState, MPMStatics
from nclaw.warp import Tape


class MPMSimFunction(autograd.Function):

    @staticmethod
    def forward(
            ctx: autograd.function.FunctionCtx,
            model: MPMModel,
            statics: MPMStatics,
            state_curr: MPMState,
            state_next: MPMState,
            x: Tensor,
            v: Tensor,
            C: Tensor,
            F: Tensor,
            stress: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        tape: Tape = Tape()
        state_curr.from_torch(x=x, v=v, C=C, F=F, stress=stress)
        model.forward(statics, state_curr, state_next, tape)

        x_next, v_next, C_next, F_next, _ = state_next.to_torch()

        ctx.model = model
        ctx.tape = tape
        ctx.statics = statics
        ctx.state_curr = state_curr
        ctx.state_next = state_next

        return x_next, v_next, C_next, F_next

    @staticmethod
    def backward(
            ctx: autograd.function.FunctionCtx,
            grad_x_next: Tensor,
            grad_v_next: Tensor,
            grad_C_next: Tensor,
            grad_F_next: Tensor) -> tuple[None, None, None, None, Tensor, Tensor, Tensor, Tensor, Tensor]:

        model: MPMModel = ctx.model
        tape: Tape = ctx.tape
        statics: MPMStatics = ctx.statics
        state_curr: MPMState = ctx.state_curr
        state_next: MPMState = ctx.state_next

        state_next.from_torch_grad(
            grad_x=grad_x_next,
            grad_v=grad_v_next,
            grad_C=grad_C_next,
            grad_F=grad_F_next)

        model.backward(statics, state_curr, state_next, tape)

        grad_x, grad_v, grad_C, grad_F, grad_stress = state_curr.to_torch_grad()

        if grad_x is not None:
            torch.nan_to_num_(grad_x, 0.0, 0.0, 0.0)
        if grad_v is not None:
            torch.nan_to_num_(grad_v, 0.0, 0.0, 0.0)
        if grad_C is not None:
            torch.nan_to_num_(grad_C, 0.0, 0.0, 0.0)
        if grad_F is not None:
            torch.nan_to_num_(grad_F, 0.0, 0.0, 0.0)
        if grad_stress is not None:
            torch.nan_to_num_(grad_stress, 0.0, 0.0, 0.0)

        return None, None, None, None, grad_x, grad_v, grad_C, grad_F, grad_stress


class MPMSim(nn.Module):

    def __init__(self, model: MPMModel) -> None:
        super().__init__()
        self.model = model

    def state(self, x: Tensor, v: Tensor, C: Tensor, F: Tensor, stress: Tensor, state: Optional[MPMState] = None) -> MPMState:
        model = self.model
        shape = x.size(0)

        if state is None:
            state: MPMState = model.state(shape)
        state.from_torch(x=x, v=v, C=C, F=F, stress=stress)

        return state


class MPMDiffSim(MPMSim):

    def __init__(self, model: MPMModel) -> None:
        super().__init__(model)

    def forward(self, statics: MPMStatics, x: Tensor, v: Tensor, C: Tensor, F: Tensor, stress: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        shape = x.size(0)
        state_curr: MPMState = self.model.state(shape)
        state_next: MPMState = self.model.state(shape)
        return MPMSimFunction.apply(self.model, statics, state_curr, state_next, x, v, C, F, stress)


class MPMCacheDiffSim(MPMSim):

    def __init__(self, model: MPMModel, num_steps: int) -> None:
        super().__init__(model)
        self.curr_states = [None for _ in range(num_steps)]
        self.next_states = [None for _ in range(num_steps)]

    def forward(self, statics: MPMStatics, step: int, x: Tensor, v: Tensor, C: Tensor, F: Tensor, stress: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        shape = x.size(0)
        if self.curr_states[step] is None:
            self.curr_states[step] = self.model.state(shape)
        if self.next_states[step] is None:
            self.next_states[step] = self.model.state(shape)
        state_curr = self.curr_states[step]
        state_next = self.next_states[step]
        return MPMSimFunction.apply(self.model, statics, state_curr, state_next, x, v, C, F, stress)


class MPMForwardSim(MPMSim):

    def __init__(self, model: MPMModel) -> None:
        super().__init__(model)

    def forward(self, statics: MPMStatics, state: MPMState) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        model = self.model
        model.forward(statics, state, state, None)
        x_next, v_next, C_next, F_next, _ = state.to_torch()
        return x_next, v_next, C_next, F_next


class MPMExtraSim(MPMSim):

    def __init__(self, model: MPMModel) -> None:
        super().__init__(model)

    def forward(self, statics: MPMStatics, state: MPMState, statics_extra: MPMStatics, state_extra: MPMState) -> Tensor:
        model = self.model
        model.forward_extra(statics, state, statics_extra, state_extra)
        x_extra, _, _, _, _ = state_extra.to_torch()
        return x_extra
