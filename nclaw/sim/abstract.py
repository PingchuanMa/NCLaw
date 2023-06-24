from typing import Any, Optional
from collections import OrderedDict

import warp as wp
import warp.sim


class State(object):
    def __init__(
            self,
            shape: Any,
            device: wp.context.Devicelike = None,
            requires_grad: bool = False) -> None:
        self.shape = shape
        self.device = wp.get_device(device)
        self.requires_grad = requires_grad

    def to_torch(self):
        raise NotImplementedError

    def to_torch_grad(self):
        raise NotImplementedError

    def from_torch(self):
        raise NotImplementedError

    def from_torch_grad(self):
        raise NotImplementedError


class Model(object):

    ConstantType = Any
    StaticsType = Any
    StateType = State

    def __init__(self, constant: ConstantType, device: wp.context.Devicelike = None, requires_grad: int = False) -> None:
        self.constant = constant
        self.device = wp.get_device(device)
        self.requires_grad = requires_grad

    def state(self, shape: Any, requires_grad: Optional[bool] = None) -> StateType:
        if requires_grad is None:
            requires_grad = self.requires_grad
        state = self.StateType(shape=shape, device=self.device, requires_grad=requires_grad)
        return state

    def statics(self, shape: Any) -> StaticsType:
        statics = self.StaticsType()
        statics.init(shape=shape, device=self.device)
        return statics


class ModelBuilder(object):

    ConstantType = Any
    StateType = State
    ModelType = Model

    def __init__(self) -> None:
        self.config = OrderedDict()

        for name in self.ConstantType.cls.__annotations__.keys():
            self.reserve(name)

    def reserve(self, name: str, init: Optional[Any] = None) -> None:
        if name in self.config:
            raise RuntimeError(f'duplicated key ({name}) reserved in ModelBuilder')
        self.config[name] = init

    @property
    def ready(self) -> bool:
        ready = True
        for k, v in self.config.items():
            if v is None:
                ready = False
                break
        return ready

    def build_constant(self) -> ConstantType:
        return self.ConstantType()

    def finalize(self, device: wp.context.Devicelike = None, requires_grad: bool = False) -> ModelType:
        if not self.ready:
            raise RuntimeError(f'config uninitialized: {self.config}')

        constant = self.build_constant()
        model = self.ModelType(constant, device, requires_grad)
        return model

class StateInitializer(object):

    StateType = State
    ModelType = Model

    def __init__(self, model: ModelType) -> None:
        self.model = model

    def finalize(self, shape: Any, requires_grad: bool = False) -> StateType:
        state = self.model.state(shape=shape, requires_grad=requires_grad)
        return state


class StaticsInitializer(object):

    StaticsType = Any
    ModelType = Model

    def __init__(self, model: ModelType) -> None:
        self.model = model

    def update(self, statics: StaticsType, step: int = 0) -> None:
        raise NotImplementedError

    def finalize(self, shape: Any) -> StaticsType:
        statics = self.model.statics(shape)
        return statics
