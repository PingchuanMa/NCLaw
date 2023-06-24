from typing import Optional

import torch.nn as nn


def get_nonlinearity(nonlinearity: Optional[str], **kwargs) -> Optional[nn.Module]:

    if nonlinearity is None:
        return nn.Identity()
    elif nonlinearity.casefold() == 'relu':
        return nn.ReLU(inplace=True, **kwargs)
    elif nonlinearity.casefold() == 'tanh':
        return nn.Tanh(**kwargs)
    elif nonlinearity.casefold() in ['silu', 'swish']:
        return nn.SiLU(**kwargs)
    elif nonlinearity.casefold() == 'gelu':
        return nn.GELU(**kwargs)
    elif nonlinearity.casefold() == 'elu':
        return nn.ELU(**kwargs)
    else:
        raise ValueError('unexpected nonlinearity: {}'.format(nonlinearity))


def get_norm(
        norm: Optional[str],
        planes: int,
        dim: int,
        affine: bool = True,
        **kwargs) -> Optional[nn.Module]:

    if dim != 1:
        raise ValueError('Unexpected dim for norm: {:d}'.format(dim))

    if norm is None:
        return nn.Identity()
    elif norm.casefold() == 'wn':
        return nn.Identity()
    elif norm.casefold() == 'ln':
        return nn.LayerNorm(planes, elementwise_affine=affine, **kwargs)
    elif norm.casefold() == 'bn':
        return nn.BatchNorm1d(planes, affine=affine, **kwargs)
    elif norm.casefold() == 'in':
        return nn.InstanceNorm1d(planes, affine=affine, **kwargs)
    else:
        raise ValueError('unexpected norm: {}'.format(norm))

def init_weight(m: nn.Module) -> None:
    if hasattr(m, 'init_weight'):
        m.init_weight()
        return
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
