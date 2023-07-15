try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message

from itertools import repeat
from enum import Enum
from PIL import Image

import torch
import collections.abc

if hasattr(Image, "Resampling"):
    _pil_interpolation_to_str = {
        Image.Resampling.NEAREST: 'nearest',
        Image.Resampling.BILINEAR: 'bilinear',
        Image.Resampling.BICUBIC: 'bicubic',
        Image.Resampling.BOX: 'box',
        Image.Resampling.HAMMING: 'hamming',
        Image.Resampling.LANCZOS: 'lanczos',
    }
else:
    _pil_interpolation_to_str = {
        Image.NEAREST: 'nearest',
        Image.BILINEAR: 'bilinear',
        Image.BICUBIC: 'bicubic',
        Image.BOX: 'box',
        Image.HAMMING: 'hamming',
        Image.LANCZOS: 'lanczos',
    }

_str_to_pil_interpolation = {b: a for a, b in _pil_interpolation_to_str.items()}

def str_to_interp_mode(mode_str):
    return _str_to_pil_interpolation[mode_str]


class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'


def nchw_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple