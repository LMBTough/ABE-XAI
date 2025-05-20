"""Utility functions for working with models and parameters."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Dict, Optional, Tuple

import functools

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler
from demo.type import ModelType
import torch.nn.functional as F
import warnings

def _check_device() -> torch.device:
    """Check and return the device of the machine (CUDA, MPS, or CPU).

    Returns:
        torch.device: The device of the machine.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
    
def _check_shuffle(dataloader: torch.utils.data.DataLoader) -> None:
    """Check if the dataloader is shuffling the data.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader to be checked.
    """
    is_shuffling = isinstance(dataloader.sampler, RandomSampler)
    if is_shuffling:
        warnings.warn(
            "The dataloader is shuffling the data. The influence \
            calculation could not be interpreted in order.",
            stacklevel=1,
        )



def DI(inputs: torch.Tensor, resize_rate: float = 0.9, diversity_prob: float = 0.5, model_type: ModelType = ModelType.IMAGECLASSIFICATION) -> torch.Tensor:
    """Apply input diversity to the input data.

    Args:
        inputs (torch.Tensor): The input data.
        resize_rate (float): The resize rate.
        diversity_prob (float): The probability of applying input diversity.
        model_type (ModelType): The type of the model.

    Returns:
        torch.Tensor: The input data after applying input diversity.
    """
    match model_type:
        case ModelType.IMAGECLASSIFICATION | ModelType.OBJECTDETECTION:
            ori_dim = inputs.dim()
            if ori_dim == 3:
                inputs = inputs.unsqueeze(0)
            h, w = inputs.shape[-2], inputs.shape[-1]
            new_h = int(h * resize_rate)
            new_w = int(w * resize_rate)

            if resize_rate < 1:
                temp_h = h
                h = new_h
                new_h = temp_h
                temp_w = w
                w = new_w
                new_w = temp_w


            rnd_h = torch.randint(low=h, high=new_h, size=(1,), dtype=torch.int32)
            rnd_w = torch.randint(low=w, high=new_w, size=(1,), dtype=torch.int32)

            rescaled = F.interpolate(inputs, size=[rnd_h.item(), rnd_w.item()], mode='bilinear', align_corners=False)

            h_rem = new_h - rnd_h
            w_rem = new_w - rnd_w
            pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
            pad_right = w_rem - pad_left

            padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
            
            if ori_dim == 3:
                return padded.squeeze(0) if torch.rand(1) < diversity_prob else inputs.squeeze(0)
            
            return padded if torch.rand(1) < diversity_prob else inputs
        case _:
            warnings.warn("The input diversity is not implemented for the model type yet.", stacklevel=1)
            return inputs


# custom weights initialization called on netG and netD
def weights_init(m: nn.Module) -> None:
    """Initialize the weights of the model.
    
    Args:
        m (nn.Module): The model to be initialized.
        
    Returns:
        None
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
