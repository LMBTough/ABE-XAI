"""This module implement the attributor base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING,Any

import torch.utils

if TYPE_CHECKING:
    from typing import List, Optional, Tuple, Union

    import torch

from demo.task import ExplanationTask
from demo.type import ModelType

import os

import torch
from tqdm import tqdm
import numpy as np

from demo.func.utils import _check_device,_check_shuffle

class BaseAttributor(ABC):
    """The base class for all explanation algorithms."""

    @abstractmethod
    def __init__(self, **kwargs:dict) -> None:
        """Initialize the explanation algorithm.

        Args:
            **kwargs: The arguments of the explanation algorithm.
            
        Returns:
            None
        """
        pass
        
    
    @abstractmethod
    def batch_attribute(self, batch: Any) -> np.ndarray:
        """Generate the attributions for the batch of data.
        
        Args:
            batch (Any): The batch of data.
            Image classification: Tuple[torch.Tensor, torch.Tensor]
            Object detection: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]
            NLP classification: Tuple[torch.Tensor, torch.Tensor]
            Time series Prediction: Tuple[torch.Tensor, torch.Tensor]
            
        Returns:
            torch.Tensor: The adversarial samples.
        """
        pass
    
    def __call__(self,batch: Any) -> np.ndarray:
        """Generate the attributions for a batch of data.
        
        Args:
            batch (Any): The batch of data.
            Image classification: Tuple[torch.Tensor, torch.Tensor]
            Object detection: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]
            NLP classification: Tuple[torch.Tensor, torch.Tensor]
            Time series Prediction: Tuple[torch.Tensor, torch.Tensor]
            
        Returns:
            np.ndarray: The attributions.
        """
        return self.batch_attribute(batch)
    


class Attributor(BaseAttributor):
    """The base class for all explanation algorithms."""
    
    def __init__(self,
                 task: ExplanationTask,
                 ) -> None:
        """Initialize the explanation algorithm.

        Args:
            task (ExplanationTask): The explanation task.
            
        Returns:
            None
        """
        self.loss_fn = task.loss_fn
        self.forward_fn = task.forward_fn
        self.model_type = task.model_type
        self.device = _check_device()
        
    def _clamp(self, grad_target: torch.Tensor) -> torch.Tensor:
        """Clamp the adversarial samples to the valid range.
        
        Args:
            grad_target (torch.Tensor): The adversarial samples.
            
        Returns:
            torch.Tensor: The clamped adversarial samples.
        """
        match self.model_type:
            case ModelType.IMAGECLASSIFICATION | ModelType.OBJECTDETECTION:
                return torch.clamp(grad_target, min=0, max=1).detach()
            case _:
                return grad_target.detach()

    def _check_inputs(self, images: torch.Tensor) -> None:
        """Check if the input images are in the range [0, 1].
        """
        if torch.max(images) > 1 or torch.min(images) < 0:
            raise ValueError('Input must have a range [0, 1] (max: {}, min: {})'.format(
                torch.max(images), torch.min(images)))
            
    def _ensure_dim(self, inputs: torch.Tensor, attribution: np.ndarray) -> np.ndarray:
        """Ensure the dimension of the attributions is the same as the inputs.
        
        Args:
            inputs (torch.Tensor): The input data.
            attribution (np.ndarray): The attributions.
            
        Returns:
            np.ndarray: The attributions with the same dimension as the inputs.
        """
        inputs_dim = len(inputs.shape)
        attribution_dim = len(attribution.shape)
        if attribution_dim < inputs_dim:
            attribution = np.expand_dims(attribution, axis=0)
        elif attribution_dim > inputs_dim:
            attribution = np.squeeze(attribution, axis=0)
        return attribution
        
    def get_loss(self, batch:Any,check_input:bool = True) -> torch.Tensor:
        """Calculate the loss of the model.
        
        Args:
            batch (Any): The batch of data.
            Image classification: Tuple[torch.Tensor, torch.Tensor]
            Object detection: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]
            NLP classification: Tuple[torch.Tensor, torch.Tensor]
            Time series Prediction: Tuple[torch.Tensor, torch.Tensor]
            
            check_input (bool): Whether to check the input images. (Default: True) For SSA to close this check.
            
        Returns:
            torch.Tensor: The loss of the model.
        """
        match self.model_type:
            case ModelType.IMAGECLASSIFICATION | ModelType.OBJECTDETECTION:
                if check_input:
                    self._check_inputs(batch[0])
            case _:
                pass
            
        loss = -self.loss_fn(batch)
        return loss
    
    def get_logits(self,batch: Any) -> torch.Tensor:
        """Get the logits of the model.
        
        Args:
            batch (Any): The batch of data.
            Image classification: Tuple[torch.Tensor, torch.Tensor]
            Object detection: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]
            NLP classification: Tuple[torch.Tensor, torch.Tensor]
            Time series Prediction: Tuple[torch.Tensor, torch.Tensor]
            
        Returns:
            torch.Tensor: The logits of the model.
        """
        return self.forward_fn(batch)