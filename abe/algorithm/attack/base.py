"""This module implement the attributor base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch.utils

if TYPE_CHECKING:
    from typing import List, Optional, Tuple, Union, Any, Callable

    import torch

    from abe.task import AttackTask

import os

import torch
from tqdm import tqdm

from abe.func.utils import _check_device
from abe.type import ModelType

class BaseAttack(ABC):
    """The base class for all attack algorithms."""

    @abstractmethod
    def __init__(self, **kwargs:dict) -> None:
        """Initialize the attack algorithm.

        Args:
            **kwargs: The arguments of the attack algorithm.
            
        Returns:
            None
        """
        pass
        
    
    @abstractmethod
    def batch_attack(self, batch: Any) -> torch.Tensor:
        """Generate the adversarial examples for a batch of data.
        
        Args:
            batch (Any): The batch of data.
            Image classification: Tuple[torch.Tensor, torch.Tensor]
            Object detection: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]
            NLP classification: Tuple[torch.Tensor, torch.Tensor]
            Time series Prediction: Tuple[torch.Tensor, torch.Tensor]
            
            
        Returns:
            torch.Tensor: The adversarial examples.
        """
        pass
    
    def __call__(self,batch: Any) -> torch.Tensor:
        """Generate the adversarial examples for a batch of data.
        
        Args:
            batch (Any): The batch of data.
            Image classification: Tuple[torch.Tensor, torch.Tensor]
            Object detection: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]
            NLP classification: Tuple[torch.Tensor, torch.Tensor]
            Time series Prediction: Tuple[torch.Tensor, torch.Tensor]
            
        Returns:
            torch.Tensor: The adversarial examples.
        """
        return self.batch_attack(batch)
    


class GradientBasedAttack(BaseAttack):
    """The base class for all gradient-based attack algorithms."""

    def __init__(self,
                 task: AttackTask,
                 ) -> None:
        """Initialize the gradient-based attack algorithm.

        Args:
            task (AttackTask): The attack task.
            
        Returns:
            None
        """
        self.loss_fn = task.loss_fn
        self.model_type = task.model_type
        self.is_targeted = task.is_targeted
        self.device = _check_device()
        
        
    def _check_inputs(self, images: torch.Tensor) -> None:
        """Check if the input images are in the range [0, 1].
        """
        if torch.max(images) > 1 or torch.min(images) < 0:
            raise ValueError('Input must have a range [0, 1] (max: {}, min: {})'.format(
                torch.max(images), torch.min(images)))
            
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
            
            
        loss = -self.loss_fn(batch) if self.is_targeted else self.loss_fn(batch)
        return loss
    
    
    

class GANBasedAttack(BaseAttack):
    """The base class for all GAN-based attack algorithms."""

    def __init__(self,
                 task: AttackTask,
                 forward_fn: Callable,
                 save_path: str,
                 ) -> None:
        """Initialize the GAN-based attack algorithm.

        Args:
            task (AttackTask): The attack task.
            save_path (str): The path to save the trained GAN model.
            
        Returns:
            None
        """
        self.forward_fn = forward_fn
        self.model_type = task.model_type
        self.save_path = save_path
        self.is_targeted = task.is_targeted
        os.makedirs(save_path, exist_ok=True)
        self.device = _check_device()
        
        
    def _check_inputs(self, images: torch.Tensor) -> None:
        """Check if the input images are in the range [0, 1].
        """
        if torch.max(images) > 1 or torch.min(images) < 0:
            raise ValueError('Input must have a range [0, 1] (max: {}, min: {})'.format(
                torch.max(images), torch.min(images)))
    
    @abstractmethod
    def train_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Train the GAN model for a batch of data.
        
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The batch of data.
        """
        pass
    
    @abstractmethod
    def train(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Train the GAN model.
        
        Args:
            dataloader (DataLoader): The dataloader of the dataset.
        """
        pass
    
    @abstractmethod
    def save(self) -> None:
        """Save the GAN model.
        """
        pass
    
    @abstractmethod
    def load(self) -> None:
        """Load the GAN model.
        """
        pass
        