"""Defines the task abstractions."""

from __future__ import annotations
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Dict, List, Optional, Tuple, Union
    
from abe.type import ModelType

import torch
from torch import nn

class Task(ABC):
    """The abstraction of the task."""
    
    def __init__(self, **kwargs:dict) -> None:
        """Initialize the attack algorithm.

        Args:
            **kwargs: The arguments of the attack algorithm.
            
        Returns:
            None
        """
        pass
    
    


class AttackTask(Task):
    """The class for the attack task."""
    
    def __init__(
        self,
        loss_fn: Callable,
        model_type: ModelType = ModelType.IMAGECLASSIFICATION,
        is_targeted: bool = False,
    ) -> None:
        """Initialize the attack task.
        
        Args:
            loss_fn (Callable): The loss function of the model training.
                The function can be quite flexible in terms of what is calculated,
                but it should take the parameters and the data as input. Other than
                that, the forwarding of model should be in `torch.func` style.
                It will be used as target function to be attributed if no other
                target function provided
                A typical example is as follows:
                ```python
                def f(data):
                    image, label = data
                    loss = nn.CrossEntropyLoss()
                    yhat = model(image)
                    return loss(yhat, label)
                ```.
                This examples calculates the CE loss of the model on the data.
                
                
            model_type (ModelType): The type of the model. Default: ModelType.IMAGECLASSIFICATION.
            
            is_targeted (bool): Whether the attack is targeted. Default: False.

        """
        self.loss_fn = loss_fn
        self.model_type = model_type
        self.is_targeted = is_targeted



        
class ExplanationTask(Task):
    """The class for the explanation task."""
    
    def __init__(
        self,
        loss_fn: Callable,
        forward_fn: Callable,
        model_type: ModelType = ModelType.IMAGECLASSIFICATION,
    ) -> None:
        """Initialize the explanation task.
        
        Args:
            loss_fn (Callable): The loss function of the model training.
                The function can be quite flexible in terms of what is calculated,
                but it should take the parameters and the data as input. Other than
                that, the forwarding of model should be in `torch.func` style.
                It will be used as target function to be attributed if no other
                target function provided
                A typical example is as follows:
                ```python
                def f(data):
                    image, label = data
                    loss = nn.CrossEntropyLoss()
                    yhat = model(image)
                    return loss(yhat, label)
                ```.
                This examples calculates the CE loss of the model on the data.
            forward_fn (Callable): The forward function of the model.
                The function can be quite flexible in terms of what is calculated,
                but it should take the parameters and the data as input. Other than
                that, the forwarding of model should be in `torch.func` style.
                A typical example is as follows:
                ```python
                def f(data):
                    image, label = data
                    return model(image)
                ```.
                This examples calculates the output of the model on the data.
            model_type (ModelType): The type of the model. Default: ModelType.IMAGECLASSIFICATION.
                
        """
        self.loss_fn = loss_fn
        self.forward_fn = forward_fn
        self.model_type = model_type