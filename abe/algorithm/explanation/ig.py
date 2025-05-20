from demo.algorithm.explanation.base import Attributor
from demo.task import ExplanationTask
from typing import Any
import torch
import numpy as np
from demo.type import ModelType


class IG(Attributor):
    r"""
    Integrated Gradient
    Arguments:
        task (ExplanationTask): The explanation task.
        gradient_steps (int): The number of steps to run.
            
    Examples::
        >>> def loss_fn(data):
        >>>     image, label = data
        >>>     loss = nn.CrossEntropyLoss()
        >>>     yhat = model(image)
        >>>     return loss(yhat, label)
        >>> def forward_fn(data):
        >>>     image, label = data
        >>>     return model(image)
        >>> task = ExplanationTask(loss_fn, forward_fn)
        >>> ig = IG(task, gradient_steps=50)
        >>> attributions = ig(data)
    """
    def __init__(self, task: ExplanationTask, gradient_steps: int = 50) -> None:
        super().__init__(task)
        self.gradient_steps = gradient_steps
        
    def batch_attribute(self, batch: Any) -> np.ndarray:
        """Generate the attributions for the batch of data.
        
        Args:
            batch (Any): The batch of data.
            Image classification: Tuple[torch.Tensor, torch.Tensor]
            Object detection: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]
            NLP classification: Tuple[torch.Tensor, torch.Tensor]
            Time series Prediction: Tuple[torch.Tensor, torch.Tensor]
            
        Returns:
            np.ndarray: The attributions.
        """
        grad_target,*extra = batch
        
        baseline = torch.zeros_like(grad_target)
        
        mean_grad = 0
        
        for i in range(self.gradient_steps + 1):
            scaled_inputs = baseline + (float(i) / self.gradient_steps) * (grad_target - baseline)
            scaled_inputs = self._clamp(scaled_inputs)
            scaled_inputs = torch.autograd.Variable(scaled_inputs, requires_grad=True)
            
            loss = self.get_loss((scaled_inputs, *extra))
            
            grad = torch.autograd.grad(loss, scaled_inputs,create_graph=False,retain_graph=False)[0]
            
            mean_grad += grad / self.gradient_steps
            
        attribution = (grad_target - baseline) * mean_grad
            
        attribution = self._ensure_dim(grad_target, attribution.detach().cpu().numpy())
        return attribution
