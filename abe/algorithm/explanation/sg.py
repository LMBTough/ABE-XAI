from demo.algorithm.explanation.base import Attributor
from demo.task import ExplanationTask
from typing import Any
import torch
import numpy as np


class SmoothGradient(Attributor):
    r"""
    SG
    Arguments:
        task (ExplanationTask): The explanation task.
        stdevs (float): The standard deviation of the noise.
        nt_samples (int): The number of noise samples. (Default: 50)
        
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
        >>> sg = SmoothGradient(task, stdevs=0.15, nt_samples=50)
        >>> attributions = sg(data)

    """
    def __init__(self, task: ExplanationTask, stdevs: float = 0.15, nt_samples: int = 50) -> None:
        super().__init__(task)
        self.stdevs = stdevs
        self.nt_samples = nt_samples
        
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
        
        gradient = 0
        
        for i in range(self.nt_samples):
            grad_target = torch.autograd.Variable(grad_target, requires_grad=True)
            noise = self.stdevs * torch.randn_like(grad_target)
            noise_input = grad_target + noise
            
            loss = self.get_loss((noise_input, *extra),check_input=False)
            
            grad = torch.autograd.grad(loss, grad_target,create_graph=False,retain_graph=False)[0]
            
            gradient += grad
            
        attribution = gradient / self.nt_samples
        
        attribution = self._ensure_dim(grad_target, attribution.detach().cpu().numpy())
        return attribution
