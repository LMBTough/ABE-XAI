from abe.algorithm.explanation.base import Attributor
from abe.task import ExplanationTask
from typing import Any
import torch
import numpy as np


class SaliencyMap(Attributor):
    """
    Saliency Map
    Arguments:
        task (ExplanationTask): The explanation task.
        
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
        >>> saliency = SaliencyMap(task)
        >>> attributions = saliency(data)
    """
    def __init__(self, task: ExplanationTask) -> None:
        super().__init__(task)

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
        
        grad_target = torch.autograd.Variable(grad_target, requires_grad=True)
        
        loss = self.get_loss((grad_target, *extra))
        
        grad = torch.autograd.grad(loss, grad_target,
                                    retain_graph=False, create_graph=False)[0]
        
        attribution = grad.abs().detach().cpu().numpy()
        attribution = self._ensure_dim(grad_target, attribution)
        return attribution