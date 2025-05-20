from demo.algorithm.explanation.base import Attributor
from demo.task import ExplanationTask,AttackTask
from demo.algorithm.attack import FGSM
from typing import Any
import torch
import numpy as np


class MFABA(Attributor):
    r"""
    MFABA
    Arguments:
        task (ExplanationTask): The explanation task.
        eps (float): The epsilon value. (Default: 80/255)
        alpha (float): The alpha value. (Default: 2/255)
        steps (int): The number of steps. (Default: 40)
        
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
        >>> mfaba = MFABA(task, eps=80/255, alpha=2/255, steps=40)
        >>> attributions = mfaba(data)

    """
    def __init__(self, task: ExplanationTask, eps: float = 80/255, alpha: float = 2/255, steps: int = 40) -> None:
        super().__init__(task)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        attack_task = AttackTask(task.loss_fn, task.model_type)
        self.fgsm = FGSM(attack_task, eps=alpha, return_grad=True)
        
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
        
        last_grad_target = grad_target.clone()
        
        ori_grad_target = grad_target.clone()
        
        attribution = 0
        
        for i in range(self.steps):
            grad_target,grad = self.fgsm([grad_target, *extra])
            delta = torch.clamp(grad_target - ori_grad_target, min=-self.eps, max=self.eps)
            grad_target = self._clamp(ori_grad_target + delta)
            
            attribution += (grad_target - last_grad_target) * grad
            
            last_grad_target = grad_target
        
        attribution = self._ensure_dim(ori_grad_target, attribution.detach().cpu().numpy())
        return attribution
            