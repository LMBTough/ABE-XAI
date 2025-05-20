from abe.algorithm.attack.base import GradientBasedAttack
from abe.task import AttackTask
from typing import Any
import torch


class FGSM(GradientBasedAttack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        task (AttackTask): The attack task.
        eps (float): The perturbation bound.

    Examples::
        >>> loss_fn = def f(data):
        >>>     image,label = data
        >>>     loss = nn.CrossEntropyLoss()
        >>>     yhat = model(image)
        >>>     return loss(yhat,label)
        >>> task = AttackTask(loss_fn)
        >>> attack = FGSM(task,eps=8/255)
        >>> adv_images = attack(batch)

    """
    def __init__(self, task: AttackTask, eps: float = 8/255, return_grad=False) -> None:
        super().__init__(task)
        self.eps = eps
        self.return_grad = return_grad
        
    def batch_attack(self, batch:Any) -> torch.Tensor:
        r"""Generate adversarial samples for a batch of data.
        
        Args:
            batch (Any): The batch of data.
            Image classification: Tuple[torch.Tensor, torch.Tensor]
            Object detection: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]
            NLP classification: Tuple[torch.Tensor, torch.Tensor]
            Time series Prediction: Tuple[torch.Tensor, torch.Tensor]
            
        Returns:
            torch.Tensor: The adversarial samples.
        """
            
            
        grad_target,*extra = batch
        
        grad_target = torch.autograd.Variable(grad_target, requires_grad=True)
        
        batch = (grad_target, *extra)
        
        # Calculate loss
        loss = self.get_loss(batch)
        
        # Update adversarial images
        grad = torch.autograd.grad(loss, grad_target,
                                   retain_graph=False, create_graph=False)[0]
        
        grad_target = grad_target + self.eps*grad.sign()
        grad_target = self._clamp(grad_target)
        
        if self.return_grad:
            return grad_target, grad
        return grad_target
