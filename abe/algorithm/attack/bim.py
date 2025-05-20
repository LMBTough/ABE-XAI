from abe.algorithm.attack.base import GradientBasedAttack
from abe.task import AttackTask
from typing import Any
import torch

class BIM(GradientBasedAttack):
    r"""
    BIM aka IFGSM in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Distance Measure : Linf

    Arguments:
        task (AttackTask): The attack task.
        eps (float): The perturbation bound.
        alpha (float): The step size.
        steps (int): The number of steps to run.

    .. note:: If steps set to 0, steps will be automatically decided following the paper.

    Examples::
        >>> loss_fn = def f(data):
        >>>     image,label = data
        >>>     loss = nn.CrossEntropyLoss()
        >>>     yhat = model(image)
        >>>     return loss(yhat,label)
        >>> task = AttackTask(loss_fn)
        >>> attack = BIM(task,eps=8/255,alpha=2/255,steps=10)
        >>> adv_images = attack(batch)
    """
    def __init__(self, task: AttackTask, eps: float = 8/255, alpha: float = 2/255, steps: int = 10) -> None:
        super().__init__(task)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps*255 + 4, 1.25*eps*255))
        else:
            self.steps = steps
        
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
        
        ori_inputs = grad_target.clone()
                
        for _ in range(self.steps):
            grad_target = torch.autograd.Variable(grad_target, requires_grad=True)
            loss = self.get_loss((grad_target, *extra))
            
            grad = torch.autograd.grad(loss, grad_target,
                                        retain_graph=False, create_graph=False)[0]
            
            grad_target = grad_target + self.alpha*grad.sign()
            delta = torch.clamp(grad_target - ori_inputs, min=-self.eps, max=self.eps)
            grad_target = self._clamp(ori_inputs + delta)
        
        return grad_target
