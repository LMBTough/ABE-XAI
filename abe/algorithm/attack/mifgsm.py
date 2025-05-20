from abe.algorithm.attack.base import GradientBasedAttack
from abe.task import AttackTask
import torch
from typing import Any

class MIFGSM(GradientBasedAttack):
    r"""
    MI-FGSM in the paper 'Boosting Adversarial Attacks with Momentum'
    [https://arxiv.org/abs/1710.06081]

    Distance Measure : Linf

    Arguments:
        task (AttackTask): The attack task.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 10)

    Examples::
        >>> loss_fn = def f(data):
        >>>     image,label = data
        >>>     loss = nn.CrossEntropyLoss()
        >>>     yhat = model(image)
        >>>     return loss(yhat,label)
        >>> task = AttackTask(loss_fn)
        >>> attack = MIFGSM(task,eps=8/255,alpha=2/255,steps=10,decay=1.0)
        >>> adv_images = attack(batch)

    """

    def __init__(self, task: AttackTask, eps: float = 8/255, alpha: float = 2/255, steps: int = 10, decay: float = 1.0) -> None:
        super().__init__(task)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        
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
        
        momentum = torch.zeros_like(grad_target)
        
        for _ in range(self.steps):
            grad_target = torch.autograd.Variable(grad_target, requires_grad=True)
            loss = self.get_loss((grad_target, *extra))
            grad = torch.autograd.grad(loss, grad_target,
                                       retain_graph=False, create_graph=False)[0]
            
            grad = grad / torch.mean(torch.abs(grad), dim=(*range(1, len(grad.shape)),), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad
            
            grad_target = grad_target.detach() + self.alpha*grad.sign()
            delta = torch.clamp(grad_target - ori_inputs, min=-self.eps, max=self.eps)
            grad_target = self._clamp(ori_inputs + delta)
            
        return grad_target
