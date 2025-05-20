from abe.algorithm.attack.base import GradientBasedAttack
from abe.task import AttackTask
import torch
from typing import Any

class PGD(GradientBasedAttack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        task (AttackTask): task to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Examples::
        >>> loss_fn = def f(data):
        >>>     image,label = data
        >>>     loss = nn.CrossEntropyLoss()
        >>>     yhat = model(image)
        >>>     return loss(yhat,label)
        >>> task = AttackTask(loss_fn)
        >>> attack = PGD(task,eps=8/255,alpha=2/255,steps=10,random_start=True)
        >>> adv_images = attack(batch)

    """
    
    def __init__(self, task: AttackTask, eps: float = 8/255, alpha: float = 2/255, steps: int = 10, random_start: bool = True) -> None:
        super().__init__(task)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        
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
        
        if self.random_start:
            # Starting at a uniformly random point
            grad_target = grad_target + torch.empty_like(grad_target).uniform_(-self.eps, self.eps)
            grad_target = self._clamp(grad_target)
            
        
        for _ in range(self.steps):
            grad_target = torch.autograd.Variable(grad_target, requires_grad=True)
            loss = self.get_loss((grad_target, *extra))
            
            grad = torch.autograd.grad(loss, grad_target,
                                        retain_graph=False, create_graph=False)[0]
            
            grad_target = grad_target + self.alpha*grad.sign()
            delta = torch.clamp(grad_target - ori_inputs, min=-self.eps, max=self.eps)
            grad_target = self._clamp(ori_inputs + delta)
            
        return grad_target
