from abe.algorithm.attack.base import GradientBasedAttack
from abe.func.utils import DI
from abe.task import AttackTask
from typing import Any
import torch

class DIFGSM(GradientBasedAttack):
    r"""
    DI2-FGSM in the paper 'Improving Transferability of Adversarial Examples with Input Diversity'
    [https://arxiv.org/abs/1803.06978]

    Distance Measure : Linf

    Arguments:
        task (AttackTask): The attack task.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 0.0)
        steps (int): number of iterations. (Default: 10)
        resize_rate (float): resize factor used in input diversity. (Default: 0.9)
        diversity_prob (float) : the probability of applying input diversity. (Default: 0.5)
        random_start (bool): using random initialization of delta. (Default: False)

    Examples::
        >>> loss_fn = def f(data):
        >>>     image,label = data
        >>>     loss = nn.CrossEntropyLoss()
        >>>     yhat = model(image)
        >>>     return loss(yhat,label)
        >>> task = AttackTask(loss_fn)
        >>> attack = DIFGSM(task,eps=8/255,alpha=2/255,steps=10,decay=0.0,resize_rate=0.9,diversity_prob=0.5,random_start=False)
        >>> adv_images = attack(batch)
    """

    def __init__(self, task: AttackTask, eps: float = 8/255, alpha: float = 2/255, steps: int = 10, decay: float = 0.0,
                 resize_rate: float = 0.9, diversity_prob: float = 0.5, random_start: bool = False) -> None:
        super().__init__(task)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
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
        
        momentum = torch.zeros_like(grad_target).to(self.device)
        
        
        if self.random_start:
            # Starting at a uniformly random point
            grad_target = grad_target + torch.empty_like(grad_target).uniform_(-self.eps, self.eps)
            grad_target = self._clamp(grad_target)
            
        
        for _ in range(self.steps):
            grad_target = torch.autograd.Variable(grad_target, requires_grad=True)
            loss = self.get_loss((DI(grad_target, self.resize_rate, self.diversity_prob,model_type=self.model_type), *extra))
            
            grad = torch.autograd.grad(loss, grad_target,
                                        retain_graph=False, create_graph=False)[0]
            
            grad = grad / torch.mean(torch.abs(grad), dim=(*range(1, len(grad.shape)),), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad
            
            grad_target = grad_target + self.alpha*grad.sign()
            delta = torch.clamp(grad_target - ori_inputs, min=-self.eps, max=self.eps)
            grad_target = self._clamp(ori_inputs + delta)
            
        return grad_target
