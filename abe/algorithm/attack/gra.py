from abe.algorithm.attack.base import GradientBasedAttack
from abe.func.dct import *
from abe.func.utils import DI
from typing import Any
from abe.task import AttackTask
import numpy as np
import scipy.stats as st
import torch.nn.functional as F
import torch
from abe.type import ModelType


class GRA(GradientBasedAttack):
    r"""
        GRA Attack
        Arguments:
            task (AttackTask): task to attack.
            steps (int): Number of iterations. (Default: 10)
            eps (float): Maximum perturbation that the attacker can introduce. (Default: 8/255)
            alpha (float): Step size of each iteration. (Default: 2/255)
            momentum (float): Momentum factor. (Default: 1.0)
            N (int): Number of random restarts. (Default: 20)
            rho (float): Rho. (Default: 0.5)
            sigma (float): Sigma. (Default: 8/255)
            resize_rate (float): Resize rate. (Default: 1.15)
            diversity_prob (float): Diversity probability. (Default: 0.5)
            len_kernel (int): Length of the kernel. (Default: 15)
            nsig (int): Radius of the Gaussian kernel. (Default: 3)

        Examples::
            >>> loss_fn = def f(data):
            >>>     image,label = data
            >>>     loss = nn.CrossEntropyLoss()
            >>>     yhat = model(image)
            >>>     return loss(yhat,label)
            >>> task = AttackTask(loss_fn)
            >>> attack = GRA(task,steps=10,eps=8/255,alpha=2/255,momentum=1.0,N=20,rho=0.5,sigma=8/255,resize_rate=1.15,diversity_prob=0.5,len_kernel=7,nsig=3)
            >>> adv_images = attack(batch)
    """

    def __init__(self, task: AttackTask, steps: int = 10, eps: float = 8/255, alpha: float = 2/255, momentum: float = 1.0, N: int = 20, beta : float = 3.5, sigma: float = 8/255, return_grad: bool = False) -> None:
        super().__init__(task)
        self.steps = steps
        self.eps = eps
        self.momentum = momentum
        self.N = N
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
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
        
        m = torch.ones_like(grad_target) * 10 / 9.4
        
        ori_inputs = grad_target.clone()
        
        grad = 0
        
        for _ in range(self.steps):
            grad_target = torch.autograd.Variable(grad_target, requires_grad=True)
            loss = self.get_loss((grad_target, *extra))
            current_grad = torch.autograd.grad(loss, grad_target,
                                        retain_graph=False, create_graph=False)[0]
            avg_grad = 0
            for _ in range(self.N):
                uniform_noise = torch.rand_like(grad_target) * 2 * (self.eps * self.beta) - self.eps * self.beta
                grad_target_neighbor = grad_target + uniform_noise
                grad_target_neighbor = torch.autograd.Variable(grad_target_neighbor, requires_grad=True)
                loss = self.get_loss((grad_target_neighbor, *extra))
                avg_grad += torch.autograd.grad(loss, grad_target_neighbor,
                                        retain_graph=False, create_graph=False)[0]
            avg_grad = avg_grad / self.N
            
            cossim = (current_grad * avg_grad).sum([i for i in range(1, len(current_grad.shape))], keepdim=True) / (
                        torch.sqrt((current_grad ** 2).sum([i for i in range(1, len(current_grad.shape))], keepdim=True)) * torch.sqrt(
                    (avg_grad ** 2).sum([i for i in range(1, len(current_grad.shape))], keepdim=True)))
            
            current_grad = cossim * current_grad + (1 - cossim) * avg_grad
            noise = self.momentum * grad + current_grad / torch.abs(current_grad).mean([i for i in range(1, len(current_grad.shape))], keepdim=True)
            eqm = (torch.sign(grad) == torch.sign(noise)).float()
            grad = noise
            dim = torch.ones_like(grad_target) - eqm
            m = m * (eqm + dim * 0.94)
            grad_target = grad_target.detach() + self.alpha*grad.sign()
            delta = torch.clamp(grad_target - ori_inputs, min=-self.eps, max=self.eps)
            grad_target = self._clamp(ori_inputs + delta)
            
        
        if self.return_grad:
            return grad_target, grad
        return grad_target

