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


class SSA(GradientBasedAttack):
    r"""
        SSA Attack
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
            >>> attack = SSA(task,steps=10,eps=8/255,alpha=2/255,momentum=1.0,N=20,rho=0.5,sigma=8/255,resize_rate=1.15,diversity_prob=0.5,len_kernel=7,nsig=3)
            >>> adv_images = attack(batch)
    """

    def __init__(self, task: AttackTask, steps: int = 10, eps: float = 8/255, alpha: float = 2/255, momentum: float = 1.0, N: int = 20, rho: float = 0.5, sigma: float = 8/255, resize_rate: float = 1.15, diversity_prob: float = 0.5, len_kernel: int = 7, nsig: int = 3, return_grad: bool = False) -> None:
        super().__init__(task)
        self.steps = steps
        self.eps = eps
        self.momentum = momentum
        self.N = N
        self.rho = rho
        self.sigma = sigma
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.return_grad = return_grad
        self.T_kernel = self.gkern(len_kernel, nsig)
        
    def gkern(self, kernlen: int = 15, nsig: int = 3) -> np.ndarray:
        """Returns a 2D Gaussian kernel array.
        
        Args:
            kernlen (int): kernel length.
            nsig (int): radius of gaussian kernel.
        """
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel.astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        gaussian_kernel = torch.from_numpy(gaussian_kernel).to(self.device)
        return gaussian_kernel
    
    
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
        
        grad = 0
        
        for _ in range(self.steps):
            grad_target = torch.autograd.Variable(grad_target, requires_grad=True)
            noise = 0
            for _ in range(self.N):
                gauss = torch.randn_like(grad_target) * self.sigma
                x_dct = dct_2d(grad_target + gauss)
                mask = torch.rand_like(grad_target) * 2 * self.rho + 1 - self.rho
                x_idct = idct_2d(x_dct * mask)
                x_idct = torch.autograd.Variable(x_idct, requires_grad=True)
                # DI-FGSM https://arxiv.org/abs/1803.06978
                loss = self.get_loss((DI(x_idct, resize_rate=self.resize_rate, diversity_prob=self.diversity_prob, model_type=self.model_type), *extra),check_input=False)
                noise += torch.autograd.grad(loss, x_idct,
                                             retain_graph=False, create_graph=False)[0]
            noise = noise / self.N
            
            if self.model_type in [ModelType.IMAGECLASSIFICATION, ModelType.OBJECTDETECTION]:
                # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
                noise = F.conv2d(noise, self.T_kernel, bias=None,
                                    stride=1, padding=(3, 3), groups=3)
            # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
            noise = noise / torch.mean(torch.abs(noise), dim=(*range(1, len(noise.shape)),), keepdim=True)
            noise = self.momentum * grad + noise
            
            grad = noise
                
            
            grad_target = grad_target + self.alpha * torch.sign(noise)
            delta = torch.clamp(grad_target - ori_inputs, min=-self.eps, max=self.eps)
            grad_target = self._clamp(ori_inputs + delta)
        
        if self.return_grad:
            return grad_target, grad
        return grad_target

