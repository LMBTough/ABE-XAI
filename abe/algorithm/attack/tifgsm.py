from abe.algorithm.attack.base import GradientBasedAttack
from abe.task import AttackTask
from abe.func.utils import DI
import torch
import numpy as np
from scipy import stats as st
import torch.nn.functional as F
from typing import Any
from abe.type import ModelType
import warnings

class TIFGSM(GradientBasedAttack):
    r"""
    TIFGSM in the paper 'Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks'
    [https://arxiv.org/abs/1904.02884]

    Distance Measure : Linf

    Arguments:
        task (AttackTask): The attack task.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 0.0)
        kernel_name (str): kernel name. (Default: gaussian)
        len_kernel (int): kernel length.  (Default: 15, which is the best according to the paper)
        nsig (int): radius of gaussian kernel. (Default: 3; see Section 3.2.2 in the paper for explanation)
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
        >>> attack = TIFGSM(task,eps=8/255,alpha=2/255,steps=10,decay=0.0,kernel_name='gaussian',len_kernel=15,nsig=3,resize_rate=0.9,diversity_prob=0.5,random_start=False)
        >>> adv_images = attack(batch)

    """
    def __init__(self, task: AttackTask, eps: float = 8/255, alpha: float = 2/255, steps: int = 10, decay: float = 0.0,
                 kernel_name: str = 'gaussian', len_kernel: int = 15, nsig: int = 3, resize_rate: float = 0.9,
                 diversity_prob: float = 0.5, random_start: bool = False) -> None:
        super().__init__(task)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation()).to(self.device)
        
    def gkern(self, kernlen: int = 15, nsig: int = 3) -> np.ndarray:
        """Returns a 2D Gaussian kernel array.
        
        Args:
            kernlen (int): kernel length.
            nsig (int): radius of gaussian kernel.
            
        Returns:
            np.ndarray: 2D Gaussian kernel array.
        """
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen: int = 15) -> np.ndarray:
        """Returns a 2D uniform kernel array.
        
        Args:
            kernlen (int): kernel length.
            
        Returns:
            np.ndarray: 2D uniform kernel array.
        """
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen: int = 15) -> np.ndarray:
        """Returns a 2D linear kernel array.
        
        Args:
            kernlen (int): kernel length.
            
        Returns:
            np.ndarray: 2D linear kernel array.
        """
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
    
    
    def kernel_generation(self) -> np.ndarray:
        """Generate the kernel.
        
        Returns:
            np.ndarray: The kernel.
        """
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel
    
    
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
            
        momentum = torch.zeros_like(grad_target)
            
        
        for _ in range(self.steps):
            grad_target = torch.autograd.Variable(grad_target, requires_grad=True)
            loss = self.get_loss((DI(grad_target, self.resize_rate, self.diversity_prob,model_type=self.model_type), *extra))
            
            grad = torch.autograd.grad(loss, grad_target,
                                        retain_graph=False, create_graph=False)[0]
            
            match self.model_type:
                case ModelType.IMAGECLASSIFICATION | ModelType.OBJECTDETECTION:
                    grad = F.conv2d(grad, self.stacked_kernel, stride=1, padding='same', groups=3)
                case _:
                    warnings.warn("The kernel is not applied to the input data.")
            grad = grad / torch.mean(torch.abs(grad), dim=(*range(1, len(grad.shape)),), keepdim=True)
            grad = grad + momentum*self.decay
            
            momentum = grad
            
            grad_target = grad_target + self.alpha*grad.sign()
            delta = torch.clamp(grad_target - ori_inputs, min=-self.eps, max=self.eps)
            grad_target = self._clamp(ori_inputs + delta)
            
        return grad_target
        
        



    