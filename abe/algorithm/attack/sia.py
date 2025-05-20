import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as st
from abe.func import dct
from abe.algorithm.attack.base import GradientBasedAttack
from abe.task import AttackTask
from typing import Any


class SIA(GradientBasedAttack):
    r"""
    SIA Attack
    Arguments:
        task (AttackTask): The attack task.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 10)
        num_copies (int): number of copies. (Default: 20)
        num_block (int): number of blocks. (Default: 3)

    Examples::
        >>> loss_fn = def f(data):
        >>>     image,label = data
        >>>     loss = nn.CrossEntropyLoss()
        >>>     yhat = model(image)
        >>>     return loss(yhat,label)
        >>> task = AttackTask(loss_fn)
        >>> attack = SIA(task,eps=8/255,alpha=2/255,steps=10,decay=1.0,num_copies=20,num_block=3)
        >>> adv_images = attack(batch)

    """
    
    def __init__(self, task: AttackTask, eps: float = 8/255, alpha: float = 2/255, steps: int = 10, decay: float = 1.0, num_copies:int=20, num_block:int=3):
        self.num_copies = num_copies
        self.num_block = num_block
        self.kernel = self.gkern()
        self.op = [self.resize, self.vertical_shift, self.horizontal_shift, self.vertical_flip, self.horizontal_flip, self.rotate180, self.scale, self.add_noise,self.dct,self.drop_out]
        
    def vertical_shift(self, x):
        _, _, w, _ = x.shape
        step = np.random.randint(low = 0, high=w, dtype=np.int32)
        return x.roll(step, dims=2)

    def horizontal_shift(self, x):
        _, _, _, h = x.shape
        step = np.random.randint(low = 0, high=h, dtype=np.int32)
        return x.roll(step, dims=3)

    def vertical_flip(self, x):
        return x.flip(dims=(2,))

    def horizontal_flip(self, x):
        return x.flip(dims=(3,))

    def rotate180(self, x):
        return x.rot90(k=2, dims=(2,3))
    
    def scale(self, x):
        return torch.rand(1)[0] * x
    
    def resize(self, x):
        """
        Resize the input
        """
        _, _, w, h = x.shape
        scale_factor = 0.8
        new_h = int(h * scale_factor)+1
        new_w = int(w * scale_factor)+1
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(w, h), mode='bilinear', align_corners=False).clamp(0, 1)
        return x
    
    def dct(self, x):
        """
        Discrete Fourier Transform
        """
        dctx = dct.dct_2d(x)
        _, _, w, h = dctx.shape
        low_ratio = 0.4
        low_w = int(w * low_ratio)
        low_h = int(h * low_ratio)
        dctx[:, :, -low_w:,:] = 0
        dctx[:, :, :, -low_h:] = 0
        idctx = dct.idct_2d(dctx)
        return idctx
    
    def add_noise(self, x):
        return torch.clip(x + torch.zeros_like(x).uniform_(-16/255,16/255), 0, 1)

    def gkern(self, kernel_size=3, nsig=3):
        x = np.linspace(-nsig, nsig, kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return torch.from_numpy(stack_kernel.astype(np.float32)).to(self.device)

    def drop_out(self, x):
        
        return F.dropout2d(x, p=0.1, training=True)

    def blocktransform(self, x, choice=-1):
        _, _, w, h = x.shape
        y_axis = [0,] + np.random.choice(list(range(1, h)), self.num_block-1, replace=False).tolist() + [h,]
        x_axis = [0,] + np.random.choice(list(range(1, w)), self.num_block-1, replace=False).tolist() + [w,]
        y_axis.sort()
        x_axis.sort()
        
        x_copy = x.clone()
        for i, idx_x in enumerate(x_axis[1:]):
            for j, idx_y in enumerate(y_axis[1:]):
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op), dtype=np.int32)
                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = self.op[chosen](x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y])

        return x_copy

    def transform(self, x, **kwargs):
        """
        Scale the input for BlockShuffle
        """
        return torch.cat([self.blocktransform(x) for _ in range(self.num_copies)])
    
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
            loss = self.get_loss((self.transform(grad_target), *extra))
            grad = torch.autograd.grad(loss, grad_target,
                                       retain_graph=False, create_graph=False)[0]
            
            grad = grad / torch.mean(torch.abs(grad), dim=(*range(1, len(grad.shape)),), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad
            
            grad_target = grad_target.detach() + self.alpha*grad.sign()
            delta = torch.clamp(grad_target - ori_inputs, min=-self.eps, max=self.eps)
            grad_target = self._clamp(ori_inputs + delta)
            
        return grad_target