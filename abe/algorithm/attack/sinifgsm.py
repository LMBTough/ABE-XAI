from abe.algorithm.attack.base import GradientBasedAttack
from abe.task import AttackTask
import torch
from typing import Any


class SINIFGSM(GradientBasedAttack):
    r"""
    SI-NI-FGSM in the paper 'NESTEROV ACCELERATED GRADIENT AND SCALEINVARIANCE FOR ADVERSARIAL ATTACKS'
    [https://arxiv.org/abs/1908.06281], Published as a conference paper at ICLR 2020
    Modified from "https://githuba.com/JHL-HUST/SI-NI-FGSM"

    Distance Measure : Linf

    Arguments:
        task (AttackTask): The attack task.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 1.0)
        m (int): number of scale copies. (Default: 5)

    Examples::
        >>> loss_fn = def f(data):
        >>>     image,label = data
        >>>     loss = nn.CrossEntropyLoss()
        >>>     yhat = model(image)
        >>>     return loss(yhat,label)
        >>> task = AttackTask(loss_fn)
        >>> attack = SINIFGSM(task,eps=8/255,alpha=2/255,steps=10,decay=1.0,m=5)
        >>> adv_images = attack(batch)

    """

    def __init__(self, task: AttackTask, eps: float = 8/255, alpha: float = 2/255, steps: int = 10, decay: float = 1.0, m: int = 5) -> None:
        super().__init__(task)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.m = m
        
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
            nes_grad_target = grad_target + self.decay*self.alpha*momentum
            # Calculate sum the gradients over the scale copies of the input image
            adv_grad = torch.zeros_like(grad_target).to(self.device)
            for i in torch.arange(self.m):
                nes_grad_targets = nes_grad_target / torch.pow(2, i)
                # Calculate loss
                loss = self.get_loss((nes_grad_targets, *extra),check_input=False)
                adv_grad += torch.autograd.grad(loss, grad_target,
                                                retain_graph=False, create_graph=False)[0]
            adv_grad = adv_grad / self.m
            
            grad = self.decay*momentum + adv_grad / torch.mean(torch.abs(adv_grad), dim=(*range(1, len(adv_grad.shape)),), keepdim=True)
            momentum = grad
            
            grad_target = grad_target.detach() + self.alpha*grad.sign()
            delta = torch.clamp(grad_target - ori_inputs, min=-self.eps, max=self.eps)
            grad_target = self._clamp(ori_inputs + delta)
            
        return grad_target
