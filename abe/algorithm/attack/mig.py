from abe.algorithm.attack.base import GradientBasedAttack
from abe.task import AttackTask,ExplanationTask
import torch
from typing import Any
from abe.algorithm.explanation.ig import IG

class MIG(GradientBasedAttack):
    r"""

    Arguments:
        task (AttackTask): The attack task.
        exp_task (ExplanationTask): The explanation task.
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
        >>> def forward_fn(data):
        >>>     image, label = data
        >>>     return model(image)
        >>> task = AttackTask(loss_fn)
        >>> exp_task = ExplanationTask(loss_fn, forward_fn)
        >>> attack = MIG(task,exp_task,eps=8/255,alpha=2/255,steps=10,decay=1.0)
        >>> adv_images = attack(batch)

    """
    def __init__(self, task: AttackTask,exp_task:ExplanationTask, eps: float = 8/255, alpha: float = 2/255, steps: int = 10, decay: float = 1.0):
        super().__init__(task)
        self.ig = IG(exp_task)
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
            grad_t = -torch.from_numpy(self.ig((grad_target, *extra)))
            grad = self.decay * momentum + grad_t / (grad_t).abs().mean(dim=[i for i in range(1, len(grad_t.shape))],keepdim=True)
            momentum = grad
            grad_target = grad_target.detach() + self.alpha*grad.sign()
            delta = torch.clamp(grad_target - ori_inputs, min=-self.eps, max=self.eps)
            grad_target = self._clamp(ori_inputs + delta)
            
        return grad_target

