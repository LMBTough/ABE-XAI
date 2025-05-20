from abe.algorithm.attack.base import GradientBasedAttack
from abe.task import AttackTask
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any,Callable
from abe.type import ModelType


class CW(GradientBasedAttack):
    r"""
    CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

    Distance Measure : L2

    Arguments:
        task (AttackTask): task to attack.
        c (float): c in the paper. parameter for box-constraint. (Default: 1)    
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (Default: 50)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)

    .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.


    Examples::
        >>> loss_fn = def f(data):
        >>>     image,label = data
        >>>     loss = nn.CrossEntropyLoss()
        >>>     yhat = model(image)
        >>>     return loss(yhat,label)
        >>> forward_fn = def f(data):
        >>>     image,label = data
        >>>     return model(image)
        >>> task = AttackTask(loss_fn)
        >>> attack = CW(task,foward_fn,c=1,kappa=0,steps=50,lr=0.01)

    .. note:: Binary search for c is NOT IMPLEMENTED methods in the paper due to time consuming.

    """
    
    def __init__(self, task: AttackTask,forward_fn:Callable,c: float = 1, kappa: float = 0, steps: int = 50, lr: float = 0.01) -> None:
        super().__init__(task)
        self.forward_fn = forward_fn
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        
    def tanh_space(self, x: torch.Tensor) -> torch.Tensor:
        """Convert x to tanh space.
        
        Args:
            x (torch.Tensor): input tensor.
            
        Returns:
            torch.Tensor: tensor in tanh space.
        """
        return 1/2*(torch.tanh(x) + 1)

    def inverse_tanh_space(self, x: torch.Tensor) -> torch.Tensor:
        """Convert x to inverse tanh space.
        
        Args:
            x (torch.Tensor): input tensor.
            
        Returns:
            torch.Tensor: tensor in inverse tanh space.
        """
        # torch.atanh is only for torch >= 1.7.0
        # atanh is defined in the range -1 to 1
        return self.atanh(torch.clamp(x*2-1, min=-1, max=1))

    def atanh(self, x: torch.Tensor) -> torch.Tensor:
        """atanh function.
        
        Args:
            x (torch.Tensor): input tensor.
            
        Returns:
            torch.Tensor: atanh(x).
        """
        return 0.5*torch.log((1+x)/(1-x))

    # f-function in the paper
    def f(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """f-function in the paper.
        
        Args:
            outputs (torch.Tensor): output tensor.
            labels (torch.Tensor): label tensor.
            
        Returns:
            torch.Tensor: f(outputs, labels).
            
        """
        if self.is_targeted:
            one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]
            other = torch.max((1-one_hot_labels)*outputs, dim=1)[0] # find the max logit other than the target class
            real = torch.max(one_hot_labels*outputs, dim=1)[0]      # get the target class's logit
            return torch.clamp((other-real), min=-self.kappa)
        else:
            target_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]
            other = torch.max((1-target_labels)*outputs, dim=1)[0] # find the max logit other than the target class
            real = torch.max(target_labels*outputs, dim=1)[0]      # get the target class's logit
            return torch.clamp((real-other), min=-self.kappa)
        
    def batch_attack(self, batch: Any) -> torch.Tensor:
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
        match self.model_type:
            case ModelType.IMAGECLASSIFICATION | ModelType.NLPCLASSIFICATION:
                pass
            case _:
                raise Exception("CW attack is not supported for this model type")
        grad_target,label,*extra = batch
        
        
        w = self.inverse_tanh_space(grad_target).detach()
        w = torch.autograd.Variable(w, requires_grad=True)
        
        best_adv_samples = grad_target.clone().detach()
        best_L2 = 1e10*torch.ones((len(grad_target))).to(self.device)
        prev_cost = 1e10
        dim = len(grad_target.shape)
        
        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()
        
        optimizer = optim.Adam([w], lr=self.lr)
        
        for step in range(self.steps):
            adv_samples = self.tanh_space(w)
            
            current_L2 = MSELoss(Flatten(adv_samples),
                                 Flatten(grad_target)).sum(dim=1)
            L2_loss = current_L2.sum()
            
            # outputs = self.get_logits(adv_samples)
            outputs = self.forward_fn((grad_target, label, *extra))
            f_loss = self.f(outputs, label).sum()
            
            cost = L2_loss + self.c*f_loss
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            pre = torch.argmax(outputs.detach(), 1)
            # condition = (pre != label).float() if self.target is None else (pre == label).float() # if targeted, we want to let pre == labels, otherwise we want to let pre != labels
            condition = (pre == label).float() if self.is_targeted else (pre != label).float()
            
            # Filter out images that get either correct predictions or non-decreasing loss, 
            # i.e., only images that are both misclassified and loss-decreasing are left 
            mask = condition*(best_L2 > current_L2.detach())
            best_L2 = mask*current_L2.detach() + (1-mask)*best_L2
            
            mask = mask.view([-1]+[1]*(dim-1))
            best_adv_samples = mask*adv_samples.detach() + (1-mask)*best_adv_samples
            
            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps//10,1) == 0:
                if cost.item() > prev_cost:
                    return best_adv_samples
                prev_cost = cost.item()
                
        return best_adv_samples