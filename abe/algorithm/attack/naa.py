from abe.algorithm.attack.base import GradientBasedAttack
from abe.task import AttackTask
from abe.func.utils import DI
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any,Callable,Union
from abe.type import ModelType



features = None


def hook_feature(module, input, output):
    global features
    features = output
    


class NAA(GradientBasedAttack):
    r"""
        NAA Attack
        Arguments:
            task (AttackTask): task to attack.
            forward_fn (Callable): forward function of the model.
            target_layer Union[str, nn.Module]: feature layer.
            eps (float): maximum perturbation. (Default: 8/255)
            alpha (float): step size. (Default: 2/255)
            steps (int): number of steps. (Default: 10)
            

            
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
            >>> target_layer = model.features[0]
            >>> attack = NAA(task,forward_fn,target_layer,eps=8/255,alpha=2/255,steps=10,num_classes=100)
            >>> adv_images = attack(batch)
    """
    def __init__(self, task: AttackTask,forward_fn:Callable,target_layer: Union[str, nn.Module], eps: float = 8/255, alpha: float = 2/255, steps: int = 10,num_classes:int=100) -> None:
        super().__init__(task)
        self.forward_fn = forward_fn
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.momentum = 1.0
        self.ens = 30.0
        self.gamma = 0.5
        self.num_classes = num_classes

        if isinstance(target_layer, str):
            for name, module in self.model.named_modules():
                if name == target_layer:
                    target_layer = module
                    break
        target_layer.register_forward_hook(hook_feature)
        
            
    def get_NAA_loss(self, adv_feature: torch.Tensor, base_feature: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Calculate the NAA loss.
        
        Args:
            adv_feature (torch.Tensor): The feature map of the adversarial image.
            base_feature (torch.Tensor): The feature map of the base image.
            weights (torch.Tensor): The weights.
            
        Returns:
            torch.Tensor: The loss.
        """
        gamma = 1.0
        attribution = (adv_feature - base_feature) * weights
        blank = torch.zeros_like(attribution)
        positive = torch.where(attribution >= 0, attribution, blank)
        negative = torch.where(attribution < 0, attribution, blank)
        # Transformation: Linear transformation performs the best
        balance_attribution = positive + gamma * negative
        loss = torch.sum(balance_attribution) / \
            (base_feature.shape[0]*base_feature.shape[1])
        return loss
    
    def normalize(self, grad: torch.Tensor, opt: int = 2) -> torch.Tensor:
        """Normalize the gradient.
        
        Args:
            grad (torch.Tensor): The gradient.
            opt (int): The option of normalization.
            
        Returns:
            torch.Tensor: The normalized gradient.
        """
        if opt == 0:
            nor_grad = grad
        elif opt == 1:
            abs_sum = torch.sum(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            nor_grad = grad/abs_sum
        elif opt == 2:
            square = torch.sum(torch.square(grad), dim=(1, 2, 3), keepdim=True)
            nor_grad = grad/torch.sqrt(square)
        return nor_grad
    

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
        match self.model_type:
            case ModelType.IMAGECLASSIFICATION:
                pass
            case _:
                raise Exception("NAA attack is not supported for this model type")
        
        grad_target,label,*extra = batch
        
        ori_inputs = grad_target.clone()
        
        one_hot_labels = F.one_hot(label, self.num_classes).to(self.device)
        
        
        grad_np = torch.zeros_like(grad_target)
        
        weight_np = None
        
        for step in range(self.steps):
            grad_target = torch.autograd.Variable(grad_target, requires_grad=True)
            if step == 0:
                if self.ens == 0:
                    logits = self.forward_fn((grad_target, label, *extra))
                    if weight_np is None:
                        weight_np = torch.autograd.grad(logits*one_hot_labels, features, grad_outputs=torch.ones_like(logits*one_hot_labels))[0]
                    else:
                        weight_np += torch.autograd.grad(logits*one_hot_labels, features, grad_outputs=torch.ones_like(logits*one_hot_labels))[0]
                        
                for l in range(int(self.ens)):
                    x_base = np.array([0.0, 0.0, 0.0])
                    images_base = grad_target.clone()
                    images_base += (torch.randn_like(grad_target)*0.2 + 0)
                    images_base = images_base.cpu().detach().numpy().transpose(0, 2, 3, 1)
                    images_base = images_base * (1 - l / self.ens) + (l / self.ens) * x_base
                    images_base = torch.from_numpy(images_base.transpose(0, 3, 1, 2)).float().to(self.device)
                    logits = self.forward_fn((images_base, label, *extra))
                    if weight_np is None:
                        weight_np = torch.autograd.grad(logits*one_hot_labels, features, grad_outputs=torch.ones_like(logits*one_hot_labels))[0]
                    else:
                        weight_np += torch.autograd.grad(logits*one_hot_labels, features, grad_outputs=torch.ones_like(logits*one_hot_labels))[0]
                weight_np = -self.normalize(weight_np, 2)
                
            images_base = torch.zeros_like(grad_target)
            _ = self.forward_fn((images_base, label, *extra))
            base_feamap = features
            _ = self.forward_fn((grad_target, label, *extra))
            adv_feamap = features
            loss = self.get_NAA_loss(adv_feamap, base_feamap, weight_np)
            grad = torch.autograd.grad(loss, grad_target, create_graph=False, retain_graph=False)[0]
            grad = grad / torch.mean(torch.abs(grad), dim=(*range(1, len(grad.shape)),), keepdim=True)
            grad = self.momentum * grad_np + grad
            grad_np = grad
            

            grad_target = grad_target + self.alpha * grad.sign()
            delta = torch.clamp(grad_target - ori_inputs, min=-self.eps, max=self.eps)
            grad_target = self._clamp(ori_inputs + delta)
            
        return grad_target

