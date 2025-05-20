from demo.algorithm.explanation.base import Attributor
from demo.task import ExplanationTask,AttackTask
from demo.algorithm.attack import FGSM
from demo.type import ModelType
from typing import Any
import torch
import numpy as np



class LA(Attributor):
    r"""
    LA
    Arguments:
        task (ExplanationTask): The explanation task.
        eps_factor (int): The epsilon factor. (Default: 10)
        steps (int): The number of steps. (Default: 30)
        num (int): The number of selected target classes. (Default: 20)
        num_classes (int): The number of classes. (Default: 1000)
        
    Examples::
        >>> def loss_fn(data):
        >>>     image, label = data
        >>>     loss = nn.CrossEntropyLoss()
        >>>     yhat = model(image)
        >>>     return loss(yhat, label)
        >>> def forward_fn(data):
        >>>     image, label = data
        >>>     return model(image)
        >>> task = ExplanationTask(loss_fn, forward_fn)
        >>> la = LA(task, eps_factor=10, steps=30, num=20, num_classes=1000)
        >>> attributions = la(data)

    """
    def __init__(self,task: ExplanationTask, eps_factor:int = 10, steps:int = 30, num:int = 20, num_classes:int = 1000) -> None:
        super().__init__(task)
        match task.model_type:
            case ModelType.IMAGECLASSIFICATION | ModelType.NLPCLASSIFICATION:
                pass
            case _:
                raise ValueError("LA is not implemented for this model type")
        self.eps_factor = eps_factor
        self.steps = steps
        self.num = num
        self.num_classes = num_classes
        self.untargeted_task = AttackTask(loss_fn=task.loss_fn, model_type=task.model_type)
        self.targeted_task = AttackTask(loss_fn=task.loss_fn, model_type=task.model_type, is_targeted=True)
        self.selected_ids = np.random.choice(num_classes, num, replace=False)
        
    def batch_attribute(self, batch: Any) -> np.ndarray:
        """Generate the attributions for the batch of data.
        
        Args:
            batch (Any): The batch of data.
            Image classification: Tuple[torch.Tensor, torch.Tensor]
            Object detection: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]
            NLP classification: Tuple[torch.Tensor, torch.Tensor]
            Time series Prediction: Tuple[torch.Tensor, torch.Tensor]
            
        Returns:
            np.ndarray: The attributions.
        """
        inputs,label,*extra = batch
        init_pred = self.get_logits(batch).argmax(-1)
        
        
        attribution = 0
        
        epsilon = inputs.clone() / self.eps_factor
        
        targeted_fgsm = FGSM(self.targeted_task, eps=epsilon, return_grad=True)
        
        untargeted_fgsm = FGSM(self.untargeted_task, eps=epsilon, return_grad=True)
        
        for i in self.selected_ids[:-1]:
            perturbed_inputs = inputs.clone()
            targeted = torch.tensor([i] * inputs.shape[0]).to(self.device)
            if i < self.num_classes - 1:
                targeted[targeted == init_pred] = i + 1
            else: 
                targeted[targeted == init_pred] = i - 1
            
            for j in range(self.steps):
                
                _,grad_targeted = targeted_fgsm((perturbed_inputs, targeted, *extra))
                
                _,grad_untargeted = untargeted_fgsm((perturbed_inputs, init_pred, *extra))
                
                delta = epsilon * grad_targeted.sign()
                
                perturbed_inputs = perturbed_inputs + delta
                perturbed_inputs = self._clamp(perturbed_inputs)
                delta = perturbed_inputs - inputs
                delta = grad_untargeted * delta
                attribution += delta
        
        perturbed_inputs = inputs.clone()
        for j in range(self.steps):
            
            perturbed_inputs,grad_untargeted = untargeted_fgsm((inputs, init_pred, *extra))
            
            attribution += (perturbed_inputs - inputs) * grad_untargeted
        
                
        attribution = self._ensure_dim(inputs, attribution.detach().cpu().numpy())
        return attribution                
        
