import numpy as np
import torch.nn.functional as F
from abe.algorithm.attack import BIM
from abe.algorithm.explanation.base import Attributor
from abe.task import ExplanationTask,AttackTask
from abe.algorithm.explanation.ig import IG
from typing import Any, List,Tuple
import torch
import numpy as np
from abe.type import ModelType



class BIG(Attributor):
    r"""
        BIG
        Arguments:
            task (ExplanationTask): The explanation task.
            attack_epsilons (List[float]): The list of attack epsilons.
            attack_step (int): The number of attack steps.
            attack_alpha (float): The attack alpha.
            gradient_steps (int): The number of steps to run.
            num_classes (int): The number of classes.
            
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
            >>> big = BIG(task, attack_epsilons=[36/255, 64/255, 0.3 , 0.5 , 0.7 , 0.9 , 1.1], attack_step=50, attack_alpha=0.001, gradient_steps=50, num_classes=1000)
            >>> attributions = big(data)
    """
    def __init__(self, task: ExplanationTask, attack_epsilons: List[float] = [36/255, 64/255, 0.3 , 0.5 , 0.7 , 0.9 , 1.1], attack_step: int = 50, attack_alpha: float = 0.001, gradient_steps: int = 50, num_classes: int = 1000) -> None:
        super().__init__(task)
        match self.model_type:
            case ModelType.IMAGECLASSIFICATION | ModelType.NLPCLASSIFICATION:
                pass
            case _:
                raise ValueError("BIG is not implemented for this model type")
        self.attack_epsilons = attack_epsilons
        self.attack_step = attack_step
        self.attack_alpha = attack_alpha
        self.gradient_steps = gradient_steps
        attack_task = AttackTask(loss_fn=task.loss_fn, model_type=task.model_type)
        self.attacks = [BIM(attack_task, eps, attack_alpha, attack_step) for eps in attack_epsilons]
        self.saliency = IG(task, gradient_steps)
        self.num_classes = num_classes
        
    
    def take_closer_bd(self, x: np.ndarray, y: np.ndarray, cls_bd: np.ndarray, dis2cls_bd: np.ndarray, boundary_points: np.ndarray, boundary_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compare and return adversarial examples that are closer to the input

        Args:
            x (np.ndarray): Benign inputs
            y (np.ndarray): Labels of benign inputs
            cls_bd (None or np.ndarray): Points on the closest boundary
            dis2cls_bd ([type]): Distance to the closest boundary
            boundary_points ([type]): New points on the closest boundary
            boundary_labels ([type]): Labels of new points on the closest boundary

        Returns:
            (np.ndarray, np.ndarray): Points on the closest boundary and distances
        """
        if cls_bd is None:
            cls_bd = boundary_points
            dis2cls_bd = np.linalg.norm(np.reshape((boundary_points - x),
                                                (x.shape[0], -1)),
                                        axis=-1)
            return cls_bd, dis2cls_bd
        else:
            d = np.linalg.norm(np.reshape((boundary_points - x), (x.shape[0], -1)),
                            axis=-1)
            for i in range(cls_bd.shape[0]):
                if d[i] < dis2cls_bd[i] and y[i] != boundary_labels[i]:
                    dis2cls_bd[i] = d[i]
                    cls_bd[i] = boundary_points[i]
        return cls_bd, dis2cls_bd
    
    def boundary_search(self, attacks: List[BIM], data: torch.Tensor, target: torch.Tensor, extra: List[Any]) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """Search for the boundary points
        
        Args:
            attacks (List[BIM]): The list of attacks
            data (torch.Tensor): The input data
            target (torch.Tensor): The target labels
            extra (List[Any]): The extra data
            
        Returns:
            Tuple[np.ndarray, np.ndarray, torch.Tensor]: The boundary points, distances to the boundary points, and the success of the attacks
        """
        dis2cls_bd = np.zeros(data.shape[0]) + 1e16
        bd = None
        batch_boundary_points = None
        batch_success = None
        boundary_points = list()
        success_total = 0
        for attack in attacks:
            c_boundary_points = attack([data, target, *extra])
            c_success = self.get_logits([c_boundary_points, target, *extra]).argmax(-1) != target
            batch_success = c_success
            success_total += torch.sum(batch_success.detach())
            if batch_boundary_points is None:
                batch_boundary_points = c_boundary_points.detach(
                ).cpu()
                batch_success = c_success.detach().cpu()
            else:
                for i in range(batch_boundary_points.shape[0]):
                    if not batch_success[i] and c_success[i]:
                        batch_boundary_points[
                            i] = c_boundary_points[i]
                        batch_success[i] = c_success[i]
        boundary_points.append(batch_boundary_points)
        boundary_points = torch.cat(boundary_points, dim=0).to(self.device)
        y_pred = self.get_logits([boundary_points, target, *extra])
        x = data.cpu().detach().numpy()
        y = target.cpu().detach().numpy()
        y_onehot = np.eye(self.num_classes)[y]
        bd, _ = self.take_closer_bd(x, y, bd,
                            dis2cls_bd, boundary_points.cpu(),
                            np.argmax(y_pred.cpu().detach().numpy(), axis=-1))
        cls_bd = None
        dis2cls_bd = None
        cls_bd, dis2cls_bd = self.take_closer_bd(x, y_onehot, cls_bd,
                                            dis2cls_bd, bd, None)
        return cls_bd, dis2cls_bd, batch_success
    
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
        grad_target, labels, *extra = batch
        
        cls_bd, _, _ = self.boundary_search(
            self.attacks, grad_target, labels, extra)
        
        cls_bd = cls_bd.to(grad_target.device)
        
        attribution = self.saliency((cls_bd, labels, *extra))
        attribution = self._ensure_dim(grad_target, attribution)
        return attribution











