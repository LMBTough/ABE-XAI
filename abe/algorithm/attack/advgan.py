from abe.task import AttackTask
from abe.algorithm.attack.base import GANBasedAttack
from abe.func.utils import weights_init
from abe.type import ModelType
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
from typing import Tuple,Any,Callable
from tqdm import tqdm
import warnings




# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(image_nc, 8, kernel_size=4,
                      stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output


class Generator(nn.Module):
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(Generator, self).__init__()

        encoder_lis = [
            nn.Conv2d(gen_input_nc, 8, kernel_size=3,
                      stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        ]

        bottle_neck_lis = [ResnetBlock(32),
                           ResnetBlock(32),
                           ResnetBlock(32),
                           ResnetBlock(32),]

        decoder_lis = [
            nn.ConvTranspose2d(32, 16, kernel_size=3,
                               stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3,
                               stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, image_nc, kernel_size=6,
                               stride=1, padding=0, bias=False),
            nn.Tanh()
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x



class AdvGAN(GANBasedAttack):
    r"""
        AdvGAN attack.
        Arguments:
            task (AttackTask): The attack task.
            forward_fn (Callable): The forward function of the model.
            save_path (str): The path to save the trained GAN model.
            image_nc (int): input image number of channels. (default: 3)
            eps (float): maximum perturbation. (default: 8/255)
            epochs (int): number of train epochs. (default: 50)
            G_lr (float): learning rate of the generator. (default: 0.001)
            D_lr (float): learning rate of the discriminator. (default: 0.001)
            num_classes (int): number of classes. (default: 1000)

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
            >>> attack = AdvGAN(task,forward_fn,save_path,image_nc=3,eps=8/255,epochs=50,G_lr=0.001,D_lr=0.001,num_classes=1000)
            >>> attack.train(train_dataloader)
            >>> adv_images = attack(batch)
    """

    def __init__(self,task: AttackTask,forward_fn:Callable,save_path: str,image_nc: int = 3,eps: float = 8/255,epochs: int = 50,G_lr: float = 0.001,D_lr: float = 0.001,num_classes=1000) -> None:
        super().__init__(task, forward_fn, save_path)
        match self.model_type:
            case ModelType.IMAGECLASSIFICATION:
                pass
            case _:
                raise ValueError('AdvGAN attack is not supported for this model type')
        self.eps = eps
        
        self.netG = Generator(image_nc, image_nc).to(self.device)
        self.netDisc = Discriminator(image_nc).to(self.device)
        
        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=G_lr)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=D_lr)

        self.epochs = epochs
        
        self.num_classes = num_classes
        
        self.load()
        
        
        
    def save(self) -> None:
        torch.save(self.netG.state_dict(), os.path.join(self.save_path, 'G.pth'))
        torch.save(self.netDisc.state_dict(), os.path.join(self.save_path, 'D.pth'))
        
    def load(self) -> None:
        if os.path.exists(os.path.join(self.save_path, 'G.pth')) and os.path.exists(os.path.join(self.save_path, 'D.pth')):
            self.netG.load_state_dict(torch.load(os.path.join(self.save_path, 'G.pth')))
            self.netDisc.load_state_dict(torch.load(os.path.join(self.save_path, 'D.pth')))
        else:
            warnings.warn("AdvGAN model not found. You need to train the model first. Example: attack.train(dataloader)")
        
    def train_batch(self, batch: Any) -> Tuple[float, float, float, float]:
        """train a batch of data.
        
        Args:
            batch (Any): The batch of data.
            Image classification: Tuple[torch.Tensor, torch.Tensor]
            Object detection: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]
            NLP classification: Tuple[torch.Tensor, torch.Tensor]
            Time series Prediction: Tuple[torch.Tensor, torch.Tensor]
            
            
        Returns:
            Tuple[float, float, float, float]: loss_D, loss_G_fake, loss_perturb, loss_adv.
        """
        
        grad_target, labels, *extra = batch
        
        perturbation = self.netG(grad_target)
        
        # add a clipping trick
        adv_images = torch.clamp(perturbation, -self.eps, self.eps) + grad_target
        adv_images = torch.clamp(adv_images, min=0, max=1)
        
        self.optimizer_D.zero_grad()
        pred_real = self.netDisc(grad_target)
        loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
        loss_D_real.backward()
        pred_fake = self.netDisc(adv_images.detach())
        loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
        loss_D_fake.backward()
        loss_D_GAN = loss_D_fake + loss_D_real
        self.optimizer_D.step()
        
        self.optimizer_G.zero_grad()
        # cal G's loss in GAN
        pred_fake = self.netDisc(adv_images)
        loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
        loss_G_fake.backward(retain_graph=True)
        
        # calculate perturbation norm
        loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))

        # cal adv loss
        # logits_model = self.model(adv_images)
        logits_model = self.forward_fn((adv_images, labels, *extra))
        probs_model = F.softmax(logits_model, dim=1)
        onehot_labels = torch.eye(
            self.num_classes, device=self.device)[labels]

        # C&W loss function
        if self.is_targeted:
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) *
                                    probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(other - real, zeros)
            loss_adv = torch.sum(loss_adv)
        else:
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) *
                                    probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(real - other, zeros)
            loss_adv = torch.sum(loss_adv)
        
        adv_lambda = 10
        pert_lambda = 1
        loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
        loss_G.backward()
        self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()
        
    def train(self, train_dataloader: torch.utils.data.DataLoader) -> None:
        """Train the generator and discriminator.
        
        Args:
            train_dataloader (DataLoader): The dataloader of the dataset.
            
        
        Returns:
            None
        """
        for epoch in range(self.epochs):
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            pbar = tqdm(train_dataloader, desc="Training epoch "+str(epoch))
            for batch in pbar:
                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = self.train_batch(batch)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch
                
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))
            
            self.save()
            
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
            case ModelType.IMAGECLASSIFICATION | ModelType.OBJECTDETECTION:
                pass
            case _:
                raise Exception("AdvGAN attack is not supported for this model type")
        
        grad_target,label,*extra = batch
        perturbation = self.netG(grad_target)
        adv_images = torch.clamp(perturbation, -self.eps, self.eps) + grad_target
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        
        return adv_images