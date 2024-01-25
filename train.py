import os
import argparse
import torch
from torchvision.utils import make_grid
from kornia.geometry.transform import resize
import numpy as np
import yaml


#import warnings
#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore",category=DeprecationWarning)
import pytorch_lightning as pl
#import lightning.pytorch as pl
# from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.utilities.cli import LightningCLI, CALLBACK_REGISTRY, MODEL_REGISTRY
#from lightning.pytorch.cli import LightningCLI

from cvdvae import CVAE
# from ema import EMA
from data import Resize, LRHRDataModule
from utils import load_cvdvae
from vae_helpers import gaussian_analytical_kl
from utils import Hyperparams


torch.backends.cudnn.benchmark = True

# @MODEL_REGISTRY
class CondVAE(pl.LightningModule):
    def __init__(
                    self,
                    enc_blocks_y : str,
                    ky : int,
                    image_size_y : int,
                    vdvae_conf_path: str = "configs/vdvae_FFHQ256.yaml",
                    vis_batch_size: int = 8,
                    lr: float = 0.0001,
                    min_lr: float = 0.000001,
                    patience: int = 5
                ):
        """Conditional VAE with pretrained generative model module

        Args:
            enc_blocks_y (str): specification of the LR encoder bottom-up pass (eg "32x17,32d2,16x7,16d2,8x5,8d2,4x5,4d4,1x4"). (rxn = resolution x, n blocks, rdf, downsample from resolution r with downscaling factor f)
            ky (int): number of latent groups to be predicted by the LR encoder
            image_size_y (int): resolution of the LR images (eg 32)
            vdvae_conf_path (str, optional): Defaults to "configs/vdvae_FFHQ256.yaml".
            vis_batch_size (int, optional): _description_. Defaults to 8.
            lr (float, optional): initial learning rate. Defaults to 0.0001.
            min_lr (float, optional): minimal learning rate value. Defaults to 0.000001.
            patience (int, optional): number of epochs without improvement on the validation loss before reducing the learning rate. Defaults to 5.
        """
        
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.min_lr = min_lr
        self.patience = patience
        H = Hyperparams()
        with open(vdvae_conf_path) as file:
            H.update(yaml.safe_load(file))
        H.update({
            'enc_blocks_y' : enc_blocks_y,
            'ky' : ky,
            'image_size_y' : image_size_y
        })
        self.cvae = CVAE(H)
        # load pretrained VDVAE weights and freeze them
        vdvae_state_dict = torch.load(H['checkpoint_path'])
        cvae_state_dict = self.cvae.state_dict()
        for k, v in vdvae_state_dict.items():
            if k in cvae_state_dict:
                cvae_state_dict[k] = v
                cvae_state_dict[k].requires_grad = False
        #cvae_state_dict = {k : vdvae_state_dict[k].requires_grad_(False) if k in vdvae_state_dict else cvae_state_dict[k] for k in cvae_state_dict}
        self.cvae.load_state_dict(cvae_state_dict)
        #nfreezed = 0
        #ntot = 0
        #for name, layer in self.cvae.named_parameters():
        #    if not layer.requires_grad:
        #        nfreezed += 1
        #    ntot += 1
        #print('{}/{} layers frozens'.format(nfreezed, ntot))
        
        # TODO : remove
        # resize function
        self.y_size = self.cvae.H.image_size_y
        self.resize = Resize(size=(self.y_size, self.y_size), quantize=True)

    def forward(self, batch):
        x, y = batch[0], batch[1]
        loss, kls = self.cvae.forward_cond_training(x, y)
        loss = loss.mean()
        kls = {str(l): kls[l].mean() for l in kls}
        return loss, kls

    def training_step(self, batch, batch_idx):
        loss, kls = self.forward(batch)
        loss = loss.mean()
        self.log('kl', kls, sync_dist=True)
        self.log('loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, index):
        loss, kls = self.forward(batch)
        y = batch[1]
        xsr = self.sample_images(y, quantize=False)
        lr_rmse = torch.mean((self.resize(xsr) - y)**2) ** 0.5
        self.log('kl_val', kls)
        self.log('lr_rmse', lr_rmse)
        self.log('hp_metric', lr_rmse)
        self.log('loss_val', loss)
        # don't update the weight if loss is NaN
        if torch.isnan(loss) or torch.isinf(loss):
            return None
        else:
            return loss

    def configure_optimizers(self):
        trainable_params = []
        # freeze vdvae parameters
        for name, param in self.cvae.named_parameters():
            if 'encoder_y' in name or 'enc_y' in name :
                trainable_params.append(param)
            else:
                # save memory by avoiding computing the gradient
                param.requires_grad = False
        #optimizer = torch.optim.Adam(trainable_params, lr=self.lr)
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)
        #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 6], gamma=0.1)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=self.patience, threshold=0.01, threshold_mode='rel', min_lr=self.min_lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "strict" : False,
                "monitor": "loss_val"
            }
        }

    def sample_images(self, y, quantize=True):
        y = y.to(torch.device(self.device))
        xsr, _ = self.cvae.forward_samples(y, quantize=quantize)
        return xsr

#@CALLBACK_REGISTRY
class TensorBoardImageSampler(pl.Callback):

    def __init__(self, iter_per_images, samples_per_images=4):
        super().__init__()
        self.x = None #HR
        self.y = None #LR (HR and LR imags will be loaded in the first validation batch)
        # number of iterations between images sampling
        self.iter_per_images = iter_per_images
        # number of demo images
        #self.nrow = x.shape[0]
        self.samples_per_image = samples_per_images

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if self.x is None:
            #self.x = batch['GT']
            #self.y = batch['LQ']
            self.x, self.y = batch[0], batch[1]
            self.y_up = torch.round(resize(self.y, size=self.x.shape[-2:], interpolation='nearest')).clamp(0, 255).type(torch.uint8)
            self.nrow = self.x.shape[0]

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx, unused=0):
        if trainer.global_step % self.iter_per_images == 0 and self.x is not None:
            samples = []
            pl_module.eval()
            for i in range(self.samples_per_image):
                with torch.no_grad():
                    # sample reconstructed images        
                    xsr = pl_module.sample_images(self.y, quantize=True)
                    samples.append(xsr)
            pl_module.train()
            samples = torch.cat(samples, dim=0)

            full_batch = torch.cat((self.y_up, samples), dim=0)
            grid = make_grid(full_batch, nrow=self.nrow)
            trainer.logger.experiment.add_image('samples', grid, global_step=trainer.global_step)

def cli_main():
    #image_callback = TensorBoardImageSampler(iter_per_images=10, samples_per_images=4)

    # cli = LightningCLI(CondVAE, LRHRDataModule, seed_everything_default=42, save_config_overwrite=True)
    cli = LightningCLI(CondVAE, LRHRDataModule, seed_everything_default=42)
    #, trainer_defaults={"callbacks" : [image_callback, lr_monitor, checkpoint_callback]})
    
    #cli.trainer.fit(cli.model, datamodule=cli.datamodule)

if __name__ == "__main__":
    cli_main()