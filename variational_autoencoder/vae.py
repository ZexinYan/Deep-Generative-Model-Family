import os
import random
from typing import Any
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from utils import sample_gaussian, log_bernoulli_with_logits, kl_normal, gaussian_parameters, log_normal, log_normal_mixture
from conv import EncoderDecoder

CHECKPOINT_PATH = './ckpt'
DATASET_PATH = '../data'
device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.set_float32_matmul_precision('medium')


class VAE(pl.LightningModule):
    def __init__(self, z_dim=10, k=500, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.network = EncoderDecoder(z_dim, k)

    def forward(self, x):
        """
        calculate negative elbo
        """
        batch_size = x.shape[0]
        prior = gaussian_parameters(self.network.z_pre, dim=1)
        x = torch.bernoulli(x)
        m, v = self.network.encoder.encode(x)
        z = sample_gaussian(m, v)
        logits_x = self.network.decoder.decode(z)
        rec = - \
            log_bernoulli_with_logits(
                x.reshape(batch_size, -1), logits_x.reshape(batch_size, -1))
        kl = log_normal(z, m, v) - log_normal_mixture(z, prior[0], prior[1])
        kl = kl.mean()
        rec = rec.mean()
        return kl, rec

    @torch.no_grad()
    def sample(self, batch_size):
        m, v = gaussian_parameters(self.network.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(
            self.network.pi).sample((batch_size,))
        m, v = m[idx], v[idx]
        z = sample_gaussian(m, v)
        logits = torch.sigmoid(self.network.decoder.decode(z))
        return torch.bernoulli(logits)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.network.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 100,  gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, x):
        kl, rec = self.forward(x[0])
        nelbo = kl + rec
        self.log('kl', kl)
        self.log('rec', rec)
        self.log('nelbo', nelbo)
        return nelbo

    def validation_step(self, x):
        kl, rec = self.forward(x[0])
        self.log('val_kl', kl)
        self.log('val_rec', rec)
        self.log('val_nelbo', kl + rec)
        return kl + rec

    def test_step(self, x):
        kl, rec = self.forward(x[0])
        self.log('test_kl', kl)
        self.log('test_rec', rec)
        self.log('vest_nelbo', kl + rec)
        return kl + rec


class GenerateCallback(pl.Callback):

    def __init__(self, sample_batch_size, every_n_epochs=5):
        super().__init__()
        # batch-size of images to generate
        self.sample_batch_size = sample_batch_size
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        # Skip for all other epochs
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Generate images
            x = pl_module.sample(self.sample_batch_size)
            imgs = torchvision.utils.make_grid(x, nrow=self.sample_batch_size)

            trainer.logger.experiment.add_image(
                f"generation_{0}", imgs, global_step=trainer.current_epoch)


def train_model(train_loader, val_loader, test_loader, sample_batch_size, **kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "VAE"),
                         accelerator="gpu" if str(
                             device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=100,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_nelbo"),
                                    GenerateCallback(
                                        sample_batch_size, every_n_epochs=10),
                                    LearningRateMonitor("epoch")])
    model = VAE(**kwargs)
    trainer.fit(model, train_loader, val_loader)
    nelbo = trainer.test(model, test_loader)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = MNIST(root=DATASET_PATH, train=True,
                      transform=transform, download=True)

    test_set = MNIST(root=DATASET_PATH, train=False,
                     transform=transform, download=True)
    train_loader = data.DataLoader(
        train_set, batch_size=256, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = data.DataLoader(
        train_set,  batch_size=256, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(
        test_set,  batch_size=128, shuffle=False, drop_last=False, num_workers=4)

    model = train_model(train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        sample_batch_size=16)
