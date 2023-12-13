import os
import random
from typing import Any
import numpy as np
from pytorch_lightning.utilities.types import OptimizerLRScheduler
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
from conv import Generator, Discriminator

CHECKPOINT_PATH = './ckpt'
DATASET_PATH = '../data'
device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.set_float32_matmul_precision('medium')


class GAN(pl.LightningModule):
    def __init__(self, dim_z=64, num_channels=1, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(dim_z=dim_z, num_channels=num_channels)
        self.discriminator = Discriminator(num_channels=num_channels)
        self.automatic_optimization = False

    def forward(self, x, with_wasserstein_gp=False):
        """
        with gradient penalty

        """
        batch_size = x.shape[0]
        z = torch.randn(batch_size, self.generator.dim_z).to(device=device)
        x_fake = self.generator(z)
        logits_fake = self.discriminator(x_fake)
        logits_real = self.discriminator(x)
        d_loss = F.binary_cross_entropy_with_logits(
            logits_real, torch.ones(batch_size, device=device)) + F.binary_cross_entropy_with_logits(logits_fake, torch.zeros(batch_size, device=device))

        if with_wasserstein_gp:
            alpha = torch.randn(batch_size, 1, 1, 1, device=device)
            x_grad_input = alpha * x + (1 - alpha) * x_fake
            x_grad_output = self.discriminator(x_grad_input).sum()

            x_grad = torch.autograd.grad(
                x_grad_output, x_grad_input, create_graph=True)
            x_grad = x_grad[0].reshape(batch_size, -1).norm(dim=1)
            d_loss += 10 * ((x_grad - 1) ** 2).mean()

        g_loss = - F.logsigmoid(logits_fake).mean()

        return d_loss, g_loss

    @torch.no_grad()
    def sample(self, batch_size):
        z = torch.randn(batch_size, self.generator.dim_z).to(device=device)
        x_fake = (self.generator(z) + 1) / 2
        return x_fake

    def configure_optimizers(self):
        g_optimizer = optim.Adam(
            self.generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))
        g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, 1, gamma=0.99)
        d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, 1, gamma=0.99)
        return ({
            'optimizer': g_optimizer,
            'lr_scheduler': {
                'scheduler': g_scheduler
            },
        },
            {
            'optimizer': d_optimizer,
            'lr_scheduler': {
                'scheduler': d_scheduler
            },
        })

    def training_step(self, batch, batch_idx):
        d_loss, g_loss = self.forward(batch[0], with_wasserstein_gp=True)
        g_optimizer, d_optimizer = self.optimizers()
        d_optimizer.zero_grad()
        d_loss.backward(retain_graph=True)
        g_optimizer.zero_grad()
        g_loss.backward()
        d_optimizer.step()
        g_optimizer.step()
        self.log('d_loss', d_loss)
        self.log('g_loss', g_loss)
        return d_loss + g_loss

    def validation_step(self, batch, batch_idx):
        d_loss, g_loss = self.forward(batch[0])
        self.log('val_d_loss', d_loss)
        self.log('val_g_loss', g_loss)
        return d_loss, g_loss

    def test_step(self, batch, batch_idx):
        d_loss, g_loss = self.forward(batch[0])
        self.log('test_d_loss', d_loss)
        self.log('test_g_loss', g_loss)
        return d_loss, g_loss


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
            imgs = torchvision.utils.make_grid(
                x, nrow=1, normalize=True, value_range=(-1, 1))

            trainer.logger.experiment.add_image(
                f"generation_{0}", imgs, global_step=trainer.current_epoch)


def train_model(train_loader, val_loader, test_loader, sample_batch_size, **kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "GAN"),
                         accelerator="gpu" if str(
                             device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=150,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_g_loss"),
                                    GenerateCallback(
                                        sample_batch_size, every_n_epochs=1),
                                    LearningRateMonitor("epoch")])
    model = GAN(**kwargs)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set, val_set = torch.utils.data.random_split(MNIST(root=DATASET_PATH, train=True,
                                                             transform=transform, download=True), [50000, 10000])

    test_set = MNIST(root=DATASET_PATH, train=False,
                     transform=transform, download=True)
    train_loader = data.DataLoader(
        train_set, batch_size=256, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = data.DataLoader(
        val_set,  batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(
        test_set,  batch_size=128, shuffle=False, drop_last=False, num_workers=4)

    model = train_model(train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        sample_batch_size=16,
                        c_in=1, c_hidden=64)
