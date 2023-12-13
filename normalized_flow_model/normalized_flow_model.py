import os
import random
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from simple_flow import create_simple_flow
from multi_scale_flow import create_multiscale_flow
import urllib.request
from urllib.error import HTTPError

CHECKPOINT_PATH = './ckpt'
DATASET_PATH = '../data'
device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.set_float32_matmul_precision('medium')


class NormalizedFlowModel(pl.LightningModule):
    def __init__(self, flows, import_samples=8):
        """
        Inputs:
            flows - A list of flows (each a nn.Module) that should be applied on the images.
            import_samples - Number of importance samples to use during testing (see explanation below). Can be changed at any time
        """
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.import_samples = import_samples
        # Create prior distribution for final latent space
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def forward(self, imgs):
        # The forward function is only used for visualizing the graph
        return self._get_likelihood(imgs)

    def encode(self, imgs):
        # Given a batch of images, return the latent representation z and ldj of the transformations
        z, ldj = imgs, torch.zeros(imgs.shape[0], device=self.device)
        for flow in self.flows:
            z, ldj = flow(z, ldj, reverse=False)
        return z, ldj

    def _get_likelihood(self, imgs, return_ll=False):
        """
        Given a batch of images, return the likelihood of those.
        If return_ll is True, this function returns the log likelihood of the input.
        Otherwise, the ouptut metric is bits per dimension (scaled negative log likelihood)
        """
        z, ldj = self.encode(imgs)
        log_pz = self.prior.log_prob(z).sum(dim=[1, 2, 3])
        log_px = ldj + log_pz
        nll = -log_px
        # Calculating bits per dimension
        bpd = nll * np.log2(np.exp(1)) / np.prod(imgs.shape[1:])
        return bpd.mean() if not return_ll else log_px

    @torch.no_grad()
    def sample(self, img_shape, z_init=None):
        """
        Sample a batch of images from the flow.
        """
        # Sample latent representation from prior
        if z_init is None:
            z = self.prior.sample(sample_shape=img_shape).to(device)
        else:
            z = z_init.to(device)

        # Transform z to x by inverting the flows
        ldj = torch.zeros(img_shape[0], device=device)
        for flow in reversed(self.flows):
            z, ldj = flow(z, ldj, reverse=True)
        return z

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # An scheduler is optional, but can help in flows to get the last bpd improvement
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # Normalizing flows are trained by maximum likelihood => return bpd
        loss = self._get_likelihood(batch[0])
        self.log('train_bpd', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_likelihood(batch[0])
        self.log('val_bpd', loss)

    def test_step(self, batch, batch_idx):
        # Perform importance sampling during testing => estimate likelihood M times for each image
        samples = []
        for _ in range(self.import_samples):
            img_ll = self._get_likelihood(batch[0], return_ll=True)
            samples.append(img_ll)
        img_ll = torch.stack(samples, dim=-1)

        # To average the probabilities, we need to go from log-space to exp, and back to log.
        # Logsumexp provides us a stable implementation for this
        img_ll = torch.logsumexp(img_ll, dim=-1) - np.log(self.import_samples)

        # Calculate final bpd
        bpd = -img_ll * np.log2(np.exp(1)) / np.prod(batch[0].shape[1:])
        bpd = bpd.mean()

        self.log('test_bpd', bpd)


class GenerateCallback(pl.Callback):

    def __init__(self, sample_shape, every_n_epochs=5):
        super().__init__()
        self.sample_shape = sample_shape         # shape of images to generate
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        # Skip for all other epochs
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Generate images
            x = pl_module.sample(self.sample_shape)
            imgs = torchvision.utils.make_grid(x, nrow=1)

            trainer.logger.experiment.add_image(
                f"generation_{0}", imgs, global_step=trainer.current_epoch)


def train_model(train_loader, test_loader, sample_shape, flow_layer, **kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "MNIST"),
                         accelerator="gpu" if str(
                             device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=200,
                         gradient_clip_val=1.0,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor='val_bpd'),
                                    GenerateCallback(
                                        sample_shape=sample_shape, every_n_epochs=1),
                                    # SamplerCallback(every_n_epochs=5),
                                    LearningRateMonitor("epoch")
                                    ])
    # Check whether pretrained model exists. If yes, load it and skip training
    pl.seed_everything(42)
    model = NormalizedFlowModel(flow_layer)
    trainer.fit(model, train_loader, test_loader)
    return model


if __name__ == '__main__':
    def discrete(sample):
        return (sample * 255).to(torch.int32)
    transform = transforms.Compose([
        transforms.ToTensor(),
        discrete
    ])
    train_set = MNIST(root=DATASET_PATH, train=True,
                      transform=transform, download=True)
    test_set = MNIST(root=DATASET_PATH, train=False,
                     transform=transform, download=True)
    train_loader = data.DataLoader(
        train_set, batch_size=256, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(
        test_set,  batch_size=128, shuffle=False, drop_last=False, num_workers=4)

    model = train_model(train_loader=train_loader,
                        test_loader=test_loader,
                        sample_shape=(32, 8, 7, 7),
                        flow_layer=create_multiscale_flow())
