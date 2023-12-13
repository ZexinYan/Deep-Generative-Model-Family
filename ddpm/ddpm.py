import os
import random
from typing import Any
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from nn import SimpleUNet
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
from utils import linear_beta_schedule, get_index_from_list


CHECKPOINT_PATH = './ckpt'
DATASET_PATH = '../data'
device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.set_float32_matmul_precision('medium')


class DDPM(pl.LightningModule):
    def __init__(self, T=300) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.T = T
        self.network = SimpleUNet()
        self.betas = linear_beta_schedule(timesteps=T)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * \
            (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def forward_diffusion_sample(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = get_index_from_list(
            self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
            + sqrt_one_minus_alphas_cumprod_t.to(device) * \
            noise.to(device), noise.to(device)

    def sample_timestep(self, x, t):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        betas_t = get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(
            self.sqrt_recip_alphas, t, x.shape)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.network(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_from_list(
            self.posterior_variance, t, x.shape)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, batch_size):
        img = torch.randn((batch_size, 1, 28, 28), device=device)
        step_size = self.T // 10
        denoise_imgs = []
        for i in range(0, self.T)[::-1]:
            t = torch.full((1, ), i, device=device, dtype=torch.long)
            img = self.sample_timestep(img, t)
            if i % step_size == 0:
                denoise_imgs.append(img)
        return denoise_imgs

    def forward(self, x):
        batch_size = x.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=device).long()
        x_noisy, noise = self.forward_diffusion_sample(x, t)
        noise_pred = self.network(x_noisy, t)
        return F.l1_loss(noise, noise_pred)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.network.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, x):
        loss = self.forward(x[0])
        self.log('loss', loss)
        return loss

    def validation_step(self, x):
        loss = self.forward(x[0])
        self.log('val_loss', loss)
        return loss

    def test_step(self, x):
        loss = self.forward(x[0])
        self.log('test_loss', loss)
        return loss


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
            imgs_list = pl_module.sample(self.sample_batch_size)
            imgs = torch.stack(imgs_list, dim=0).reshape(-1, 1, 28, 28)
            torchvision.utils.save_image(
                imgs, f'ddpm-{trainer.current_epoch}.png')

            # trainer.logger.experiment.add_image(
            #     f"generation_{0}", imgs, global_step=trainer.current_epoch)


class VisualizeImage(pl.Callback):

    def __init__(self, sample_batch_size):
        super().__init__()
        self.sample_batch_size = sample_batch_size

    def on_train_start(self, trainer, pl_module) -> None:
        # Simulate forward diffusion
        image = next(iter(trainer.train_dataloader))[0]

        num_images = 10
        step_size = int(pl_module.T / num_images)
        imgs_list = []
        image = image[:self.sample_batch_size]
        for idx in range(0, pl_module.T, step_size):
            t = torch.Tensor([idx]).type(torch.int64)
            image, noise = pl_module.forward_diffusion_sample(image, t)
            imgs_list.append(image)
        imgs = torch.stack(imgs_list, dim=0).reshape(-1, 1, 28, 28)
        torchvision.utils.save_image(imgs, f'noise_image.png')


def train_model(train_loader, val_loader, test_loader, sample_batch_size, **kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "DDPM"),
                         accelerator="gpu" if str(
                             device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=300,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                                    GenerateCallback(
                                        sample_batch_size, every_n_epochs=10),
                                    LearningRateMonitor("epoch"),
                                    VisualizeImage(sample_batch_size=sample_batch_size)])
    model = DDPM(**kwargs)
    trainer.fit(model, train_loader, val_loader)
    loss = trainer.test(model, test_loader)


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
