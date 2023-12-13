import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dequantization(nn.Module):

    def __init__(self, alpha=1e-5, quants=256):
        """
        Inputs:
            alpha - small constant that is used to scale the original input.
                    Prevents dealing with values very close to 0 and 1 when inverting the sigmoid
            quants - Number of possible discrete values (usually 256 for 8-bit image)
        """
        super().__init__()
        self.alpha = alpha
        self.quants = quants

    def forward(self, z, ldj, reverse=False):
        if not reverse:
            # x -> z
            # z = sigmoid-inverse((x + u) / 256) * (1 - alpha) + 0.5 * alpha)
            z, ldj = self.dequant(z, ldj)
            z, ldj = self.sigmoid(z, ldj, reverse=True)
        else:
            # z -> x
            # x = clamp((sigmoid(z) - 0.5 * alpha) / (1 - alpha) * 256, 0, 255)
            z, ldj = self.sigmoid(z, ldj, reverse=False)
            z = z * self.quants
            ldj += np.log(self.quants) * np.prod(z.shape[1:])
            z = torch.floor(z).clamp(min=0, max=self.quants-1).to(torch.int32)
        return z, ldj

    def sigmoid(self, z, ldj, reverse=False):
        # Applies an invertible sigmoid transformation
        if not reverse:
            # z -> x
            ldj += (-z - 2 * F.softplus(-z)).sum(dim=[1, 2, 3])
            z = torch.sigmoid(z)
            # Reversing scaling for numerical stability
            ldj -= np.log(1 - self.alpha) * np.prod(z.shape[1:])
            z = (z - 0.5 * self.alpha) / (1 - self.alpha)
        else:
            # x -> z
            # Scale to prevent boundaries 0 and 1
            z = z * (1 - self.alpha) + 0.5 * self.alpha
            ldj += np.log(1 - self.alpha) * np.prod(z.shape[1:])
            ldj += (-torch.log(z) - torch.log(1-z)).sum(dim=[1, 2, 3])
            z = torch.log(z) - torch.log(1-z)
        return z, ldj

    def dequant(self, z, ldj):
        # Transform discrete values to continuous volumes
        # x -> z
        z = z.to(torch.float32)
        z = z + torch.rand_like(z).detach()
        z = z / self.quants
        ldj -= np.log(self.quants) * np.prod(z.shape[1:])
        return z, ldj


class VariationalDequantization(Dequantization):

    def __init__(self, var_flows, alpha=1e-5):
        """
        Inputs:
            var_flows - A list of flow transformations to use for modeling q(u|x)
            alpha - Small constant, see Dequantization for details
        """
        super().__init__(alpha=alpha)
        self.flows = nn.ModuleList(var_flows)

    def dequant(self, x, ldj):
        # x -> z
        x = x.to(torch.float32)
        # We condition the flows on x, i.e. the original image
        img = (x / 255.0) * 2 - 1

        # Prior of u is a uniform distribution as before
        # As most flow transformations are defined on [-infinity,+infinity], we apply an inverse sigmoid first.
        deq_noise = torch.rand_like(x).detach()  # [0, 1]
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=True)
        for flow in self.flows:
            deq_noise, ldj = flow(deq_noise, ldj, reverse=False, orig_img=img)
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=False)

        # After the flows, apply u as in standard dequantization
        z = (x + deq_noise) / 256.0
        ldj -= np.log(256.0) * np.prod(z.shape[1:])
        return z, ldj


if __name__ == '__main__':
    from torchvision.datasets import MNIST
    from torchvision import transforms
    import pytorch_lightning as pl
    DATASET_PATH = '../data'

    def discrete(sample):
        return (sample * 255).to(torch.int32)
    transform = transforms.Compose([
        transforms.ToTensor(),
        discrete
    ])
    train_set = MNIST(root=DATASET_PATH, train=True,
                      transform=transform, download=True)
    pl.seed_everything(42)
    orig_img = train_set[0][0].unsqueeze(dim=0)
    ldj = torch.zeros(1,)
    dequant_module = Dequantization()
    deq_img, ldj = dequant_module(orig_img, ldj, reverse=False)
    reconst_img, ldj = dequant_module(deq_img, ldj, reverse=True)

    d1, d2 = torch.where(orig_img.squeeze() != reconst_img.squeeze())
    if len(d1) != 0:
        print("Dequantization was not invertible.")
        for i in range(d1.shape[0]):
            print("Original value:", orig_img[0, 0, d1[i], d2[i]].item())
            print("Reconstructed value:",
                  reconst_img[0, 0, d1[i], d2[i]].item())
    else:
        print("Successfully inverted dequantization")
