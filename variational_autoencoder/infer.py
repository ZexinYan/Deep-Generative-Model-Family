import math
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch
from vae import VAE


device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def show_imgs(imgs, title=None, row_size=4):
    # Form a grid of pictures (we use max. 8 columns)
    num_imgs = imgs.shape[0] if isinstance(imgs, torch.Tensor) else len(imgs)
    is_int = imgs.dtype == torch.int32 if isinstance(
        imgs, torch.Tensor) else imgs[0].dtype == torch.int32
    nrow = min(num_imgs, row_size)
    ncol = int(math.ceil(num_imgs/nrow))
    imgs = torchvision.utils.make_grid(
        imgs, nrow=nrow)
    print(f'imgs: {imgs.min()} {imgs.max()} {imgs.shape}')
    np_imgs = imgs.cpu().numpy()
    # Plot the grid
    plt.figure(figsize=(1.5*nrow, 1.5*ncol))
    plt.imshow(np.transpose(np_imgs, (1, 2, 0)), interpolation='nearest')
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.savefig('./debug.png')
    # plt.show()
    # plt.close()


model = VAE()
pretrained_filename = './ckpt/VAE/lightning_logs/version_0/checkpoints/epoch=147-step=34632.ckpt'
ckpt = torch.load(pretrained_filename, map_location=device)
model.load_state_dict(ckpt['state_dict'])
model.to(device)
x = model.sample(32)
print(f'x: {x.shape}')
show_imgs(x)
