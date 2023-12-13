import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class MaskedConvolution(nn.Module):

    def __init__(self, c_in, c_out, mask, **kwargs):
        """
        Implements a convolution with mask applied on its weights.
        Inputs:
            c_in - Number of input channels
            c_out - Number of output channels
            mask - Tensor of shape [kernel_size_H, kernel_size_W] with 0s where
                   the convolution should be masked, and 1s otherwise.
            kwargs - Additional arguments for the convolution
        """
        super().__init__()
        # For simplicity: calculate padding automatically
        # h - (k - 1) * d + 2p = h => p = (d * (k - 1) // 2)
        kernel_size = (mask.shape[0], mask.shape[1])
        dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
        padding = tuple([dilation*(kernel_size[i]-1)//2 for i in range(2)])
        # Actual convolution
        self.conv = nn.Conv2d(c_in, c_out, kernel_size,
                              padding=padding, **kwargs)

        # Mask as buffer => it is no parameter but still a tensor of the module
        # (must be moved with the devices)
        self.register_buffer('mask', mask[None, None])

    def forward(self, x):
        self.conv.weight.data *= self.mask  # Ensures zero's at masked positions
        return self.conv(x)


class VerticalStackConvolution(MaskedConvolution):

    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height, but for simplicity, we stick with masking here.
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size//2+1:, :] = 0

        # For the very first convolution, we will also mask the center row
        if mask_center:
            mask[kernel_size//2, :] = 0

        super().__init__(c_in, c_out, mask, **kwargs)


class HorizontalStackConvolution(MaskedConvolution):

    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        mask = torch.ones(1, kernel_size)
        mask[0, kernel_size//2+1:] = 0

        # For the very first convolution, we will also mask the center pixel
        if mask_center:
            mask[0, kernel_size//2] = 0

        super().__init__(c_in, c_out, mask, **kwargs)


class GatedMaskedConv(nn.Module):

    def __init__(self, c_in, **kwargs):
        """
        Gated Convolution block implemented the computation graph shown above.
        """
        super().__init__()
        self.conv_vert = VerticalStackConvolution(c_in, c_out=2*c_in, **kwargs)
        self.conv_horiz = HorizontalStackConvolution(
            c_in, c_out=2*c_in, **kwargs)
        self.conv_vert_to_horiz = nn.Conv2d(
            2*c_in, 2*c_in, kernel_size=1, padding=0)
        self.conv_horiz_1x1 = nn.Conv2d(c_in, c_in, kernel_size=1, padding=0)

    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out


if __name__ == '__main__':
    inp_img = torch.zeros(1, 1, 11, 11)
    inp_img.requires_grad_()

    def show_center_recep_field(img, out, filename):
        """
        Calculates the gradients of the input with respect to the output center pixel,
        and visualizes the overall receptive field.
        Inputs:
            img - Input image for which we want to calculate the receptive field on.
            out - Output features/loss which is used for backpropagation, and should be
                the output of the network/computation graph.
        """
        # Determine gradients
        loss = out[0, :, img.shape[2]//2, img.shape[3] //
                   2].sum()  # L1 loss for simplicity
        # Retain graph as we want to stack multiple layers and show the receptive field of all of them
        loss.backward(retain_graph=True)
        img_grads = img.grad.abs()
        img.grad.fill_(0)  # Reset grads

        # Plot receptive field
        img = img_grads.squeeze().cpu().numpy()
        fig, ax = plt.subplots(1, 2)
        pos = ax[0].imshow(img)
        ax[1].imshow(img > 0)
        # Mark the center pixel in red if it doesn't have any gradients (should be the case for standard autoregressive models)
        show_center = (img[img.shape[0]//2, img.shape[1]//2] == 0)
        if show_center:
            center_pixel = np.zeros(img.shape + (4,))
            center_pixel[center_pixel.shape[0]//2, center_pixel.shape[1] //
                         2, :] = np.array([1.0, 0.0, 0.0, 1.0])
        for i in range(2):
            ax[i].axis('off')
            if show_center:
                ax[i].imshow(center_pixel)
        ax[0].set_title("Weighted receptive field")
        ax[1].set_title("Binary receptive field")
        plt.savefig(filename)

    show_center_recep_field(inp_img, inp_img, filename='0.png')

    horiz_conv = HorizontalStackConvolution(
        c_in=1, c_out=1, kernel_size=3, mask_center=True)
    horiz_conv.conv.weight.data.fill_(1)
    horiz_conv.conv.bias.data.fill_(0)
    horiz_img = horiz_conv(inp_img)
    show_center_recep_field(inp_img, horiz_img, filename='conv_h_0.png')

    vert_conv = VerticalStackConvolution(
        c_in=1, c_out=1, kernel_size=3, mask_center=True)
    vert_conv.conv.weight.data.fill_(1)
    vert_conv.conv.bias.data.fill_(0)
    vert_img = vert_conv(inp_img)
    show_center_recep_field(inp_img, vert_img, filename='conv_v_0.png')

    horiz_img = vert_img + horiz_img
    show_center_recep_field(inp_img, horiz_img, filename='conv_1.png')

    # Initialize convolutions with equal weight to all input pixels
    # mask-center should be False
    horiz_conv = HorizontalStackConvolution(
        c_in=1, c_out=1, kernel_size=3, mask_center=False)
    horiz_conv.conv.weight.data.fill_(1)
    horiz_conv.conv.bias.data.fill_(0)
    vert_conv = VerticalStackConvolution(
        c_in=1, c_out=1, kernel_size=3, mask_center=False)
    vert_conv.conv.weight.data.fill_(1)
    vert_conv.conv.bias.data.fill_(0)

    # We reuse our convolutions for the 4 layers here. Note that in a standard network,
    # we don't do that, and instead learn 4 separate convolution. As this cell is only for
    # visualization purposes, we reuse the convolutions for all layers.
    for l_idx in range(4):
        vert_img = vert_conv(vert_img)
        horiz_img = horiz_conv(horiz_img) + vert_img
        show_center_recep_field(
            inp_img, horiz_img, filename=f'conv_{l_idx + 2}.png')
