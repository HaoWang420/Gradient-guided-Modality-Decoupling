"""Adapted from https://github.com/milesial/Pytorch-UNet/tree/master/unet"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.ensemble.modules import *


class UNet(nn.Module):
    def __init__(
        self, 
        n_channels, 
        n_classes, 
        width_multiplier=1, 
        trilinear=True, 
        conv_type=conv_para, 
        num_modalities=4, 
        parallel=False, 
        exchange=False, 
        feature=False):
        """A simple 3D Unet, adapted from a 2D Unet from https://github.com/milesial/Pytorch-UNet/tree/master/unet
        Arguments:
          n_channels = number of input channels; 3 for RGB, 1 for grayscale input
          n_classes = number of output channels/classes
          width_multiplier = how much 'wider' your UNet should be compared with a standard UNet
                  default is 1;, meaning 32 -> 64 -> 128 -> 256 -> 512 -> 256 -> 128 -> 64 -> 32
                  higher values increase the number of kernels pay layer, by that factor
          trilinear = use trilinear interpolation to upsample; if false, 3D convtranspose layers will be used instead
          use_ds_conv = if True, we use depthwise-separable convolutional layers. in my experience, this is of little help. This
                  appears to be because with 3D data, the vast vast majority of GPU RAM is the input data/labels, not the params, so little
                  VRAM is saved by using ds_conv, and yet performance suffers."""
        super(UNet, self).__init__()
        _channels = (32, 64, 128, 256, 512)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = [int(c*width_multiplier) for c in _channels]
        self.trilinear = trilinear
        self.conv_type = conv_type
        self.parallel = parallel
        self.feature = feature

        self.inc = DoubleConv(
            n_channels, self.channels[0], conv_type=self.conv_type, num_modalities=num_modalities, exchange=exchange)
        self.down1 = Down(self.channels[0], self.channels[1], conv_type=self.conv_type,
                          num_modalities=num_modalities, exchange=exchange)
        self.down2 = Down(self.channels[1], self.channels[2], conv_type=self.conv_type,
                          num_modalities=num_modalities, exchange=exchange)
        self.down3 = Down(self.channels[2], self.channels[3], conv_type=self.conv_type,
                          num_modalities=num_modalities, exchange=exchange)
        factor = 2 if trilinear else 1
        self.down4 = Down(self.channels[3], self.channels[4] // factor,
                          conv_type=self.conv_type, num_modalities=num_modalities, exchange=exchange)
        self.up1 = Up(self.channels[4], self.channels[3] // factor, trilinear, conv_type=self.conv_type,
                      num_modalities=num_modalities, parallel=parallel, exchange=exchange)
        self.up2 = Up(self.channels[3], self.channels[2] // factor, trilinear, conv_type=self.conv_type,
                      num_modalities=num_modalities, parallel=parallel, exchange=exchange)
        self.up3 = Up(self.channels[2], self.channels[1] // factor, trilinear, conv_type=self.conv_type,
                      num_modalities=num_modalities, parallel=parallel, exchange=exchange)
        self.up4 = Up(self.channels[1], self.channels[0], trilinear, conv_type=self.conv_type,
                      num_modalities=num_modalities, parallel=parallel, exchange=exchange)
        self.outc = OutConv(
            self.channels[0], n_classes, conv_type=self.conv_type)

    def forward(self, x: list, modality=0):
        # x a list of modalities
        x1 = self.inc(x, modality=modality)
        x2 = self.down1(x1, modality=modality)
        x3 = self.down2(x2, modality=modality)
        x4 = self.down3(x3, modality=modality)
        x5 = self.down4(x4, modality=modality)

        x = self.up1(x5, x4, modality=modality)
        x = self.up2(x, x3, modality=modality)
        x = self.up3(x, x2, modality=modality)
        x = self.up4(x, x1, modality=modality)
        logits = self.outc(x)

        if self.feature and self.training:
            return logits, x5
        return logits


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(
            nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv3d(
            nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
