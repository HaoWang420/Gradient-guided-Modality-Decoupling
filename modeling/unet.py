import torch.nn.functional as F
import torch

from modeling.unet_component import *

# TODO
# modify to encoder, decoder version
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, feature=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature = feature

        self.inc = DoubleConv(n_channels, 64, bn=False)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.factor = factor

    def forward(self, x, feature_only=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if feature_only:
            return x5
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x ,x1)
        logits = self.outc(x)

        if self.training and self.feature:
            return (logits, x5)
        else:
            return logits