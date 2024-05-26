# Retrieved from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU

# building brick for unet
class DoubleConv(nn.Module):
    """
    (convolution -> BN -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(DoubleConv, self).__init__()
        self.bn = bn
        self.double_conv = None

        if not mid_channels:
            mid_channels = out_channels

        if self.bn:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )


    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    maxpooling -> double_conv
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        mid_channels = out_channels
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling -> double conv
    """
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX //2,
                        diffY // 2, diffY - diffY //2])

        # addition/cat?
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, i_channels, o_channels=512, bilinear=True, num_layers=5, feature_only=False):
        """
        param i_channels: number of input channels
        
        """
        super(Encoder, self).__init__()
        self.n_channels = i_channels
        self.i_channels = i_channels
        self.o_channels = o_channels
        self.bilinear = bilinear
        self.num_layers = num_layers
        self.feature_only = feature_only

        self.inc = DoubleConv(self.i_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.down4 = Down(64, 128)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.feature_only:
            return x5
        return (x1, x2, x3, x4, x5)

class Decoder(nn.Module):
    def __init__(self, o_channels, bilinear=False) -> None:
        super().__init__()
        self.factor = factor = 2 if bilinear else 1
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        # self.outc1 = OutConv(128, o_channels)
        # self.outc2 = OutConv(64, o_channels)
        self.outc3 = OutConv(32, o_channels)
    
    def forward(self, x, mstage=False):
        x1, x2, x3, x4, x5 = x
        f1 = self.up1(x5, x4)
        f2 = self.up2(f1, x3)
        f3 = self.up3(f2 ,x2)
        f4 = self.up4(f3 ,x1)

        o3 = self.outc3(f4)

        return o3

        # if mstage:
        #     o1 = self.outc1(f2)
        #     o2 = self.outc2(f3)

        #     return (o1, o2, o3)
        # else:
        #     return o3