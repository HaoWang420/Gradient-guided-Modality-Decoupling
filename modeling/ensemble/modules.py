import torch
import torch.nn as nn
import torch.nn.functional as F

class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]

class GroupNormPara(nn.Module):
    def __init__(self, num_groups, num_channles, num_parallel):
        super(GroupNormPara, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'norm_' + str(i), nn.GroupNorm(num_groups, num_channles))

    def forward(self, x_parallel):
        return [getattr(self, 'norm_' + str(i))(x) for i, x in enumerate(x_parallel)]

class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, threshold=1e-2):
        n = x[0].shape[0]
        c = x[0].shape[1]
        out = []
        for i in range(len(x)):
            xi_out = torch.zeros_like(x[i])
            var = torch.var(x[i].view(n, c, -1), dim=2)
            xi_out[var >= threshold] = x[i][var >= threshold]

            modal_to_exchange = list(range(len(x)))
            modal_to_exchange.remove(i)
            xi_out[var < threshold] = torch.mean(
                                        torch.stack(
                                            [x[j][var < threshold] for j in modal_to_exchange], dim=0), dim=0)

            out.append(xi_out)

        return out

def conv_para(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return ModuleParallel(
            nn.Conv3d(in_channels, 
                      out_channels, 
                      kernel_size, 
                      stride=stride, 
                      padding=padding, 
                      dilation=dilation, 
                      groups=groups, 
                      bias=bias))

def relu_para(inplace):
    return ModuleParallel(nn.ReLU(inplace=inplace))

def upsample_para(scale_factor, mode, align_corners):
    return ModuleParallel(nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners))

def conv_trans3d_para(in_channels, out_channnels, kernel_size, stride):
    return ModuleParallel(nn.ConvTranspose3d(in_channels, out_channnels, kernel_size, stride))

def maxpool3d_para(kernel_size):
    return ModuleParallel(nn.MaxPool3d(kernel_size))

def padding_para(x1_list, x2_list):
    out = []
    for x1, x2 in zip(x1_list, x2_list):
        out.append(padding(x1, x2))

    return out

def padding(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    x = torch.cat([x2, x1], dim=1)

    return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, mid_channels=None, num_groups=8, num_modalities=4, exchange=False):
        super().__init__()
        self.num_modalities = num_modalities
        self.exchange = exchange
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = conv_type(in_channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = GroupNormPara(num_groups, mid_channels, num_modalities)
        if self.exchange:
            self.exchange1 = Exchange()
        self.relu1 = relu_para(inplace=True)
    
        self.conv2 = conv_type(mid_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = GroupNormPara(num_groups, out_channels, num_modalities)
        self.relu2 = relu_para(inplace=True)

    def forward(self, x, modality=0):
        x = self.conv1(x)
        x = self.norm1(x)
        if self.exchange:
            x = self.exchange1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        if self.exchange:
            x = self.exchange1(x)
        x = self.relu2(x)

        return x
        
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, num_modalities=4, exchange=False):
        super().__init__()
        self.pool = maxpool3d_para(2)
        self.double_conv = DoubleConv(in_channels, out_channels, conv_type=conv_type, num_modalities=num_modalities, exchange=exchange)

    def forward(self, x, modality=0):
        x = self.pool(x)
        x = self.double_conv(x, modality=modality)

        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True, conv_type=nn.Conv3d, num_modalities=4, parallel=False, exchange=False):
        super().__init__()
        self.parallel = parallel

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = upsample_para(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(
                            in_channels, 
                            out_channels, 
                            conv_type=conv_type, 
                            mid_channels=in_channels // 2, 
                            num_modalities=num_modalities, 
                            exchange=exchange)
        else:
            self.up = conv_trans3d_para(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(
                            in_channels, 
                            out_channels, 
                            conv_type=conv_type, 
                            num_modalities=num_modalities, 
                            exchange=exchange)

        if self.parallel:
            self.padding = padding_para
        else:
            self.padding = padding

    def forward(self, x1, x2, modality=0):
        x1 = self.up(x1)

        x = self.padding(x1, x2)

        return self.conv(x, modality=modality)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d):
        super(OutConv, self).__init__()
        self.conv = conv_type(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)