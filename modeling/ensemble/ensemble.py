import torch
import torch.nn as nn
from modeling.ensemble.unet3d_parallel import UNet as UNetPara
from modeling.unet3d.unet3d import UNet


class Ensemble(nn.Module):
    def __init__(self, in_channels, out_channels,
                 output='list', exchange=False, feature=False, modality_specific_norm=True, width_ratio=1., sharing=True, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.output = output
        self.feature = feature
        self.modality_specific_norm = modality_specific_norm
        self.width_ratio = width_ratio
        self.sharing = sharing
        if self.modality_specific_norm and sharing:
            self.module = UNetPara(1, out_channels, num_modalities=in_channels, parallel=True,
                            exchange=exchange, feature=feature, width_multiplier=width_ratio)
        else:
            if sharing:
                self.module = UNet(1, out_channels, width_multiplier=width_ratio)
            elif not sharing:
                self.module = nn.ModuleList()
                for ii in range(in_channels):
                    self.module.append(UNet(1, out_channels, width_multiplier=width_ratio))
            else:
                raise NotImplementedError()

        # self.weights = nn.parameter.Parameter(torch.ones(in_channels) / in_channels, requires_grad=True)

    def forward(self, x, channel=[], weights=None):
        x = [x[:, i:i + 1] for i in range(self.in_channels)]

        if self.modality_specific_norm:
            out = self.module(x)
        else:
            if self.sharing:
                out = [self.module(x_i) for x_i in x]
            else:
                out = [self.module[ii](x_i) for ii, x_i in enumerate(x)]

        if self.output == 'list' and self.training:
            return out

        out = torch.stack(out, dim=0)
        preserved = list(range(self.in_channels))
        for c in channel:
            preserved.remove(c)

        if weights is None:
            return torch.mean(out[preserved], dim=0)
        else:
            w = weights[preserved] / weights[preserved].sum()

            return torch.einsum('mncwhd,m->ncwhd', out[preserved], w)

    def shared_module_zero_grad(self,):
        for module in self.shared_modules:
            module.zero_grad()
