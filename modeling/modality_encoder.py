from modeling.unet_component import *

class ModalityEncoder(nn.Module):
    def __init__(self, nmod, nout, feature_only=False):
        """
        param nmod: number of input modality
        """
        super().__init__()

        self.nmod = nmod
        self.nout = nout
        self.feature_only = feature_only
        self.nlayers = 5

        for ii in range(self.nmod):
            self.add_module(f"enc{ii}", Encoder(1, 512, feature_only=feature_only))

        self.skip1 = SkipConnect(128, 512, self.nmod)
        self.skip2 = SkipConnect(64, 256, self.nmod)
        self.skip3 = SkipConnect(32, 128, self.nmod)
        self.skip4 = SkipConnect(16, 64, self.nmod)
        self.skip5 = SkipConnect(8, 32, self.nmod)

    def forward(self, x):
        mout = []
        for ii in range(self.nmod):
            mout.append(self._modules[f"enc{ii}"](x[:, ii:ii+1, ...]))
        
        out = []
        for ii in range(1, self.nlayers + 1):
            feature = []
            for jj in range(self.nmod):
                feature.append(mout[jj][ii-1])

            out.append(self._modules[f"skip{self.nlayers - ii + 1}"](feature))

        return out

class SkipConnect(nn.Module):
    def __init__(self, in_channel, out_channel, nmod):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.nmod = nmod

        self.layer = nn.Sequential(
            nn.Conv2d(in_channel * nmod, out_channel, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = torch.cat(x, dim=1)

        return self.layer(x)