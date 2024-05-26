from modeling.modality_encoder import *

class MUNet(nn.Module):
    """
    Args:
        nmod: number of input modalities;
        nclass: number of output classes(channels);
        adv: adversarial training with feature discriminator;
    """
    def __init__(self, nmod, nclass, adv=True):
        super().__init__()

        self.nmod = nmod
        self.nclass = nclass
        self.adv = adv

        self.encoder = ModalityEncoder(nmod, nclass)

        self.decoder = Decoder(nclass, bilinear=False)

    def forward(self, x, feature_only=False):
        f = self.encoder(x)

        if feature_only:
            return f[-1]

        x = self.decoder(f, mstage=True)

        if self.adv and self.training:
            return x, f[-1]
        else:
            return x

