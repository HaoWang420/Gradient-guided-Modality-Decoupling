from modeling.unet import *
from modeling.munet import *
from modeling.unet3d.unet3d import UNet as UNet3D
from modeling.discriminator import *
from modeling.ensemble.ensemble import Ensemble

def build_model(args, nclass, nchannels, model='unet', recons=False):
    if model == 'unet':
        return UNet(n_channels=nchannels, n_classes=nclass, bilinear=True, feature=args.feature)
    elif model == 'munet':
        return MUNet(nmod=nchannels, nclass=nclass, adv=args.feature)
    elif model == 'disc':
        use_3d = True if args.name in ['unet3d'] else False
        return FeatureDiscriminator(nchannels, use3d=use_3d)
    elif model == 'unet3d':
        return UNet3D(nchannels, nclass)
    elif model == 'ensemble':
        return Ensemble(
            nchannels,
            nclass,
            output=args.output,
            exchange=args.exchange,
            feature=args.feature,
            width_ratio=args.width_ratio,
            modality_specific_norm=args.modality_specific_norm,
            sharing=args.sharing
        )
    else:
        raise NotImplementedError