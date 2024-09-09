import itertools
from matplotlib.ft2font import LOAD_TARGET_LIGHT
from torch.nn.modules.flatten import Flatten
from dataloaders.datasets.brats import BraTSSet
from PIL.Image import NONE
import torch
import numpy as np
from torch import flatten, logit

import torch.nn as nn
import torch.nn.functional as F

class SegmentationLosses(object):
    def __init__(
            self, 
            args,
            nclass=3, 
        ):
        self.args = args
        self.cuda = self.args.cuda
        self.nclass = nclass

    def build_loss(self, mode='ce', disc=None, nchannels=4):
        """Choices: ['ce', 'dice', 'mstage-dice']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'dice':
            return self.DiceCoef
        elif mode == 'mdice':
            return self.mstage_dice
        elif mode == 'baseline':
            return self.BaselineLoss
        elif mode == 'feature-adv':
            self.disc = disc
            self.weights = self.args.loss.weights
            return self.FeatureAdv
        elif mode == 'feature-sim':
            self.distance = self.args.loss.distance
            self.weights = self.args.loss.weights
            return self.FeatureSim
        elif mode == 'cross-patient':
            self.distance = self.args.loss.distance
            self.weights = self.args.loss.weights
            return self.CrossPatientFeatureSim
        elif mode == 'input-sim':
            self.distance = self.args.loss.distance
            self.weights = self.args.loss.weights
            return self.InputFeatureSim
        elif mode == 'multi-sim':
            self.distance = self.args.loss.distance
            self.weights = self.args.loss.weights
            return self.MultiLevelFeatureSim
        elif mode == 'mse':
            self.weights = self.args.loss.weights
            return self.Reconstruction
        elif mode == 'enumeration':
            return self.EnumerationLoss
        elif mode == 'feature-enum':
            self.weights = self.args.loss.weights
            return self.FeatureEnum
        elif mode == 'weighted-enum':
            return self.WeightedEnum
        elif mode == 'full_enum':
            return self.FullEnumerationLoss
        elif mode == 'modality-dice':
            return self.ModalityDice
        else:
            print(f'Loss {mode} not available.')
            raise NotImplementedError

    def FullEnumerationLoss(self, logits, target, weights=None):
        M = len(logits)
        cnt = 0

        if self.args.loss.output == 'list':
            loss = []
        else:
            loss = 0.

        for missing_num in [0, 1, 2, 3]:
            for subset in itertools.combinations(list(range(M)), M - missing_num):
                # let weights sum to 1
                missing_logits = torch.stack([logits[l] for l in subset], dim=0)

                missing_logits = torch.mean(missing_logits, dim=0)

                if self.args.loss.output == 'list':
                    loss.append(self.DiceCoef(missing_logits, target))
                else:
                    loss += self.DiceCoef(missing_logits, target)
                cnt += 1

        if self.args.loss.output == 'mean':
            loss /= cnt

        return loss
    
    def Reconstruction(self, logits, target):
        recons, output = logits
        recons_target, target = target
        mse = nn.MSELoss()
        return self.weights.mse * mse(recons, recons_target) \
             + self.weights.dice * self.DiceCoef(output, target)
    
    def EnumerationLoss(self, logits, target, weights=None):
        M = len(logits)

        if self.args.loss.output == 'list':
            loss = []
        else:
            loss = 0.
        for subset in itertools.combinations(list(range(M)), M - self.args.loss.missing_num):
            # let weights sum to 1
            missing_logits = torch.stack([logits[l] for l in subset], dim=0)

            if weights is not None:
                w = weights[list(subset)] / weights[list(subset)].sum()
                missing_logits = torch.einsum('mncwhd,m->ncwhd', missing_logits, w)
            else:
                missing_logits = torch.mean(missing_logits, dim=0)

            if self.args.loss.output == 'list':
                loss.append(self.DiceCoef(missing_logits, target))
            else:
                loss += self.DiceCoef(missing_logits, target)
        
        if self.args.loss.output == 'mean':
            loss /= len(list(itertools.combinations(list(range(M)), M - self.args.loss.missing_num)))

        return loss
    
    def ModalityDice(self, logits, target):
        if self.args.loss.output == 'list':
            loss = []
        else:
            loss = 0.
        for l in logits:
            if self.args.loss.output == 'list':
                loss.append(self.DiceCoef(l, target))
            else:
                loss += self.DiceCoef(l, target)
        
        if self.args.loss.output == 'mean':
            loss /= len(logits)

        return loss

    def WeightedEnum(self, logits, target):
        M = len(logits)

        if self.args.loss.output == 'list':
            loss = []
        else:
            loss = 0.
        for subset in itertools.combinations(list(range(M)), M - self.args.loss.missing_num):
            missing_logits = [logits[l] for l in subset]
            missing_logits = torch.mean(torch.stack(missing_logits, dim=0), dim=0)
            if self.args.loss.output == 'list':
                loss.append(self.DiceCoef(missing_logits, target))
            else:
                loss_i = self.DiceCoef(missing_logits, target)
                loss += (1. + 1. / (1. - loss_i.item())) * loss_i
        
        if self.args.loss.output == 'mean':
            loss /= M

        return loss
        

    def FeatureEnum(self, logits, target):
        logits, features = logits

        loss = 0.

        for i in range(len(features)):
            recons_feature = [features[j] for j in range(len(features)) if j != i]
            recons_feature = torch.mean(torch.stack(recons_feature, dim=0), dim=0)
            recons_feature = recons_feature / torch.norm(recons_feature, dim=1, keepdim=True)
            missing_feature = features[i] / torch.norm(features[i], dim=1, keepdim=True)

            sim = F.mse_loss(recons_feature, missing_feature.detach())

            loss += sim
        loss /= len(features)
        loss *= self.weights.feature
        
        loss += self.weights.dice * self.EnumerationLoss(logits, target)
        return loss


    def FeatureAdv(self, logits, target):
        bce = nn.BCEWithLogitsLoss()

        full, drop = logits
        flogit, ffeature = full
        dlogit, dfeature = drop

        sim_loss = bce(
                self.disc(dfeature)[:, 0], 
                torch.ones(dfeature.size()[0], device=dfeature.device)
                )
        seg_full = self.DiceCoef(flogit, target)
        seg_drop = self.DiceCoef(dlogit, target)

        return self.weights.sim * sim_loss \
             + self.weights.full_dice * seg_full \
             + self.weights.drop_dice * seg_drop
    
    def FeatureSim(self, logits, target, weights=None):
        full, drop = logits
        flogit, ffeature = full
        dlogit, dfeature = drop
        ffeature = ffeature

        if self.distance == 'l2':
            dis = nn.MSELoss()
        elif self.distance == 'l1':
            dis = nn.L1Loss()
        elif self.distance == 'cosine':
            dis = self.NegativeCosineSimilarity
        
        return self.weights.sim * dis(ffeature, dfeature) \
             + self.weights.full_dice * self.DiceCoef(flogit, target) \
             + self.weights.drop_dice * self.DiceCoef(dlogit, target)
 
    def CrossPatientFeatureSim(self, logits, target):
        full, drop = logits
        ftarget, dtarget = target
        flogit, ffeature = full
        dlogit, dfeature = drop

        gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        f_down_target = interpolate3d(ftarget, ffeature.size()[-3:])
        d_down_target = interpolate3d(dtarget, dfeature.size()[-3:])
        ffeature = gap(ffeature * f_down_target[0, 2])
        dfeature = gap(dfeature * d_down_target[0, 2])

        if self.distance == 'l2':
            dis = nn.MSELoss()
        elif self.distance == 'l1':
            dis = nn.L1Loss()
        elif self.distance == 'cosine':
            dis = self.NegativeCosineSimilarity
        
        return self.weights.sim * dis(ffeature, dfeature) \
             + self.weights.full_dice * self.DiceCoef(flogit, ftarget) \
             + self.weights.drop_dice * self.DiceCoef(dlogit, dtarget)
            
    def InputFeatureSim(self, logits, target):
        full, drop = logits
        flogit, ffeature = full
        dlogit, dfeature = drop
        ffeature = ffeature[-1]
        dfeature = dfeature[-1]

        if self.distance == 'l2':
            dis = nn.MSELoss()
        elif self.distance == 'l1':
            dis = nn.L1Loss()
        elif self.distance == 'cosine':
            dis = self.NegativeCosineSimilarity
        
        return self.weights.sim * dis(ffeature, dfeature) \
             + self.weights.full_dice * self.DiceCoef(flogit, target) \
             + self.weights.drop_dice * self.DiceCoef(dlogit, target)

    def MultiLevelFeatureSim(self, logits, target):
        full, drop = logits
        flogit, ffeature = full
        dlogit, dfeature = drop

        if self.distance == 'l2':
            dis = nn.MSELoss()
        elif self.distance == 'l1':
            dis = nn.L1Loss()
        elif self.distance == 'cosine':
            dis = self.NegativeCosineSimilarity
        
        similarity = 0.
        for ii in range(len(ffeature)):
            similarity += dis(ffeature[ii], dfeature[ii])
        
        return self.weights.sim * similarity \
             + self.weights.full_dice * self.DiceCoef(flogit, target) \
             + self.weights.drop_dice * self.DiceCoef(dlogit, target)

    def NegativeCosineSimilarity(self, l1, l2):
        f = nn.Flatten()
        cosine = nn.CosineSimilarity()
        l1 = f(l1)
        l2 = f(l2)

        return 1 - torch.mean(cosine(l1, l2))


    def BaselineLoss(self, logits, target, weights=None):
        return self.mstage_dice(logits, target, weights=weights)
    
    def mstage_ce(self, logits, target, weights=None, epoch=0):
        m = len(logits)

        loss = 0.0
        for ii in range(m):
            target = interpolate2d(target, logits[ii].size()[2:])
            l = self.CrossEntropyLoss(logits[ii], target, epoch=epoch)

            if weights is not None:
                loss += weights[ii] * l
            else:
                loss += l
        
        return loss

    def mstage_dice(self, logits, target, weights=None, epoch=0):
        m = len(logits)

        loss = 0.0
        for ii in range(m):
            # downsample the ground truth
            target = interpolate2d(target, logits[ii].size()[2:])
            l = self.DiceCoef(logits[ii], target)
            
            if weights is not None:
                loss += weights[ii] * l
            else:
                loss += l
        
        return loss
         
    def CrossEntropyLoss(self, logit, target, epoch=0):
        n, c, h, w = logit.size()
        weight = self.weight * pow(0.99, epoch) + 1
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def DiceCoef(self, preds, targets):
        smooth = 1.0
        class_num = self.nclass
        sigmoid = nn.Sigmoid()
        preds = sigmoid(preds)
        loss = torch.zeros(class_num, device=preds.device)
        for i in range(class_num):
            pred = preds[:, i, :, :]
            target = targets[:, i, :, :]
            intersection = (pred * target).sum()

            loss[i] += 1 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
        
        return torch.mean(loss)

def interpolate3d(input, size):
    H, W, D = input.size()[-3:]
    h, w, d = size

    wi = torch.linspace(0, W-1, w).long()
    hi =  torch.linspace(0, H-1, h).long()
    di = torch.linspace(0, D-1, d).long()

    return input[..., hi[:, None, None], wi[:, None], di]

def interpolate2d(input, size):
    H, W = input.size()[-2:]
