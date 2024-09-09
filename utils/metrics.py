import numpy as np

import torch.nn.functional as F
from dataloaders.datasets.brats import BraTSSet

class Evaluator(object):
    def __init__(self, loss='ce', metrics=['dice'], preprocess=True):
        self.loss = loss
        self.metrics = metrics

        self.mdice = []
        self.dice_class = []
        self.preprocess = preprocess
    
    def __preprocess(self, y, activation=False):
        if activation and self.preprocess:
            if self.loss == 'ce':
                y = np.argmax(y, axis=1)

                y = BraTSSet.transform_label(y)
            else:
                y = self.__sigmoid(y)

        return y
        
    def __sigmoid(self, x):
        # prevent numerical overflow
        x = np.clip(x, -88.72, 88.73)

        return 1 / (1 + np.exp(-x))

    def _dice_coef(self, gt_image, pre_image, smooth=1.0):
        c = gt_image.shape[1]

        _dice = np.zeros(c, dtype=np.float)
        for ii in range(c):
            gt = gt_image[:, ii]
            pre = pre_image[:, ii]
            intersection = np.sum(gt * pre)
            summed = np.sum(gt + pre)
            _dice[ii] += (2. * intersection + smooth) / (summed + smooth)

        return np.mean(_dice), _dice


    def add_batch(self, gt_image, pre_image):
        pre_image = self.__preprocess(pre_image, True)
        gt_image = self.__preprocess(gt_image)

        result = self._dice_coef(gt_image, pre_image)
        if 'dice' in self.metrics:
            self.mdice.append(result[0])
            self.dice_class.append(result[1])

    def reset(self):
        self.mdice.clear()
        self.dice_class.clear()

    def Dice_score(self):
        result = np.mean(self.mdice)
        return result

    def Dice_score_class(self):
        result = np.mean(np.array(self.dice_class), axis=0)
        return result
