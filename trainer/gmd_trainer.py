import torch
import numpy as np

from tqdm import tqdm

from trainer.trainer import Trainer
from utils.loss import SegmentationLosses
from optim.gmd import GMD


class GMDTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        if args.trainer.method == 'gmd':
            self.gmd = GMD(self.optimizer, reduction='mean', writer=self.writer)
        else:
            raise ValueError('invalid optim method')

    def _init_loss(self, ):
        # segmentation losss
        self.criterion = SegmentationLosses(
            self.args,
            nclass=self.nclass,
        ).build_loss(mode=self.args.loss.name)

    def training(self, epoch):
        self.model.train()

        train_loss = 0.0

        if self.args.distributed:
            self.train_loader.sampler.set_epoch(epoch)
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image = image.cuda()
                target = target.cuda()
            self.scheduler(self.optimizer, epoch, i, self.best_pred)
            self.gmd.zero_grad()

            output, loss = self.forward_batch(image, target)

            # pcgrad backward
            self.gmd.pc_backward(loss, self.model)

            self.gmd.step()
            losses = [l.item() for l in loss]
            loss = np.mean([l.item() for l in loss])
            train_loss += loss

            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            if self.cuda_device == 0:
                self.writer.add_scalar(
                    'train/total_loss_iter', loss, i + num_img_tr * epoch)
                for ii, l in enumerate(losses):
                    self.writer.add_scalar(f'modality/loss_{ii}_iter', l, i + num_img_tr * epoch)

        if self.cuda_device == 0:
            print('[Epoch: {}]'.format(epoch))
            print('Loss: {:.3f}'.format(train_loss))
            self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)

    def forward_batch(self, image, target):
        output = self.model(image)

        loss = self.criterion(output, target)

        return output, loss
