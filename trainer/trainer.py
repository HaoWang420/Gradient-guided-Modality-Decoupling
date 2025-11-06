import itertools
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.optim import lr_scheduler

from tqdm import tqdm

from dataloaders import make_data_loader
from modeling import build_model
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.cuda_device = 0
        if args.distributed:
            self.cuda_device = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(self.cuda_device)
            torch.distributed.init_process_group(
                backend='nccl',
                rank=self.cuda_device, world_size=args.world_size
            )

        # Define Saver
        self.saver = None
        if self.cuda_device == 0:
            self.saver = Saver(args)
            self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = None
        self.writer = None

        if self.cuda_device == 0:
            self.summary = TensorboardSummary(self.saver.experiment_dir)
            self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass, self.nchannels = make_data_loader(
            args, **kwargs)
        print('number of classes: ', self.nclass)

        self._init_model(self.nchannels, self.nclass)

        # segmentation losss
        self._init_loss()

        # Define Evaluator
        self.evaluator = Evaluator(loss=args.loss.name, metrics=args.metrics)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.optim.lr,
                                      args.epochs, len(self.train_loader))

        # Resuming checkpoint
        self.resume()

        self.best_pred = 0

    def _init_model(self, nchannel, nclass):
        # Define network
        self.model = build_model(self.args.model,
                                 nclass=nclass,
                                 nchannels=nchannel,
                                 model=self.args.model.name)

        # Using cuda
        if self.args.cuda:
            self.model = self.model.cuda()
        if self.args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.cuda_device], output_device=[self.cuda_device])
        else:
            self.model = torch.nn.parallel.DataParallel(
                self.model, self.args.gpu_ids)

        train_params = None
        train_params = [
            {'params': self.model.parameters(), 'lr': self.args.optim.lr}]

        # Define Optimizer
        if self.args.optim.name == 'sgd':
            self.optimizer = torch.optim.SGD(
                train_params, momentum=self.args.optim.momentum, weight_decay=self.args.optim.weight_decay)
        elif self.args.optim.name == 'adam':
            self.optimizer = torch.optim.Adam(
                train_params, weight_decay=self.args.optim.weight_decay)

    def _init_loss(self, ):
        # segmentation losses
        self.criterion = SegmentationLosses(
            self.args,
            nclass=self.nclass,
        ).build_loss(mode=self.args.loss.name)

    def resume(self, ):
        # Resuming checkpoint
        self.best_pred = 0.0
        if self.args.resume is not None:
            if not os.path.isfile(self.args.resume):
                raise RuntimeError(
                    "=> no checkpoint found at '{}'" .format(self.args.resume))
            checkpoint = torch.load(
                self.args.resume, map_location=f'cuda:{self.cuda_device}')
            # self.args.start_epoch = checkpoint['epoch']
            if self.args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])

            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.args.resume, checkpoint['epoch']))
            self.args.start_epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])

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

            output, loss = self.forward_batch(image, target)

            train_loss += loss.item()

            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            if self.cuda_device == 0:
                self.writer.add_scalar(
                    'train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        if self.cuda_device == 0:
            print('[Epoch: {}]'.format(epoch))
            print('Loss: {:.3f}'.format(train_loss))
            self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)

        return train_loss / len(self.train_loader.dataset)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        num_img_tr = len(self.val_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']

            if self.args.cuda:
                image = image.cuda()
                target = target.cuda()

            with torch.no_grad():
                output = self.predict(image)

            # loss = self.criterion(output, target)

            # test_loss += loss.item()
            # tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            tbar.set_description(f'Val/Epoch {epoch}')

            # Add batch sample into evaluator
            self.evaluator.add_batch(
                target.cpu().numpy(), output.data.cpu().numpy())

            if self.args.distributed:
                dist.barrier()

        global_step = i + num_img_tr * epoch
        # self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        # Fast test during the training
        dice = self.evaluator.Dice_score()
        dice_class = self.evaluator.Dice_score_class()

        if self.args.distributed:
            dice, dice_class = self.gather_test_score(dice, dice_class)

        if self.cuda_device == 0:

            new_pred = dice
            is_best = False

            self.writer.add_scalar('val/dice', dice, epoch)
            self.writer.add_scalar('val/dice_WT', dice_class[0], epoch)
            self.writer.add_scalar('val/dice_TC', dice_class[1], epoch)
            self.writer.add_scalar('val/dice_ET', dice_class[2], epoch)

            if new_pred > self.best_pred:
                is_best = True
                self.best_pred = new_pred

                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, is_best)

            print('Validation:')
            print('[Epoch: %d, numImages: %5d]' %
                  (epoch, i * self.args.batch_size + image.data.shape[0]))
            print(f'Dice: {dice:.4f}')
            print('Loss: %.3f' % test_loss)

    def test(self, drop=None, epoch=0):
        dices = []
        dices_class = {}
        for l in reversed(range(self.nchannels)):
            for subset in itertools.combinations(list(range(self.nchannels)), l):
                dice, dice_class = self._test(drop=subset, epoch=epoch)
                dices.append(dice)
                dices_class[str(subset)] = dice_class
        if self.cuda_device == 0:
            self.save(np.mean(dices), dices_class, epoch)

    def _test(self, drop=[], epoch=0):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']

            if self.args.cuda:
                image = image.cuda()
                target = target.cuda()

            for d in drop:
                image[:, d] = 0

            with torch.no_grad():
                output = self.predict(image, channel=drop)

            pred = output
            # Add batch sample into evaluator
            self.evaluator.add_batch(
                target.cpu().numpy(), pred.data.cpu().numpy())

            if self.args.distributed:
                dist.barrier()

        drop = str(drop).replace(',', '_').replace(' ', '').removeprefix('(').removesuffix(')')

        dice = self.evaluator.Dice_score()
        dice_class = self.evaluator.Dice_score_class()

        if self.args.distributed:
            dice, dice_class = self.gather_test_score(dice, dice_class)

        if self.cuda_device == 0:

            self.writer.add_scalar(f'test/dice_drop{drop}', dice, epoch)
            self.writer.add_scalar(
                f'test/dice_WT_drop{drop}', dice_class[0], epoch)
            self.writer.add_scalar(
                f'test/dice_TC_drop{drop}', dice_class[1], epoch)
            self.writer.add_scalar(
                f'test/dice_ET_drop{drop}', dice_class[2], epoch)

            print(f'Evaluating with modality {drop} dropped')
            print(f'Dice: {dice:.4f}')
            print(f'Dice: {dice_class}')

        return dice, dice_class

    def gather_test_score(self, dice, dice_class):
        dice_list = [dice for _ in range(dist.get_world_size())]
        dice_class_list = [dice_class for _ in range(dist.get_world_size())]
        dist.all_gather_object(dice_list, dice)
        dist.all_gather_object(dice_class_list, dice_class)
        dice = np.mean(dice_list)
        dice_class = np.mean(np.stack(dice_class_list, axis=0), axis=0)
        return dice, dice_class

    def save(self, best_pred, dice_classes, epoch):
        new_pred = best_pred
        is_best = False

        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred

            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

            with open(os.path.join(self.saver.experiment_dir, 'results.csv'), 'w') as f:
                for ii in dice_classes:
                    missing = str(ii).replace(',', '_').replace(' ', '')
                    f.writelines('{},{:.3f},{:.3f},{:.3f}\n'.format(missing, *[i * 100 for i in dice_classes[ii]]))

    def predict(self, image, channel=-1):
        return self.model(x=image, channel=channel)

    def visualize(self, image, target, output, epoch, channel):
        pass

    def forward_batch(self, image, target):
        output = self.model(image)

        loss = self.criterion(output, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return output, loss
