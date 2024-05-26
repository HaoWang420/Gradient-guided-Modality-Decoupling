from trainer import build_trainer
from dataloaders.datasets.brats import BraTSSet, BraTSVolume

import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
import hydra
from omegaconf import OmegaConf, DictConfig
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm import tqdm

from trainer.trainer import Trainer

@record
@hydra.main(config_path="config", config_name="config")
def main(args: DictConfig):
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # default settings for epochs, batch_size and lr
    assert args.epochs is not None
    assert args.batch_size is not None
    assert args.checkname is not None

    # torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trainer = build_trainer(args)

    if args.mode == 'train':
        train(trainer, args)
    elif args.mode == 'eval':
        trainer.test(epoch=0)
    elif args.mode == 'test':
        testing(trainer)

    if trainer.cuda_device == 0:
        trainer.writer.close()

    if args.distributed and dist.get_rank() == 0:
        dist.destroy_process_group()


def train(trainer, args):
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.test(epoch=epoch)


def testing(trainer):
    for jj in range(0, 4):
        for ii in range(-1, BraTSVolume.NMODALITY):
            test_skip_connection(trainer, ii, on_skip=jj)


@torch.no_grad()
def test_skip_connection(trainer: Trainer, channel=-1, on_skip=0):
    model = trainer.model.module

    model.eval()
    trainer.evaluator.reset()

    pbar = tqdm(trainer.val_loader, desc='\r')

    for ii, sample in enumerate(pbar):
        image, target = sample['image'], sample['label']
        if trainer.args.cuda:
            image = image.cuda()
            target = target.cuda()
        ffeat = encode(model, image)

        if channel != -1:
            image[:, channel] = 0

        dfeat = encode(model, image)

        # dfeat[3 - on_skip] = ffeat[3 - on_skip]
        for ii in range(len(dfeat)):
            dfeat[ii] = dfeat[ii] / 3. * 4.

        output = decode(model, dfeat)
        output = model.final_conv(output)

        trainer.evaluator.add_batch(target.cpu().numpy(), output.data.cpu().numpy())

        if trainer.args.distributed:
            dist.barrier()

    dice = trainer.evaluator.Dice_score()
    dice_class = trainer.evaluator.Dice_score_class()

    if trainer.args.distributed:
        trainer.gather_test_score(dice, dice_class)

    if trainer.cuda_device == 0:
        print(f'Testing with modality {channel} dropped, full feature at {on_skip}')
        print(f'Dice: {dice:.4f}')
        print(f'Dice: {dice_class}')


def finetune_weights(trainer):
    model = trainer.model
    train_loader = trainer.train_loader
    val_loader = trainer.val_loader
    weights = nn.Parameter(torch.ones(4, dtype=torch.float64, device=model.device), requires_grad=True)


@torch.no_grad()
def encode(model, x):
    encoder_features = []

    for encoder in model.encoders:
        x = encoder(x)
        encoder_features.insert(0, x)

    return encoder_features


@torch.no_grad()
def decode(model, x):
    encoder_features = x
    x = x[0]
    encoder_features = encoder_features[1:]

    for decoder, encoder_features in zip(model.decoders, encoder_features):
        x = decoder(encoder_features, x)

    return x


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "8"
    main()
