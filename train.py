from trainer import build_trainer
from dataloaders.datasets.brats import BraTSSet, BraTSVolume

import os
import numpy as np
import torch
import itertools
import nibabel as nib

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
    elif args.mode == 'save_seg':
        evaluate_and_save_segmentation_maps(trainer, args)

    if trainer.cuda_device == 0:
        trainer.writer.close()

    if args.distributed and dist.get_rank() == 0:
        dist.destroy_process_group()

# evaluate the model under all missing modality scenarios on the validation set and save the segmentation results
@torch.no_grad()
def evaluate_and_save_segmentation_maps(trainer, args):
    trainer.model.eval()
    trainer.evaluator.reset()

    save_dir = os.path.join(trainer.saver.experiment_dir, 'segmentation_results')
    os.makedirs(save_dir, exist_ok=True)

    pbar = tqdm(trainer.val_loader, desc='\r')

    # disable pre dice sigmoid/softmax
    trainer.evaluator.preprocess = False
    for l in reversed(range(trainer.nchannels)):
        for subset in itertools.combinations(list(range(trainer.nchannels)), l):
            trainer.evaluator.reset()
            for i, sample in enumerate(pbar):
                # image, target = sample['image'], sample['label']
                image = sample['image']
                if trainer.args.cuda:
                    image = image.cuda()

                # forward pass with all missing modality scenarios
                image_copy = image.clone()
                for j in subset:
                    image_copy[:, j] = 0
                output = trainer.model(x=image_copy, channel=subset)
                pred = output.data.cpu().numpy()

                # pred_thres = sigmoid(pred) > 0.2
                # result = np.zeros((pred_thres.shape[0], *pred_thres.shape[2:]))
                # result[pred_thres[:, 0]==1] = 2
                # result[pred_thres[:, 1]==1] = 1
                # result[pred_thres[:, 2]==1] = 4

                # wt = result > 0
                # tc = np.logical_or(result==1, result==4)
                # et = result==4

                # result = np.stack([wt, tc, et], axis=1).astype("float32")

                # # evaluate threshold result
                # trainer.evaluator.add_batch(
                #     target.cpu().numpy(), 
                #     result,
                #     # pred,
                # )

                # Save each sample's segmentation maps as nii.gz files
                for j in range(pred.shape[0]):
                    pred_map = sigmoid(pred[j]) > 0.2
                    result = np.zeros(pred_map.shape[1:])
                    result[pred_map[0]==1] = 2
                    result[pred_map[1]==1] = 1
                    result[pred_map[2]==1] = 4
                    

                    # padding to sample['origin_shape']
                    origin_shape = sample['origin_shape']

                    # pad size at each dimension
                    pad_size = [(origin_shape[i][j].item() - result.shape[i]) // 2 for i in range(3)]
                    result = np.pad(result, ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2], pad_size[2])))
                    # print(result.shape)
                    result = result.astype(np.uint8)

                    save_path = os.path.join(save_dir, 
                            f'missing_{str(subset).replace(" ", "_").replace("(", "").replace(")", "")}', 
                            f'{sample["filename"][j]}.nii.gz')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    nib.save(nib.Nifti1Image(result, np.eye(4)), save_path)
            
            # dice = trainer.evaluator.Dice_score()
            # dice_class = trainer.evaluator.Dice_score_class()

            print(f'Missing modality: {subset}')
            # print(f'Dice: {dice:.4f}')
            # print(f'Dice: {dice_class}')

def sigmoid(x):
    # prevent numerical overflow
    x = np.clip(x, -88.72, 88.73)

    return 1 / (1 + np.exp(-x))

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
