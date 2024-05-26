import os
import torch
import numpy as np
from torch.utils import data

from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from tqdm.utils import FormatReplace
from dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self) -> SummaryWriter:
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        if dataset == 'brats':
            grid_image = make_grid(image[:, -1:].clone().cpu().data, 3, normalize=True)
            writer.add_image('Image', grid_image, global_step)
            grid_image = make_grid(target[:, -1:].clone().cpu().data.float() * 25., 3, normalize=False)
            writer.add_image('Groundtruth label', grid_image, global_step)
            grid_image = make_grid(np.argmax(output[:, None, ...].clone().cpu().data.float(), axis=1) * 25., 3, normalize=False)
            writer.add_image('Predicted label', grid_image, global_step)
        else:
            raise NotImplementedError('Visualization for {} not available.'.format(dataset))
    
    def visualize_param(self, writer: SummaryWriter, model, global_step):
        for name, param in model.module.named_parameters():
            if 'bn' not in name:
                writer.add_histogram(name, param, global_step)
    
    @staticmethod
    def __sigmoid(pred):
        pred = np.clip(pred, -88.72, 88.72)
        return 1 / (np.exp(-pred) + 1)