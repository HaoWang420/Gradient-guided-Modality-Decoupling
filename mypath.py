import os

from torch.utils.data.dataloader import DataLoader

DATASET_ROOT = './datasets/'


class Path(object):
    @staticmethod
    def getPath(dataset):
        if dataset == 'brats':
            path = os.path.join(DATASET_ROOT, 'braTS17_missing_modality')
        elif dataset == 'brats3d':
            path = os.path.join(DATASET_ROOT, 'BraTS18TrainingData')
        elif dataset == 'brats3d-acn':
            path = os.path.join(DATASET_ROOT, 'brats18_val/train')
        elif dataset == 'brats3d-val':
            path = os.path.join(DATASET_ROOT, 'brats18_val/val')
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

        return os.path.realpath(path)
