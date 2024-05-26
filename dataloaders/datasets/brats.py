from os import scandir
import torch
import numpy as np
import nibabel as nib
import os
from PIL import Image
from torch.autograd.grad_mode import F
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from mypath import Path
from tqdm import tqdm

# L0 = 0
# L1 = 1
# L2 = 2
# L3 = 3

class BraTSSet(torch.utils.data.Dataset):
    """
    Brain Tumor Segmentation dataset loader for preprocessed 2D slices

    Args:
        root: the path to the root directory of dataset
    """
    NCLASS = 3
    NMODALITY = 4
    NSLICES = 96
    CLASS_FREQ = [0.98009453, 0.00489941, 0.01138634, 0.00361971]
    def __init__(self, path=Path.getPath('brats')):
        """
        """
        self.path = path 
        img_paths = []

        for ii in range(5):
            img_paths.append([])

        for ii, scan_type in enumerate(['t1', 't1ce', 't2', 'flair', 'seg']):
            with os.scandir(os.path.join(path, scan_type)) as img_dir:
                # volume_xxx
                for volume_dir in img_dir:
                    if volume_dir.is_dir():
                        with os.scandir(volume_dir.path) as v:
                            for slice in v:
                                img_paths[ii].append(slice.path)
    
        dataset = []
        for scan in img_paths:
            dataset.append(sorted(scan))
        
        self.imgs = dataset[:4]
        self.labels = dataset[4]

    def __getitem__(self, index):
        img_path = []
        for ii in range(self.NMODALITY):
            img_path.append(self.imgs[ii][index])
        label_path = self.labels[index]

        x = [nib.load(a).get_fdata().astype(np.float) for a in img_path]
        y = nib.load(label_path).get_fdata()

        x = np.stack(x, axis=0)

        # get regions of brats
        y = self.transform_label(y)
        
        sample = self._transform(x, y)

        return sample

    def _transform(self, x, y):
        composed = transforms.Compose([
            ]) 

        return composed({'image': torch.from_numpy(x).float(), 'label': torch.from_numpy(y).long()})

    @staticmethod
    def transform_label(y):
        # whole tumor
        y0 = y > 0
        # tumor core
        y1 = np.logical_or(y==1, y==3)
        # enhancing tumor
        y2 = y==3

        if y.ndim == 4:
            return np.stack([y0, y1, y2], axis=1)
        else:
            return np.stack([y0, y1, y2], axis=0)
    
    def class_weights(self, ):
        freq = np.zeros(self.NCLASS, dtype=np.long)
        dataset = DataLoader(self, batch_size=64, num_workers=4, shuffle=False)

        for ii in tqdm(dataset):
            for jj in range(self.NCLASS):
                freq[jj] += torch.sum(ii['label'] == jj)

        freq = freq / np.sum(freq)
        return freq

    def __len__(self):
        return len(self.labels)

class BraTSVolume(torch.utils.data.Dataset):
    """
    Brain Tumor Segmentation dataset loader for preprocessed 2D slices, \n
        for volume split

    Args:
        root: the path to the root directory of dataset
    """
    NCLASS = 3
    NMODALITY = 4
    NVOLUME = 284
    CLASS_FREQ = [0.98009453, 0.00489941, 0.01138634, 0.00361971]
    def __init__(self, path=Path.getPath('brats'), indices=None):
        """
        """
        self.path = path 
        volume_paths = []

        for ii in range(5):
            volume_paths.append([])

        for ii, scan_type in enumerate(['t1', 't1ce', 't2', 'flair', 'seg']):
            with os.scandir(os.path.join(path, scan_type)) as img_dir:
                # volume_xxx
                for volume_dir in img_dir:
                    if volume_dir.is_dir():
                        with os.scandir(volume_dir.path) as v:
                            slices = []
                            for slice in v:
                                slices.append(slice.path)
                            volume_paths[ii].append(slices)
    
        dataset = []
        for scan_dir in volume_paths:
            scans = []
            for volume in scan_dir:
                scans.append(sorted(volume))
            dataset.append(sorted(scans, key=lambda x:x[0]))
        
        self.imgs = []
        self.labels = []
        for ii in range(self.NMODALITY):
            self.imgs.append([])

        if indices is not None:
            for index in indices:
                for ii in range(self.NMODALITY):
                    self.imgs[ii] += dataset[ii][index]
                # segmentation mask
                self.labels += dataset[self.NMODALITY][index]


    def __getitem__(self, index):
        img_path = []
        for ii in range(self.NMODALITY):
            img_path.append(self.imgs[ii][index])
        label_path = self.labels[index]

        x = [nib.load(a).get_fdata().astype(np.float) for a in img_path]
        y = nib.load(label_path).get_fdata()

        x = np.stack(x, axis=0)

        # get regions of brats
        y = self.transform_label(y)
        
        sample = self._transform(x, y)

        return sample

    def _transform(self, x, y):
        composed = transforms.Compose([
            ]) 

        return composed({'image': torch.from_numpy(x).float(), 'label': torch.from_numpy(y).long()})

    @staticmethod
    def transform_label(y):
        # whole tumor
        y0 = y > 0
        # tumor core
        y1 = np.logical_or(y==1, y==3)
        # enhancing tumor
        y2 = y==3
    
        if y.ndim == 4:
            return np.stack([y0, y1, y2], axis=1)
        else:
            return np.stack([y0, y1, y2], axis=0)
    
    def class_weights(self, ):
        freq = np.zeros(self.NCLASS, dtype=np.long)
        dataset = DataLoader(self, batch_size=64, num_workers=4, shuffle=False)

        for ii in tqdm(dataset):
            for jj in range(self.NCLASS):
                freq[jj] += torch.sum(ii['label'] == jj)

        freq = freq / np.sum(freq)
        return freq

    def __len__(self):
        return len(self.labels)