import os
import torch
import numpy as np
import numexpr as ne
import nibabel as nib
import SimpleITK as itk

from monai.transforms import *
from torch._C import dtype
from mypath import Path
from tqdm import tqdm


class BraTS3d(torch.utils.data.Dataset):
    NCLASS = 3
    NMODALITY = 4
    def __init__(self, 
                 root=Path.getPath('brats3d'), 
                 mode='train', 
                 indices=None, 
                 channel_shuffle=False, 
                 crop_size=[128, 128, 128]):

        self.mode = mode
        self.indices = indices
        self.channel_shuffle = channel_shuffle
        self.mod_paths = {
            't1': [],
            't1ce': [],
            't2': [],
            'flair': [],
            'seg': []
        }

        with os.scandir(root) as root_dir:
            for volume_dir in root_dir:
                if not volume_dir.is_dir():
                    continue
                with os.scandir(volume_dir.path) as modalities:
                    for mod in modalities:
                        # '.' to avoid overlapping between 't1' and 't1ce'
                        for mod_name in ['t1.', 't1ce.', 't2.', 'flair.', 'seg.']:
                            if mod.name.find(mod_name) != -1:
                                self.mod_paths[mod_name[:-1]].append(mod.path)
        
        if self.mode == 'train':
            self.transform = Compose(
                [
                    # CropForegroundd(keys=['image', 'label'], source_key='image', margin=20),
                    RandSpatialCropd(keys=['image', 'label'], roi_size=crop_size, random_size=False),
                    ToTensord(keys=['image'], dtype=torch.float32),
                    ToTensord(keys=['label'], dtype=torch.long)
                ]
            )
        elif self.mode == 'val':
            self.transform = Compose(
                [
                    CenterSpatialCropd(keys=['image', 'label'], roi_size=crop_size),
                    ToTensord(keys=['image'], dtype=torch.float32),
                    ToTensord(keys=['label'], dtype=torch.long),
                ]
            )

    def __getitem__(self, index, normalize=True):
        if self.indices != None:
            index = self.indices[index]
        img = []
        y = itk.GetArrayFromImage(itk.ReadImage(self.mod_paths['seg'][index])).transpose(2, 1, 0)
        # whole tumor
        y0 = y > 0
        # tumor core
        y1 = np.logical_or(y==1, y==4)
        # enhancing tumor
        y2 = y==4

        for mod in ['t1', 't1ce', 't2', 'flair']:
            data = itk.GetArrayFromImage(itk.ReadImage(self.mod_paths[mod][index])).transpose(2, 1, 0)
            # data = self.modalities[mod][index]
            if normalize:
                data = self.__normalize(data)
            img.append(data)
        
        if self.channel_shuffle and self.mode == 'train':
            np.random.shuffle(img)
        img = np.stack(img, axis=0)
        y = np.stack([y0, y1, y2], axis=0)
        data = {'image': img, 'label': y}

        data = self.transform(data)

        return data

    def __len__(self):
        if self.indices != None:
            return len(self.indices)
        return len(self.mod_paths['seg'])
    
    @staticmethod
    def __normalize(data):
        slices = np.nonzero(data)
        # percentile clipping
        p1 = np.percentile(data[slices], 1)
        p99 = np.percentile(data[slices], 99)
        data[slices] = np.clip(data[slices], p1, p99)

        # z-score normalization
        std = np.std(data[slices])
        std = 1 if std == 0 else std
        data[slices] = (data[slices] - np.mean(data[slices])) / std

        return data


class BraTS3dMem(torch.utils.data.Dataset):
    NCLASS = 3
    NMODALITY = 4
    def __init__(self, 
                 root=Path.getPath('brats3d'), 
                 mode='train', 
                 indices=None, 
                 channel_shuffle=False, 
                 crop_size=[128, 128, 128]):

        self.mode = mode
        self.indices = indices
        self.channel_shuffle = channel_shuffle
        self.mod_paths = {
            't1': [],
            't1ce': [],
            't2': [],
            'flair': [],
            'seg': []
        }

        with os.scandir(root) as root_dir:
            for volume_dir in root_dir:
                if not volume_dir.is_dir():
                    continue
                with os.scandir(volume_dir.path) as modalities:
                    for mod in modalities:
                        # '.' to avoid overlapping between 't1' and 't1ce'
                        for mod_name in ['t1.', 't1ce.', 't2.', 'flair.', 'seg.']:
                            if mod.name.find(mod_name) != -1:
                                self.mod_paths[mod_name[:-1]].append(mod.path)
        
        if self.mode == 'train':
            self.transform = Compose(
                [
                    # CropForegroundd(keys=['image', 'label'], source_key='image', margin=20),
                    RandSpatialCropd(keys=['image', 'label'], roi_size=crop_size, random_size=False),
                    ToTensord(keys=['image'], dtype=torch.float32),
                    ToTensord(keys=['label'], dtype=torch.long)
                ]
            )
        elif self.mode == 'val':
            self.transform = Compose(
                [
                    CenterSpatialCropd(keys=['image', 'label'], roi_size=crop_size),
                    ToTensord(keys=['image'], dtype=torch.float32),
                    ToTensord(keys=['label'], dtype=torch.long),
                ]
            )
        
        self.modalities = {}
        if self.indices is not None:
            for mod  in ['t1', 't1ce', 't2', 'flair', 'seg']:
                self.modalities[mod] = []
                for ii in tqdm(self.indices):
                    # self.modalities[mod].append(nib.load(self.mod_paths[mod][ii]).get_fdata())
                    image = itk.ReadImage(self.mod_paths[mod][ii])
                    arr = itk.GetArrayFromImage(image).transpose(2, 1, 0)
                    if mod == 'seg':
                        # whole tumor
                        y0 = arr > 0
                        # tumor core
                        y1 = np.logical_or(arr==1, arr==4)
                        # enhancing tumor
                        y2 = arr==4
                        arr = np.stack([y0, y1, y2], axis=0)
                    else:
                        arr = self.__normalize(arr)
                        # pass

                    self.modalities[mod].append(arr)

    def __getitem__(self, index, normalize=True):
        img = []
        # y = nib.load(self.mod_paths['seg'][index]).get_fdata()
        y = self.modalities['seg'][index]

        for mod in ['t1', 't1ce', 't2', 'flair']:
            data = self.modalities[mod][index]
            img.append(data)
        
        if self.channel_shuffle and self.mode == 'train':
            np.random.shuffle(img)
        img = np.stack(img, axis=0)
        data = {'image': img, 'label': y}

        data = self.transform(data)

        return data

    def __len__(self):
        if self.indices != None:
            return len(self.indices)
        return len(self.mod_paths['seg'])
    
    @staticmethod
    def __normalize(data):
        data = (data - data.min()) / (data.max() - data.min())

        return data