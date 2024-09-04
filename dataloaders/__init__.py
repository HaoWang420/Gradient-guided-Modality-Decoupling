import numpy as np

from torch.utils.data import sampler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mypath import Path
from dataloaders.datasets.brats import BraTSSet, BraTSVolume
from dataloaders.datasets import SplitWrapper
from dataloaders.datasets.brats3d import BraTS3d, BraTS3dMem
from dataloaders.datasets.brats_acn import split_dataset, Brats2018

def make_data_loader(args, **kwargs):

    if args.dataset.name == 'brats':

        dataset = BraTSSet()
        num_class = dataset.NCLASS
        num_channels = dataset.NMODALITY

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.1 * dataset_size))
        np.random.shuffle(indices)

        train_ratio = int((1 - args.dataset.val_ratio) * 10)

        train_indices, val_indices = indices[:(train_ratio * split)], indices[(train_ratio * split):]

        train_set = SplitWrapper(dataset=dataset, indices=train_indices)
        val_set = SplitWrapper(dataset=dataset, indices=val_indices)


    elif args.dataset.name == 'brats-volume':

        nvolume = BraTSVolume.NVOLUME
        num_class = BraTSVolume.NCLASS
        num_channels = BraTSVolume.NMODALITY
        indices = list(range(nvolume))

        train_ratio = int(np.floor((1 - args.dataset.val_ratio) * nvolume))
        val_ratio = nvolume - train_ratio

        np.random.shuffle(indices)

        train_indices, val_indices = indices[:train_ratio], indices[train_ratio:]

        train_set = BraTSVolume(indices=train_indices)
        val_set = BraTSVolume(indices=val_indices)
    
    elif args.dataset.name == 'brats3d':
        num_class = BraTS3d.NCLASS
        num_channels = BraTS3d.NMODALITY

        dataset = BraTS3d()
        nvolume = len(dataset)
        indices = list(range(nvolume))

        train_ratio = int(np.floor((1 - args.dataset.val_ratio) * nvolume))
        val_ratio = nvolume - train_ratio

        np.random.shuffle(indices)

        train_indices, val_indices = indices[:train_ratio], indices[train_ratio:]

        train_set = BraTS3d(indices=train_indices, mode='train', crop_size=args.dataset.crop_size)
        val_set = BraTS3d(indices=val_indices, mode='val', crop_size=args.dataset.val_size)

    elif args.dataset.name == 'brats3d-acn':
        num_channels = 4
        num_class = 3
        print(Path.getPath('brats3d-acn'))
        train_list, val_list = split_dataset(Path.getPath('brats3d-acn'), 5, 0)
        train_set = Brats2018(train_list, crop_size=args.dataset.crop_size, modes=("t1", "t1ce", "t2", "flair"), train=True)
        val_set = Brats2018(val_list, crop_size=args.dataset.val_size, modes=("t1", "t1ce", "t2", "flair"), train=False)

    elif args.dataset.name == 'brats3d-val':
        num_channels = 4
        num_class = 3
        patients_dir = glob.glob(os.path.join(Path.getPath(args.dataset.name), "Brats18*"))
        train_set = Brats2018(patients_dir, crop_size=args.dataset.val_size, modes=("t1", "t1ce", "t2", "flair"), train=False, has_label=False)
        val_set = Brats2018(patients_dir, crop_size=args.dataset.val_size, modes=("t1", "t1ce", "t2", "flair"), train=False, has_label=False)
        
    else:
        raise NotImplementedError
    
    if args.distributed:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        val_sampler = DistributedSampler(val_set, shuffle=False)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size, 
            num_workers=args.workers,
            pin_memory=False,
            sampler=train_sampler,
            prefetch_factor=2,
            )
        val_loader = DataLoader(
            val_set,
            batch_size=args.test_batch_size, 
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            prefetch_factor=2,
            )
        test_loader = None
    else:
        train_loader = DataLoader(
                train_set, 
                batch_size=args.batch_size, 
                num_workers=args.workers,
                shuffle=True)
        val_loader = DataLoader(
                val_set, 
                batch_size=args.test_batch_size, 
                num_workers=args.workers,
                shuffle=False)
        test_loader = None 
    
    return train_loader, val_loader, test_loader, num_class, num_channels
