import numpy as np
from PIL import Image
import nibabel as nib
from skimage.io import imread, imsave
import os
import argparse

# label of the converted img
# 
T1 = 0
T2 = 1
T3 = 2
T4 = 4

def volume2slice(volume: dict, count, root_to_save):
    assert len(volume.keys()) == 5
    grey_count = 0.0

    # load and normalization
    for itype in volume.keys():
        data = nib.load(volume[itype]).get_fdata()
        data = data[27:213, 20:220, :]

        if itype != 'seg':

            slices = np.nonzero(data)
            # percentile clipping
            p1 = np.percentile(data[slices], 1)
            p99 = np.percentile(data[slices], 99)
            data[slices] = np.clip(data[slices], p1, p99)

            # z-score normalization
            std = np.std(data[slices])
            std = 1 if std == 0 else std
            data[slices] = (data[slices] - np.mean(data[slices])) / std
        else:
            data[data==4] = 3
        volume[itype] = data

    for ii in range(volume['t1'].shape[2]):
        grey_count = np.sum(volume['t1'][:, :, ii] != 0)
        if grey_count < 200:
            continue
        for itype in volume.keys():
            data = volume[itype]
            img_path = os.path.join(root_to_save, itype, 'volume_{:0>3d}/{:0>3d}.nii.gz'.format(count, ii))
            img = data[:, :, ii]

            # mkdir recursively
            os.makedirs(os.path.join(root_to_save, itype, f'volume_{count:0>3d}'), mode=0o777, exist_ok=True)

            # save to root_to_save/$itype/$itype_$count.jpg
            img = nib.Nifti1Image(img, np.eye(4))

            img.to_filename(img_path)

def brats_preprocess(root, root_to_save):
    with open(os.path.join(root, "name_mapping.csv")) as mapping:
        ii = 0
        # remove table heading
        line = mapping.readline()
        for line in mapping:
            for scan_type in ['flair', 't1', 't1ce', 't2', 'seg']:
                # obtain filenames from mapping 
                line = line.strip()
                name = line.split(',')[-1]
                data_path = os.path.join(root, name, name+'_'+ scan_type + '.nii.gz')

                volume2slice(data_path, ii, root_to_save, scan_type)
            ii = ii + 1
                
def brats2017_preprocess(root, root_to_save):
    ii = 0
    for level in ['HGG', 'LGG']:
        with os.scandir(os.path.join(root, level)) as subroot:
            for entry in subroot:
                if entry.is_dir():
                    with os.scandir(entry.path) as patient:
                        patient_dict = {}
                        for scan in patient:
                            name = scan.name.split('_')
                            if name[0] == 'ROI':
                                continue
                            scan_type = name[-1].split('.')
                            if scan_type[-1] != 'gz':
                                continue
                            scan_type = scan_type[0]
                            if scan_type in ['flair', 't1', 't1ce', 't2', 'seg']:
                                patient_dict[scan_type] = scan.path
                        volume2slice(patient_dict, ii, root_to_save)
                        ii = ii + 1

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="The path to root directory of BraTS17 dataset")
    parser.add_argument("--to-save", type=str, help="path for storing processed 2D slices")
    args = parser.parse_args()
    brats2017_preprocess(args.root, args.to_save)