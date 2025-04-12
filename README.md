# Gradient-guided-Modality-Decoupling-for-Missing-Modality-Robustness
Official implementation for paper: Gradient-guided Modality Decoupling for Missing Modality Robustness

## Environment
The required libraries are listed in `environment.yml`
```
conda env create -n gmd -f environment.yml
```
## Data preparation
download [BraTS18](https://www.med.upenn.edu/sbia/brats2018/registration.html) and modify paths in `mypath.py`

The structure of the dataset directory should be as follows:
```
<data_root>/<DATA_NAME>/*/case_name/*_flair.nii.gz      
<data_root>/<DATA_NAME>/*/case_name/*_t1.nii.gz   
<data_root>/<DATA_NAME>/*/case_name/*_t1ce.nii.gz   
<data_root>/<DATA_NAME>/*/case_name/*_flair.nii.gz
<data_root>/<DATA_NAME>/*/case_name/*_seg.nii.gz     # groundtruth 
```

## training & eval
run `sh cli/train_brats3d_gmd.sh`
