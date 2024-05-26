# Gradient-guided-Modality-Decoupling-for-Missing-Modality-Robustness
Official implementatation for paper: Gradient-guided Modality Decoupling for Missing Modality Robustness

## Environment
The required libraries are listed in `environment.yml`
```
cond create -n gmd -f environment.yml
```
## Data preparation
download [BraTS18](https://www.med.upenn.edu/sbia/brats2018/registration.html) and modify paths in `mypath.py`

## training & eval
run `sh cli/train_brats3d_gmd.sh`
