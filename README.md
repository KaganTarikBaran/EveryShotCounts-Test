# EveryShotCounts - Testing Only Version

This repository is a **modified version** of the original **EveryShotCounts** repository. The original code was published as part of the paper **"Every Shot Counts: Using Exemplars for Repetition Counting in Videos"** by:

- [Saptarshi Sinha](https://sinhasaptarshi.github.io)
- [Alexandros Stergiou](https://alexandrosstergiou.github.io)
- [Dima Damen](https://dimadamen.github.io)

[Published at ACCV 2024](https://accv2024.org) | Links: [[arXiv]](https://arxiv.org/abs/2403.18074) [[Original Repo]](https://github.com/sinhasaptarshi/EveryShotCounts)

![Python](https://img.shields.io/badge/python-3.x-brightgreen/?style=flat&logo=python&color=green)
![Library](https://img.shields.io/badge/library-PyTorch-blue/?style=flat&logo=pytorch&color=informational)
![GitHub license](https://img.shields.io/cocoapods/l/AFNetworking)

## About This Version
This repository has been **modified for testing purposes only**. Training-related functionalities have been removed. The goal of this version is to allow easy evaluation of repetition counting models without requiring additional training steps.

## Installation
### 1️⃣ Create and activate a virtual environment
Run the following commands to set up a virtual environment:
```bash
conda create -n repcount python=3.8
conda activate repcount
```

### 2️⃣ Install dependencies
All required dependencies are listed in `requirements.txt`. To install them, run:
```bash
pip install -r requirements.txt
```

### 3️⃣ Install Additional Dependencies
Since some dependencies require special installation, install them separately:
```bash
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install -e git+https://github.com/facebookresearch/pytorchvideo.git@fae0d89a194a2c1ca99e59eab6eedd40bde38726#egg=pytorchvideo
mim install mmcv-full
```

## Dataset & Pretrained Model Download
### 📂 **Dataset**
Place datasets in the following directories:
- **RepCount dataset:** `data/RepCount/`
- **UCF101 dataset:** `data/UCFRep/`

Datasets can be downloaded from:
- [RepCount dataset](https://svip-lab.github.io/dataset/RepCount_dataset.html)
- [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php)

### 🔥 **Pretrained Model**
Download the pretrained model and place it in the `models/pretrained_models` directory.
- [VideoMAE pretrained encoder](https://dl.fbaipublicfiles.com/pyslowfast/masked_models/VIT_B_16x4_MAE_PT.pyth)

## Running the Model for Testing
Use the `test_model.py` script to evaluate the model on a given video:

```bash
python test_model.py --dataset 'RepCount' --resource 'cpu'
```

Modify the `--resource` argument to `'cuda'` if running on GPU.

### Expected Output
The script will output:
```
Overall MAE: X
OBO: X
```

## Citation
If you use this repository, please consider citing the original paper:

```
@InProceedings{sinha2024every,
title = {Every Shot Counts: Using Exemplars for Repetition Counting in Videos},
author = {Sinha, Saptarshi and Stergiou, Alexandros and Damen, Dima},
booktitle={Proceedings of the Asian conference on computer vision (ACCV)},
year = {2024},
}
```

---
This is a **testing-focused** fork of EveryShotCounts. The original repository can be found [here](https://github.com/sinhasaptarshi/EveryShotCounts).