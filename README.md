# VFM-Guided Semi-Supervised Detection Transformer for Source-Free Object Detection in Remote Sensing Images

By Jianhong Han, Liang Chen and Yupei Wang.

This repository contains the implementation accompanying our paper VFM-Guided Semi-Supervised Detection Transformer for Source-Free Object Detection in Remote Sensing Images.


![](/figs/Figure1.png)

## Acknowledgment
This implementation is bulit upon [DINO](https://github.com/IDEA-Research/DINO/) and [RemoteSensingTeacher](https://github.com/h751410234/RemoteSensingTeacher).

## Installation
Please refer to the instructions [here](requirements.txt). We leave our system information for reference.

* OS: Ubuntu 16.04
* Python: 3.10.9
* CUDA: 11.8
* PyTorch: 2.0.1 (The lower versions of Torch can cause some bugs.)
* torchvision: 0.15.2

## Dataset Preparation
Please construct the datasets following these steps:

- Download the datasets from their sources.

- Convert the annotation files into COCO-format annotations.

- Modify the dataset path setting within the script [coco_self_training.py](./datasets/coco_self_training.py)

```
    PATHS = {
        "train": ('train_images',
                'train.json'),
        "val": ('val_images',
                'val.json'),
    }
```
# Stage 1 · Offline Extraction of Referenced Features & Prototypes

A lightweight **pre-processing pipeline** that prepares the target-domain data for all subsequent stages of our framework.

| Step | Script / Resource | Description |
|------|------------------|-------------|
| **1** | [`random_select_data.py`](./random_select_data.py) | Randomly selects a user-defined percentage of images from the complete target-domain dataset. |
| **2** | `extract_features_dinov2.py` <br><sup>(to be released after paper acceptance)</sup> | Extracts dense visual features for all selected images using **DINOv2**. |
| **3** | `offline_prototype_clustering.py` <br><sup>(to be released after paper acceptance)</sup> | Clusters the extracted features to generate **class-specific prototypes** for later stages. |

---

# Stage 2 · Training

*Details and code will be released once the paper is accepted.*

---

## Evaluation / Inference

We provide scripts for both evaluation and inference:

### Evaluation  
| Task | Command |
|------|---------|
| Standard model | `sh scripts/DINO_eval_model.sh` |
| EMA model | `sh scripts/DINO_eval_ema.sh` |

### Inference  
| Task | Command |
|------|---------|
| Standard model | `python inference_model.py` |
| EMA model | `python inference_ema_model.py` |

## 📦 Pre-trained Weights

Below are the mAP@50 scores and download links for models fine-tuned with different proportions of labeled target data.

### Cross-Satellite Adaptation

| Labeled Data (%) | mAP@50    | Model |
|------------------|-----------|-------|
| **1 %**  | **70.5%** | [Baidu Disk](https://pan.baidu.com/s/14UEWbQSKTF9tdTtaFaB_Lw?pwd=qa8r) <sub>(pwd: qa8r)</sub> |
| **5 %**  | **77.5%** | [Baidu Disk](https://pan.baidu.com/s/14UEWbQSKTF9tdTtaFaB_Lw?pwd=qa8r) <sub>(pwd: qa8r)</sub> |
| **10 %** | **78.4%** | [Baidu Disk](https://pan.baidu.com/s/14UEWbQSKTF9tdTtaFaB_Lw?pwd=qa8r) <sub>(pwd: qa8r)</sub> |

---

### Synthetic-to-Real Adaptation

| Labeled Data (%) | mAP@50    | Model |
|------------------|-----------|-------|
| **1 %**  | **60.7%** | [Baidu Disk](https://pan.baidu.com/s/14UEWbQSKTF9tdTtaFaB_Lw?pwd=qa8r) <sub>(pwd: qa8r)</sub> |
| **5 %**  | **65.9%** | [Baidu Disk](https://pan.baidu.com/s/14UEWbQSKTF9tdTtaFaB_Lw?pwd=qa8r) <sub>(pwd: qa8r)</sub> |
| **10 %** | **67.4%** | [Baidu Disk](https://pan.baidu.com/s/14UEWbQSKTF9tdTtaFaB_Lw?pwd=qa8r) <sub>(pwd: qa8r)</sub> |

### Cross-modal Adaptation

| Labeled Data (%) | mAP@50    | Model |
|------------------|-----------|-------|
| **1 %**  | **61.4%** | [Baidu Disk](https://pan.baidu.com/s/14UEWbQSKTF9tdTtaFaB_Lw?pwd=qa8r) <sub>(pwd: qa8r)</sub> |
| **5 %**  | **70.6%** | [Baidu Disk](https://pan.baidu.com/s/14UEWbQSKTF9tdTtaFaB_Lw?pwd=qa8r) <sub>(pwd: qa8r)</sub> |
| **10 %** | **78.0%** | [Baidu Disk](https://pan.baidu.com/s/14UEWbQSKTF9tdTtaFaB_Lw?pwd=qa8r) <sub>(pwd: qa8r)</sub> |




## Reference
https://github.com/IDEA-Research/DINO

https://github.com/h751410234/RemoteSensingTeacher