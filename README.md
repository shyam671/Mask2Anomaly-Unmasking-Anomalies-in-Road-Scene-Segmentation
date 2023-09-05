# [**ICCV'23 Oral**] Unmasking Anomalies in Road-Scene Segmentation 
[[`arXiv`](https://arxiv.org/abs/2307.13316)]

### Installation
Please follow the [Installation Instruction](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md) to set up the codebase.

### Datasets
* **Inlier Dataset(Cityscapes):** can be prepared by following the same structure as given [here](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md).
* **Outlier Supervision Dataset(MS-COCO):** is created by using [this script](https://github.com/robin-chan/meta-ood/blob/master/preparation/prepare_coco_segmentation.py) and change the ``cfg.MODEL.MASK_FORMER.ANOMALY_FILEPATH`` accordingly.
* **Anomaly Dataset (validation):** can be downloaded using this [link](https://drive.google.com/file/d/1r2eFANvSlcUjxcerjC8l6dRa0slowMpx/view?usp=share_link). Please unzip the file and place it accordingly.

### Training and Inference

* We provide all the commands for training, ood-fine tune, and anomaly inference in ``run.sh``. 

