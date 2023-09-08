# [**ICCV'23 Oral**] Unmasking Anomalies in Road-Scene Segmentation 
[[`arXiv`](https://arxiv.org/abs/2307.13316)]

https://github.com/shyam671/Mask2Anomaly-Unmasking-Anomalies-in-Road-Scene-Segmentation/assets/17329284/096effcd-51c8-4b1b-9b2b-f6746d4f6437


Run our demo using Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iMF5lWj3J8zlIJFkekXC3ipQo2semJfL?usp=sharing)

### Installation
Please follow the [Installation Instruction](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md) to set up the codebase.

### Datasets
We have three different sets of dataset used for training, ood-fine-tuning, and anomaly inference. Please follow the below steps to set-up each set. 
* **Inlier Dataset(Cityscapes/Streethazard):** consists of only inlier classes that can be prepared by following the same structure as given [here](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md).
* **Outlier Supervision Dataset(MS-COCO):** helps fine-tune the model pre-trained on the inlier dataset on ood-objects. The outlier dataset is created by using [this script](https://github.com/robin-chan/meta-ood/blob/master/preparation/prepare_coco_segmentation.py) and then changing the ``cfg.MODEL.MASK_FORMER.ANOMALY_FILEPATH`` accordingly.
* **Anomaly Dataset (validation):** can be downloaded using this [link](https://drive.google.com/file/d/1r2eFANvSlcUjxcerjC8l6dRa0slowMpx/view?usp=share_link). Please unzip the file and place it preferably in the dataset folder.

### Training and Inference
* We provide all the commands for training, ood-fine-tune, and anomaly inference in ``run.sh`` and corresponding config files at ``/configs/cityscapes
/semantic-segmentation/``.
* To perform anomaly segmentation using pre-trained models, download the model from shared Google Drive link and then change the model weight path in ``/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml``.
  +  Single Pretained model for SMIYC, Fishyscapes, LostAndFound, and Road Anomaly: [Link](https://drive.google.com/file/d/1mlLYq8ADU7hDyKQdzCtXAkMb7tKIPO-H/view?usp=share_link)
  +  StreetHazard: [Link](https://drive.google.com/file/d/1s_ctryZtFmawXkU2nWqSm6lXkeJkOrxg/view?usp=share_link)

### Docker Image
*  We provide a singularity image similar to docker that provides anomaly output without needing library installation/GPUs.
*  Install Singularity following the instructions. [Link](https://singularity-admindoc.readthedocs.io/en/latest/admin_quickstart.html)
*  Download the .sif image from [link](https://drive.google.com/file/d/1djIP2PelLyzNgfWIzg78eq51t3ZAssO_/view?usp=share_link)
* Run the command ``singularity run --bind {input-path-to-image-datset}:/input,{output-path-to-save-segmentation-maps}:/output mask2former.sif``

### Acknowledgement

We thank the authors of the codebases mentioned below, which helped build the repository.
* [Meta-OOD](https://github.com/robin-chan/meta-ood)
* [PEBEL](https://github.com/tianyu0207/PEBAL/)
* [Mask2Former](https://github.com/facebookresearch/Mask2Former/tree/main)


