# [**ICCV'23 Oral**] Unmasking Anomalies in Road-Scene Segmentation 
[[`arXiv`](https://arxiv.org/abs/2307.13316)]

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unmasking-anomalies-in-road-scene/anomaly-detection-on-lost-and-found)](https://paperswithcode.com/sota/anomaly-detection-on-lost-and-found?p=unmasking-anomalies-in-road-scene)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unmasking-anomalies-in-road-scene/scene-segmentation-on-streethazards)](https://paperswithcode.com/sota/scene-segmentation-on-streethazards?p=unmasking-anomalies-in-road-scene)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unmasking-anomalies-in-road-scene/anomaly-detection-on-fishyscapes-1)](https://paperswithcode.com/sota/anomaly-detection-on-fishyscapes-1?p=unmasking-anomalies-in-road-scene)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unmasking-anomalies-in-road-scene/anomaly-detection-on-road-anomaly)](https://paperswithcode.com/sota/anomaly-detection-on-road-anomaly?p=unmasking-anomalies-in-road-scene)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unmasking-anomalies-in-road-scene/anomaly-detection-on-fishyscapes-l-f)](https://paperswithcode.com/sota/anomaly-detection-on-fishyscapes-l-f?p=unmasking-anomalies-in-road-scene)

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
* The training of the model takes place in two stages:
  + [Stage1] Training of inlier dataset performed using the command: Cityscapes Dataset: ``CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1  --config-file configs/cityscapes/semantic-segmentation/anomaly_train.yaml``. Streethazard Dataset: ``CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 --config-file 'configs/streethazard/streethazard_training.yaml'``.
  + [Stage2] We fine-tune the weights of the model using following command: Cityscapes Dataset: ``CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1  --config-file configs/cityscapes/semantic-segmentation/anomaly_ft.yaml``. Streethazard Dataset: ``CUDA_VISIBLE_DEVICES=1 python train_net.py --num-gpus 1 --config-file 'configs/streethazard/streethazard_ft.yaml' ``.
* During inference, we use the final-weights obtained after fine-tuning: ``CUDA_VISIBLE_DEVICES=1 python anomaly_utils/anomaly_inference.py --input '/path-to-images-files/*.jpg' --config-file '/path-to-anomaly-inference-config/anomaly_inference.yaml'``

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

## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The majority of Mask2Anomaly is licensed under a [MIT License](LICENSE).


However portions of the project are available under separate license terms: Mask2Former and Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

### Acknowledgement

We thank the authors of the codebases mentioned below, which helped build the repository.
* [Meta-OOD](https://github.com/robin-chan/meta-ood)
* [PEBEL](https://github.com/tianyu0207/PEBAL/)
* [Mask2Former](https://github.com/facebookresearch/Mask2Former/tree/main)


