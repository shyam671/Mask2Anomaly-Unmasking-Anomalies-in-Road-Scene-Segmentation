# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F
import glob, random
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

__all__ = ["MaskFormerSemanticDatasetMapper"]

#Source: https://github.com/tianyu0207/PEBAL/blob/main/code/dataset/data_loader.py
def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """

    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)

    for i in range(mask.shape[-1]):

        m = mask[:, :, i]

        # Bounding box.

        horizontal_indicies = np.where(np.any(m, axis=0))[0]

        vertical_indicies = np.where(np.any(m, axis=1))[0]

        if horizontal_indicies.shape[0]:

            x1, x2 = horizontal_indicies[[0, -1]]

            y1, y2 = vertical_indicies[[0, -1]]

            # x2 and y2 should not be part of the box. Increment by 1.

            x2 += 1

            y2 += 1

        else:

            # No mask for this instance. Might happen due to

            # resizing or cropping. Set bbox to zeros

            x1, x2, y1, y2 = 0, 0, 0, 0

        boxes[i] = np.array([y1, x1, y2, x2])

    return boxes.astype(np.int32)

#Source: https://github.com/tianyu0207/PEBAL/blob/main/code/dataset/data_loader.py
def mix_object(current_labeled_image=None, current_labeled_mask=None, cut_object_image=None, cut_object_mask=None):

    train_id_out = 254

    cut_object_mask[cut_object_mask == train_id_out] = 254

    mask = cut_object_mask == 254

    ood_mask = np.expand_dims(mask, axis=2)
    ood_boxes = extract_bboxes(ood_mask)
    ood_boxes = ood_boxes[0, :]  # (y1, x1, y2, x2)
    y1, x1, y2, x2 = ood_boxes[0], ood_boxes[1], ood_boxes[2], ood_boxes[3]
    cut_object_mask = cut_object_mask[y1:y2, x1:x2]
    cut_object_image = cut_object_image[y1:y2, x1:x2, :]

    mask = cut_object_mask == 254

    idx = np.transpose(np.repeat(np.expand_dims(cut_object_mask, axis=0), 3, axis=0), (1, 2, 0))

    if mask.shape[0] != 0:
        h_start_point = random.randint(0, current_labeled_mask.shape[0] - cut_object_mask.shape[0])
        h_end_point = h_start_point + cut_object_mask.shape[0]
        w_start_point = random.randint(0, current_labeled_mask.shape[1] - cut_object_mask.shape[1])
        w_end_point = w_start_point + cut_object_mask.shape[1]
    else:
        h_start_point = 0
        h_end_point = 0
        w_start_point = 0
        w_end_point = 0

    current_labeled_image[h_start_point:h_end_point, w_start_point:w_end_point, :][np.where(idx == 254)] = \
        cut_object_image[np.where(idx == 254)]
    current_labeled_mask[h_start_point:h_end_point, w_start_point:w_end_point][np.where(cut_object_mask == 254)] = \
        cut_object_mask[np.where(cut_object_mask == 254)]

    return current_labeled_image, current_labeled_mask

class MaskFormerSemanticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        is_ood_ft,
        anomaly_mix_ratio,
        anomaly_file_path,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.is_ood_ft = is_ood_ft
        self.anomaly_mix_ratio = anomaly_mix_ratio
        self.anomaly_file_path = [anomaly_file_path]
        self.gt_list = glob.glob(self.anomaly_file_path)

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "is_ood_ft": cfg.MODEL.MASK_FORMER.OOD_FINETUNE,
            "anomaly_mix_ratio": cfg.MODEL.MASK_FORMER.ANOMALY_MIX_RATIO,
            "anomaly_file_path": cfg.MODEL.MASK_FORMER.ANOMALY_FILEPATH,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        #Anomaly Mix#
        if self.is_ood_ft: 
            if np.random.uniform() < self.anomaly_mix_ratio:
                coco_gt_path = random.choice(self.gt_list)
                coco_img_path = coco_gt_path.replace('ood_annotations','images')
                coco_img_path = coco_img_path.replace('png','jpg')
                coco_img = utils.read_image(coco_img_path, format=self.img_format)
                coco_gt = utils.read_image(coco_gt_path).astype("double")
                image, sem_seg_gt = mix_object(current_labeled_image=image, current_labeled_mask=sem_seg_gt, cut_object_image=coco_img, cut_object_mask=coco_gt)

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]

            if self.is_ood_ft:         
                classes = classes[classes != 254]

            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            ood_masks = []

            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if self.is_ood_ft:  
                ood_masks.append(sem_seg_gt == 254)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

            if self.is_ood_ft:         
                if len(ood_masks) == 0:
                    # Some image does not have annotation (all ignored)
                    ood_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
                else:
                    ood_masks = BitMasks(
                        torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in ood_masks])
                    )
                    ood_masks = ood_masks.tensor
                dataset_dict["ood_mask"] = ood_masks

        return dataset_dict
