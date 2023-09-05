# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
from PIL import Image, ImageDraw, ImageChops
import torch.nn.functional as F
from easydict import EasyDict
import scipy.stats as st

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on
import tempfile
import random
import time
import warnings
import torch
import cv2
import numpy as np
import tqdm
import torch.nn as nn
import math

import seaborn as sns
import matplotlib.pylab as plt
from matplotlib import cm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision import transforms
from mask2former import add_maskformer2_config, SemanticSegmentorWithTTA
from demo.predictor import VisualizationDemo
from torch.autograd import Variable
from matplotlib import pyplot
from torchvision.utils import save_image
import kornia as K

from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/home/shyam/Mask2Former/configs/cityscapes/semantic-segmentation/test_unk_na.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="/home/shyam/Mask2Former/unk-eval/results/",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--aug",
        default=True,
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    
    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')
    
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    anomaly_score_list = []
    ood_gts_list = []


    if args.input:
        for path in glob.glob(os.path.expanduser(str(args.input[0]))):
            img = read_image(path, format="BGR")
            start_time = time.time()
            
            img_ud = np.flipud(img) # Aug 1

            img_lr = np.fliplr(img) # Aug 2
            
            predictions_na, _ = demo.run_on_image(img)
            predictions_lr, _ = demo.run_on_image(img_lr)

            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions_na["instances"]))
                    if "instances" in predictions_na
                    else "finished",
                    time.time() - start_time,
                )
            )
            


            if args.output:

                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output

                predictions_naa =  predictions_na["sem_seg"].unsqueeze(0)
                outputs_na = 1 - torch.max(predictions_naa[0:19,:,:], axis = 1)[0]
                if predictions_na["sem_seg"][19:,:,:].shape[0] > 1:
                    outputs_na_mask = torch.max(predictions_na["sem_seg"][19:,:,:].unsqueeze(0),  axis = 1)[0]
                    outputs_na_mask[outputs_na_mask < 0.5] = 0
                    outputs_na_mask[outputs_na_mask >= 0.5] = 1
                    outputs_na_mask = 1 - outputs_na_mask
                    outputs_na_save = outputs_na.clone().detach().cpu().numpy().squeeze().squeeze()
                    outputs_na = outputs_na*outputs_na_mask.detach()
                    outputs_na_mask = outputs_na_mask.detach().cpu().numpy().squeeze().squeeze()
                outputs_na = outputs_na.detach().cpu().numpy().squeeze().squeeze()

                #left-right
                predictions_lrr =  predictions_lr["sem_seg"].unsqueeze(0)
                outputs_lr = 1 - torch.max(predictions_lrr[0:19,:,:], axis = 1)[0]
                if predictions_lr["sem_seg"][19:,:,:].shape[0] > 1:
                    outputs_lr_mask = torch.max(predictions_lr["sem_seg"][19:,:,:].unsqueeze(0),  axis = 1)[0]
                    outputs_lr_mask[outputs_lr_mask < 0.5] = 0
                    outputs_lr_mask[outputs_lr_mask >= 0.5] = 1
                    outputs_lr_mask = 1 - outputs_lr_mask
                    outputs_lr_save = outputs_lr.clone()
                    outputs_lr = outputs_lr*outputs_lr_mask.detach()
                outputs_lr = outputs_lr.detach().cpu().numpy().squeeze().squeeze()
                outputs_lr = np.flip(outputs_lr.squeeze(), 1)

                
                outputs = np.expand_dims((outputs_lr + outputs_na )/2.0, 0).astype(np.float32)
                pathGT = path.replace("images", "labels_masks")                

                if "RoadObsticle21" in pathGT:
                   pathGT = pathGT.replace("webp", "png")
                if "fs_static" in pathGT:
                   pathGT = pathGT.replace("jpg", "png")                
                if "RoadAnomaly" in pathGT:
                   pathGT = pathGT.replace("jpg", "png")  

                mask = Image.open(pathGT)
                ood_gts = np.array(mask)

                if "RoadAnomaly" in pathGT:
                    ood_gts = np.where((ood_gts==2), 1, ood_gts)
                if "LostAndFound" in pathGT:
                    ood_gts = np.where((ood_gts==0), 255, ood_gts)
                    ood_gts = np.where((ood_gts==1), 0, ood_gts)
                    ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

                if "Streethazard" in pathGT:
                    ood_gts = np.where((ood_gts==14), 255, ood_gts)
                    ood_gts = np.where((ood_gts<20), 0, ood_gts)
                    ood_gts = np.where((ood_gts==255), 1, ood_gts)

                if 1 not in np.unique(ood_gts):
                    continue              
                else:
                     ood_gts_list.append(np.expand_dims(ood_gts, 0))
                     anomaly_score_list.append(outputs)

    file.write( "\n")
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)
    # drop void pixels
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    fpr, tpr, _ = roc_curve(val_label, val_out)    
    roc_auc = auc(fpr, tpr)
    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUROC score: {roc_auc}')
    print(f'AUPRC score: {prc_auc}')
    print(f'FPR@TPR95: {fpr}')

    file.write(('AUPRC score:' + str(prc_auc) + '   FPR@TPR95:' + str(fpr) ))
    file.close()