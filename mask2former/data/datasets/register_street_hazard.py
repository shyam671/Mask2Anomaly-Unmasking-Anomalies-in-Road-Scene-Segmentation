# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

#colors = np.array([[ 0,   0,   0],# // unlabeled     =   0,
#        [ 70,  70,  70],# // building      =   1,
#        [190, 153, 153],# // fence         =   2,
#        [250, 170, 160],# // other         =   3,
#        [220,  20,  60],# // pedestrian    =   4,
#        [153, 153, 153],# // pole          =   5,
#        [157, 234,  50],# // road line     =   6,
#        [128,  64, 128],# // road          =   7,
#        [244,  35, 232],# // sidewalk      =   8,
#        [107, 142,  35],# // vegetation    =   9,
#        [  0,   0, 142],# // car           =  10,
#        [102, 102, 156],# // wall          =  11,
#        [220, 220,   0],# // traffic sign  =  12,
#        [ 60, 250, 240],# // anomaly       =  13,
#
#        ])                

STREET_HAZARD_SEM_SEG_CATEGORIES = [
    {
        "color": [180, 165, 180],
        "instances": False,
        "readable": "Background",
        "name": "Background",
        "evaluate": True,
    }, 
    {
        "color": [0, 192, 0],
        "instances": True,
        "readable": "Buildings",
        "name": "Buildings",
        "evaluate": True,
    },
    {
        "color": [190, 153, 153],
        "instances": False,
        "readable": "Fence",
        "name": "fence",
        "evaluate": True,
    },
    {
        "color": [90, 120, 150],
        "instances": False,
        "readable": "Pedestrian",
        "name": "Pedestrian",
        "evaluate": True,
    },
    {
        "color": [102, 102, 156],
        "instances": False,
        "readable": "Pole",
        "name": "Pole",
        "evaluate": True,
    },
    {
        "color": [128, 64, 255],
        "instances": False,
        "readable": "Roadline",
        "name": "Roadline",
        "evaluate": True,
    },
    {
        "color": [140, 140, 200],
        "instances": True,
        "readable": "Road",
        "name": "Road",
        "evaluate": True,
    },
    {
        "color": [170, 170, 170],
        "instances": False,
        "readable": "Sidewalk",
        "name": "Sidewalk",
        "evaluate": True,
    },
    {
        "color": [250, 170, 160],
        "instances": False,
        "readable": "Vegetation",
        "name": "Vegetation",
        "evaluate": True,
    },
    {
        "color": [96, 96, 96],
        "instances": False,
        "readable": "Car",
        "name": "Car",
        "evaluate": True,
    },
    {
        "color": [230, 150, 140],
        "instances": False,
        "readable": "Wall",
        "name": "Wall",
        "evaluate": True,
    },
    {
        "color": [128, 64, 128],
        "instances": False,
        "readable": "TrafficSign",
        "name": "TrafficSign",
        "evaluate": True,
    },
    {
        "color": [165, 42, 42],
        "instances": True,
        "readable": "void",
        "name": "void",
        "evaluate": True,
    },
]


def _get_street_hazard_meta():
    stuff_classes = [k["readable"] for k in STREET_HAZARD_SEM_SEG_CATEGORIES if k["evaluate"]]
    stuff_colors = [k["color"] for k in STREET_HAZARD_SEM_SEG_CATEGORIES if k["evaluate"]]

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret


def register_street_hazard(root):
    root = os.path.join(root, "streethazard")
    meta = _get_street_hazard_meta()
    for name, dirname in [("train", "train"), ("val", "val")]:
        image_dir = os.path.join(root, dirname, "images")
        gt_dir = os.path.join(root, dirname, "labels")
        name = f"street_hazard_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=12,
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_street_hazard(_root)