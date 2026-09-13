"""
Dataset configuration for Conformal Calibration Pipeline.
Defines paths, URLs, splits, and normalization parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


# ─── COCO 2017 ───────────────────────────────────────────────────────────────
@dataclass
class COCOConfig:
    name: str = "MS-COCO 2017"
    task: str = "object_detection + instance_segmentation"
    root: str = os.path.join(DATA_DIR, "coco")

    # Download URLs
    urls: Dict[str, str] = field(default_factory=lambda: {
        "train_images": "http://images.cocodataset.org/zips/train2017.zip",
        "val_images": "http://images.cocodataset.org/zips/val2017.zip",
        "train_val_annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    })

    # Paths after extraction
    val_images_dir: str = "val2017"
    val_ann_file: str = "annotations/instances_val2017.json"

    # Dataset stats
    num_classes: int = 80
    num_val_images: int = 5000
    num_train_images: int = 118287

    # Conformal split ratios (applied to val set)
    cal_ratio: float = 0.5   # 50% for calibration
    test_ratio: float = 0.5  # 50% for test

    # COCO categories with supercategories
    supercategories: List[str] = field(default_factory=lambda: [
        "person", "vehicle", "outdoor", "animal", "accessory",
        "sports", "kitchen", "food", "furniture", "electronic",
        "appliance", "indoor"
    ])


# ─── PASCAL VOC 2012 ─────────────────────────────────────────────────────────
@dataclass
class VOCConfig:
    name: str = "PASCAL VOC 2012"
    task: str = "object_detection + semantic_segmentation"
    root: str = os.path.join(DATA_DIR, "voc")

    urls: Dict[str, str] = field(default_factory=lambda: {
        "trainval": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
    })

    num_classes: int = 20
    num_images: int = 11540

    class_names: List[str] = field(default_factory=lambda: [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ])

    # VOC <-> COCO shared classes (for cross-dataset analysis)
    shared_with_coco: List[str] = field(default_factory=lambda: [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ])


# ─── Normalization ────────────────────────────────────────────────────────────
@dataclass
class NormConfig:
    """ImageNet normalization (used by most pretrained detectors)."""
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    input_size: int = 640  # default for YOLO / DETR
