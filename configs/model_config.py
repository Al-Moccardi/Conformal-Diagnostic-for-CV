# -*- coding: utf-8 -*-
"""
model_config.py — Centralized model zoo and metric definitions.

Every script imports MODEL_ZOO from here.
"""

# ═══════════════════════════════════════════════════════════════
#  MODEL ZOO — canonical config for all 7 active models
# ═══════════════════════════════════════════════════════════════

MODEL_ZOO = {
    # ── Ultralytics Detection ──
    "yolov8x": {
        "architecture": "YOLOv8x",
        "family": "YOLO",
        "framework": "ultralytics",
        "weight_key": "yolov8x.pt",
        "tasks": ["detection"],
    },
    "yolov11x": {
        "architecture": "YOLOv11x",
        "family": "YOLO",
        "framework": "ultralytics",
        "weight_key": "yolo11x.pt",
        "tasks": ["detection"],
    },
    "rtdetr-l": {
        "architecture": "RT-DETR-L",
        "family": "DETR",
        "framework": "ultralytics",
        "weight_key": "rtdetr-l.pt",
        "tasks": ["detection"],
    },
    # ── HuggingFace Detection ──
    "detr-resnet101": {
        "architecture": "DETR-ResNet-101",
        "family": "DETR",
        "framework": "huggingface",
        "model_id": "facebook/detr-resnet-101",
        "tasks": ["detection"],
    },
    # ── Ultralytics Segmentation ──
    "yolov8x-seg": {
        "architecture": "YOLOv8x-seg",
        "family": "YOLO",
        "framework": "ultralytics",
        "weight_key": "yolov8x-seg.pt",
        "tasks": ["detection", "instance_segmentation"],
    },
    "yolov11x-seg": {
        "architecture": "YOLOv11x-seg",
        "family": "YOLO",
        "framework": "ultralytics",
        "weight_key": "yolo11x-seg.pt",
        "tasks": ["detection", "instance_segmentation"],
    },
    # ── HuggingFace Segmentation (EXCLUDED from paper) ──
    "mask2former-swin-l": {
        "architecture": "Mask2Former-Swin-L",
        "family": "Transformer",
        "framework": "huggingface",
        "model_id": "facebook/mask2former-swin-large-coco-instance",
        "tasks": ["detection", "instance_segmentation"],
        "exclude_from_paper": True,
        "exclude_reason": "mAP=0.126, broken integration",
    },
}

# Models to use in adversarial experiments
ADVERSARIAL_MODELS = ["yolov8x", "yolov11x", "detr-resnet101"]

# Models to use in segmentation attack experiments
SEGMENTATION_MODELS = ["yolov8x-seg", "yolov11x-seg"]

# ═══════════════════════════════════════════════════════════════
#  METRIC DEFINITIONS
# ═══════════════════════════════════════════════════════════════

DETECTION_METRICS = [
    "mAP@[0.5:0.95]", "mAP@0.5", "mAP@0.75",
    "mAP_small", "mAP_medium", "mAP_large",
    "AR@1", "AR@10", "AR@100",
]

SEGMENTATION_METRICS = [
    "mask_mAP@[0.5:0.95]", "mask_mAP@0.5", "mask_mAP@0.75",
]

CALIBRATION_METRICS = ["ECE", "MCE", "Brier", "NLL"]

# COCO 80-class index to 91-category-id mapping
COCO_80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
