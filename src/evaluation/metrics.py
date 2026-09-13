"""
metrics.py — Comprehensive evaluation metrics for object detection and segmentation.

Computes COCO-standard mAP/AR + calibration metrics (ECE, Brier, NLL)
+ segmentation-specific metrics (IoU, Dice, Boundary IoU).
"""

import numpy as np
import json
import os
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════
#  COCO EVALUATION (mAP, AR)
# ═══════════════════════════════════════════════════════════════

def run_coco_eval(predictions: List[Dict], annotation_file: str,
                  iou_type: str = "bbox") -> Dict[str, float]:
    """
    Run official COCO evaluation.

    Args:
        predictions: list of {image_id, category_id, bbox/segmentation, score}
        annotation_file: path to COCO ground-truth JSON
        iou_type: "bbox" or "segm"

    Returns:
        dict with all 12 COCO metrics
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(annotation_file)

    if len(predictions) == 0:
        return {k: 0.0 for k in [
            "mAP@[0.5:0.95]", "mAP@0.5", "mAP@0.75",
            "mAP_small", "mAP_medium", "mAP_large",
            "AR@1", "AR@10", "AR@100",
            "AR_small", "AR_medium", "AR_large",
        ]}

    # Write predictions to temp file (COCO API requires file)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(predictions, f)
        pred_file = f.name

    try:
        coco_dt = coco_gt.loadRes(pred_file)
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        return {
            "mAP@[0.5:0.95]": float(stats[0]),
            "mAP@0.5": float(stats[1]),
            "mAP@0.75": float(stats[2]),
            "mAP_small": float(stats[3]),
            "mAP_medium": float(stats[4]),
            "mAP_large": float(stats[5]),
            "AR@1": float(stats[6]),
            "AR@10": float(stats[7]),
            "AR@100": float(stats[8]),
            "AR_small": float(stats[9]),
            "AR_medium": float(stats[10]),
            "AR_large": float(stats[11]),
        }
    finally:
        os.unlink(pred_file)


def format_predictions_coco(image_id: int, result: Dict,
                            coco_cat_ids: List[int],
                            iou_type: str = "bbox") -> List[Dict]:
    """Convert model output to COCO prediction format.

    Handles two label conventions:
      - 0-indexed class indices (Ultralytics): labels[i]=0 -> coco_cat_ids[0]
      - COCO category IDs (DETR, Mask2Former): labels[i]=1 -> cat_id=1 directly
    """
    preds = []
    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["labels"]
    masks = result.get("masks")
    label_is_coco_id = result.get("label_is_coco_id", False)

    # Build a set of valid COCO category IDs for validation
    valid_cat_ids = set(coco_cat_ids)

    for i in range(len(scores)):
        if label_is_coco_id:
            # Labels are already COCO category IDs (DETR, Mask2Former)
            cat_id = int(labels[i])
            if cat_id not in valid_cat_ids:
                continue
        else:
            # Labels are 0-indexed class indices (Ultralytics, Detectron2)
            cls_idx = int(labels[i])
            if cls_idx < len(coco_cat_ids):
                cat_id = coco_cat_ids[cls_idx]
            else:
                continue

        pred = {
            "image_id": image_id,
            "category_id": cat_id,
            "score": float(scores[i]),
        }

        if iou_type == "bbox":
            x1, y1, x2, y2 = boxes[i]
            pred["bbox"] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]  # COCO: [x,y,w,h]
        elif iou_type == "segm" and masks is not None:
            from pycocotools import mask as mask_util
            binary_mask = masks[i].astype(np.uint8)
            rle = mask_util.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            pred["segmentation"] = rle

        preds.append(pred)

    return preds


# ═══════════════════════════════════════════════════════════════
#  CALIBRATION METRICS
# ═══════════════════════════════════════════════════════════════

def expected_calibration_error(confidences: np.ndarray, accuracies: np.ndarray,
                                n_bins: int = 15) -> Dict[str, float]:
    """
    Compute ECE, MCE, and per-bin calibration data.

    Args:
        confidences: predicted confidence scores (N,)
        accuracies: binary correctness indicators (N,) — 1 if correct, 0 if wrong

    Returns:
        dict with ECE, MCE, overconfidence, underconfidence, and bin data
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_data = []
    n_total = len(confidences)

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            bin_data.append({"bin_center": (bin_edges[i] + bin_edges[i+1]) / 2,
                             "avg_conf": 0, "avg_acc": 0, "count": 0, "gap": 0})
            continue

        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        gap = abs(avg_acc - avg_conf)
        ece += (count / n_total) * gap
        mce = max(mce, gap)

        bin_data.append({
            "bin_center": float((bin_edges[i] + bin_edges[i+1]) / 2),
            "avg_conf": float(avg_conf),
            "avg_acc": float(avg_acc),
            "count": int(count),
            "gap": float(gap),
        })

    # Overconfidence vs underconfidence
    overconf = np.mean(confidences[confidences > accuracies] - accuracies[confidences > accuracies]) \
        if (confidences > accuracies).any() else 0.0

    return {
        "ECE": float(ece),
        "MCE": float(mce),
        "overconfidence": float(overconf),
        "n_bins": n_bins,
        "bin_data": bin_data,
    }


def brier_score(confidences: np.ndarray, accuracies: np.ndarray) -> float:
    """Brier score (lower is better)."""
    return float(np.mean((confidences - accuracies) ** 2))


def negative_log_likelihood(confidences: np.ndarray, accuracies: np.ndarray,
                            eps: float = 1e-7) -> float:
    """NLL of the confidence as a probability estimate."""
    probs = np.clip(confidences, eps, 1 - eps)
    nll = -np.mean(accuracies * np.log(probs) + (1 - accuracies) * np.log(1 - probs))
    return float(nll)


def compute_calibration_metrics(predictions: List[Dict],
                                 ground_truths: Dict,
                                 iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute calibration metrics by matching predictions to ground truth.

    For each prediction, determine if it's a TP (IoU > threshold with correct class)
    then use the confidence as the "probability" and TP/FP as the "accuracy".
    """
    all_confs = []
    all_correct = []

    for pred in predictions:
        img_id = pred["image_id"]
        gt_anns = ground_truths.get(img_id, [])

        conf = pred["score"]
        pred_cat = pred["category_id"]
        pred_box = pred["bbox"]  # [x, y, w, h]
        px1, py1, pw, ph = pred_box
        pred_xyxy = [px1, py1, px1 + pw, py1 + ph]

        matched = False
        best_iou = 0.0
        for gt in gt_anns:
            if gt["category_id"] != pred_cat:
                continue
            gx, gy, gw, gh = gt["bbox"]
            gt_xyxy = [gx, gy, gx + gw, gy + gh]
            iou = _box_iou(pred_xyxy, gt_xyxy)
            if iou > best_iou:
                best_iou = iou
                if iou >= iou_threshold:
                    matched = True

        all_confs.append(conf)
        all_correct.append(1.0 if matched else 0.0)

    if len(all_confs) == 0:
        return {"ECE": 0.0, "MCE": 0.0, "Brier": 0.0, "NLL": 0.0}

    confs = np.array(all_confs)
    accs = np.array(all_correct)

    ece_results = expected_calibration_error(confs, accs)
    return {
        "ECE": ece_results["ECE"],
        "MCE": ece_results["MCE"],
        "Brier": brier_score(confs, accs),
        "NLL": negative_log_likelihood(confs, accs),
        "overconfidence": ece_results["overconfidence"],
        "n_predictions": len(confs),
        "precision_at_conf50": float(accs[confs >= 0.5].mean()) if (confs >= 0.5).any() else 0.0,
        "bin_data": ece_results["bin_data"],
    }


# ═══════════════════════════════════════════════════════════════
#  SEGMENTATION METRICS
# ═══════════════════════════════════════════════════════════════

def compute_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """IoU between two binary masks."""
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    return float(intersection / union) if union > 0 else 0.0


def compute_dice(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """Dice coefficient (F1) between two binary masks."""
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    total = mask_pred.sum() + mask_gt.sum()
    return float(2 * intersection / total) if total > 0 else 0.0


def compute_boundary_iou(mask_pred: np.ndarray, mask_gt: np.ndarray,
                          dilation: int = 2) -> float:
    """Boundary IoU — IoU computed only on boundary pixels."""
    from scipy.ndimage import binary_dilation, binary_erosion

    struct = np.ones((2 * dilation + 1, 2 * dilation + 1))

    boundary_pred = binary_dilation(mask_pred, struct) ^ binary_erosion(mask_pred, struct)
    boundary_gt = binary_dilation(mask_gt, struct) ^ binary_erosion(mask_gt, struct)

    intersection = np.logical_and(boundary_pred, boundary_gt).sum()
    union = np.logical_or(boundary_pred, boundary_gt).sum()
    return float(intersection / union) if union > 0 else 0.0


def compute_segmentation_metrics(pred_masks: List[np.ndarray],
                                  gt_masks: List[np.ndarray]) -> Dict[str, float]:
    """Aggregate segmentation metrics over a list of mask pairs."""
    ious, dices, bious = [], [], []

    for pm, gm in zip(pred_masks, gt_masks):
        ious.append(compute_iou(pm, gm))
        dices.append(compute_dice(pm, gm))
        if pm.shape == gm.shape:
            bious.append(compute_boundary_iou(pm, gm))

    return {
        "mean_IoU": float(np.mean(ious)) if ious else 0.0,
        "mean_Dice": float(np.mean(dices)) if dices else 0.0,
        "boundary_IoU": float(np.mean(bious)) if bious else 0.0,
        "std_IoU": float(np.std(ious)) if ious else 0.0,
        "n_masks_evaluated": len(ious),
    }


# ═══════════════════════════════════════════════════════════════
#  F1 SCORE
# ═══════════════════════════════════════════════════════════════

def compute_f1_at_threshold(predictions: List[Dict], ground_truths: Dict,
                             iou_threshold: float = 0.5,
                             conf_threshold: float = 0.5) -> Dict[str, float]:
    """Compute F1, precision, recall at a given confidence threshold."""
    tp, fp, fn = 0, 0, 0

    for img_id, gt_anns in ground_truths.items():
        img_preds = [p for p in predictions if p["image_id"] == img_id and p["score"] >= conf_threshold]
        gt_matched = [False] * len(gt_anns)

        for pred in sorted(img_preds, key=lambda x: -x["score"]):
            px, py, pw, ph = pred["bbox"]
            pred_xyxy = [px, py, px + pw, py + ph]
            best_iou, best_j = 0, -1

            for j, gt in enumerate(gt_anns):
                if gt_matched[j] or gt["category_id"] != pred["category_id"]:
                    continue
                gx, gy, gw, gh = gt["bbox"]
                gt_xyxy = [gx, gy, gx + gw, gy + gh]
                iou = _box_iou(pred_xyxy, gt_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_threshold and best_j >= 0:
                tp += 1
                gt_matched[best_j] = True
            else:
                fp += 1

        fn += sum(1 for m in gt_matched if not m)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "F1@0.5": float(f1),
        "precision@0.5": float(precision),
        "recall@0.5": float(recall),
        "TP": tp, "FP": fp, "FN": fn,
    }


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _box_iou(box1: List[float], box2: List[float]) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0
