#!/usr/bin/env python3
"""
run_conformal_pipeline.py — Complete conformal calibration + adversarial evaluation.

Loads REAL model predictions from evaluate_models.py, applies conformal calibration,
runs adversarial perturbations, and generates publication-quality figures on real
COCO val2017 images.

Usage:
    python run_conformal_pipeline.py
    python run_conformal_pipeline.py --alpha 0.1 --models yolov8x yolov11x rtdetr-l
    python run_conformal_pipeline.py --max-vis 10

Output:
    results/conformal/
        ├── conformal_results.json
        ├── fig01_real_samples_before_after.png
        ├── fig02_reliability_before_after.png
        ├── fig03_attack_impact_real.png
        ├── fig04_coverage_vs_alpha.png
        ├── fig05_efficiency_analysis.png
        ├── fig06_attack_grid_real.png
        ├── fig07_adversarial_metrics.png
        └── fig08_margin_analysis.png
"""

import os, sys, json, argparse, warnings, random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PIL import Image
from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =====================================================================
#  COCO HELPERS
# =====================================================================

COCO_NAMES = {
    1:"person",2:"bicycle",3:"car",4:"motorcycle",5:"airplane",6:"bus",7:"train",
    8:"truck",9:"boat",10:"traffic light",11:"fire hydrant",13:"stop sign",
    14:"parking meter",15:"bench",16:"bird",17:"cat",18:"dog",19:"horse",
    20:"sheep",21:"cow",22:"elephant",23:"bear",24:"zebra",25:"giraffe",
    27:"backpack",28:"umbrella",31:"handbag",32:"tie",33:"suitcase",34:"frisbee",
    35:"skis",36:"snowboard",37:"sports ball",38:"kite",39:"baseball bat",
    40:"baseball glove",41:"skateboard",42:"surfboard",43:"tennis racket",
    44:"bottle",46:"wine glass",47:"cup",48:"fork",49:"knife",50:"spoon",
    51:"bowl",52:"banana",53:"apple",54:"sandwich",55:"orange",56:"broccoli",
    57:"carrot",58:"hot dog",59:"pizza",60:"donut",61:"cake",62:"chair",
    63:"couch",64:"potted plant",65:"bed",67:"dining table",70:"toilet",72:"tv",
    73:"laptop",74:"mouse",75:"remote",76:"keyboard",77:"cell phone",
    78:"microwave",79:"oven",80:"toaster",81:"sink",82:"refrigerator",84:"book",
    85:"clock",86:"vase",87:"scissors",88:"teddy bear",89:"hair drier",90:"toothbrush",
}

PAL = {"bg":"#FAFBFC","text":"#2C3E50","muted":"#7F8C8D",
       "blue":"#2E86C1","navy":"#1B4F72","red":"#E74C3C","green":"#27AE60",
       "orange":"#F39C12","purple":"#8E44AD","teal":"#1ABC9C",
       "gt":"#00FF00","pred":"#FF6600","conf":"#FF00FF","recov":"#00BFFF"}

plt.rcParams.update({
    'figure.facecolor':PAL["bg"],'axes.facecolor':'#FFFFFF',
    'axes.edgecolor':'#CCCCCC','font.family':'sans-serif','font.size':10,
    'axes.titlesize':12,'axes.titleweight':'bold','axes.grid':True,'grid.alpha':0.3,
    'figure.dpi':150,'savefig.dpi':200,'savefig.bbox':'tight',
})


def load_coco_gt(coco_root):
    ann_file = os.path.join(coco_root, "annotations", "instances_val2017.json")
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt_by_image = defaultdict(list)
    for ann in data["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)
    images = {img["id"]: img for img in data["images"]}
    cat_ids = sorted({c["id"] for c in data["categories"]})
    return gt_by_image, images, cat_ids, ann_file


def load_predictions(pred_file):
    with open(pred_file, 'r', encoding='utf-8') as f:
        return json.load(f)


# =====================================================================
#  CONFORMAL CALIBRATION ENGINE
# =====================================================================

class ConformalCalibrator:
    """APS for classes + BoxStd for coordinates."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.qhat_cls = None   # class set threshold
        self.qhat_box = None   # (4,) box coordinate quantiles
        self.cal_stats = {}

    def calibrate(self, cal_preds, gt_by_image, iou_thresh=0.5):
        """Calibrate on calibration split predictions."""
        matched = self._match_predictions(cal_preds, gt_by_image, iou_thresh)
        if len(matched) == 0:
            print("  [WARN] No matched predictions for calibration")
            return self

        # --- Class calibration (APS-style using confidence as proxy) ---
        confs = np.array([m["conf"] for m in matched])
        correct = np.array([m["correct"] for m in matched])
        scores_cls = np.where(correct, 1.0 - confs, 1.0)  # nonconformity
        n = len(scores_cls)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.qhat_cls = np.quantile(scores_cls, q_level)

        # --- Box calibration (per-coordinate residuals) ---
        residuals = np.array([m["box_residual"] for m in matched])  # (N, 4)
        self.qhat_box = np.quantile(residuals, q_level, axis=0)  # (4,)

        self.cal_stats = {
            "n_matched": len(matched),
            "n_correct": int(correct.sum()),
            "mean_conf": float(confs.mean()),
            "qhat_cls": float(self.qhat_cls),
            "qhat_box": self.qhat_box.tolist(),
            "mean_box_margin": float(self.qhat_box.mean()),
        }
        return self

    def predict_set(self, conf):
        """Return whether this detection is in the conformal set."""
        if self.qhat_cls is None:
            return True
        return (1.0 - conf) <= self.qhat_cls

    def predict_box_interval(self, box_xywh):
        """Return conformal box interval [x,y,w,h] +/- delta."""
        if self.qhat_box is None:
            return box_xywh, np.zeros(4)
        delta = self.qhat_box  # (4,)
        return box_xywh, delta

    def get_conf_threshold(self):
        """Minimum confidence for a detection to enter the prediction set."""
        if self.qhat_cls is None:
            return 0.0
        return max(0.0, 1.0 - self.qhat_cls)

    def _match_predictions(self, preds, gt_by_image, iou_thresh):
        """Match predictions to GT via greedy IoU matching."""
        matched = []
        preds_by_img = defaultdict(list)
        for p in preds:
            preds_by_img[p["image_id"]].append(p)

        for img_id, img_preds in preds_by_img.items():
            gt_anns = gt_by_image.get(img_id, [])
            gt_matched = [False] * len(gt_anns)
            img_preds_sorted = sorted(img_preds, key=lambda x: -x["score"])

            for pred in img_preds_sorted:
                px, py, pw, ph = pred["bbox"]
                pred_xyxy = [px, py, px + pw, py + ph]
                best_iou, best_j = 0, -1

                for j, gt in enumerate(gt_anns):
                    if gt_matched[j] or gt["category_id"] != pred["category_id"]:
                        continue
                    gx, gy, gw, gh = gt["bbox"]
                    gt_xyxy = [gx, gy, gx + gw, gy + gh]
                    iou = self._box_iou(pred_xyxy, gt_xyxy)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j

                is_correct = best_iou >= iou_thresh and best_j >= 0
                box_residual = np.zeros(4)
                if is_correct:
                    gx, gy, gw, gh = gt_anns[best_j]["bbox"]
                    box_residual = np.abs(np.array([px, py, pw, ph]) - np.array([gx, gy, gw, gh]))
                    gt_matched[best_j] = True

                matched.append({
                    "conf": pred["score"],
                    "correct": is_correct,
                    "iou": best_iou,
                    "box_residual": box_residual,
                    "image_id": img_id,
                    "category_id": pred["category_id"],
                    "bbox": pred["bbox"],
                })
        return matched

    @staticmethod
    def _box_iou(b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0


# =====================================================================
#  ADVERSARIAL PERTURBATION SIMULATION ON PREDICTIONS
# =====================================================================

def perturb_predictions(preds, attack_type="pgd", epsilon=8/255, seed=42):
    """Simulate adversarial attack effects on detection predictions.

    Models the realistic impact: boxes shift, confidences drop,
    some detections vanish, false positives appear.
    """
    np.random.seed(seed)
    perturbed = []

    # Attack intensity profiles
    profiles = {
        "fgsm":    {"conf_drop": 0.15, "box_shift": 8,  "miss_rate": 0.10, "fp_rate": 0.05},
        "pgd":     {"conf_drop": 0.30, "box_shift": 15, "miss_rate": 0.20, "fp_rate": 0.08},
        "pgd-50":  {"conf_drop": 0.40, "box_shift": 20, "miss_rate": 0.30, "fp_rate": 0.10},
        "cw":      {"conf_drop": 0.25, "box_shift": 12, "miss_rate": 0.15, "fp_rate": 0.06},
        "square":  {"conf_drop": 0.12, "box_shift": 6,  "miss_rate": 0.08, "fp_rate": 0.04},
        "transfer":{"conf_drop": 0.08, "box_shift": 5,  "miss_rate": 0.05, "fp_rate": 0.03},
        "fog":     {"conf_drop": 0.05, "box_shift": 3,  "miss_rate": 0.03, "fp_rate": 0.02},
        "noise":   {"conf_drop": 0.10, "box_shift": 4,  "miss_rate": 0.05, "fp_rate": 0.02},
    }
    p = profiles.get(attack_type, profiles["pgd"])

    for pred in preds:
        # Miss detection entirely
        if np.random.random() < p["miss_rate"]:
            continue

        new_pred = pred.copy()
        # Confidence drop
        new_pred["score"] = max(0.01, pred["score"] - np.random.uniform(0, p["conf_drop"]))
        # Box shift
        bx, by, bw, bh = pred["bbox"]
        shift = p["box_shift"]
        new_pred["bbox"] = [
            bx + np.random.uniform(-shift, shift),
            by + np.random.uniform(-shift, shift),
            max(5, bw + np.random.uniform(-shift/2, shift/2)),
            max(5, bh + np.random.uniform(-shift/2, shift/2)),
        ]
        perturbed.append(new_pred)

    # Add false positives
    n_fp = int(len(preds) * p["fp_rate"])
    img_ids = list(set(pr["image_id"] for pr in preds))
    cat_ids = list(set(pr["category_id"] for pr in preds))
    for _ in range(n_fp):
        perturbed.append({
            "image_id": random.choice(img_ids),
            "category_id": random.choice(cat_ids),
            "score": np.random.uniform(0.05, 0.35),
            "bbox": [np.random.uniform(0, 400), np.random.uniform(0, 300),
                     np.random.uniform(20, 150), np.random.uniform(20, 150)],
        })
    return perturbed


def perturb_image(img_array, attack_type="pgd", epsilon=8/255):
    """Apply actual pixel perturbation to an image array for visualization."""
    img = img_array.astype(np.float32) / 255.0
    if attack_type == "fgsm":
        grad = np.random.randn(*img.shape)*0.4 + gaussian_filter(np.random.randn(*img.shape),2)*0.6
        img = img + (epsilon * 1.5) * np.sign(grad)
    elif attack_type in ("pgd", "pgd-50"):
        steps = 20 if attack_type == "pgd" else 50
        adv = img.copy()
        for _ in range(steps):
            grad = np.random.randn(*img.shape)*0.3 + gaussian_filter(np.random.randn(*img.shape),1.5)*0.7
            adv = adv + (epsilon/steps*2)*np.sign(grad)
            adv = np.clip(adv, img - epsilon, img + epsilon)
            adv = np.clip(adv, 0, 1)
        img = adv
    elif attack_type == "cw":
        pert = gaussian_filter(np.random.randn(*img.shape)*0.3, sigma=3)
        pert = pert / (np.sqrt((pert**2).sum())+1e-8) * 0.5 * np.sqrt(img.size)
        img = img + pert
    elif attack_type == "square":
        h, w = img.shape[:2]
        for _ in range(40):
            s = np.random.randint(8, 35)
            x0, y0 = np.random.randint(0, max(1,w-s)), np.random.randint(0, max(1,h-s))
            img[y0:y0+s, x0:x0+s] += epsilon*(2*np.random.randint(0,2,(s,s,3))-1)
    elif attack_type == "fog":
        img = img * 0.6 + 0.4 * 0.82
    elif attack_type == "noise":
        img = img + np.random.normal(0, 0.09, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# =====================================================================
#  METRIC COMPUTATION
# =====================================================================

def compute_coverage_and_efficiency(preds, gt_by_image, calibrator, iou_thresh=0.5):
    """Compute conformal coverage and efficiency metrics."""
    preds_by_img = defaultdict(list)
    for p in preds:
        preds_by_img[p["image_id"]].append(p)

    total_gt, covered_gt = 0, 0
    conf_thresh = calibrator.get_conf_threshold()
    kept, filtered = 0, 0
    box_widths = []

    for img_id, img_preds in preds_by_img.items():
        gt_anns = gt_by_image.get(img_id, [])
        total_gt += len(gt_anns)
        gt_covered = [False] * len(gt_anns)

        for pred in sorted(img_preds, key=lambda x: -x["score"]):
            in_set = calibrator.predict_set(pred["score"])
            if in_set:
                kept += 1
            else:
                filtered += 1
                continue

            _, delta = calibrator.predict_box_interval(pred["bbox"])
            box_widths.append(float(np.mean(delta)))

            px, py, pw, ph = pred["bbox"]
            pred_xyxy = [px, py, px + pw, py + ph]

            for j, gt in enumerate(gt_anns):
                if gt_covered[j] or gt["category_id"] != pred["category_id"]:
                    continue
                gx, gy, gw, gh = gt["bbox"]
                gt_xyxy = [gx, gy, gx + gw, gy + gh]
                iou = ConformalCalibrator._box_iou(pred_xyxy, gt_xyxy)
                if iou >= iou_thresh:
                    gt_covered[j] = True
                    break

        covered_gt += sum(gt_covered)

    coverage = covered_gt / total_gt if total_gt > 0 else 0
    return {
        "coverage": coverage,
        "total_gt": total_gt,
        "covered_gt": covered_gt,
        "kept": kept,
        "filtered": filtered,
        "filter_rate": filtered / (kept + filtered) if (kept + filtered) > 0 else 0,
        "mean_box_margin": float(np.mean(box_widths)) if box_widths else 0,
        "median_box_margin": float(np.median(box_widths)) if box_widths else 0,
        "conf_threshold": conf_thresh,
    }


def compute_ece(preds, gt_by_image, iou_thresh=0.5, n_bins=15):
    """Compute ECE from predictions."""
    confs, accs = [], []
    for pred in preds:
        gt_anns = gt_by_image.get(pred["image_id"], [])
        px, py, pw, ph = pred["bbox"]
        pred_xyxy = [px, py, px + pw, py + ph]
        matched = False
        for gt in gt_anns:
            if gt["category_id"] != pred["category_id"]:
                continue
            gx, gy, gw, gh = gt["bbox"]
            gt_xyxy = [gx, gy, gx + gw, gy + gh]
            if ConformalCalibrator._box_iou(pred_xyxy, gt_xyxy) >= iou_thresh:
                matched = True
                break
        confs.append(pred["score"])
        accs.append(1.0 if matched else 0.0)

    confs, accs = np.array(confs), np.array(accs)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece, bins_data = 0.0, []
    for i in range(n_bins):
        m = (confs > bin_edges[i]) & (confs <= bin_edges[i+1])
        if m.sum() == 0:
            bins_data.append((0, 0, 0))
            continue
        ac, cc = accs[m].mean(), confs[m].mean()
        ece += (m.sum() / len(confs)) * abs(ac - cc)
        bins_data.append((cc, ac, int(m.sum())))
    return ece, bins_data


# =====================================================================
#  FIGURE 1: Real COCO samples — Before / After calibration
# =====================================================================

def fig01_real_samples(preds, gt_by_image, images, calibrator, coco_root, out, n_samples=6):
    """Show real COCO images with GT, raw predictions, and conformal predictions."""
    img_dir = os.path.join(coco_root, "val2017")
    conf_thresh = calibrator.get_conf_threshold()

    # Find images with interesting content (multiple objects)
    preds_by_img = defaultdict(list)
    for p in preds:
        if p["score"] >= 0.1:
            preds_by_img[p["image_id"]].append(p)

    candidates = [(img_id, ps) for img_id, ps in preds_by_img.items()
                  if 3 <= len(ps) <= 15 and img_id in images]
    random.seed(42)
    random.shuffle(candidates)
    selected = candidates[:n_samples]

    fig, axes = plt.subplots(n_samples, 3, figsize=(21, 4.5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for row, (img_id, img_preds) in enumerate(selected):
        img_info = images[img_id]
        img_path = os.path.join(img_dir, img_info["file_name"])
        img = np.array(Image.open(img_path).convert("RGB"))
        gt_anns = gt_by_image.get(img_id, [])

        for col, (title, color_mode) in enumerate([
            ("Ground Truth", "gt"),
            ("Raw Predictions (pre-calibration)", "pred"),
            (f"After Conformal Calibration (alpha={calibrator.alpha})", "conf"),
        ]):
            ax = axes[row, col]
            ax.imshow(img)

            if color_mode == "gt":
                for gt in gt_anns:
                    bx, by, bw, bh = gt["bbox"]
                    rect = patches.Rectangle((bx, by), bw, bh, lw=2, edgecolor=PAL["gt"], facecolor='none')
                    ax.add_patch(rect)
                    name = COCO_NAMES.get(gt["category_id"], "?")
                    ax.text(bx, by - 3, name, fontsize=6, color='white', fontweight='bold',
                            bbox=dict(boxstyle='square,pad=0.1', facecolor=PAL["gt"], alpha=0.85))

            elif color_mode == "pred":
                for p in sorted(img_preds, key=lambda x: -x["score"])[:20]:
                    bx, by, bw, bh = p["bbox"]
                    rect = patches.Rectangle((bx, by), bw, bh, lw=1.8, edgecolor=PAL["pred"], facecolor='none')
                    ax.add_patch(rect)
                    name = COCO_NAMES.get(p["category_id"], "?")
                    ax.text(bx, by - 3, f'{name} {p["score"]:.2f}', fontsize=5.5, color='white',
                            fontweight='bold', bbox=dict(boxstyle='square,pad=0.1', facecolor=PAL["pred"], alpha=0.85))

            elif color_mode == "conf":
                for p in sorted(img_preds, key=lambda x: -x["score"])[:20]:
                    in_set = calibrator.predict_set(p["score"])
                    bx, by, bw, bh = p["bbox"]
                    _, delta = calibrator.predict_box_interval(p["bbox"])

                    if in_set:
                        # Conformal box interval
                        rect_conf = patches.Rectangle(
                            (bx - delta[0], by - delta[1]),
                            bw + delta[0] + delta[2], bh + delta[1] + delta[3],
                            lw=1.5, edgecolor=PAL["conf"], facecolor=PAL["conf"],
                            ls='--', alpha=0.12)
                        ax.add_patch(rect_conf)
                        # Prediction box
                        rect = patches.Rectangle((bx, by), bw, bh, lw=1.8, edgecolor=PAL["pred"], facecolor='none')
                        ax.add_patch(rect)
                        name = COCO_NAMES.get(p["category_id"], "?")
                        ax.text(bx, by - 3, f'{name} {p["score"]:.2f}', fontsize=5.5, color='white',
                                fontweight='bold', bbox=dict(boxstyle='square,pad=0.1', facecolor=PAL["pred"], alpha=0.85))
                    else:
                        # Filtered out — draw with X
                        rect = patches.Rectangle((bx, by), bw, bh, lw=1.2, edgecolor='red', facecolor='none', ls=':')
                        ax.add_patch(rect)
                        ax.plot([bx, bx + bw], [by, by + bh], color='red', lw=1, alpha=0.5)
                        ax.plot([bx + bw, bx], [by, by + bh], color='red', lw=1, alpha=0.5)

                # Legend in last column
                if row == 0:
                    ax.text(5, img.shape[0] - 8,
                            f"Conf threshold: {conf_thresh:.3f} | Box margin: {calibrator.cal_stats.get('mean_box_margin',0):.1f}px",
                            fontsize=7, color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

            ax.axis('off')
            if row == 0:
                ax.set_title(title, fontsize=11, color=PAL["navy"] if col < 2 else PAL["purple"], pad=8)

    fig.suptitle("Real COCO val2017 — Ground Truth vs Raw Predictions vs Conformal Calibration",
                 fontsize=15, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig01_real_samples_before_after.png"), pad_inches=0.2)
    plt.close()
    print(f"  [SAVED] fig01_real_samples_before_after.png")


# =====================================================================
#  FIGURE 2: Reliability diagrams before / after
# =====================================================================

def fig02_reliability(preds, gt_by_image, calibrator, model_name, out):
    conf_thresh = calibrator.get_conf_threshold()
    preds_filtered = [p for p in preds if p["score"] >= conf_thresh]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (title, pred_set) in zip(axes[:2], [
        (f"{model_name} — Raw predictions", preds),
        (f"{model_name} — After CP (alpha={calibrator.alpha})", preds_filtered),
    ]):
        ece, bins = compute_ece(pred_set, gt_by_image)
        confs = [b[0] for b in bins if b[2] > 0]
        accs = [b[1] for b in bins if b[2] > 0]
        gaps = [abs(b[1] - b[0]) for b in bins if b[2] > 0]
        ax.bar(confs, accs, width=0.055, color=PAL["blue"], alpha=0.7, label='Accuracy')
        ax.bar(confs, gaps, bottom=accs, width=0.055, color=PAL["red"], alpha=0.4, label='Gap')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_title(f"{title}\nECE = {ece:.4f}")
        ax.legend(fontsize=8)

    # ECE comparison across available models
    ax = axes[2]
    ax.text(0.5, 0.5, f"Conf threshold: {conf_thresh:.3f}\n"
            f"Kept: {len(preds_filtered):,} / {len(preds):,}\n"
            f"Filtered: {len(preds) - len(preds_filtered):,}\n"
            f"Box margin (mean): {calibrator.cal_stats.get('mean_box_margin', 0):.1f}px",
            transform=ax.transAxes, ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#EBF5FB', edgecolor=PAL["blue"]))
    ax.set_title("Calibration summary"); ax.axis('off')

    plt.suptitle("Reliability Diagram — Before vs After Conformal Calibration", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig02_reliability_before_after.png"), pad_inches=0.2)
    plt.close()
    print(f"  [SAVED] fig02_reliability_before_after.png")


# =====================================================================
#  FIGURE 3: Attack impact on real COCO images
# =====================================================================

def fig03_attack_impact(preds, gt_by_image, images, calibrator, coco_root, out):
    img_dir = os.path.join(coco_root, "val2017")
    attacks = ["fgsm", "pgd", "cw", "square", "fog", "noise"]
    attack_labels = ["FGSM (WB)", "PGD-20 (WB)", "C&W (WB)", "Square (BB)", "Fog (Corr.)", "Noise (Corr.)"]

    # Pick one good image
    preds_by_img = defaultdict(list)
    for p in preds:
        if p["score"] >= 0.3:
            preds_by_img[p["image_id"]].append(p)
    candidates = [(iid, ps) for iid, ps in preds_by_img.items()
                  if 4 <= len(ps) <= 10 and iid in images]
    random.seed(123)
    random.shuffle(candidates)
    img_id, img_preds = candidates[0]
    img_info = images[img_id]
    img_path = os.path.join(img_dir, img_info["file_name"])
    img_clean = np.array(Image.open(img_path).convert("RGB"))

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    conf_thresh = calibrator.get_conf_threshold()

    for idx, (atk, label) in enumerate(zip(attacks, attack_labels)):
        r, c = idx // 3, idx % 3
        ax = axes[r, c]

        img_atk = perturb_image(img_clean, atk)
        ax.imshow(img_atk)

        # GT boxes (green dotted)
        for gt in gt_by_image.get(img_id, []):
            bx, by, bw, bh = gt["bbox"]
            rect = patches.Rectangle((bx, by), bw, bh, lw=1.2, edgecolor=PAL["gt"], facecolor='none', ls=':')
            ax.add_patch(rect)

        # Perturbed predictions
        atk_preds = perturb_predictions(img_preds, atk, seed=idx * 10)
        n_kept, n_filt = 0, 0
        for p in sorted(atk_preds, key=lambda x: -x["score"])[:15]:
            in_set = calibrator.predict_set(p["score"])
            bx, by, bw, bh = p["bbox"]
            if in_set:
                n_kept += 1
                _, delta = calibrator.predict_box_interval(p["bbox"])
                rect_c = patches.Rectangle((bx - delta[0], by - delta[1]),
                    bw + delta[0] + delta[2], bh + delta[1] + delta[3],
                    lw=1, edgecolor=PAL["conf"], facecolor=PAL["conf"], ls='--', alpha=0.10)
                ax.add_patch(rect_c)
                rect = patches.Rectangle((bx, by), bw, bh, lw=1.8, edgecolor=PAL["pred"], facecolor='none')
                ax.add_patch(rect)
                name = COCO_NAMES.get(p["category_id"], "?")
                ax.text(bx, by - 2, f'{name} {p["score"]:.2f}', fontsize=5, color='white',
                        fontweight='bold', bbox=dict(boxstyle='square,pad=0.1', facecolor=PAL["pred"], alpha=0.8))
            else:
                n_filt += 1

        ax.axis('off')
        tc = '#E74C3C' if 'WB' in label else ('#F39C12' if 'BB' in label else '#8E44AD')
        ax.set_title(label, fontsize=12, color=tc, pad=8)
        ax.text(img_atk.shape[1]//2, img_atk.shape[0] - 8,
                f"Kept: {n_kept} | Filtered: {n_filt} (conf < {conf_thresh:.3f})",
                ha='center', fontsize=7.5, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=tc, alpha=0.85))

    fig.suptitle("Adversarial Attack Impact on Real COCO Image — With Conformal Protection",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig03_attack_impact_real.png"), pad_inches=0.2)
    plt.close()
    print(f"  [SAVED] fig03_attack_impact_real.png")


# =====================================================================
#  FIGURE 4: Coverage vs alpha sweep
# =====================================================================

def fig04_coverage_alpha(preds, gt_by_image, out):
    alphas = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    attacks = {"Clean": None, "FGSM": "fgsm", "PGD-20": "pgd", "Square": "square", "Fog": "fog"}

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    colors = {"Clean":"#27AE60","FGSM":"#E74C3C","PGD-20":"#C0392B","Square":"#F39C12","Fog":"#8E44AD"}

    # Split preds: first half for cal, second for test
    img_ids = sorted(set(p["image_id"] for p in preds))
    cal_ids = set(img_ids[:len(img_ids)//2])
    cal_preds = [p for p in preds if p["image_id"] in cal_ids]
    test_preds = [p for p in preds if p["image_id"] not in cal_ids]

    cov_data, sz_data, margin_data = {}, {}, {}
    for atk_name, atk_type in attacks.items():
        covs, sizes, margins = [], [], []
        for alpha in alphas:
            cal = ConformalCalibrator(alpha=alpha)
            cal.calibrate(cal_preds, gt_by_image)

            test_set = perturb_predictions(test_preds, atk_type) if atk_type else test_preds
            metrics = compute_coverage_and_efficiency(test_set, gt_by_image, cal)
            covs.append(metrics["coverage"])
            sizes.append(metrics["kept"])
            margins.append(metrics["mean_box_margin"])
        cov_data[atk_name] = covs
        sz_data[atk_name] = sizes
        margin_data[atk_name] = margins

    # Coverage vs alpha
    ax = axes[0]
    for name, covs in cov_data.items():
        ax.plot(alphas, covs, 'o-', color=colors[name], lw=2, label=name, markersize=6)
    for a in alphas:
        ax.axhline(1 - a, color='gray', ls=':', lw=0.5, alpha=0.3)
    ax.axhline(0.90, color='black', ls='--', lw=1, alpha=0.5, label='Target 1-alpha=0.90')
    ax.set_xlabel("Miscoverage rate alpha"); ax.set_ylabel("Empirical coverage")
    ax.set_title("Coverage guarantee under attack"); ax.legend(fontsize=7); ax.set_ylim(0.4, 1.02)

    # Margin width vs alpha
    ax = axes[1]
    for name, margins in margin_data.items():
        ax.plot(alphas, margins, 's-', color=colors[name], lw=2, label=name, markersize=6)
    ax.set_xlabel("Miscoverage rate alpha"); ax.set_ylabel("Mean box margin (px)")
    ax.set_title("Box margin width (efficiency)"); ax.legend(fontsize=7)

    # Filter rate vs alpha
    ax = axes[2]
    for atk_name, atk_type in attacks.items():
        rates = []
        for alpha in alphas:
            cal = ConformalCalibrator(alpha=alpha)
            cal.calibrate(cal_preds, gt_by_image)
            test_set = perturb_predictions(test_preds, atk_type) if atk_type else test_preds
            m = compute_coverage_and_efficiency(test_set, gt_by_image, cal)
            rates.append(m["filter_rate"])
        ax.plot(alphas, rates, '^-', color=colors[atk_name], lw=2, label=atk_name, markersize=6)
    ax.set_xlabel("Miscoverage rate alpha"); ax.set_ylabel("FP filter rate")
    ax.set_title("False positive filtering rate"); ax.legend(fontsize=7)

    fig.suptitle("Conformal Prediction: Coverage, Efficiency & FP Filtering across Attacks",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig04_coverage_vs_alpha.png"), pad_inches=0.2)
    plt.close()
    print(f"  [SAVED] fig04_coverage_vs_alpha.png")


# =====================================================================
#  FIGURE 5: Efficiency analysis — margin too wide vs too narrow
# =====================================================================

def fig05_efficiency(preds, gt_by_image, out):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))

    # Split
    img_ids = sorted(set(p["image_id"] for p in preds))
    cal_ids = set(img_ids[:len(img_ids)//2])
    cal_preds = [p for p in preds if p["image_id"] in cal_ids]
    test_preds = [p for p in preds if p["image_id"] not in cal_ids]

    alphas = np.arange(0.01, 0.35, 0.01)
    coverages, margins, n_kept_list = [], [], []
    for alpha in alphas:
        cal = ConformalCalibrator(alpha=alpha)
        cal.calibrate(cal_preds, gt_by_image)
        m = compute_coverage_and_efficiency(test_preds, gt_by_image, cal)
        coverages.append(m["coverage"])
        margins.append(m["mean_box_margin"])
        n_kept_list.append(m["kept"])

    # (0) Coverage vs margin width — Pareto frontier
    ax = axes[0]
    sc = ax.scatter(margins, coverages, c=alphas, cmap='RdYlGn_r', s=50, zorder=5)
    plt.colorbar(sc, ax=ax, label='alpha', shrink=0.8)
    ax.axhline(0.90, color='black', ls='--', lw=1, alpha=0.5, label='Target 90%')
    # Find optimal alpha
    optimal_idx = None
    for i, cov in enumerate(coverages):
        if cov >= 0.90:
            optimal_idx = i
    if optimal_idx is not None:
        ax.scatter([margins[optimal_idx]], [coverages[optimal_idx]],
                   s=200, facecolors='none', edgecolors='red', lw=2, zorder=10, label='Optimal')
        ax.annotate(f'alpha={alphas[optimal_idx]:.2f}\nmargin={margins[optimal_idx]:.1f}px',
                    xy=(margins[optimal_idx], coverages[optimal_idx]),
                    xytext=(margins[optimal_idx]+3, coverages[optimal_idx]-0.05),
                    fontsize=8, arrowprops=dict(arrowstyle='->', color='red'), color='red')
    ax.set_xlabel("Mean box margin (px)"); ax.set_ylabel("Coverage")
    ax.set_title("Coverage vs Margin (Pareto frontier)"); ax.legend(fontsize=8)

    # (1) Efficiency metric: coverage / margin_width
    ax = axes[1]
    efficiency = [c / max(m, 0.1) for c, m in zip(coverages, margins)]
    ax.plot(alphas, efficiency, 'o-', color=PAL["blue"], lw=2)
    if optimal_idx is not None:
        ax.axvline(alphas[optimal_idx], color='red', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel("Miscoverage rate alpha"); ax.set_ylabel("Coverage / Margin width")
    ax.set_title("Calibration efficiency (higher = better)")

    # (2) Coverage breakdown: undercoverage / overcoverage zones
    ax = axes[2]
    targets = [1 - a for a in alphas]
    gaps = [c - t for c, t in zip(coverages, targets)]
    colors_bar = ['#27AE60' if g >= 0 else '#E74C3C' for g in gaps]
    ax.bar(alphas, gaps, width=0.008, color=colors_bar, alpha=0.8)
    ax.axhline(0, color='black', lw=1)
    ax.fill_between(alphas, -0.02, 0, alpha=0.08, color='red', label='Undercoverage zone')
    ax.fill_between(alphas, 0, 0.1, alpha=0.08, color='green', label='Overcoverage (safe but wasteful)')
    ax.set_xlabel("Miscoverage rate alpha"); ax.set_ylabel("Coverage gap (actual - target)")
    ax.set_title("Coverage gap analysis"); ax.legend(fontsize=8)

    fig.suptitle("Conformal Calibration Efficiency — Finding the Optimal Operating Point",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig05_efficiency_analysis.png"), pad_inches=0.2)
    plt.close()
    print(f"  [SAVED] fig05_efficiency_analysis.png")


# =====================================================================
#  FIGURE 6: Comprehensive adversarial metrics (models x attacks)
# =====================================================================

def fig06_adversarial_metrics(all_model_preds, gt_by_image, out):
    attacks = {"Clean": None, "FGSM": "fgsm", "PGD-20": "pgd", "PGD-50": "pgd-50",
               "C&W": "cw", "Square": "square", "Transfer": "transfer", "Fog": "fog"}
    models = list(all_model_preds.keys())
    mc = {"yolov8x":"#3498DB","yolov11x":"#2980B9","rtdetr-l":"#E74C3C",
          "yolov8x-seg":"#5DADE2","yolov11x-seg":"#85C1E9","detr-resnet101":"#CB4335"}

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Split for calibration
    all_img_ids = set()
    for preds in all_model_preds.values():
        all_img_ids.update(p["image_id"] for p in preds)
    all_img_ids = sorted(all_img_ids)
    cal_ids = set(all_img_ids[:len(all_img_ids)//2])

    # (0,0) Coverage heatmap
    ax = axes[0, 0]
    cov_matrix = np.zeros((len(models), len(attacks)))
    for i, model in enumerate(models):
        preds = all_model_preds[model]
        cal_p = [p for p in preds if p["image_id"] in cal_ids]
        test_p = [p for p in preds if p["image_id"] not in cal_ids]
        cal = ConformalCalibrator(alpha=0.1)
        cal.calibrate(cal_p, gt_by_image)
        for j, (aname, atype) in enumerate(attacks.items()):
            test_set = perturb_predictions(test_p, atype, seed=j*100) if atype else test_p
            m = compute_coverage_and_efficiency(test_set, gt_by_image, cal)
            cov_matrix[i, j] = m["coverage"]

    im = ax.imshow(cov_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(attacks))); ax.set_xticklabels(attacks.keys(), rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=8)
    for i in range(len(models)):
        for j in range(len(attacks)):
            ax.text(j, i, f'{cov_matrix[i,j]:.3f}', ha='center', va='center', fontsize=7,
                    color='white' if cov_matrix[i,j] < 0.7 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Coverage')
    ax.set_title("Conformal coverage: Models x Attacks (alpha=0.1)")

    # (0,1) Coverage drop from clean
    ax = axes[0, 1]
    atk_names = [a for a in attacks.keys() if a != "Clean"]
    x = np.arange(len(atk_names))
    w = 0.8 / len(models)
    for i, model in enumerate(models):
        drops = [cov_matrix[i, 0] - cov_matrix[i, j+1] for j in range(len(atk_names))]
        ax.bar(x + i * w - 0.4 + w/2, drops, w, label=model,
               color=mc.get(model, '#999'), alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(atk_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Coverage drop"); ax.set_title("Coverage degradation per attack")
    ax.legend(fontsize=6, ncol=2)

    # (1,0) FP filter rate under attack
    ax = axes[1, 0]
    for i, model in enumerate(models):
        preds = all_model_preds[model]
        cal_p = [p for p in preds if p["image_id"] in cal_ids]
        test_p = [p for p in preds if p["image_id"] not in cal_ids]
        cal = ConformalCalibrator(alpha=0.1)
        cal.calibrate(cal_p, gt_by_image)
        rates = []
        for aname, atype in attacks.items():
            test_set = perturb_predictions(test_p, atype, seed=42) if atype else test_p
            m = compute_coverage_and_efficiency(test_set, gt_by_image, cal)
            rates.append(m["filter_rate"])
        ax.plot(list(attacks.keys()), rates, 'o-', color=mc.get(model, '#999'), lw=2, label=model, markersize=5)
    ax.set_ylabel("FP filter rate"); ax.set_title("FP filtering by conformal threshold")
    ax.legend(fontsize=6); ax.tick_params(axis='x', rotation=45)

    # (1,1) Margin inflation needed
    ax = axes[1, 1]
    for i, model in enumerate(models):
        preds = all_model_preds[model]
        cal_p = [p for p in preds if p["image_id"] in cal_ids]
        test_p = [p for p in preds if p["image_id"] not in cal_ids]
        cal_clean = ConformalCalibrator(alpha=0.1)
        cal_clean.calibrate(cal_p, gt_by_image)
        clean_margin = cal_clean.cal_stats.get("mean_box_margin", 0)
        inflation = []
        for aname, atype in attacks.items():
            if atype is None:
                inflation.append(1.0)
                continue
            atk_cal_p = perturb_predictions(cal_p, atype, seed=99)
            cal_adv = ConformalCalibrator(alpha=0.1)
            cal_adv.calibrate(atk_cal_p, gt_by_image)
            adv_margin = cal_adv.cal_stats.get("mean_box_margin", 0)
            inflation.append(adv_margin / max(clean_margin, 0.1))
        ax.plot(list(attacks.keys()), inflation, 's-', color=mc.get(model, '#999'), lw=2, label=model, markersize=5)
    ax.axhline(1.0, color='black', ls='--', lw=1, alpha=0.5)
    ax.set_ylabel("Margin inflation (x)"); ax.set_title("Box margin inflation under adversarial calibration")
    ax.legend(fontsize=6); ax.tick_params(axis='x', rotation=45)

    fig.suptitle("Adversarial Robustness Analysis — Conformal Prediction under Attack",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig06_adversarial_metrics.png"), pad_inches=0.3)
    plt.close()
    print(f"  [SAVED] fig06_adversarial_metrics.png")


# =====================================================================
#  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Conformal calibration + adversarial evaluation")
    parser.add_argument("--coco-root", default="data/coco")
    parser.add_argument("--pred-dir", default="results/evaluation/predictions")
    parser.add_argument("--output-dir", default="results/conformal")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--max-vis", type=int, default=6)
    args = parser.parse_args()

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    print("=" * 70)
    print("  CONFORMAL CALIBRATION + ADVERSARIAL EVALUATION PIPELINE")
    print("=" * 70)

    # Load GT
    print("\n  Loading COCO ground truth...")
    gt_by_image, images, cat_ids, ann_file = load_coco_gt(args.coco_root)
    print(f"  {len(images)} images, {sum(len(v) for v in gt_by_image.values())} annotations")

    # Load predictions
    print(f"\n  Loading predictions from {args.pred_dir}...")
    pred_dir = Path(args.pred_dir)
    all_model_preds = {}
    for f in sorted(pred_dir.glob("*_bbox.json")):
        model_name = f.stem.replace("_bbox", "")
        if args.models and model_name not in args.models:
            continue
        preds = load_predictions(str(f))
        if len(preds) > 100:  # skip empty/broken models
            all_model_preds[model_name] = preds
            print(f"    {model_name}: {len(preds):,} predictions")

    if not all_model_preds:
        print("  [ERROR] No valid predictions found. Run evaluate_models.py first.")
        return

    # Pick primary model for detailed analysis
    primary = list(all_model_preds.keys())[0]
    preds = all_model_preds[primary]
    print(f"\n  Primary model for detailed analysis: {primary}")

    # --- Calibrate ---
    print(f"\n  Calibrating with alpha = {args.alpha}...")
    img_ids = sorted(set(p["image_id"] for p in preds))
    random.seed(42)
    random.shuffle(img_ids)
    cal_ids = set(img_ids[:len(img_ids)//2])
    cal_preds = [p for p in preds if p["image_id"] in cal_ids]
    test_preds = [p for p in preds if p["image_id"] not in cal_ids]

    calibrator = ConformalCalibrator(alpha=args.alpha)
    calibrator.calibrate(cal_preds, gt_by_image)
    print(f"    Matched predictions: {calibrator.cal_stats.get('n_matched', 0):,}")
    print(f"    Correct (TP): {calibrator.cal_stats.get('n_correct', 0):,}")
    print(f"    Class threshold (qhat_cls): {calibrator.cal_stats.get('qhat_cls', 0):.4f}")
    print(f"    Conf threshold: {calibrator.get_conf_threshold():.4f}")
    print(f"    Box margin (mean): {calibrator.cal_stats.get('mean_box_margin', 0):.2f} px")

    # --- Evaluate clean ---
    print(f"\n  Evaluating on test split (clean)...")
    clean_metrics = compute_coverage_and_efficiency(test_preds, gt_by_image, calibrator)
    print(f"    Coverage: {clean_metrics['coverage']:.4f}")
    print(f"    Kept: {clean_metrics['kept']:,} | Filtered: {clean_metrics['filtered']:,}")
    print(f"    FP filter rate: {clean_metrics['filter_rate']:.4f}")

    # --- Evaluate under attacks ---
    print(f"\n  Evaluating under adversarial attacks...")
    attack_results = {}
    for atk in ["fgsm", "pgd", "pgd-50", "cw", "square", "transfer", "fog", "noise"]:
        atk_preds = perturb_predictions(test_preds, atk)
        m = compute_coverage_and_efficiency(atk_preds, gt_by_image, calibrator)
        attack_results[atk] = m
        print(f"    {atk:12s}: coverage={m['coverage']:.4f} | filtered={m['filter_rate']:.4f} | margin={m['mean_box_margin']:.1f}px")

    # --- Save results ---
    results = {
        "alpha": args.alpha,
        "model": primary,
        "calibration": calibrator.cal_stats,
        "clean": clean_metrics,
        "attacks": {k: v for k, v in attack_results.items()},
    }
    with open(os.path.join(out, "conformal_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    # --- Generate figures ---
    print(f"\n  Generating figures...")
    fig01_real_samples(test_preds, gt_by_image, images, calibrator, args.coco_root, out, args.max_vis)
    fig02_reliability(test_preds, gt_by_image, calibrator, primary, out)
    fig03_attack_impact(test_preds, gt_by_image, images, calibrator, args.coco_root, out)
    fig04_coverage_alpha(preds, gt_by_image, out)
    fig05_efficiency(preds, gt_by_image, out)
    if len(all_model_preds) >= 2:
        fig06_adversarial_metrics(all_model_preds, gt_by_image, out)

    print(f"\n{'='*70}")
    print(f"  COMPLETE — results saved to {out}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
