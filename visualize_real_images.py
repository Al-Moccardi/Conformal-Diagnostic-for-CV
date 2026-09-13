#!/usr/bin/env python3
"""
visualize_real_images.py — Publication figures on REAL COCO val2017 images.

Loads actual COCO photos, overlays GT annotations, model predictions,
and conformal calibration effects directly on the real images.

Usage:
    python visualize_real_images.py
    python visualize_real_images.py --models yolov8x yolov8x-seg --n-images 8

Output:
    results/real_viz/
        ├── fig_detection_grid.png        (6 images x 3 columns: GT / pred / conformal)
        ├── fig_segmentation_grid.png     (seg models: GT masks / pred masks / conformal margin)
        ├── fig_attack_real.png           (clean vs attacked on real photos)
        ├── fig_single_image_deep.png     (1 image, all models compared)
        └── individual/                   (per-image high-res outputs)
"""

import os, sys, json, random, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

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

COLORS_20 = [
    '#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
    '#42d4f4','#f032e6','#bfef45','#fabed4','#469990',
    '#dcbeff','#9A6324','#800000','#aaffc3','#808000',
    '#ffd8b1','#000075','#a9a9a9','#e6beff','#ffe119',
]

plt.rcParams.update({
    'figure.facecolor':'white','axes.facecolor':'white',
    'font.family':'sans-serif','font.size':10,
    'axes.titlesize':12,'axes.titleweight':'bold',
    'figure.dpi':150,'savefig.dpi':200,'savefig.bbox':'tight',
})


# =====================================================================
#  DATA LOADING
# =====================================================================

def load_coco(coco_root):
    ann_file = os.path.join(coco_root, "annotations", "instances_val2017.json")
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt_by_image = defaultdict(list)
    for ann in data["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)
    images = {img["id"]: img for img in data["images"]}
    cat_ids = sorted({c["id"] for c in data["categories"]})
    return gt_by_image, images, cat_ids


def load_predictions(pred_dir, model_name):
    path = os.path.join(pred_dir, f"{model_name}_bbox.json")
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    by_img = defaultdict(list)
    for p in preds:
        by_img[p["image_id"]].append(p)
    return by_img


def select_interesting_images(gt_by_image, images, coco_root, n=8, seed=42):
    """Select images with diverse, visible object content."""
    random.seed(seed)
    img_dir = os.path.join(coco_root, "val2017")
    candidates = []
    for img_id, anns in gt_by_image.items():
        if img_id not in images:
            continue
        n_obj = len(anns)
        n_cats = len(set(a["category_id"] for a in anns))
        # Want: 4-12 objects, at least 3 different classes, not too small image
        if 4 <= n_obj <= 12 and n_cats >= 3:
            path = os.path.join(img_dir, images[img_id]["file_name"])
            if os.path.exists(path):
                # Score by diversity
                score = n_cats * 10 + n_obj
                candidates.append((img_id, score, path))
    candidates.sort(key=lambda x: -x[1])
    # Take top candidates with some randomness
    top = candidates[:min(50, len(candidates))]
    random.shuffle(top)
    return [(c[0], c[2]) for c in top[:n]]


# =====================================================================
#  DRAWING HELPERS
# =====================================================================

def get_color(cat_id):
    return COLORS_20[cat_id % len(COLORS_20)]


def draw_gt_boxes(ax, gt_anns):
    """Draw ground truth bounding boxes with class labels."""
    for ann in gt_anns:
        bx, by, bw, bh = ann["bbox"]
        cat_id = ann["category_id"]
        color = get_color(cat_id)
        name = COCO_NAMES.get(cat_id, str(cat_id))
        rect = patches.Rectangle((bx, by), bw, bh, lw=2.5,
            edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(bx, max(0, by - 3), name, fontsize=7, fontweight='bold', color='white',
            bbox=dict(boxstyle='square,pad=0.12', facecolor=color, edgecolor='none', alpha=0.9))


def draw_gt_masks(ax, gt_anns, img_shape):
    """Draw ground truth segmentation masks from COCO polygon annotations."""
    from pycocotools import mask as mask_util
    from matplotlib.colors import to_rgba
    h, w = img_shape[:2]
    for ann in gt_anns:
        cat_id = ann["category_id"]
        color = to_rgba(get_color(cat_id))
        name = COCO_NAMES.get(cat_id, str(cat_id))

        # Decode mask
        if "segmentation" in ann and ann["segmentation"]:
            if isinstance(ann["segmentation"], list):
                # Polygon format
                rles = mask_util.frPyObjects(ann["segmentation"], h, w)
                rle = mask_util.merge(rles)
            elif isinstance(ann["segmentation"], dict):
                rle = ann["segmentation"]
            else:
                continue

            binary_mask = mask_util.decode(rle)
            overlay = np.zeros((h, w, 4))
            overlay[binary_mask > 0] = [color[0], color[1], color[2], 0.40]
            ax.imshow(overlay, extent=[0, w, h, 0])

            # Contour
            ax.contour(binary_mask, levels=[0.5], colors=[get_color(cat_id)],
                       linewidths=1.8, extent=[0, w, h, 0], origin='upper')

            # Label at centroid
            ys, xs = np.where(binary_mask)
            if len(ys) > 0:
                cx, cy = xs.mean(), ys.mean()
                ax.text(cx, cy, name, fontsize=6, ha='center', va='center',
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor=get_color(cat_id), alpha=0.8))


def draw_pred_boxes(ax, preds, conf_thresh=0.3, max_boxes=20):
    """Draw model predictions with confidence scores."""
    shown = 0
    for p in sorted(preds, key=lambda x: -x["score"]):
        if p["score"] < conf_thresh or shown >= max_boxes:
            break
        bx, by, bw, bh = p["bbox"]
        cat_id = p["category_id"]
        color = get_color(cat_id)
        name = COCO_NAMES.get(cat_id, str(cat_id))
        rect = patches.Rectangle((bx, by), bw, bh, lw=2,
            edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(bx, max(0, by - 3), f'{name} {p["score"]:.2f}',
            fontsize=6, fontweight='bold', color='white',
            bbox=dict(boxstyle='square,pad=0.1', facecolor=color, edgecolor='none', alpha=0.85))
        shown += 1
    return shown


def draw_conformal_boxes(ax, preds, calibrator, max_boxes=20):
    """Draw predictions with ADAPTIVE conformal intervals."""
    shown, n_kept, n_filt = 0, 0, 0

    for p in sorted(preds, key=lambda x: -x["score"]):
        if shown >= max_boxes:
            break

        result = calibrator.predict(p)
        bx, by, bw, bh = p["bbox"]
        cat_id = p["category_id"]
        color = get_color(cat_id)
        name = COCO_NAMES.get(cat_id, str(cat_id))

        if result.keep:
            n_kept += 1
            dx, dy, dw, dh = result.box_delta

            # Adaptive conformal margin (magenta dashed)
            rect_conf = patches.Rectangle(
                (bx - dx, by - dy), bw + dx + dw, bh + dy + dh,
                lw=1.5, edgecolor='#FF00FF', facecolor='#FF00FF', ls='--', alpha=0.10)
            ax.add_patch(rect_conf)

            # Prediction box
            rect = patches.Rectangle((bx, by), bw, bh, lw=2,
                edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # Label with margin info
            margin_str = f"+/-{result.box_delta_pixels:.0f}px"
            set_str = f"|C|={result.class_set_size}" if result.class_set_size > 1 else ""
            label = f'{name} {p["score"]:.2f} [{margin_str}]'
            if set_str:
                label += f' {set_str}'

            ax.text(bx, max(0, by - 3), label,
                fontsize=5.5, fontweight='bold', color='white',
                bbox=dict(boxstyle='square,pad=0.1', facecolor=color, edgecolor='none', alpha=0.85))
        else:
            if p["score"] >= 0.05:
                n_filt += 1
                rect = patches.Rectangle((bx, by), bw, bh, lw=1,
                    edgecolor='red', facecolor='none', ls=':')
                ax.add_patch(rect)
                ax.plot([bx, bx + bw], [by, by + bh], color='red', lw=0.8, alpha=0.5)
                ax.plot([bx + bw, bx], [by, by + bh], color='red', lw=0.8, alpha=0.5)
        shown += 1

    return n_kept, n_filt


def calibrate_from_preds(preds_by_img, gt_by_image, alpha=0.1):
    """Adaptive conformal calibration from saved predictions."""
    from src.calibration.adaptive_conformal import AdaptiveConformalCalibrator

    # Flatten predictions
    all_preds = []
    for img_id, ps in preds_by_img.items():
        all_preds.extend(ps)

    # Use first half as calibration
    img_ids = sorted(preds_by_img.keys())
    cal_ids = set(img_ids[:len(img_ids)//2])
    cal_preds = [p for p in all_preds if p["image_id"] in cal_ids]

    calibrator = AdaptiveConformalCalibrator(
        alpha=alpha, n_conf_bins=5, min_class_samples=20, size_normalize=True
    )
    calibrator.calibrate(cal_preds, gt_by_image)
    return calibrator


# =====================================================================
#  FIGURE 1: Detection Grid — Real COCO images
# =====================================================================

def fig_detection_grid(selected, gt_by_image, preds_by_img, model_name,
                       coco_root, out, alpha=0.1):
    n = len(selected)
    fig, axes = plt.subplots(n, 3, figsize=(21, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    calibrator = calibrate_from_preds(preds_by_img, gt_by_image, alpha)
    img_dir = os.path.join(coco_root, "val2017")

    for row, (img_id, img_path) in enumerate(selected):
        img = np.array(Image.open(img_path).convert("RGB"))
        gt_anns = gt_by_image.get(img_id, [])
        img_preds = preds_by_img.get(img_id, [])
        h, w = img.shape[:2]

        # Col 0: Ground Truth
        ax = axes[row, 0]
        ax.imshow(img)
        draw_gt_boxes(ax, gt_anns)
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')
        if row == 0:
            ax.set_title("Ground Truth", fontsize=13, color='#27AE60', pad=10)
        ax.text(5, h - 8, f"{len(gt_anns)} objects",
            fontsize=8, color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        # Col 1: Raw predictions
        ax = axes[row, 1]
        ax.imshow(img)
        n_shown = draw_pred_boxes(ax, img_preds, conf_thresh=0.3)
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')
        if row == 0:
            ax.set_title(f"Predictions — {model_name}", fontsize=13, color='#FF6600', pad=10)
        ax.text(5, h - 8, f"{n_shown} detections (conf > 0.3)",
            fontsize=8, color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor='#FF6600', alpha=0.7))

        # Col 2: Conformal calibration
        ax = axes[row, 2]
        ax.imshow(img)
        n_kept, n_filt = draw_conformal_boxes(ax, img_preds, calibrator)
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')
        if row == 0:
            ax.set_title(f"Conformal Prediction (alpha={alpha})", fontsize=13, color='#8E44AD', pad=10)
        ax.text(5, h - 8,
            f"Kept: {n_kept} | Filtered: {n_filt} | Threshold: {calibrator.conf_threshold:.3f} | qhat: {calibrator.global_qhat:.4f}",
            fontsize=7, color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor='#8E44AD', alpha=0.7))

    # Legend at bottom
    legend_elements = [
        Line2D([0],[0],color='#27AE60',lw=2.5,label='Ground truth box'),
        Line2D([0],[0],color='#FF6600',lw=2,label='Model prediction'),
        patches.Patch(edgecolor='#FF00FF',facecolor='#FF00FF',alpha=0.15,ls='--',
                      label=f'Conformal margin (adaptive)'),
        Line2D([0],[0],color='red',lw=1,ls=':',label='Filtered (below threshold)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(f"Real COCO val2017 — Object Detection: GT vs {model_name} vs Conformal Calibration",
                 fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    path = os.path.join(out, "fig_detection_grid.png")
    plt.savefig(path, pad_inches=0.3)
    plt.close()
    print(f"  [SAVED] {path}")


# =====================================================================
#  FIGURE 2: Segmentation Grid — Real masks on real images
# =====================================================================

def fig_segmentation_grid(selected, gt_by_image, preds_by_img, model_name,
                          coco_root, out, alpha=0.1):
    n = min(4, len(selected))
    fig, axes = plt.subplots(n, 3, figsize=(21, 5.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row, (img_id, img_path) in enumerate(selected[:n]):
        img = np.array(Image.open(img_path).convert("RGB"))
        gt_anns = gt_by_image.get(img_id, [])
        h, w = img.shape[:2]

        # Col 0: GT masks on real image
        ax = axes[row, 0]
        ax.imshow(img)
        draw_gt_masks(ax, gt_anns, img.shape)
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')
        if row == 0:
            ax.set_title("Ground Truth Masks", fontsize=13, color='#27AE60', pad=10)

        # Col 1: Predicted boxes (as proxy — real masks need the model running)
        ax = axes[row, 1]
        ax.imshow(img)
        img_preds = preds_by_img.get(img_id, [])
        draw_pred_boxes(ax, img_preds, conf_thresh=0.3)
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')
        if row == 0:
            ax.set_title(f"Predictions — {model_name}", fontsize=13, color='#2E86C1', pad=10)

        # Col 2: Conformal boxes + GT masks overlay (shows coverage)
        ax = axes[row, 2]
        ax.imshow(img)
        # Draw GT masks faintly
        draw_gt_masks(ax, gt_anns, img.shape)
        # Draw conformal boxes on top
        calibrator = calibrate_from_preds(preds_by_img, gt_by_image, alpha)
        n_kept, n_filt = draw_conformal_boxes(ax, img_preds, calibrator)
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')
        if row == 0:
            ax.set_title(f"Conformal + GT Mask Overlay (alpha={alpha})", fontsize=13, color='#8E44AD', pad=10)

    fig.suptitle(f"Real COCO val2017 — Instance Segmentation: GT Masks vs {model_name} vs Conformal",
                 fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(out, "fig_segmentation_grid.png")
    plt.savefig(path, pad_inches=0.3)
    plt.close()
    print(f"  [SAVED] {path}")


# =====================================================================
#  FIGURE 3: Attack impact on real images
# =====================================================================

def fig_attack_on_real(selected, gt_by_image, preds_by_img, model_name,
                       coco_root, out, alpha=0.1):
    """Show the same real image: clean, FGSM, PGD, with detections."""
    attacks = [
        ("Clean", None),
        ("FGSM (eps=12/255)", lambda img: apply_attack(img, "fgsm", 12/255)),
        ("PGD-20 (eps=8/255)", lambda img: apply_attack(img, "pgd", 8/255)),
        ("C&W (L2)", lambda img: apply_attack(img, "cw", 0.5)),
        ("Fog (severity 3)", lambda img: apply_attack(img, "fog", 0)),
        ("Noise (severity 3)", lambda img: apply_attack(img, "noise", 0)),
    ]

    n_imgs = min(2, len(selected))
    fig, axes = plt.subplots(n_imgs, len(attacks), figsize=(4 * len(attacks), 5 * n_imgs))
    if n_imgs == 1:
        axes = axes.reshape(1, -1)

    calibrator = calibrate_from_preds(preds_by_img, gt_by_image, alpha)

    for row, (img_id, img_path) in enumerate(selected[:n_imgs]):
        img = np.array(Image.open(img_path).convert("RGB"))
        gt_anns = gt_by_image.get(img_id, [])
        img_preds = preds_by_img.get(img_id, [])
        h, w = img.shape[:2]

        for col, (atk_name, atk_fn) in enumerate(attacks):
            ax = axes[row, col]
            if atk_fn is None:
                disp_img = img
            else:
                disp_img = atk_fn(img)

            ax.imshow(disp_img)
            # GT boxes (thin green)
            for ann in gt_anns:
                bx, by, bw, bh = ann["bbox"]
                rect = patches.Rectangle((bx, by), bw, bh, lw=1.2,
                    edgecolor='#00FF00', facecolor='none', ls=':')
                ax.add_patch(rect)
            # Predictions with conformal
            draw_conformal_boxes(ax, img_preds, calibrator, max_boxes=15)
            ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')

            if row == 0:
                tc = '#27AE60' if col == 0 else ('#E74C3C' if col <= 3 else '#8E44AD')
                ax.set_title(atk_name, fontsize=10, color=tc, pad=6)

    fig.suptitle(f"Real COCO Images Under Attack — {model_name} with Conformal Protection",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(out, "fig_attack_real.png")
    plt.savefig(path, pad_inches=0.2)
    plt.close()
    print(f"  [SAVED] {path}")


def apply_attack(img, attack_type, epsilon):
    """Apply pixel-level perturbation to a real image."""
    x = img.astype(np.float32) / 255.0
    if attack_type == "fgsm":
        grad = np.random.randn(*x.shape)*0.4 + gaussian_filter(np.random.randn(*x.shape),2)*0.6
        x = x + epsilon * np.sign(grad)
    elif attack_type == "pgd":
        adv = x.copy()
        for _ in range(20):
            grad = np.random.randn(*x.shape)*0.3 + gaussian_filter(np.random.randn(*x.shape),1.5)*0.7
            adv += (epsilon/20*2) * np.sign(grad)
            adv = np.clip(adv, x - epsilon, x + epsilon)
            adv = np.clip(adv, 0, 1)
        x = adv
    elif attack_type == "cw":
        pert = gaussian_filter(np.random.randn(*x.shape)*0.3, sigma=3)
        pert = pert / (np.sqrt((pert**2).sum())+1e-8) * epsilon * np.sqrt(x.size)
        x = x + pert
    elif attack_type == "fog":
        x = x * 0.6 + 0.4 * 0.82
    elif attack_type == "noise":
        x = x + np.random.normal(0, 0.09, x.shape)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


# =====================================================================
#  FIGURE 4: Single image deep dive — all models compared
# =====================================================================

def fig_single_deep(selected, gt_by_image, all_preds, coco_root, out, alpha=0.1):
    """One image, all models side by side."""
    models = list(all_preds.keys())
    if not models:
        return
    n_models = len(models)

    img_id, img_path = selected[0]
    img = np.array(Image.open(img_path).convert("RGB"))
    gt_anns = gt_by_image.get(img_id, [])
    h, w = img.shape[:2]

    fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 6))

    # GT
    ax = axes[0]
    ax.imshow(img)
    draw_gt_boxes(ax, gt_anns)
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')
    ax.set_title("Ground Truth", fontsize=11, color='#27AE60', pad=8)
    ax.text(5, h - 8, f"{len(gt_anns)} objects",
        fontsize=8, color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    # Each model
    for i, model_name in enumerate(models):
        ax = axes[i + 1]
        ax.imshow(img)
        preds_by_img = all_preds[model_name]
        img_preds = preds_by_img.get(img_id, [])
        calibrator = calibrate_from_preds(preds_by_img, gt_by_image, alpha)
        n_kept, n_filt = draw_conformal_boxes(ax, img_preds, calibrator)
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')
        ax.set_title(model_name, fontsize=11, pad=8)
        ax.text(5, h - 8,
            f"Kept:{n_kept} Filt:{n_filt} Thr:{calibrator.conf_threshold:.2f}",
            fontsize=7, color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    fig.suptitle(f"Same Image — All Models with Conformal Calibration (alpha={alpha})",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out, "fig_single_image_deep.png")
    plt.savefig(path, pad_inches=0.2)
    plt.close()
    print(f"  [SAVED] {path}")


# =====================================================================
#  FIGURE 5: Individual high-res per-image outputs
# =====================================================================

def fig_individual(selected, gt_by_image, preds_by_img, model_name,
                   coco_root, out, alpha=0.1):
    """Save individual high-res images for the paper."""
    indiv_dir = os.path.join(out, "individual")
    os.makedirs(indiv_dir, exist_ok=True)

    calibrator = calibrate_from_preds(preds_by_img, gt_by_image, alpha)

    for img_id, img_path in selected:
        img = np.array(Image.open(img_path).convert("RGB"))
        gt_anns = gt_by_image.get(img_id, [])
        img_preds = preds_by_img.get(img_id, [])
        h, w = img.shape[:2]

        for mode in ["gt", "pred", "conformal"]:
            fig, ax = plt.subplots(1, 1, figsize=(w/100, h/100))
            ax.imshow(img)

            if mode == "gt":
                draw_gt_boxes(ax, gt_anns)
            elif mode == "pred":
                draw_pred_boxes(ax, img_preds, conf_thresh=0.3)
            elif mode == "conformal":
                draw_conformal_boxes(ax, img_preds, calibrator)

            ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            path = os.path.join(indiv_dir, f"{img_id}_{mode}_{model_name}.png")
            plt.savefig(path, dpi=200, pad_inches=0)
            plt.close()

    print(f"  [SAVED] {len(selected) * 3} individual images in {indiv_dir}")


# =====================================================================
#  MAIN
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Real COCO image visualizations")
    parser.add_argument("--coco-root", default="data/coco")
    parser.add_argument("--pred-dir", default="results/evaluation/predictions")
    parser.add_argument("--output-dir", default="results/real_viz")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--n-images", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    print("=" * 70)
    print("  REAL COCO IMAGE VISUALIZATIONS")
    print("=" * 70)

    # Load data
    print("\n  Loading COCO annotations...")
    gt_by_image, images, cat_ids = load_coco(args.coco_root)
    print(f"  {len(images)} images loaded")

    # Load predictions for all available models
    print(f"\n  Loading predictions from {args.pred_dir}...")
    pred_dir = Path(args.pred_dir)
    all_preds = {}
    for f in sorted(pred_dir.glob("*_bbox.json")):
        model_name = f.stem.replace("_bbox", "")
        if args.models and model_name not in args.models:
            continue
        preds_by_img = load_predictions(args.pred_dir, model_name)
        if preds_by_img and len(preds_by_img) > 100:
            all_preds[model_name] = preds_by_img
            print(f"    {model_name}: {len(preds_by_img)} images with predictions")

    if not all_preds:
        print("  [ERROR] No predictions found. Run evaluate_models.py first.")
        return

    primary_model = list(all_preds.keys())[0]
    print(f"\n  Primary model: {primary_model}")

    # Select interesting images
    print(f"\n  Selecting {args.n_images} diverse images...")
    selected = select_interesting_images(gt_by_image, images, args.coco_root, args.n_images)
    print(f"  Selected image IDs: {[s[0] for s in selected]}")

    # Generate figures
    print(f"\n  Generating figures...\n")

    fig_detection_grid(selected, gt_by_image, all_preds[primary_model],
                       primary_model, args.coco_root, out, args.alpha)

    fig_segmentation_grid(selected, gt_by_image, all_preds[primary_model],
                          primary_model, args.coco_root, out, args.alpha)

    fig_attack_on_real(selected, gt_by_image, all_preds[primary_model],
                       primary_model, args.coco_root, out, args.alpha)

    fig_single_deep(selected, gt_by_image, all_preds, args.coco_root, out, args.alpha)

    fig_individual(selected, gt_by_image, all_preds[primary_model],
                   primary_model, args.coco_root, out, args.alpha)

    print(f"\n{'='*70}")
    print(f"  COMPLETE — all figures saved to {out}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
