#!/usr/bin/env python3
"""
visualize_box_calibration.py — Show what TRUE box calibration does on real images.

Shows:
  1. Original box vs Corrected box vs GT — with IoU before/after
  2. Filtering: which detections are rejected and why
  3. IoU distribution improvement histogram
  4. Per-class bias vectors (which direction each class is biased)

Usage:
    python visualize_box_calibration.py
    python visualize_box_calibration.py --models yolov8x rtdetr-l --n-images 6
"""

import os, sys, json, argparse, warnings, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PIL import Image

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from src.calibration.box_calibration import ConformalBoxCalibrator

COCO_NAMES = {
    1:"person",2:"bicycle",3:"car",4:"motorcycle",5:"airplane",6:"bus",7:"train",
    8:"truck",9:"boat",10:"traffic light",11:"fire hydrant",13:"stop sign",
    14:"parking meter",15:"bench",16:"bird",17:"cat",18:"dog",19:"horse",
    20:"sheep",21:"cow",22:"elephant",23:"bear",24:"zebra",25:"giraffe",
    27:"backpack",28:"umbrella",31:"handbag",32:"tie",33:"suitcase",
    44:"bottle",46:"wine glass",47:"cup",48:"fork",49:"knife",50:"spoon",
    51:"bowl",52:"banana",53:"apple",54:"sandwich",55:"orange",56:"broccoli",
    57:"carrot",58:"hot dog",59:"pizza",60:"donut",61:"cake",62:"chair",
    63:"couch",64:"potted plant",65:"bed",67:"dining table",70:"toilet",72:"tv",
    73:"laptop",74:"mouse",75:"remote",76:"keyboard",77:"cell phone",
    78:"microwave",79:"oven",80:"toaster",81:"sink",82:"refrigerator",84:"book",
    85:"clock",86:"vase",87:"scissors",88:"teddy bear",89:"hair drier",90:"toothbrush",
}

COLORS = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
          '#42d4f4','#f032e6','#bfef45','#fabed4','#469990']

plt.rcParams.update({
    'figure.facecolor':'white','axes.facecolor':'white','axes.grid':True,
    'grid.alpha':0.3,'font.size':10,'axes.titleweight':'bold',
    'figure.dpi':150,'savefig.dpi':250,'savefig.bbox':'tight',
})


def load_data(coco_root, pred_dir, model_name):
    ann_file = os.path.join(coco_root, "annotations", "instances_val2017.json")
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt = defaultdict(list)
    for ann in data["annotations"]:
        gt[ann["image_id"]].append(ann)
    images = {img["id"]: img for img in data["images"]}

    pred_path = os.path.join(pred_dir, f"{model_name}_bbox.json")
    with open(pred_path, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    by_img = defaultdict(list)
    for p in preds:
        by_img[p["image_id"]].append(p)

    return gt, images, preds, by_img


def iou_xywh(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2]); y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = b1[2]*b1[3]; a2 = b2[2]*b2[3]
    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0


# =====================================================================
#  FIGURE 1: Original vs Corrected vs GT on real images
# =====================================================================

def fig_correction_grid(calibrator, preds_by_img, gt_by_image, images,
                        coco_root, out, n_imgs=6):
    """The money figure: same image, 3 columns — GT / Original / Corrected."""
    img_dir = os.path.join(coco_root, "val2017")

    # Select images with good matches
    candidates = []
    for img_id, ps in preds_by_img.items():
        gt_anns = gt_by_image.get(img_id, [])
        n_high = sum(1 for p in ps if p["score"] >= 0.5)
        if 3 <= len(gt_anns) <= 10 and n_high >= 3 and img_id in images:
            candidates.append(img_id)
    random.seed(42)
    random.shuffle(candidates)
    selected = candidates[:n_imgs]

    fig, axes = plt.subplots(n_imgs, 3, figsize=(21, 5.5 * n_imgs))
    if n_imgs == 1:
        axes = axes.reshape(1, -1)

    for row, img_id in enumerate(selected):
        img_info = images[img_id]
        img = np.array(Image.open(os.path.join(img_dir, img_info["file_name"])).convert("RGB"))
        gt_anns = gt_by_image.get(img_id, [])
        img_preds = sorted(preds_by_img.get(img_id, []), key=lambda x: -x["score"])
        h, w = img.shape[:2]

        # ─── Column 0: Ground Truth ───
        ax = axes[row, 0]
        ax.imshow(img)
        for ann in gt_anns:
            bx,by,bw,bh = ann["bbox"]
            color = COLORS[ann["category_id"] % len(COLORS)]
            rect = patches.Rectangle((bx,by), bw, bh, lw=2.5, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            name = COCO_NAMES.get(ann["category_id"], "?")
            ax.text(bx, max(0,by-3), name, fontsize=7, fontweight='bold', color='white',
                    bbox=dict(boxstyle='square,pad=0.1', facecolor=color, alpha=0.9))
        ax.axis('off')
        if row == 0:
            ax.set_title("Ground Truth", fontsize=13, color='#27AE60', pad=10)

        # ─── Column 1: Original predictions ───
        ax = axes[row, 1]
        ax.imshow(img)
        # Draw GT faintly
        for ann in gt_anns:
            bx,by,bw,bh = ann["bbox"]
            rect = patches.Rectangle((bx,by), bw, bh, lw=1, edgecolor='#00FF00', facecolor='none', ls=':', alpha=0.4)
            ax.add_patch(rect)

        shown = 0
        for p in img_preds:
            if p["score"] < 0.3 or shown >= 15:
                break
            bx,by,bw,bh = p["bbox"]
            color = COLORS[p["category_id"] % len(COLORS)]

            # Compute IoU with best matching GT
            best_iou = 0
            for gt in gt_anns:
                if gt["category_id"] == p["category_id"]:
                    iou = iou_xywh(p["bbox"], gt["bbox"])
                    best_iou = max(best_iou, iou)

            rect = patches.Rectangle((bx,by), bw, bh, lw=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            name = COCO_NAMES.get(p["category_id"], "?")
            iou_color = '#27AE60' if best_iou >= 0.7 else ('#F39C12' if best_iou >= 0.5 else '#E74C3C')
            ax.text(bx, max(0,by-3), f'{name} {p["score"]:.2f} IoU={best_iou:.2f}',
                    fontsize=5.5, fontweight='bold', color='white',
                    bbox=dict(boxstyle='square,pad=0.1', facecolor=iou_color, alpha=0.9))
            shown += 1
        ax.axis('off')
        if row == 0:
            ax.set_title("Original predictions + IoU", fontsize=13, color='#FF6600', pad=10)

        # ─── Column 2: Corrected predictions ───
        ax = axes[row, 2]
        ax.imshow(img)
        # Draw GT faintly
        for ann in gt_anns:
            bx,by,bw,bh = ann["bbox"]
            rect = patches.Rectangle((bx,by), bw, bh, lw=1, edgecolor='#00FF00', facecolor='none', ls=':', alpha=0.4)
            ax.add_patch(rect)

        shown = 0
        n_improved, n_kept, n_filt = 0, 0, 0
        for p in img_preds:
            if shown >= 15:
                break
            r = calibrator.predict(p)

            if not r.keep:
                if p["score"] >= 0.05:
                    n_filt += 1
                    bx,by,bw,bh = p["bbox"]
                    rect = patches.Rectangle((bx,by), bw, bh, lw=0.8, edgecolor='red', facecolor='none', ls=':')
                    ax.add_patch(rect)
                    ax.plot([bx,bx+bw],[by,by+bh], color='red', lw=0.6, alpha=0.4)
                continue

            n_kept += 1
            cbx, cby, cbw, cbh = r.corrected_box
            color = COLORS[p["category_id"] % len(COLORS)]

            # Compute IoU with corrected box
            best_iou_orig, best_iou_corr = 0, 0
            for gt in gt_anns:
                if gt["category_id"] == p["category_id"]:
                    iou_o = iou_xywh(p["bbox"], gt["bbox"])
                    iou_c = iou_xywh(r.corrected_box, gt["bbox"])
                    best_iou_orig = max(best_iou_orig, iou_o)
                    best_iou_corr = max(best_iou_corr, iou_c)

            improved = best_iou_corr > best_iou_orig + 0.005
            if improved:
                n_improved += 1

            # Draw corrected box
            rect = patches.Rectangle((cbx,cby), cbw, cbh, lw=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # Show original box faintly if different
            if np.abs(r.correction_applied).max() > 1:
                bx,by,bw,bh = p["bbox"]
                rect_old = patches.Rectangle((bx,by), bw, bh, lw=1, edgecolor=color,
                                              facecolor='none', ls=':', alpha=0.3)
                ax.add_patch(rect_old)

            name = COCO_NAMES.get(p["category_id"], "?")
            delta_iou = best_iou_corr - best_iou_orig
            delta_str = f"+{delta_iou:.2f}" if delta_iou > 0 else f"{delta_iou:.2f}"
            iou_color = '#27AE60' if best_iou_corr >= 0.7 else ('#F39C12' if best_iou_corr >= 0.5 else '#E74C3C')

            label = f'{name} IoU={best_iou_corr:.2f} ({delta_str})'
            ax.text(cbx, max(0,cby-3), label,
                    fontsize=5.5, fontweight='bold', color='white',
                    bbox=dict(boxstyle='square,pad=0.1', facecolor=iou_color, alpha=0.9))
            shown += 1

        ax.axis('off')
        if row == 0:
            ax.set_title("Corrected predictions + IoU change", fontsize=13, color='#2E86C1', pad=10)
        ax.text(w//2, h-8, f"Kept:{n_kept} Filt:{n_filt} Improved:{n_improved}",
                ha='center', fontsize=7, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    # Legend
    legend_elements = [
        Line2D([0],[0], color='#00FF00', lw=1, ls=':', label='Ground truth'),
        Line2D([0],[0], color='#FF6600', lw=2, label='Original prediction'),
        Line2D([0],[0], color='#2E86C1', lw=2, label='Corrected prediction'),
        Line2D([0],[0], color='red', lw=1, ls=':', label='Filtered (rejected)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("True Box Calibration: Bias Correction + Conformal Filtering",
                 fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    path = os.path.join(out, "fig_box_correction_grid.png")
    plt.savefig(path, pad_inches=0.3); plt.close()
    print(f"  [SAVED] {path}")


# =====================================================================
#  FIGURE 2: IoU improvement histogram
# =====================================================================

def fig_iou_histogram(eval_results, model_name, out):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) IoU distribution: original vs corrected
    ax = axes[0]
    ax.text(0.5, 0.5, f"Mean IoU original: {eval_results['mean_iou_original']:.4f}\n"
            f"Mean IoU corrected: {eval_results['mean_iou_corrected']:.4f}\n"
            f"Improvement: {eval_results['iou_improvement']:+.4f}\n\n"
            f"Improved: {eval_results['n_improved']}/{eval_results['n_evaluated']} "
            f"({eval_results['pct_improved']*100:.1f}%)\n"
            f"Degraded: {eval_results['n_degraded']}/{eval_results['n_evaluated']} "
            f"({eval_results['pct_degraded']*100:.1f}%)",
            transform=ax.transAxes, ha='center', va='center', fontsize=12,
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E6F1FB', edgecolor='#378ADD'))
    ax.set_title(f"(a) IoU improvement summary — {model_name}")
    ax.axis('off')

    # (b) Coverage guarantee
    ax = axes[1]
    coverage = eval_results["empirical_coverage"]
    target = eval_results["target_coverage"]
    holds = eval_results["coverage_holds"]

    bar_color = '#27AE60' if holds else '#E74C3C'
    ax.bar(["Empirical\ncoverage", "Target\n(1-alpha)"], [coverage, target],
           color=[bar_color, '#B4B2A9'], edgecolor='white', lw=2)
    ax.set_ylabel("Coverage")
    ax.set_title(f"(b) IoU coverage guarantee\n"
                 f"P(IoU >= {eval_results['iou_lower_bound']:.3f}) >= {target:.2f}")
    ax.set_ylim(0, 1.1)
    for i, v in enumerate([coverage, target]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=11, fontweight='bold')

    # (c) Practical value
    ax = axes[2]
    labels = ["Original\n(no CP)", "Margin\ninflation", "Bias\ncorrection\n(ours)"]
    values = [0, 0, eval_results["iou_improvement"]]
    colors_bar = ['#B4B2A9', '#E74C3C', '#27AE60']
    descriptions = ["Baseline", "Useless\n(wider box)", f"IoU +{eval_results['iou_improvement']:.4f}"]

    ax.bar(labels, values, color=colors_bar, edgecolor='white', lw=2)
    ax.set_ylabel("IoU improvement over baseline")
    ax.set_title("(c) What each approach actually does to IoU")
    for i, (v, d) in enumerate(zip(values, descriptions)):
        ax.text(i, max(v, 0) + 0.001, d, ha='center', fontsize=9, fontweight='bold',
                color=colors_bar[i])

    fig.suptitle(f"Box Calibration Evaluation — {model_name}",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out, "fig_iou_improvement.png")
    plt.savefig(path); plt.close()
    print(f"  [SAVED] {path}")


# =====================================================================
#  FIGURE 3: Per-class bias vectors
# =====================================================================

def fig_bias_vectors(calibrator, out):
    """Show which direction each class is systematically biased."""
    if not calibrator.class_bias:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cats = list(calibrator.class_bias.keys())[:15]
    biases = np.array([calibrator.class_bias[c] for c in cats])
    names = [COCO_NAMES.get(c, str(c))[:12] for c in cats]

    # (a) Bias magnitude per coordinate
    ax = axes[0]
    x = np.arange(len(cats))
    w = 0.2
    coords = ['dx (x-pos)', 'dy (y-pos)', 'dw (width)', 'dh (height)']
    colors_c = ['#2E86C1', '#27AE60', '#E74C3C', '#F39C12']
    for i, (coord, col) in enumerate(zip(coords, colors_c)):
        ax.bar(x + i*w - 1.5*w, biases[:, i], w, label=coord, color=col, alpha=0.8)
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Signed bias (pixels)"); ax.set_title("(a) Per-class systematic bias")
    ax.legend(fontsize=7)

    # (b) Bias direction arrows
    ax = axes[1]
    for i, (cat_id, name) in enumerate(zip(cats, names)):
        b = calibrator.class_bias[cat_id]
        ax.arrow(i, 0, 0, b[0], head_width=0.3, head_length=0.5,
                 fc='#2E86C1', ec='#2E86C1', alpha=0.7)
        ax.text(i, b[0] + np.sign(b[0])*1, f"x:{b[0]:+.1f}\ny:{b[1]:+.1f}",
                ha='center', fontsize=6, color='#2C3E50')
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Bias direction (px)"); ax.set_title("(b) Correction direction per class")

    fig.suptitle("Learned Bias Vectors — systematic detector errors per class",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out, "fig_bias_vectors.png")
    plt.savefig(path); plt.close()
    print(f"  [SAVED] {path}")


# =====================================================================
#  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", default="data/coco")
    parser.add_argument("--pred-dir", default="results/evaluation/predictions")
    parser.add_argument("--output-dir", default="results/box_calibration")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--n-images", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    print("=" * 70)
    print("  TRUE BOX CALIBRATION — Correct, Don't Inflate")
    print("=" * 70)

    # Find models
    pred_dir = Path(args.pred_dir)
    models = []
    for f in sorted(pred_dir.glob("*_bbox.json")):
        m = f.stem.replace("_bbox", "")
        if args.models and m not in args.models:
            continue
        models.append(m)

    if not models:
        print("  [ERROR] No predictions found")
        return

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*60}")

        gt, images, preds, by_img = load_data(args.coco_root, args.pred_dir, model_name)

        # Split
        img_ids = sorted(by_img.keys())
        random.seed(42)
        random.shuffle(img_ids)
        n = len(img_ids) // 2
        cal_preds = [p for p in preds if p["image_id"] in set(img_ids[:n])]
        test_preds = [p for p in preds if p["image_id"] in set(img_ids[n:])]

        print(f"  Cal: {len(cal_preds):,} | Test: {len(test_preds):,}")

        # Calibrate
        calibrator = ConformalBoxCalibrator(alpha=args.alpha, n_size_bins=3,
                                            n_conf_bins=4, min_samples=15)
        calibrator.calibrate(cal_preds, gt)

        stats = calibrator.get_summary()
        print(f"  Correct matches: {stats.get('n_correct_matches', 0):,}")
        print(f"  Mean IoU raw: {stats.get('mean_iou_raw', 0):.4f}")
        print(f"  Mean IoU corrected: {stats.get('mean_iou_corrected', 0):.4f}")
        print(f"  IoU improvement: {stats.get('iou_improvement', 0):+.4f}")
        print(f"  IoU lower bound: {stats.get('iou_lower_bound_quantile', 0):.4f}")
        print(f"  Conf threshold: {stats.get('conf_threshold', 0):.4f}")
        print(f"  Bias bins: {stats.get('n_bias_bins', 0)}")
        print(f"  Class biases: {stats.get('n_class_bias', 0)}")

        # Evaluate
        print(f"\n  Evaluating correction quality on test set...")
        eval_results = calibrator.evaluate_correction(test_preds, gt)
        print(f"  Mean IoU original: {eval_results.get('mean_iou_original', 0):.4f}")
        print(f"  Mean IoU corrected: {eval_results.get('mean_iou_corrected', 0):.4f}")
        print(f"  Improvement: {eval_results.get('iou_improvement', 0):+.4f}")
        print(f"  Improved: {eval_results.get('n_improved', 0)}/{eval_results.get('n_evaluated', 0)} "
              f"({eval_results.get('pct_improved', 0)*100:.1f}%)")
        print(f"  Coverage (IoU >= {eval_results.get('iou_lower_bound', 0):.3f}): "
              f"{eval_results.get('empirical_coverage', 0):.4f} "
              f"(target: {eval_results.get('target_coverage', 0):.2f}) "
              f"{'HOLDS' if eval_results.get('coverage_holds', False) else 'VIOLATED'}")

        # Figures
        print(f"\n  Generating figures...")
        fig_correction_grid(calibrator, by_img, gt, images, args.coco_root, out, args.n_images)
        fig_iou_histogram(eval_results, model_name, out)
        fig_bias_vectors(calibrator, out)

        # Save results
        all_res = {"calibration": stats, "evaluation": eval_results}
        with open(os.path.join(out, f"results_{model_name}.json"), 'w', encoding='utf-8') as f:
            json.dump(all_res, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  COMPLETE — {out}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
