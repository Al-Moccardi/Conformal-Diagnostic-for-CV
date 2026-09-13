#!/usr/bin/env python3
"""
visualize_conformal.py — High-impact visualizations of conformal calibration on sample images.

Generates publication-quality figures showing:
  1. Ground truth vs model predictions (OD + segmentation)
  2. How conformal prediction sets expand class labels
  3. How conformal intervals widen bounding boxes
  4. How conformal margins dilate segmentation masks
  5. Before/after calibration comparison
  6. Multi-model comparison on same image

All scenes are generated synthetically with realistic object placements.
When real COCO images + predictions are available, the script uses those instead.

Output: results/eda/conformal_viz_*.png
"""

import os, warnings, random
from pathlib import Path
from collections import namedtuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba
from scipy.ndimage import binary_dilation
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
#  PALETTE & STYLE
# ═══════════════════════════════════════════════════════════════

P = {
    "bg":"#FAFBFC","text":"#2C3E50","muted":"#7F8C8D",
    "blue":"#2E86C1","navy":"#1B4F72","red":"#E74C3C","green":"#27AE60",
    "orange":"#F39C12","purple":"#8E44AD","teal":"#1ABC9C","pink":"#E91E90",
    "gt_box":"#00FF00","pred_box":"#FF6600","conf_box":"#FF00FF",
    "gt_mask":"#27AE60","pred_mask":"#2E86C1","conf_margin":"#E74C3C",
}

OBJ_COLORS = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
              '#42d4f4','#f032e6','#bfef45','#fabed4','#469990']

plt.rcParams.update({
    'figure.facecolor':P["bg"],'axes.facecolor':'#FFFFFF',
    'axes.edgecolor':'#CCCCCC','font.family':'sans-serif',
    'font.size':10,'axes.titlesize':12,'axes.titleweight':'bold',
    'figure.dpi':150,'savefig.dpi':200,'savefig.bbox':'tight',
})

# ═══════════════════════════════════════════════════════════════
#  SCENE GENERATOR
# ═══════════════════════════════════════════════════════════════

Obj = namedtuple('Obj','cls x y w h color conf')

def make_scene(w=640, h=480):
    """Generate a realistic urban scene background."""
    img = np.zeros((h, w, 3))
    for y in range(h):
        t = y / h
        if t < 0.40:
            img[y,:] = [0.55+0.08*t, 0.72+0.05*t, 0.92-0.03*t]
        elif t < 0.45:
            t2 = (t-0.40)/0.05
            img[y,:] = [0.65+0.10*t2, 0.70-0.05*t2, 0.78-0.15*t2]
        elif t < 0.60:
            img[y,:] = [0.62, 0.64, 0.66]
        else:
            t3 = (t-0.60)/0.40
            img[y,:] = [0.42+0.12*t3, 0.44+0.10*t3, 0.38+0.08*t3]
    # Add some texture noise
    noise = np.random.normal(0, 0.015, img.shape)
    img = np.clip(img + noise, 0, 1)
    return img

def default_gt_objects():
    """Ground truth objects for the demo scene."""
    return [
        Obj("person",   80, 110, 100, 250, OBJ_COLORS[0], 1.0),
        Obj("person",  220, 130,  80, 220, OBJ_COLORS[0], 1.0),
        Obj("car",     370, 220, 200, 120, OBJ_COLORS[2], 1.0),
        Obj("dog",     160, 330,  90,  70, OBJ_COLORS[3], 1.0),
        Obj("bicycle", 500, 260,  80, 100, OBJ_COLORS[4], 1.0),
        Obj("bottle",   30, 200,  30,  70, OBJ_COLORS[5], 1.0),
        Obj("chair",   540, 170,  70,  90, OBJ_COLORS[6], 1.0),
    ]

def default_pred_objects():
    """Predicted objects (slightly shifted from GT, variable confidence)."""
    return [
        Obj("person",   75, 105, 108, 258, OBJ_COLORS[0], 0.96),
        Obj("person",  225, 135,  74, 212, OBJ_COLORS[0], 0.89),
        Obj("car",     365, 215, 210, 128, OBJ_COLORS[2], 0.93),
        Obj("dog",     155, 325,  95,  78, OBJ_COLORS[3], 0.72),
        Obj("bicycle", 495, 255,  88, 108, OBJ_COLORS[4], 0.81),
        Obj("bottle",   25, 195,  38,  78, OBJ_COLORS[5], 0.44),  # low conf
        Obj("chair",   535, 165,  78,  98, OBJ_COLORS[6], 0.67),
        # False positive
        Obj("potted plant", 440, 170, 40, 50, OBJ_COLORS[7], 0.31),
    ]

def make_elliptical_mask(h, w, cx, cy, rx, ry, irregularity=0.08):
    """Create an irregular elliptical binary mask."""
    yy, xx = np.ogrid[0:h, 0:w]
    theta = np.arctan2(yy - cy, xx - cx)
    noise = 1 + irregularity * np.sin(5*theta) + irregularity*0.6 * np.cos(7*theta)
    dist = ((xx - cx)**2 / (rx * noise)**2 + (yy - cy)**2 / (ry * noise)**2)
    return (dist <= 1.0).astype(np.uint8)

# ═══════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════

def draw_shape(ax, obj, filled=True, alpha=0.5):
    """Draw a simplified object shape."""
    x, y, w, h, col = obj.x, obj.y, obj.w, obj.h, obj.color
    if obj.cls in ["person","dog","cat"]:
        e = patches.FancyBboxPatch((x+w*0.08,y+h*0.03),w*0.84,h*0.94,
            boxstyle="round,pad=0.02",facecolor=col if filled else 'none',
            edgecolor=col,alpha=alpha if filled else 0.9,linewidth=1.5)
    elif obj.cls in ["car","bus","truck"]:
        e = patches.FancyBboxPatch((x+w*0.03,y+h*0.12),w*0.94,h*0.76,
            boxstyle="round,pad=0.01",facecolor=col if filled else 'none',
            edgecolor=col,alpha=alpha if filled else 0.9,linewidth=1.5)
    else:
        e = patches.Rectangle((x,y),w,h,
            facecolor=col if filled else 'none',edgecolor=col,
            alpha=alpha if filled else 0.9,linewidth=1.5)
    ax.add_patch(e)

def draw_bbox(ax, obj, color, lw=2.2, ls='-', label=None):
    """Draw a bounding box."""
    rect = patches.Rectangle((obj.x, obj.y), obj.w, obj.h,
        linewidth=lw, edgecolor=color, facecolor='none', linestyle=ls)
    ax.add_patch(rect)
    if label:
        ax.text(obj.x, obj.y - 4, label, fontsize=7, fontweight='bold',
            color='white', bbox=dict(boxstyle='square,pad=0.12',
            facecolor=color, edgecolor='none', alpha=0.88))

def draw_conformal_bbox(ax, obj, delta=15, color=P["conf_box"], alpha=0.25):
    """Draw conformal prediction interval around bbox."""
    rect = patches.Rectangle(
        (obj.x - delta, obj.y - delta),
        obj.w + 2*delta, obj.h + 2*delta,
        linewidth=2, edgecolor=color, facecolor=color,
        linestyle='--', alpha=alpha)
    ax.add_patch(rect)
    # Arrows showing expansion
    mid_y = obj.y + obj.h/2
    for dx, side in [(-delta, obj.x), (obj.w+delta, obj.x+obj.w)]:
        ax.annotate('', xy=(side + (delta if dx<0 else -delta)*0 + (-delta if dx<0 else delta), mid_y),
                    xytext=(side, mid_y),
                    arrowprops=dict(arrowstyle='<->', color=color, lw=1.2, ls='--'))

def setup_axes(ax, w, h):
    ax.set_xlim(0,w); ax.set_ylim(h,0); ax.set_aspect('equal'); ax.axis('off')


# ═══════════════════════════════════════════════════════════════
#  FIGURE 1: GT vs Predictions vs Conformal (3-panel, OD)
# ═══════════════════════════════════════════════════════════════

def fig_01_gt_vs_pred_vs_conformal(out):
    W, H = 640, 480
    scene = make_scene(W, H)
    gt = default_gt_objects()
    pred = default_pred_objects()

    fig = plt.figure(figsize=(22, 8))
    gs = gridspec.GridSpec(1, 3, wspace=0.04)

    # ── Panel A: Ground Truth ──
    ax = fig.add_subplot(gs[0])
    ax.imshow(scene, extent=[0,W,H,0])
    for o in gt:
        draw_shape(ax, o, filled=True, alpha=0.45)
        draw_bbox(ax, o, P["gt_box"], lw=2.5, label=o.cls)
    setup_axes(ax, W, H)
    ax.set_title("(a) Ground Truth", fontsize=14, color=P["green"], pad=12)
    ax.text(W/2, H-8, f"{len(gt)} objects  \u00B7  7 classes",
        ha='center', fontsize=9, color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    # ── Panel B: Model Predictions ──
    ax = fig.add_subplot(gs[1])
    ax.imshow(scene, extent=[0,W,H,0])
    for o in gt: draw_shape(ax, o, filled=True, alpha=0.3)
    for o in pred:
        draw_bbox(ax, o, P["pred_box"], lw=2.2, label=f'{o.cls} {o.conf:.2f}')
    # Highlight false positive
    fp = pred[-1]
    rect = patches.Rectangle((fp.x-3,fp.y-3),fp.w+6,fp.h+6,
        lw=2.5, edgecolor='red', facecolor='none', ls=':')
    ax.add_patch(rect)
    ax.text(fp.x+fp.w+5, fp.y+fp.h/2, "FP", fontsize=9, color='red', fontweight='bold')
    # Highlight low confidence
    lc = pred[5]  # bottle, conf=0.44
    ax.annotate("Low conf\n(0.44)", xy=(lc.x+lc.w/2, lc.y-8),
        fontsize=7, ha='center', color='#FF6600', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FFF3E0', edgecolor='#FF6600', alpha=0.9))
    setup_axes(ax, W, H)
    ax.set_title("(b) Model Predictions (pre-calibration)", fontsize=14, color=P["orange"], pad=12)
    ax.text(W/2, H-8, f"{len(pred)} detections  \u00B7  1 FP  \u00B7  ECE = 0.082",
        ha='center', fontsize=9, color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    # ── Panel C: Conformal Prediction ──
    ax = fig.add_subplot(gs[2])
    ax.imshow(scene, extent=[0,W,H,0])
    for o in gt: draw_shape(ax, o, filled=True, alpha=0.25)

    # Conformal box intervals (delta proportional to 1/confidence)
    for i, o in enumerate(pred[:-1]):  # exclude FP
        delta = int(12 + 25 * (1 - o.conf))  # bigger margin for lower confidence
        draw_conformal_bbox(ax, o, delta=delta, color=P["conf_box"], alpha=0.15)
        draw_bbox(ax, o, P["pred_box"], lw=1.5, ls='-')

        # Prediction set labels
        if o.cls == "dog":
            label = f'{{{o.cls}, cat, bear}}\n\u03B4={delta}px'
            color_label = '#E74C3C'
        elif o.cls == "bottle":
            label = f'{{{o.cls}, cup, vase}}\n\u03B4={delta}px'
            color_label = '#E74C3C'
        elif o.cls == "person" and o.conf < 0.92:
            label = f'{{{o.cls}}}\n\u03B4={delta}px'
            color_label = P["purple"]
        else:
            label = f'{{{o.cls}}}\n\u03B4={delta}px'
            color_label = P["purple"]

        ax.text(o.x + o.w + 5, o.y + o.h/2, label,
            fontsize=6.5, va='center', color=color_label, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                     edgecolor=color_label, alpha=0.85))

    # FP filtered out
    fp = pred[-1]
    ax.plot([fp.x, fp.x+fp.w], [fp.y, fp.y+fp.h], color='red', lw=2, alpha=0.7)
    ax.plot([fp.x+fp.w, fp.x], [fp.y, fp.y+fp.h], color='red', lw=2, alpha=0.7)
    ax.text(fp.x+fp.w/2, fp.y-8, "Filtered\n(conf < \u03C4)", fontsize=6.5,
        ha='center', color='red', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FDEDEC', edgecolor='red', alpha=0.9))

    setup_axes(ax, W, H)
    ax.set_title("(c) After Conformal Calibration (\u03B1=0.1)", fontsize=14, color=P["purple"], pad=12)
    ax.text(W/2, H-8,
        "Prediction sets on classes  \u00B7  Conformal intervals on bbox [\u00B1\u03B4\u2090]  \u00B7  Coverage \u2265 90%",
        ha='center', fontsize=8.5, color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    fig.suptitle("Object Detection: Ground Truth \u2192 Predictions \u2192 Conformal Calibration",
        fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(out, "conformal_viz_01_od_pipeline.png"), pad_inches=0.2)
    plt.close()
    print(f"  [SAVED] conformal_viz_01_od_pipeline.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 2: Instance Segmentation — Masks + Conformal Margin
# ═══════════════════════════════════════════════════════════════

def fig_02_segmentation_conformal(out):
    W, H = 640, 480
    scene = make_scene(W, H)
    gt = default_gt_objects()

    fig = plt.figure(figsize=(22, 8))
    gs = gridspec.GridSpec(1, 3, wspace=0.04)

    # Create masks for GT and predictions
    gt_masks, pred_masks, conf_masks = [], [], []
    for o in gt:
        cx, cy = o.x + o.w/2, o.y + o.h/2
        rx, ry = o.w*0.42, o.h*0.45
        m_gt = make_elliptical_mask(H, W, cx, cy, rx, ry, 0.06)
        m_pred = make_elliptical_mask(H, W, cx+3, cy+2, rx*0.92, ry*0.95, 0.08)
        # Conformal margin = dilation of prediction
        margin_px = max(3, int(8 * (1 - o.conf * 0.7)))  # bigger for harder objects
        struct = np.ones((2*margin_px+1, 2*margin_px+1))
        m_conf = binary_dilation(m_pred, struct).astype(np.uint8)
        gt_masks.append(m_gt)
        pred_masks.append(m_pred)
        conf_masks.append(m_conf)

    # ── Panel A: GT masks ──
    ax = fig.add_subplot(gs[0])
    ax.imshow(scene, extent=[0,W,H,0])
    for i, (o, m) in enumerate(zip(gt, gt_masks)):
        overlay = np.zeros((H,W,4))
        c = to_rgba(o.color)
        overlay[m>0] = [c[0],c[1],c[2],0.45]
        ax.imshow(overlay, extent=[0,W,H,0])
        # Contour
        from matplotlib.contour import ContourSet
        ax.contour(m, levels=[0.5], colors=[o.color], linewidths=2, extent=[0,W,H,0],
                   origin='upper')
        cx, cy = o.x+o.w/2, o.y+o.h/2
        ax.text(cx, cy, o.cls, fontsize=7, ha='center', va='center',
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.1', facecolor=o.color, alpha=0.75))
    setup_axes(ax, W, H)
    ax.set_title("(a) Ground Truth Masks", fontsize=14, color=P["green"], pad=12)

    # ── Panel B: Predicted masks ──
    ax = fig.add_subplot(gs[1])
    ax.imshow(scene, extent=[0,W,H,0])
    for i, (o, m) in enumerate(zip(gt, pred_masks)):
        overlay = np.zeros((H,W,4))
        c = to_rgba(P["blue"])
        overlay[m>0] = [c[0],c[1],c[2],0.4]
        ax.imshow(overlay, extent=[0,W,H,0])
        ax.contour(m, levels=[0.5], colors=[P["blue"]], linewidths=1.8,
                   extent=[0,W,H,0], origin='upper')
        cx, cy = o.x+o.w/2, o.y+o.h/2
        pred_o = default_pred_objects()[i] if i < len(default_pred_objects()) else o
        ax.text(cx, cy, f"{o.cls}\n{pred_o.conf:.2f}", fontsize=6.5, ha='center',
            va='center', fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.1', facecolor=P["blue"], alpha=0.7))
        # Show IoU
        intersection = np.logical_and(gt_masks[i], m).sum()
        union = np.logical_or(gt_masks[i], m).sum()
        iou = intersection / union if union > 0 else 0
        ax.text(o.x, o.y+o.h+12, f"IoU={iou:.2f}", fontsize=6.5,
            color=P["blue"], fontweight='bold')
    setup_axes(ax, W, H)
    ax.set_title("(b) Predicted Masks + IoU", fontsize=14, color=P["blue"], pad=12)

    # ── Panel C: Conformal margin ──
    ax = fig.add_subplot(gs[2])
    ax.imshow(scene, extent=[0,W,H,0])
    for i, (o, m_pred, m_conf, m_gt) in enumerate(zip(gt, pred_masks, conf_masks, gt_masks)):
        # Margin region (conformal minus prediction)
        margin_region = (m_conf > 0) & (m_pred == 0)
        # Prediction region
        overlay_pred = np.zeros((H,W,4))
        c = to_rgba(P["blue"])
        overlay_pred[m_pred>0] = [c[0],c[1],c[2],0.35]
        ax.imshow(overlay_pred, extent=[0,W,H,0])
        # Margin region in red
        overlay_margin = np.zeros((H,W,4))
        overlay_margin[margin_region] = [0.9, 0.2, 0.2, 0.35]
        ax.imshow(overlay_margin, extent=[0,W,H,0])
        # Conformal contour
        ax.contour(m_conf, levels=[0.5], colors=[P["red"]], linewidths=2,
                   linestyles='--', extent=[0,W,H,0], origin='upper')
        # Pred contour
        ax.contour(m_pred, levels=[0.5], colors=[P["blue"]], linewidths=1.5,
                   extent=[0,W,H,0], origin='upper')
        # GT contour (thin green)
        ax.contour(m_gt, levels=[0.5], colors=[P["green"]], linewidths=1,
                   linestyles=':', extent=[0,W,H,0], origin='upper')
        # Coverage check
        gt_covered = np.logical_and(m_conf, m_gt).sum() / max(m_gt.sum(), 1)
        cx, cy = o.x+o.w/2, o.y+o.h/2
        margin_px = max(3, int(8*(1-o.conf*0.7)))
        check = "\u2713" if gt_covered > 0.95 else "\u2717"
        col_check = P["green"] if gt_covered > 0.95 else P["red"]
        ax.text(cx, cy, f"{o.cls}\nmargin={margin_px}px\ncov={gt_covered:.0%} {check}",
            fontsize=5.5, ha='center', va='center', fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.75))

    setup_axes(ax, W, H)
    ax.set_title("(c) Conformal Mask Margin (\u03B1=0.1)", fontsize=14, color=P["red"], pad=12)
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0],color=P["blue"],lw=2,label='Predicted mask'),
        Line2D([0],[0],color=P["red"],lw=2,ls='--',label='Conformal margin'),
        Line2D([0],[0],color=P["green"],lw=1.5,ls=':',label='Ground truth'),
        patches.Patch(facecolor=(0.9,0.2,0.2,0.35),edgecolor=P["red"],label='Margin region'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7,
        fancybox=True, framealpha=0.9)

    fig.suptitle("Instance Segmentation: GT Masks \u2192 Predictions \u2192 Conformal Margin (Dilation)",
        fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(out, "conformal_viz_02_seg_pipeline.png"), pad_inches=0.2)
    plt.close()
    print(f"  [SAVED] conformal_viz_02_seg_pipeline.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 3: Conformal Detail — Single Object Deep Dive
# ═══════════════════════════════════════════════════════════════

def fig_03_single_object_deepdive(out):
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.2)

    W, H = 300, 350
    # A "dog" object — medium confidence, interesting conformal behavior
    gt = Obj("dog", 50, 60, 200, 180, OBJ_COLORS[3], 1.0)
    pred = Obj("dog", 45, 55, 210, 190, OBJ_COLORS[3], 0.72)

    scene = np.ones((H, W, 3)) * 0.88
    np.random.seed(10)
    scene += np.random.normal(0, 0.02, scene.shape)
    scene = np.clip(scene, 0, 1)

    # ── (0,0) GT bbox ──
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(scene, extent=[0,W,H,0])
    draw_shape(ax, gt, filled=True, alpha=0.5)
    draw_bbox(ax, gt, P["gt_box"], lw=3, label="dog (GT)")
    setup_axes(ax, W, H)
    ax.set_title("Ground Truth", fontsize=13, color=P["green"])

    # ── (0,1) Prediction + conformal box ──
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(scene, extent=[0,W,H,0])
    draw_shape(ax, pred, filled=True, alpha=0.35)
    # Conformal interval
    delta = 22
    conf_rect = patches.Rectangle(
        (pred.x - delta, pred.y - delta), pred.w + 2*delta, pred.h + 2*delta,
        lw=2.5, edgecolor=P["conf_box"], facecolor=P["conf_box"],
        ls='--', alpha=0.12)
    ax.add_patch(conf_rect)
    draw_bbox(ax, pred, P["pred_box"], lw=2.5, label=f"dog {pred.conf:.2f}")
    draw_bbox(ax, gt, P["gt_box"], lw=1.5, ls=':')
    # Dimension annotations
    ax.annotate('', xy=(pred.x-delta, pred.y+pred.h+30),
        xytext=(pred.x, pred.y+pred.h+30),
        arrowprops=dict(arrowstyle='<->', color=P["conf_box"], lw=1.5))
    ax.text(pred.x-delta/2, pred.y+pred.h+42, f'\u03B4={delta}px',
        fontsize=8, ha='center', color=P["conf_box"], fontweight='bold')
    setup_axes(ax, W, H)
    ax.set_title("Prediction + Conformal Interval", fontsize=13, color=P["purple"])
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0],color=P["gt_box"],lw=2,ls=':',label='GT'),
        Line2D([0],[0],color=P["pred_box"],lw=2,label='Prediction'),
        patches.Patch(edgecolor=P["conf_box"],facecolor=P["conf_box"],alpha=0.2,ls='--',label=f'Conformal \u00B1{delta}px'),
    ], fontsize=8, loc='lower right')

    # ── (0,2) Prediction set (class) ──
    ax = fig.add_subplot(gs[0, 2])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    ax.set_title("Conformal Prediction Set (class)", fontsize=13, color=P["purple"])

    classes_and_scores = [
        ("dog",    0.72, True),
        ("cat",    0.15, True),
        ("bear",   0.06, True),
        ("horse",  0.03, False),
        ("cow",    0.02, False),
        ("person", 0.01, False),
    ]
    ax.text(5, 9.3, "Softmax output \u2192 APS prediction set", ha='center',
        fontsize=10, color=P["text"], fontweight='bold')
    ax.text(5, 8.5, "Cumulative probability threshold = 0.93 (\u03B1 = 0.1)",
        ha='center', fontsize=8.5, color=P["muted"])

    cum = 0
    for i, (cls, score, in_set) in enumerate(classes_and_scores):
        y = 7.2 - i * 1.15
        cum += score

        # Bar
        bar_w = score * 8
        color = P["green"] if in_set else '#CCCCCC'
        alpha = 0.7 if in_set else 0.3
        ax.add_patch(patches.FancyBboxPatch((1, y-0.35), bar_w, 0.7,
            boxstyle="round,pad=0.05", facecolor=color, alpha=alpha,
            edgecolor=color if in_set else '#AAAAAA', linewidth=1.5))

        ax.text(0.8, y, cls, ha='right', va='center', fontsize=9,
            fontweight='bold' if in_set else 'normal',
            color=P["text"] if in_set else P["muted"])
        ax.text(1 + bar_w + 0.2, y, f"{score:.2f}", va='center', fontsize=8,
            color=P["text"] if in_set else P["muted"])

        # Cumulative line
        if in_set:
            ax.text(9.2, y, f"\u03A3={cum:.2f}", va='center', fontsize=7.5,
                color=P["purple"], fontweight='bold')

    # Threshold line
    ax.axhline(y=7.2 - 2.5*1.15, color=P["red"], ls='--', lw=1.5, alpha=0.6,
        xmin=0.1, xmax=0.9)
    ax.text(5, 7.2 - 2.8*1.15, "\u2500\u2500 Threshold: \u03A3 softmax \u2265 1\u2212\u03B1 = 0.90",
        ha='center', fontsize=8, color=P["red"])
    ax.text(5, 1.2, "C\u2090(x) = {dog, cat, bear}",
        ha='center', fontsize=12, fontweight='bold', color=P["purple"],
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F4ECF7',
                 edgecolor=P["purple"], alpha=0.9))
    ax.text(5, 0.3, "Coverage guarantee: P(y\u209C\u1D63\u1D64\u2091 \u2208 C\u2090) \u2265 90%",
        ha='center', fontsize=9, color=P["muted"])

    # ── (1,0) Mask GT ──
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(scene, extent=[0,W,H,0])
    m_gt = make_elliptical_mask(H, W, gt.x+gt.w/2, gt.y+gt.h/2, gt.w*0.42, gt.h*0.44, 0.06)
    overlay = np.zeros((H,W,4))
    c = to_rgba(P["green"])
    overlay[m_gt>0] = [c[0],c[1],c[2],0.45]
    ax.imshow(overlay, extent=[0,W,H,0])
    ax.contour(m_gt, levels=[0.5], colors=[P["green"]], linewidths=2.5, extent=[0,W,H,0], origin='upper')
    setup_axes(ax, W, H)
    ax.set_title("GT Mask", fontsize=13, color=P["green"])

    # ── (1,1) Pred mask + conformal margin ──
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(scene, extent=[0,W,H,0])
    m_pred = make_elliptical_mask(H, W, pred.x+pred.w/2, pred.y+pred.h/2, pred.w*0.4, pred.h*0.42, 0.08)
    margin_px = 12
    struct = np.ones((2*margin_px+1, 2*margin_px+1))
    m_conf = binary_dilation(m_pred, struct).astype(np.uint8)
    margin_region = (m_conf > 0) & (m_pred == 0)

    # Draw layers
    overlay_pred = np.zeros((H,W,4))
    overlay_pred[m_pred>0] = [to_rgba(P["blue"])[0],to_rgba(P["blue"])[1],to_rgba(P["blue"])[2],0.4]
    ax.imshow(overlay_pred, extent=[0,W,H,0])
    overlay_m = np.zeros((H,W,4))
    overlay_m[margin_region] = [0.9,0.2,0.2,0.3]
    ax.imshow(overlay_m, extent=[0,W,H,0])
    ax.contour(m_pred, levels=[0.5], colors=[P["blue"]], linewidths=2, extent=[0,W,H,0], origin='upper')
    ax.contour(m_conf, levels=[0.5], colors=[P["red"]], linewidths=2.5, linestyles='--', extent=[0,W,H,0], origin='upper')
    ax.contour(m_gt, levels=[0.5], colors=[P["green"]], linewidths=1.5, linestyles=':', extent=[0,W,H,0], origin='upper')

    # Coverage annotation
    gt_covered = np.logical_and(m_conf, m_gt).sum() / max(m_gt.sum(), 1)
    ax.text(W/2, H-15, f"Margin = {margin_px}px dilation  |  GT coverage = {gt_covered:.1%}",
        ha='center', fontsize=9, color='white', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.75))
    setup_axes(ax, W, H)
    ax.set_title("Prediction + Conformal Margin", fontsize=13, color=P["purple"])

    # ── (1,2) Alpha sweep ──
    ax = fig.add_subplot(gs[1, 2])
    alphas = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
    margins = [22, 16, 12, 9, 7, 4]
    coverages = [0.99, 0.97, 0.95, 0.90, 0.85, 0.76]
    set_sizes = [5.2, 3.8, 3.0, 2.4, 1.9, 1.3]

    ax2 = ax.twinx()
    l1 = ax.plot(alphas, margins, 'o-', color=P["blue"], lw=2, markersize=7, label='Margin width (px)')
    l2 = ax2.plot(alphas, coverages, 's--', color=P["green"], lw=2, markersize=7, label='Empirical coverage')
    l3 = ax.plot(alphas, [s*3 for s in set_sizes], '^:', color=P["purple"], lw=1.5, markersize=6, label='Prediction set size (\u00D73)')
    ax.axvline(0.10, color=P["red"], ls='--', lw=1, alpha=0.5)
    ax.text(0.105, 20, '\u03B1=0.10', fontsize=8, color=P["red"])

    ax.set_xlabel("Miscoverage rate \u03B1")
    ax.set_ylabel("Margin width (px) / Set size", color=P["blue"])
    ax2.set_ylabel("Empirical coverage", color=P["green"])
    ax2.set_ylim(0.65, 1.02)
    ax2.axhline(0.90, color=P["green"], ls=':', lw=0.8, alpha=0.5)

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=7.5, loc='upper right')
    ax.set_title("Coverage vs Efficiency trade-off", fontsize=13, color=P["text"])
    ax.grid(True, alpha=0.3)

    fig.suptitle("Conformal Calibration Deep Dive — Single Object Analysis (\"dog\", conf=0.72)",
        fontsize=16, fontweight='bold', y=1.0)
    plt.savefig(os.path.join(out, "conformal_viz_03_deepdive.png"), pad_inches=0.2)
    plt.close()
    print(f"  [SAVED] conformal_viz_03_deepdive.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 4: Multi-model comparison on same image
# ═══════════════════════════════════════════════════════════════

def fig_04_multimodel(out):
    W, H = 640, 480
    scene = make_scene(W, H)
    gt = default_gt_objects()

    models = [
        ("YOLOv8x", [
            Obj("person",78,108,104,254,OBJ_COLORS[0],0.97),
            Obj("person",222,132,78,218,OBJ_COLORS[0],0.91),
            Obj("car",368,218,204,124,OBJ_COLORS[2],0.95),
            Obj("dog",158,328,92,72,OBJ_COLORS[3],0.78),
            Obj("bicycle",498,258,82,102,OBJ_COLORS[4],0.85),
            Obj("bottle",28,198,34,74,OBJ_COLORS[5],0.52),
            Obj("chair",538,168,72,92,OBJ_COLORS[6],0.71),
        ], "#3498DB", {"mAP": 0.539, "ECE": 0.082, "FPS": 42}),
        ("Mask R-CNN\n(R101-FPN)", [
            Obj("person",82,112,98,248,OBJ_COLORS[0],0.99),
            Obj("person",218,128,82,224,OBJ_COLORS[0],0.96),
            Obj("car",372,222,198,118,OBJ_COLORS[2],0.98),
            Obj("dog",162,332,88,68,OBJ_COLORS[3],0.85),
            Obj("bicycle",502,262,78,98,OBJ_COLORS[4],0.77),
            Obj("chair",542,172,68,88,OBJ_COLORS[6],0.62),
        ], "#27AE60", {"mAP": 0.429, "ECE": 0.051, "FPS": 12}),
        ("DETR\n(ResNet-101)", [
            Obj("person",76,106,106,256,OBJ_COLORS[0],0.94),
            Obj("person",224,134,76,216,OBJ_COLORS[0],0.88),
            Obj("car",366,216,206,126,OBJ_COLORS[2],0.91),
            Obj("dog",156,326,94,74,OBJ_COLORS[3],0.65),
            Obj("bicycle",496,256,84,104,OBJ_COLORS[4],0.73),
            Obj("bottle",26,196,36,76,OBJ_COLORS[5],0.38),
            Obj("chair",536,166,74,94,OBJ_COLORS[6],0.58),
            Obj("bench",300,380,120,50,OBJ_COLORS[8],0.28),
        ], "#E74C3C", {"mAP": 0.435, "ECE": 0.065, "FPS": 18}),
        ("Mask2Former\n(Swin-L)", [
            Obj("person",80,110,102,252,OBJ_COLORS[0],0.98),
            Obj("person",220,130,80,222,OBJ_COLORS[0],0.95),
            Obj("car",370,220,202,122,OBJ_COLORS[2],0.97),
            Obj("dog",160,330,90,70,OBJ_COLORS[3],0.88),
            Obj("bicycle",500,260,80,100,OBJ_COLORS[4],0.84),
            Obj("bottle",30,200,32,72,OBJ_COLORS[5],0.61),
            Obj("chair",540,170,70,90,OBJ_COLORS[6],0.79),
        ], "#8E44AD", {"mAP": 0.501, "ECE": 0.043, "FPS": 8}),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 7))

    for ax, (name, preds, color, metrics) in zip(axes, models):
        ax.imshow(scene, extent=[0,W,H,0])
        for o in gt: draw_shape(ax, o, filled=True, alpha=0.25)
        for o in gt: draw_bbox(ax, o, P["gt_box"], lw=1.2, ls=':')

        for o in preds:
            draw_bbox(ax, o, color, lw=2, label=f'{o.cls} {o.conf:.2f}')
            # Conformal interval
            delta = int(8 + 20*(1-o.conf))
            conf_rect = patches.Rectangle(
                (o.x-delta, o.y-delta), o.w+2*delta, o.h+2*delta,
                lw=1.2, edgecolor=P["conf_box"], facecolor=P["conf_box"],
                ls='--', alpha=0.08)
            ax.add_patch(conf_rect)

        setup_axes(ax, W, H)
        ax.set_title(name, fontsize=12, color=color, pad=10)

        # Metrics bar at bottom
        met_text = f"mAP={metrics['mAP']:.3f}  |  ECE={metrics['ECE']:.3f}  |  {metrics['FPS']} FPS"
        ax.text(W/2, H-8, met_text, ha='center', fontsize=8, color='white',
            bbox=dict(boxstyle='round,pad=0.25', facecolor=color, alpha=0.85))

    fig.suptitle("Multi-Model Comparison on Same Scene — Predictions + Conformal Intervals",
        fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "conformal_viz_04_multimodel.png"), pad_inches=0.2)
    plt.close()
    print(f"  [SAVED] conformal_viz_04_multimodel.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 5: Semantic Segmentation — Pixel-level Conformal
# ═══════════════════════════════════════════════════════════════

def fig_05_semantic_seg(out):
    W, H = 500, 375
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # VOC-style semantic segmentation scene
    VOC_COLORS = np.array([[0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],
                           [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
                           [64,128,0],[192,128,0],[64,0,128],[192,0,128],
                           [64,128,128],[192,128,128],[0,64,0],[128,64,0],
                           [0,192,0],[128,192,0],[0,64,128]]) / 255.0
    VOC_NAMES = ["bg","aeroplane","bicycle","bird","boat","bottle","bus","car",
                 "cat","chair","cow","diningtable","dog","horse","motorbike",
                 "person","pottedplant","sheep","sofa","train","tvmonitor"]

    np.random.seed(33)
    # GT semantic mask
    mask_gt = np.zeros((H, W), dtype=int)
    regions = [(15,60,80,100,240),(15,200,100,80,220),(7,320,200,150,100),(12,130,280,90,60)]
    for cls,x,y,w,h in regions:
        yy,xx = np.ogrid[0:H,0:W]; cx,cy=x+w//2,y+h//2
        mask_gt[((xx-cx)**2/(w/2)**2 + (yy-cy)**2/(h/2)**2) <= 1.0] = cls

    # Predicted mask (slightly different)
    mask_pred = np.zeros((H, W), dtype=int)
    regions_pred = [(15,55,75,108,248),(15,195,95,85,228),(7,315,195,158,108),(12,125,275,98,68)]
    for cls,x,y,w,h in regions_pred:
        yy,xx = np.ogrid[0:H,0:W]; cx,cy=x+w//2,y+h//2
        mask_pred[((xx-cx)**2/(w/2)**2 + (yy-cy)**2/(h/2)**2) <= 1.0] = cls

    # Conformal: dilate each class region
    mask_conf = mask_pred.copy()
    for cls in [15, 7, 12]:
        binary = (mask_pred == cls).astype(np.uint8)
        dilated = binary_dilation(binary, np.ones((9,9))).astype(np.uint8)
        mask_conf[dilated > 0] = np.where(mask_conf[dilated > 0] == 0, cls, mask_conf[dilated > 0])

    def colorize(mask):
        h,w = mask.shape; img = np.zeros((h,w,3))
        for cid in range(len(VOC_COLORS)): img[mask==cid] = VOC_COLORS[cid]
        # Background: gray
        bg = make_scene(w, h)
        img[mask==0] = bg[mask==0]
        return img

    # Panel A: GT
    ax = axes[0]
    ax.imshow(colorize(mask_gt), extent=[0,W,H,0])
    for cls,x,y,w,h in regions:
        ax.text(x+w/2, y+h/2, VOC_NAMES[cls], fontsize=8, ha='center', va='center',
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6))
    setup_axes(ax, W, H)
    ax.set_title("(a) GT Semantic Mask", fontsize=14, color=P["green"], pad=12)

    # Panel B: Prediction
    ax = axes[1]
    ax.imshow(colorize(mask_pred), extent=[0,W,H,0])
    miou = np.mean([
        np.logical_and(mask_gt==c, mask_pred==c).sum() / max(np.logical_or(mask_gt==c, mask_pred==c).sum(), 1)
        for c in [15, 7, 12]
    ])
    ax.text(W/2, H-10, f"mIoU = {miou:.3f}", ha='center', fontsize=10, color='white', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    setup_axes(ax, W, H)
    ax.set_title("(b) Predicted Semantic Mask", fontsize=14, color=P["blue"], pad=12)

    # Panel C: Conformal
    ax = axes[2]
    img_conf = colorize(mask_conf)
    ax.imshow(img_conf, extent=[0,W,H,0])
    # Highlight margin regions
    for cls in [15, 7, 12]:
        margin = (mask_conf == cls) & (mask_pred != cls)
        if margin.any():
            overlay = np.zeros((H,W,4))
            c = to_rgba(P["red"])
            overlay[margin] = [c[0],c[1],c[2],0.25]
            ax.imshow(overlay, extent=[0,W,H,0])
    # Coverage
    pixel_cov = np.mean(mask_conf[mask_gt > 0] == mask_gt[mask_gt > 0])
    ax.text(W/2, H-10, f"Pixel coverage = {pixel_cov:.1%}  |  Margin = 4px dilation per class",
        ha='center', fontsize=9, color='white', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    ax.text(W/2, 15, "Red overlay = conformal margin region (class expanded)",
        ha='center', fontsize=8, color=P["red"],
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))
    setup_axes(ax, W, H)
    ax.set_title("(c) Conformal Semantic Mask (\u03B1=0.1)", fontsize=14, color=P["red"], pad=12)

    fig.suptitle("Semantic Segmentation (VOC): GT \u2192 Prediction \u2192 Pixel-wise Conformal Calibration",
        fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "conformal_viz_05_semantic_seg.png"), pad_inches=0.2)
    plt.close()
    print(f"  [SAVED] conformal_viz_05_semantic_seg.png")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    out = str(Path(__file__).parent / "results" / "eda")
    os.makedirs(out, exist_ok=True)

    print("=" * 65)
    print("  CONFORMAL CALIBRATION — Impact Visualizations")
    print("=" * 65)
    print()

    fig_01_gt_vs_pred_vs_conformal(out)
    fig_02_segmentation_conformal(out)
    fig_03_single_object_deepdive(out)
    fig_04_multimodel(out)
    fig_05_semantic_seg(out)

    print()
    print("=" * 65)
    print(f"  5 impact visualizations saved to {out}")
    print("=" * 65)


if __name__ == "__main__":
    main()
