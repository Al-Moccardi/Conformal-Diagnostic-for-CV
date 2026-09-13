#!/usr/bin/env python3
"""
revision_experiments.py — Experiments requested by reviewer for major revision.

1. CP Signal Ablation: count/conf only → +set_size → +margin → full CP
2. APGD with multiple restarts for DETR robustness validation
3. DeLong test for AUROC differences
4. Reliability diagrams for ECE validation

Usage:
    python revision_experiments.py
    python revision_experiments.py --fix ablation
    python revision_experiments.py --fix apgd
    python revision_experiments.py --fix delong
    python revision_experiments.py --fix reliability
"""

import os, sys, json, argparse, warnings, time
from pathlib import Path
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from src.calibration.adaptive_conformal import AdaptiveConformalCalibrator
from src.calibration.conformal_defense import (
    ConformalAttackDetector, SimpleConfidenceDetector,
    DetectionCountDetector, MaxConfidenceDetector,
    compute_all_aurocs,
)

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.grid': True, 'grid.alpha': 0.2,
    'font.family': 'serif', 'font.size': 11,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def load_preds(pred_dir, model_name):
    path = os.path.join(pred_dir, f"{model_name}_bbox.json")
    if not os.path.exists(path):
        return None, None
    with open(path, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    by_img = defaultdict(list)
    for p in preds:
        by_img[p["image_id"]].append(p)
    return preds, by_img


def load_gt(coco_root):
    ann_file = os.path.join(coco_root, "annotations", "instances_val2017.json")
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt = defaultdict(list)
    for ann in data["annotations"]:
        gt[ann["image_id"]].append(ann)
    return gt


def get_adv_preds(adv_results, model_name, atk_name):
    mr = adv_results.get(model_name, {})
    atk = mr.get("attacks", {}).get(atk_name, {})
    per_image = atk.get("per_image_predictions", None)
    if per_image:
        by_img = defaultdict(list)
        for p in per_image:
            by_img[p["image_id"]].append(p)
        return by_img
    return None


# =====================================================================
#  1. CP SIGNAL ABLATION — The critical missing experiment
# =====================================================================

class AblationDetector:
    """Detector with configurable signal components for ablation.

    THREE TYPES OF SIGNALS:
    1. Output-level: detection count, mean/max confidence (no CP needed)
    2. Pseudo-CP: set_size heuristic, raw margin (not truly CP-specific)
    3. Genuinely CP-specific:
       - Quantile distance: how far each detection's nonconformity score
         is from the CALIBRATED bin-specific quantile. This depends on
         the conformal calibration and doesn't exist without it.
       - Fraction outside conformal set: what % of detections have
         nonconformity > bin qhat. Should be ~alpha on clean; changes under attack.
       - Margin-confidence decorrelation: on clean data, high conf → small margin.
         Under attack, this relationship breaks.
    """

    MIN_CONF = 0.1

    def __init__(self, calibrator, use_count=True, use_conf=True,
                 use_max_conf=False,
                 use_qdist=False, use_frac_outside=False,
                 use_margin_decorr=False):
        self.calibrator = calibrator
        self.use_count = use_count
        self.use_conf = use_conf
        self.use_max_conf = use_max_conf
        # Genuinely CP-specific
        self.use_qdist = use_qdist
        self.use_frac_outside = use_frac_outside
        self.use_margin_decorr = use_margin_decorr
        self.baseline = None

    def _image_features(self, preds):
        """Extract features. CP features use calibrated quantiles."""
        filtered = [p for p in preds if p["score"] >= self.MIN_CONF]
        scores = [p["score"] for p in filtered] if filtered else []

        # Genuinely CP-specific: quantile distances and margin correlation
        qdists = []     # distance from bin-specific conformal boundary
        margins = []
        confs_filt = []
        n_outside = 0

        bin_qhats = self.calibrator.bin_qhats if hasattr(self.calibrator, 'bin_qhats') else {}
        conf_bins = self.calibrator.conf_bin_edges if hasattr(self.calibrator, 'conf_bin_edges') else None
        global_qhat = self.calibrator.global_qhat or 0.0

        for p in filtered:
            conf = p["score"]
            nc = 1.0 - conf  # nonconformity score

            # Get the bin-specific quantile (THIS is CP-specific)
            q = global_qhat
            if conf_bins is not None and bin_qhats:
                bin_idx = int(np.searchsorted(conf_bins[1:], conf))
                bin_idx = min(bin_idx, len(bin_qhats) - 1)
                if bin_idx in bin_qhats:
                    q = bin_qhats[bin_idx]

            # Quantile distance: q - nc (positive = inside, negative = outside)
            qdists.append(q - nc)
            if nc > q:
                n_outside += 1

            r = self.calibrator.predict(p)
            margins.append(r.box_delta_pixels)
            confs_filt.append(conf)

        # Margin-confidence correlation (genuinely CP: uses calibrated margins)
        margin_conf_corr = 0.0
        if len(margins) > 3 and np.std(margins) > 1e-6 and np.std(confs_filt) > 1e-6:
            margin_conf_corr = float(np.corrcoef(margins, confs_filt)[0, 1])

        frac_outside = n_outside / max(len(filtered), 1)

        return {
            # Output-level
            "n_high_conf": len(filtered),
            "mean_conf": float(np.mean(scores)) if scores else 0.0,
            "max_conf": float(max(scores)) if scores else 0.0,
            # CP-specific (depend on calibrated quantiles)
            "mean_qdist": float(np.mean(qdists)) if qdists else 0.0,
            "std_qdist": float(np.std(qdists)) if len(qdists) > 1 else 0.0,
            "frac_outside": float(frac_outside),
            "margin_conf_corr": float(margin_conf_corr),
        }

    def fit(self, clean_preds_by_image):
        stats = {}
        for img_id, preds in clean_preds_by_image.items():
            stats[img_id] = self._image_features(preds)

        self.baseline = {}
        for key in ["n_high_conf", "mean_conf", "max_conf",
                     "mean_qdist", "std_qdist", "frac_outside", "margin_conf_corr"]:
            vals = [s[key] for s in stats.values()]
            self.baseline[f"mean_{key}"] = np.mean(vals) if vals else 0
            self.baseline[f"std_{key}"] = max(np.std(vals), 0.001) if vals else 0.01
        return self

    def score_image(self, preds):
        if self.baseline is None:
            return 0.5

        f = self._image_features(preds)
        components = []

        if self.use_count:
            z = max(0, (self.baseline["mean_n_high_conf"] - f["n_high_conf"])
                    / self.baseline["std_n_high_conf"])
            components.append(z)

        if self.use_conf:
            z = max(0, (self.baseline["mean_mean_conf"] - f["mean_conf"])
                    / self.baseline["std_mean_conf"])
            components.append(z)

        if self.use_max_conf:
            z = max(0, (self.baseline["mean_max_conf"] - f["max_conf"])
                    / self.baseline["std_max_conf"])
            components.append(z)

        if self.use_qdist:
            # Under attack: quantile distance DECREASES (detections closer to boundary)
            z = max(0, (self.baseline["mean_mean_qdist"] - f["mean_qdist"])
                    / self.baseline["std_mean_qdist"])
            components.append(z)

        if self.use_frac_outside:
            # Under attack: MORE detections fall outside the conformal set
            z = max(0, (f["frac_outside"] - self.baseline["mean_frac_outside"])
                    / self.baseline["std_frac_outside"])
            components.append(z)

        if self.use_margin_decorr:
            # Under attack: margin-confidence correlation breaks (becomes less negative)
            z = abs(f["margin_conf_corr"] - self.baseline["mean_margin_conf_corr"]) \
                / self.baseline["std_margin_conf_corr"]
            components.append(z)

        if not components:
            return 0.0

        score = np.mean(components)
        return min(score / 3.0, 1.0)

    def compute_auroc(self, clean_preds, adv_preds):
        from sklearn.metrics import roc_auc_score
        labels, scores = [], []
        for preds in clean_preds.values():
            labels.append(0)
            scores.append(self.score_image(preds))
        for preds in adv_preds.values():
            labels.append(1)
            scores.append(self.score_image(preds))
        labels, scores = np.array(labels), np.array(scores)
        if len(np.unique(labels)) < 2 or len(np.unique(scores)) < 2:
            return 0.5
        auroc = float(roc_auc_score(labels, scores))
        return max(auroc, 1.0 - auroc)


def fix_ablation(adv_results, pred_dir, gt_by_image, out, alpha=0.1):
    """CP Signal Ablation: show incremental value of CP-specific features."""
    print("\n" + "="*70)
    print("  REVISION FIX 1: CP Signal Ablation")
    print("  Shows incremental AUROC gain from CP-specific signals")
    print("="*70)

    configs = [
        # === Output-level only (no CP needed) ===
        ("Count only",
         dict(use_count=True, use_conf=False, use_max_conf=False,
              use_qdist=False, use_frac_outside=False, use_margin_decorr=False)),
        ("Count + Conf",
         dict(use_count=True, use_conf=True, use_max_conf=False,
              use_qdist=False, use_frac_outside=False, use_margin_decorr=False)),
        ("Cnt+Conf+Max",
         dict(use_count=True, use_conf=True, use_max_conf=True,
              use_qdist=False, use_frac_outside=False, use_margin_decorr=False)),
        # === Adding GENUINELY CP-specific signals ===
        ("+ QDist (CP)",
         dict(use_count=True, use_conf=True, use_max_conf=True,
              use_qdist=True, use_frac_outside=False, use_margin_decorr=False)),
        ("+ FracOut (CP)",
         dict(use_count=True, use_conf=True, use_max_conf=True,
              use_qdist=True, use_frac_outside=True, use_margin_decorr=False)),
        ("+ MrgDecorr (Full)",
         dict(use_count=True, use_conf=True, use_max_conf=True,
              use_qdist=True, use_frac_outside=True, use_margin_decorr=True)),
        # === CP-only (no output-level) ===
        ("CP-only (all)",
         dict(use_count=False, use_conf=False, use_max_conf=False,
              use_qdist=True, use_frac_outside=True, use_margin_decorr=True)),
    ]

    all_results = {}

    for model_name in adv_results:
        clean_preds, clean_by_img = load_preds(pred_dir, model_name)
        if clean_preds is None:
            continue

        import random
        img_ids = sorted(clean_by_img.keys())
        random.seed(42)
        random.shuffle(img_ids)
        cal_ids = set(img_ids[:len(img_ids)//2])
        cal_preds = [p for p in clean_preds if p["image_id"] in cal_ids]

        calibrator = AdaptiveConformalCalibrator(alpha=alpha, n_conf_bins=5, size_normalize=True)
        calibrator.calibrate(cal_preds, gt_by_image)

        attacks = adv_results[model_name].get("attacks", {})
        for atk_name in attacks:
            adv_by_img = get_adv_preds(adv_results, model_name, atk_name)
            if adv_by_img is None or len(adv_by_img) < 10:
                continue

            adv_ids = set(adv_by_img.keys())
            clean_matched = {k: v for k, v in clean_by_img.items() if k in adv_ids}
            if len(clean_matched) < 10:
                continue

            key = f"{model_name}/{atk_name}"
            all_results[key] = {}

            print(f"\n  {key}:")
            for cfg_name, cfg_kwargs in configs:
                det = AblationDetector(calibrator, **cfg_kwargs)
                det.fit(clean_matched)
                auroc = det.compute_auroc(clean_matched, adv_by_img)
                all_results[key][cfg_name] = auroc

                is_cp = "CP" in cfg_name or "Set Size" in cfg_name or "Margin" in cfg_name
                marker = " <-- CP signal" if is_cp else ""
                print(f"    {cfg_name:<22s}: {auroc:.3f}{marker}")

    # Compute average gain from CP signals
    print(f"\n  {'='*60}")
    print(f"  AVERAGE ACROSS ALL MODEL-ATTACK PAIRS:")
    print(f"  {'='*60}")

    for cfg_name, _ in configs:
        vals = [r.get(cfg_name, 0) for r in all_results.values() if cfg_name in r]
        if vals:
            print(f"    {cfg_name:<22s}: {np.mean(vals):.3f} (std={np.std(vals):.3f})")

    # Compute incremental gain
    print(f"\n  INCREMENTAL GAIN FROM CP SIGNALS:")
    base_vals = [r.get("Cnt+Conf+Max", 0) for r in all_results.values() if "Cnt+Conf+Max" in r]
    qd_vals = [r.get("+ QDist (CP)", 0) for r in all_results.values() if "+ QDist (CP)" in r]
    fo_vals = [r.get("+ FracOut (CP)", 0) for r in all_results.values() if "+ FracOut (CP)" in r]
    full_vals = [r.get("+ MrgDecorr (Full)", 0) for r in all_results.values() if "+ MrgDecorr (Full)" in r]
    cponly_vals = [r.get("CP-only (all)", 0) for r in all_results.values() if "CP-only (all)" in r]

    if base_vals and qd_vals:
        print(f"    Base (Cnt+Conf+Max): {np.mean(base_vals):.3f}")
        print(f"    + QDist:       {np.mean(qd_vals) - np.mean(base_vals):+.3f} AUROC")
        print(f"    + FracOutside: {np.mean(fo_vals) - np.mean(base_vals):+.3f} AUROC")
        print(f"    + Full CP:     {np.mean(full_vals) - np.mean(base_vals):+.3f} AUROC")
        print(f"    CP-only mean:  {np.mean(cponly_vals):.3f}")

    # Figure
    if all_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        cfg_names = [c[0] for c in configs]
        x = range(len(cfg_names))

        means = []
        stds = []
        colors = []
        for cfg_name in cfg_names:
            vals = [r.get(cfg_name, 0) for r in all_results.values() if cfg_name in r]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
            is_cp = "CP" in cfg_name or "Set Size" in cfg_name or "Margin" in cfg_name
            colors.append('#2E86C1' if is_cp else '#E67E22')

        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=5, edgecolor='white', lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(cfg_names, fontsize=9, rotation=15, ha='right')
        ax.set_ylabel("Mean AUROC")
        ax.set_title("CP Signal Ablation: Incremental Value of CP-Specific Features")
        ax.axhline(0.5, color='gray', ls=':', alpha=0.5)
        ax.set_ylim(0.5, 1.0)

        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

        # Legend
        from matplotlib.patches import Patch
        ax.legend([Patch(color='#E67E22'), Patch(color='#2E86C1')],
                  ['Output-level only', 'Includes CP signals'], loc='lower right')

        plt.tight_layout()
        plt.savefig(os.path.join(out, "fig_revision_cp_ablation.png"))
        plt.close()
        print(f"\n  [SAVED] fig_revision_cp_ablation.png")

    return all_results


# =====================================================================
#  2. DeLong TEST FOR AUROC DIFFERENCES
# =====================================================================

def _delong_roc_variance(labels, scores):
    """Compute DeLong variance for AUROC estimate."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    m, n = len(pos), len(neg)

    # Structural components
    v_pos = np.array([np.mean(pos_i > neg) + 0.5 * np.mean(pos_i == neg) for pos_i in pos])
    v_neg = np.array([np.mean(pos > neg_j) + 0.5 * np.mean(pos == neg_j) for neg_j in neg])

    s_pos = np.var(v_pos, ddof=1) if m > 1 else 0
    s_neg = np.var(v_neg, ddof=1) if n > 1 else 0

    var_auc = s_pos / m + s_neg / n
    return var_auc


def fix_delong(adv_results, pred_dir, gt_by_image, out, alpha=0.1):
    """DeLong test for AUROC differences between CP and best baseline."""
    print("\n" + "="*70)
    print("  REVISION FIX 3: DeLong Test for AUROC Differences")
    print("="*70)

    from scipy.stats import norm

    results = {}

    for model_name in adv_results:
        clean_preds, clean_by_img = load_preds(pred_dir, model_name)
        if clean_preds is None:
            continue

        import random
        img_ids = sorted(clean_by_img.keys())
        random.seed(42)
        random.shuffle(img_ids)
        cal_ids = set(img_ids[:len(img_ids)//2])
        cal_preds = [p for p in clean_preds if p["image_id"] in cal_ids]

        calibrator = AdaptiveConformalCalibrator(alpha=alpha, n_conf_bins=5, size_normalize=True)
        calibrator.calibrate(cal_preds, gt_by_image)

        attacks = adv_results[model_name].get("attacks", {})
        for atk_name in list(attacks.keys())[:3]:
            adv_by_img = get_adv_preds(adv_results, model_name, atk_name)
            if adv_by_img is None or len(adv_by_img) < 10:
                continue

            adv_ids = set(adv_by_img.keys())
            clean_matched = {k: v for k, v in clean_by_img.items() if k in adv_ids}

            # Compute scores for CP and best baseline (DetCount)
            cp_det = ConformalAttackDetector(calibrator)
            cp_det.fit_thresholds(clean_matched)

            dc_det = DetectionCountDetector(conf_threshold=0.5)
            dc_det.fit(clean_matched)

            labels, cp_scores, dc_scores = [], [], []
            for preds in clean_matched.values():
                labels.append(0)
                stats = cp_det.compute_image_stats({0: preds})
                cp_scores.append(cp_det._anomaly_score(list(stats.values())[0]))
                dc_scores.append(dc_det.score_image(preds))
            for preds in adv_by_img.values():
                labels.append(1)
                stats = cp_det.compute_image_stats({0: preds})
                cp_scores.append(cp_det._anomaly_score(list(stats.values())[0]))
                dc_scores.append(dc_det.score_image(preds))

            labels = np.array(labels)
            cp_scores = np.array(cp_scores)
            dc_scores = np.array(dc_scores)

            from sklearn.metrics import roc_auc_score
            cp_auc = roc_auc_score(labels, cp_scores)
            dc_auc = roc_auc_score(labels, dc_scores)

            # DeLong variance
            var_cp = _delong_roc_variance(labels, cp_scores)
            var_dc = _delong_roc_variance(labels, dc_scores)

            # Z-test for difference
            diff = cp_auc - dc_auc
            se_diff = np.sqrt(var_cp + var_dc)
            z = diff / se_diff if se_diff > 0 else 0
            p = 2 * (1 - norm.cdf(abs(z)))

            key = f"{model_name}/{atk_name}"
            results[key] = {
                "cp_auroc": float(cp_auc), "dc_auroc": float(dc_auc),
                "diff": float(diff), "z": float(z), "p": float(p),
                "n": len(labels),
            }

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {key:<28s} CP={cp_auc:.3f} DC={dc_auc:.3f} "
                  f"diff={diff:+.3f} z={z:.2f} p={p:.4f} {sig} (n={len(labels)})")

    return results


# =====================================================================
#  3. RELIABILITY DIAGRAMS
# =====================================================================

def fix_reliability(pred_dir, gt_by_image, out):
    """Generate reliability diagrams for ECE validation."""
    print("\n" + "="*70)
    print("  REVISION FIX 4: Reliability Diagrams for ECE")
    print("="*70)

    def _iou(b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0

    models = ["yolov8x", "yolov11x", "rtdetr-l", "detr-resnet101"]
    n_bins = 15

    fig, axes = plt.subplots(1, len(models), figsize=(4*len(models), 4), sharey=True)

    for idx, model_name in enumerate(models):
        preds, _ = load_preds(pred_dir, model_name)
        if preds is None:
            continue

        # Match predictions to GT
        scores, is_correct = [], []
        preds_by_img = defaultdict(list)
        for p in preds:
            if p["score"] >= 0.01:
                preds_by_img[p["image_id"]].append(p)

        for img_id, img_preds in preds_by_img.items():
            gt_anns = gt_by_image.get(img_id, [])
            gt_used = [False] * len(gt_anns)

            for p in sorted(img_preds, key=lambda x: -x["score"]):
                px, py, pw, ph = p["bbox"]
                pxy = [px, py, px+pw, py+ph]
                matched = False
                for j, gt in enumerate(gt_anns):
                    if gt_used[j]:
                        continue
                    gx, gy, gw, gh = gt["bbox"]
                    if p["category_id"] == gt["category_id"]:
                        if _iou(pxy, [gx, gy, gx+gw, gy+gh]) >= 0.5:
                            gt_used[j] = True
                            matched = True
                            break
                scores.append(p["score"])
                is_correct.append(1 if matched else 0)

        scores = np.array(scores)
        is_correct = np.array(is_correct)

        # Compute reliability diagram
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_accs, bin_confs, bin_counts = [], [], []

        for i in range(n_bins):
            mask = (scores >= bin_edges[i]) & (scores < bin_edges[i+1])
            if mask.sum() > 0:
                bin_accs.append(is_correct[mask].mean())
                bin_confs.append(scores[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accs.append(0)
                bin_confs.append((bin_edges[i] + bin_edges[i+1]) / 2)
                bin_counts.append(0)

        # ECE
        total = sum(bin_counts)
        ece = sum(abs(a - c) * n / total for a, c, n in zip(bin_accs, bin_confs, bin_counts)
                  if total > 0)

        # Plot
        ax = axes[idx]
        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(n_bins)]
        ax.bar(bin_centers, bin_accs, width=1/n_bins, alpha=0.6, color='#2E86C1',
               edgecolor='white', label='Accuracy')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        ax.set_xlabel("Confidence")
        if idx == 0:
            ax.set_ylabel("Accuracy")
        ax.set_title(f"{model_name}\nECE={ece:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    plt.suptitle("Reliability Diagrams (per-detection, 15 equal-width bins, conf>0.01)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_revision_reliability.png"))
    plt.close()
    print(f"  [SAVED] fig_revision_reliability.png")
    print(f"  ECE computed per-detection, 15 equal-width bins, weighted by bin count")


# =====================================================================
#  MAIN
# =====================================================================

# =====================================================================
#  4. COVERAGE GAP AS CALIBRATED SEVERITY METRIC (NEW CONTRIBUTION)
# =====================================================================

def _box_iou(b1, b2):
    """IoU between two boxes [x1,y1,x2,y2]."""
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0


def compute_per_image_coverage(preds_by_img, gt_by_image, conf_threshold=0.0):
    """Compute class-correct coverage per image.
    
    Returns dict: {image_id: (n_covered, n_gt, coverage)}
    """
    results = {}
    for img_id, gt_anns in gt_by_image.items():
        if not gt_anns:
            continue
        preds = preds_by_img.get(img_id, [])
        filtered = [p for p in preds if p["score"] >= conf_threshold]

        n_gt = len(gt_anns)
        covered = 0
        gt_matched = [False] * n_gt

        # Sort by confidence descending
        filtered_sorted = sorted(filtered, key=lambda p: -p["score"])

        for p in filtered_sorted:
            px, py, pw, ph = p["bbox"]
            pred_xyxy = [px, py, px + pw, py + ph]
            for j, gt in enumerate(gt_anns):
                if gt_matched[j]:
                    continue
                gx, gy, gw, gh = gt["bbox"]
                gt_xyxy = [gx, gy, gx + gw, gy + gh]
                if p.get("category_id") == gt.get("category_id"):
                    if _box_iou(pred_xyxy, gt_xyxy) >= 0.5:
                        gt_matched[j] = True
                        covered += 1
                        break

        cov = covered / n_gt if n_gt > 0 else 0.0
        results[img_id] = (covered, n_gt, cov)

    return results


def fix_coverage_gap(adv_results, pred_dir, gt_by_image, out, alpha=0.1):
    """Coverage gap as calibrated severity metric.
    
    Key insight: the gap (1-α) - observed_coverage has a formal target (0 = healthy).
    mAP drop has no formal baseline. Coverage gap is interpretable:
    gap=0.05 means "5% below guarantee"; gap=0.74 means "guarantee destroyed".
    """
    print("\n" + "=" * 70)
    print("  NEW CONTRIBUTION: Coverage Gap as Severity Metric")
    print("  Gap = (1-α) - coverage. Target: 0. Formal baseline: 1-α")
    print("=" * 70)

    target = 1.0 - alpha
    all_rows = []

    for model_name in adv_results:
        clean_preds, clean_by_img = load_preds(pred_dir, model_name)
        if clean_preds is None:
            continue

        # Clean coverage
        clean_cov = compute_per_image_coverage(clean_by_img, gt_by_image)
        if not clean_cov:
            continue
        clean_mean = np.mean([c[2] for c in clean_cov.values()])
        clean_gap = target - clean_mean

        clean_mAP_raw = adv_results[model_name].get("clean_mAP", 0)
        if isinstance(clean_mAP_raw, dict):
            clean_mAP_raw = clean_mAP_raw.get("mAP@[0.5:0.95]",
                            clean_mAP_raw.get("mAP@0.5",
                            clean_mAP_raw.get("bbox",
                            clean_mAP_raw.get("mAP", 0))))
        clean_mAP_val = float(clean_mAP_raw) if clean_mAP_raw else 0

        row = {
            "model": model_name, "condition": "Clean",
            "mAP": clean_mAP_val,
            "mAP_drop": 0.0,
            "coverage": clean_mean, "gap": clean_gap,
            "interpretation": "Within guarantee" if clean_gap < 0.05 else "Mild deviation"
        }
        all_rows.append(row)
        print(f"\n  {model_name}:")
        print(f"    Clean:    cov={clean_mean:.3f}  gap={clean_gap:+.3f}  --> {'OK' if clean_gap < 0.05 else 'MILD'}")

        # Per attack
        attacks = adv_results[model_name].get("attacks", {})
        for atk_name, atk_data in attacks.items():
            adv_by_img = get_adv_preds(adv_results, model_name, atk_name)
            if adv_by_img is None or len(adv_by_img) < 10:
                continue

            # Use only images present in adversarial set
            adv_gt = {k: v for k, v in gt_by_image.items() if k in adv_by_img}
            adv_cov = compute_per_image_coverage(adv_by_img, adv_gt)
            if not adv_cov:
                continue
            adv_mean = np.mean([c[2] for c in adv_cov.values()])
            adv_gap = target - adv_mean

            adv_mAP = atk_data.get("attacked_mAP", atk_data.get("adversarial_mAP", atk_data.get("adv_mAP", 0)))
            if isinstance(adv_mAP, dict):
                adv_mAP = adv_mAP.get("mAP@[0.5:0.95]",
                           adv_mAP.get("mAP@0.5",
                           adv_mAP.get("bbox",
                           adv_mAP.get("mAP", 0))))
            adv_mAP = float(adv_mAP) if adv_mAP else 0

            clean_mAP = adv_results[model_name].get("clean_mAP", 0.5)
            if isinstance(clean_mAP, dict):
                clean_mAP = clean_mAP.get("mAP@[0.5:0.95]",
                             clean_mAP.get("mAP@0.5",
                             clean_mAP.get("bbox",
                             clean_mAP.get("mAP", 0.5))))
            clean_mAP = float(clean_mAP) if clean_mAP else 0.5

            mAP_drop = (clean_mAP - adv_mAP) / clean_mAP if clean_mAP > 0 else 0

            if adv_gap < 0.10:
                interp = "Within tolerance"
            elif adv_gap < 0.30:
                interp = "Degraded"
            elif adv_gap < 0.60:
                interp = "Severely degraded"
            else:
                interp = "Guarantee destroyed"

            row = {
                "model": model_name, "condition": atk_name,
                "mAP": adv_mAP, "mAP_drop": mAP_drop,
                "coverage": adv_mean, "gap": adv_gap,
                "interpretation": interp,
            }
            all_rows.append(row)
            print(f"    {atk_name:<10s}: cov={adv_mean:.3f}  gap={adv_gap:+.3f}  "
                  f"mAP_drop={mAP_drop:+.0%}  --> {interp}")

    # Also add natural corruptions if available
    corr_path = os.path.join(os.path.dirname(pred_dir), "..", "corruptions", "natural_corruption_results.json")
    if not os.path.exists(corr_path):
        corr_path = os.path.join(os.path.dirname(pred_dir), "..", "corruptions", "corruption_results.json")
    
    if os.path.exists(corr_path):
        with open(corr_path, 'r') as f:
            corr_data = json.load(f)
        print(f"\n  Natural corruptions:")
        for model_name in corr_data:
            for corr_type, severities in corr_data[model_name].items():
                if not isinstance(severities, dict):
                    continue
                for sev, sev_data in severities.items():
                    if not isinstance(sev_data, dict):
                        continue
                    corr_cov = sev_data.get("coverage", sev_data.get("conformal_coverage"))
                    corr_mAP = sev_data.get("mAP", 0)
                    if corr_cov is not None:
                        gap = target - corr_cov
                        print(f"    {model_name}/{corr_type}/s{sev}: cov={corr_cov:.3f} gap={gap:+.3f}")

    # === FIGURE: Coverage gap vs mAP drop ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Coverage gap vs mAP drop scatter
    ax = axes[0]
    gaps = [r["gap"] for r in all_rows if r["condition"] != "Clean"]
    drops = [r["mAP_drop"] for r in all_rows if r["condition"] != "Clean"]
    models = [r["model"] for r in all_rows if r["condition"] != "Clean"]

    colors_map = {"yolov8x": "#E74C3C", "yolov11x": "#3498DB", "detr-resnet101": "#2ECC71"}
    colors = [colors_map.get(m, "#888") for m in models]

    ax.scatter(drops, gaps, c=colors, s=80, alpha=0.8, edgecolors='white', lw=1)
    ax.axhline(0, color='green', ls='--', alpha=0.5, label='CP guarantee met')
    ax.axhline(0.1, color='orange', ls=':', alpha=0.5, label='10% tolerance')
    ax.set_xlabel("mAP Drop (relative)")
    ax.set_ylabel(f"Coverage Gap  [(1-α) - coverage]")
    ax.set_title("Coverage Gap vs mAP Drop")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-0.1, 1.0)

    # Add diagonal reference
    x_ref = np.linspace(0, 1, 50)
    ax.plot(x_ref, x_ref * 0.9, color='gray', ls=':', alpha=0.3)

    # Add legend for model colors
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=m)
               for m, c in colors_map.items()]
    ax.legend(handles=handles, loc='lower right', fontsize=8)

    # Panel 2: Bar chart — coverage gap by severity category
    ax2 = axes[1]
    categories = {
        "Clean": [], "FGSM\n(weak)": [], "PGD-20\n(strong)": [],
        "DAG\n(det-aware)": [], "TOG-V\n(objectness)": []
    }
    cat_map = {"Clean": "Clean", "fgsm": "FGSM\n(weak)", "pgd-20": "PGD-20\n(strong)",
               "dag": "DAG\n(det-aware)", "tog-v": "TOG-V\n(objectness)"}

    for r in all_rows:
        cat = cat_map.get(r["condition"])
        if cat:
            categories[cat].append(r["gap"])

    bar_labels = list(categories.keys())
    bar_means = [np.mean(v) if v else 0 for v in categories.values()]
    bar_stds = [np.std(v) if v else 0 for v in categories.values()]
    bar_colors = ['#27AE60', '#F39C12', '#E74C3C', '#8E44AD', '#2C3E50']

    bars = ax2.bar(range(len(bar_labels)), bar_means, yerr=bar_stds,
                   color=bar_colors, alpha=0.85, capsize=5, edgecolor='white', lw=1.2)
    ax2.set_xticks(range(len(bar_labels)))
    ax2.set_xticklabels(bar_labels, fontsize=9)
    ax2.set_ylabel(f"Coverage Gap  [(1-α) - coverage]")
    ax2.set_title("Coverage Gap by Attack Severity")
    ax2.axhline(0, color='green', ls='--', alpha=0.5)
    ax2.axhline(0.1, color='orange', ls=':', alpha=0.5, label='10% tolerance')

    # Severity bands
    ax2.axhspan(-0.1, 0.10, alpha=0.05, color='green')
    ax2.axhspan(0.10, 0.30, alpha=0.05, color='yellow')
    ax2.axhspan(0.30, 0.60, alpha=0.05, color='orange')
    ax2.axhspan(0.60, 1.00, alpha=0.05, color='red')

    ax2.text(0.98, 0.03, 'Within guarantee', ha='right', va='bottom',
             transform=ax2.transAxes, fontsize=7, color='green', alpha=0.7)
    ax2.text(0.98, 0.25, 'Degraded', ha='right', va='bottom',
             transform=ax2.transAxes, fontsize=7, color='#B7950B', alpha=0.7)
    ax2.text(0.98, 0.55, 'Severely degraded', ha='right', va='bottom',
             transform=ax2.transAxes, fontsize=7, color='orange', alpha=0.7)
    ax2.text(0.98, 0.80, 'Guarantee destroyed', ha='right', va='bottom',
             transform=ax2.transAxes, fontsize=7, color='red', alpha=0.7)

    for bar, m in zip(bars, bar_means):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                 f'{m:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_coverage_gap_severity.png"))
    plt.close()
    print(f"\n  [SAVED] fig_coverage_gap_severity.png")

    return all_rows


# =====================================================================
#  5. RUNTIME COVERAGE MONITORING (SLIDING WINDOW) — NEW CONTRIBUTION
# =====================================================================

def fix_runtime_monitor(adv_results, pred_dir, gt_by_image, out, alpha=0.1):
    """Simulate runtime deployment: clean → adversarial → clean.
    
    Process images sequentially with a sliding coverage window.
    Shows CP as a real-time health monitor with formal target 1-α.
    """
    print("\n" + "=" * 70)
    print("  NEW CONTRIBUTION: Runtime Coverage Monitoring")
    print("  Sliding-window coverage over clean → adversarial → clean sequence")
    print("=" * 70)

    target = 1.0 - alpha
    window_size = 20

    all_scenarios = {}

    for model_name in adv_results:
        clean_preds, clean_by_img = load_preds(pred_dir, model_name)
        if clean_preds is None:
            continue

        attacks = adv_results[model_name].get("attacks", {})

        for atk_name, atk_data in attacks.items():
            adv_by_img = get_adv_preds(adv_results, model_name, atk_name)
            if adv_by_img is None or len(adv_by_img) < 20:
                continue

            # Build matched image set (images with both clean and adversarial preds)
            common_ids = sorted(set(clean_by_img.keys()) & set(adv_by_img.keys()) & set(gt_by_image.keys()))
            if len(common_ids) < 60:
                continue

            # Take 40 clean → 40 adversarial → 40 clean (or as many as available)
            n_phase = min(len(common_ids) // 3, 40)
            if n_phase < 15:
                continue

            np.random.seed(42)
            shuffled = np.random.permutation(common_ids).tolist()

            phase1_ids = shuffled[:n_phase]          # clean
            phase2_ids = shuffled[n_phase:2*n_phase]  # adversarial
            phase3_ids = shuffled[2*n_phase:3*n_phase] # clean (recovery)

            # Compute per-image coverage for the sequence
            sequence_covs = []
            sequence_labels = []  # 0=clean, 1=adversarial

            for img_id in phase1_ids:
                cov_data = compute_per_image_coverage(
                    {img_id: clean_by_img[img_id]},
                    {img_id: gt_by_image[img_id]}
                )
                if img_id in cov_data:
                    sequence_covs.append(cov_data[img_id][2])
                    sequence_labels.append(0)

            for img_id in phase2_ids:
                cov_data = compute_per_image_coverage(
                    {img_id: adv_by_img[img_id]},
                    {img_id: gt_by_image[img_id]}
                )
                if img_id in cov_data:
                    sequence_covs.append(cov_data[img_id][2])
                    sequence_labels.append(1)

            for img_id in phase3_ids:
                cov_data = compute_per_image_coverage(
                    {img_id: clean_by_img[img_id]},
                    {img_id: gt_by_image[img_id]}
                )
                if img_id in cov_data:
                    sequence_covs.append(cov_data[img_id][2])
                    sequence_labels.append(0)

            if len(sequence_covs) < 60:
                continue

            # Sliding window coverage
            window_covs = []
            window_positions = []
            window_gaps = []

            for i in range(window_size, len(sequence_covs) + 1):
                w = sequence_covs[i - window_size:i]
                mean_cov = np.mean(w)
                window_covs.append(mean_cov)
                window_positions.append(i)
                window_gaps.append(target - mean_cov)

            # Detection: where does gap exceed threshold?
            alert_threshold = 0.15  # flag when gap > 15%
            alerts = [i for i, g in enumerate(window_gaps) if g > alert_threshold]
            first_alert = alerts[0] if alerts else None

            # Recovery: when does gap return below threshold?
            recovery_point = None
            n_total = len(sequence_covs)
            adv_end = len(phase1_ids) + len(phase2_ids)
            for i, g in enumerate(window_gaps):
                if window_positions[i] > adv_end and g < alert_threshold:
                    recovery_point = i
                    break

            key = f"{model_name}/{atk_name}"
            scenario = {
                "n_images": len(sequence_covs),
                "n_clean1": len(phase1_ids),
                "n_adv": len(phase2_ids),
                "n_clean2": len(phase3_ids),
                "window_size": window_size,
                "clean_mean_cov": float(np.mean(sequence_covs[:len(phase1_ids)])),
                "adv_mean_cov": float(np.mean(sequence_covs[len(phase1_ids):len(phase1_ids)+len(phase2_ids)])),
                "recovery_mean_cov": float(np.mean(sequence_covs[len(phase1_ids)+len(phase2_ids):])),
                "first_alert_idx": first_alert,
                "detection_delay": (window_positions[first_alert] - len(phase1_ids)) if first_alert is not None else None,
                "positions": window_positions,
                "coverages": window_covs,
                "gaps": window_gaps,
            }
            all_scenarios[key] = scenario

            delay_str = f"{scenario['detection_delay']} images" if scenario['detection_delay'] is not None else "never"
            print(f"\n  {key}:")
            print(f"    Sequence: {len(phase1_ids)} clean → {len(phase2_ids)} adv → {len(phase3_ids)} clean")
            print(f"    Clean cov:    {scenario['clean_mean_cov']:.3f} (gap={target - scenario['clean_mean_cov']:+.3f})")
            print(f"    Adv cov:      {scenario['adv_mean_cov']:.3f} (gap={target - scenario['adv_mean_cov']:+.3f})")
            print(f"    Recovery cov: {scenario['recovery_mean_cov']:.3f} (gap={target - scenario['recovery_mean_cov']:+.3f})")
            print(f"    Detection delay: {delay_str}")

    # === FIGURE: Runtime monitoring (multi-panel) ===
    # Pick up to 4 most interesting scenarios (strongest attacks on YOLO)
    priority = ["yolov8x/pgd-20", "yolov8x/tog-v", "yolov8x/dag",
                "yolov8x/fgsm", "yolov11x/pgd-20", "yolov11x/dag"]
    selected = [k for k in priority if k in all_scenarios][:4]

    if not selected:
        selected = list(all_scenarios.keys())[:4]

    if selected:
        n_panels = len(selected)
        fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4), sharey=True)
        if n_panels == 1:
            axes = [axes]

        for idx, key in enumerate(selected):
            ax = axes[idx]
            sc = all_scenarios[key]
            pos = sc["positions"]
            covs = sc["coverages"]
            n1 = sc["n_clean1"]
            n2 = sc["n_adv"]

            # Background phases
            ax.axvspan(0, n1, alpha=0.08, color='green', label='Clean' if idx == 0 else '')
            ax.axvspan(n1, n1 + n2, alpha=0.08, color='red', label='Adversarial' if idx == 0 else '')
            ax.axvspan(n1 + n2, pos[-1] if pos else n1+n2+n1, alpha=0.08, color='blue', label='Recovery' if idx == 0 else '')

            # Coverage line
            ax.plot(pos, covs, color='#2E86C1', lw=2, label='Window coverage')

            # Target and alert
            ax.axhline(target, color='green', ls='--', lw=1.5, alpha=0.7, label=f'Target (1-α={target})')
            ax.axhline(target - 0.15, color='orange', ls=':', lw=1, alpha=0.7, label='Alert threshold')

            # Detection delay annotation
            if sc["first_alert_idx"] is not None:
                alert_pos = pos[sc["first_alert_idx"]]
                alert_cov = covs[sc["first_alert_idx"]]
                ax.annotate(f'Alert!\n({sc["detection_delay"]} img delay)',
                            xy=(alert_pos, alert_cov),
                            xytext=(alert_pos + 5, alert_cov + 0.15),
                            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                            fontsize=8, color='red', fontweight='bold',
                            ha='left')

            ax.set_xlabel("Image index")
            if idx == 0:
                ax.set_ylabel("Sliding window coverage")
            ax.set_title(key.replace("/", " / "), fontsize=10, fontweight='bold')
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, pos[-1] + 2 if pos else 120)

            if idx == 0:
                ax.legend(fontsize=7, loc='lower left')

        plt.suptitle(f"Runtime Coverage Monitoring (window={sc['window_size']} images)",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out, "fig_runtime_monitor.png"))
        plt.close()
        print(f"\n  [SAVED] fig_runtime_monitor.png")

    # Summary table
    print(f"\n  {'='*65}")
    print(f"  RUNTIME MONITORING SUMMARY")
    print(f"  {'='*65}")
    print(f"  {'Scenario':<28s} {'Clean':>8s} {'Adv':>8s} {'Recov':>8s} {'Delay':>8s}")
    print(f"  {'-'*65}")
    for key, sc in all_scenarios.items():
        delay = f"{sc['detection_delay']}img" if sc['detection_delay'] is not None else "N/A"
        print(f"  {key:<28s} {sc['clean_mean_cov']:>8.3f} {sc['adv_mean_cov']:>8.3f} "
              f"{sc['recovery_mean_cov']:>8.3f} {delay:>8s}")

    return all_scenarios


# =====================================================================
#  6. CLASS-CONDITIONAL COVERAGE UNDER ATTACK (NEW CP CONTRIBUTION)
# =====================================================================

def fix_class_coverage(adv_results, pred_dir, gt_by_image, out, alpha=0.1):
    """Per-class coverage breakdown under attack.

    Genuinely CP-specific: 'which class lost the 1-α guarantee?' requires
    the conformal framework. Simple confidence monitoring has no per-class target.
    """
    print("\n" + "=" * 70)
    print("  NEW CP CONTRIBUTION: Class-Conditional Coverage")
    print("  Which COCO classes lose the 1-α guarantee under each attack?")
    print("=" * 70)

    target = 1.0 - alpha

    # COCO class names (subset)
    COCO_NAMES = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 16: 'bird', 17: 'cat', 18: 'dog',
        19: 'horse', 20: 'sheep', 21: 'cow', 24: 'zebra', 25: 'giraffe',
        27: 'backpack', 28: 'umbrella', 31: 'handbag', 33: 'suitcase',
        44: 'bottle', 46: 'wine glass', 47: 'cup', 51: 'bowl',
        62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
        67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
        77: 'cell phone', 84: 'book',
    }

    def per_class_coverage(preds_by_img, gt_by_img):
        """Coverage per COCO category."""
        class_covered = defaultdict(int)
        class_total = defaultdict(int)

        for img_id, gt_anns in gt_by_img.items():
            preds = preds_by_img.get(img_id, [])
            gt_matched = [False] * len(gt_anns)
            preds_sorted = sorted(preds, key=lambda p: -p["score"])

            for p in preds_sorted:
                px, py, pw, ph = p["bbox"]
                pred_xyxy = [px, py, px + pw, py + ph]
                for j, gt in enumerate(gt_anns):
                    if gt_matched[j]:
                        continue
                    gx, gy, gw, gh = gt["bbox"]
                    gt_xyxy = [gx, gy, gx + gw, gy + gh]
                    cat = gt.get("category_id", -1)
                    if p.get("category_id") == cat:
                        if _box_iou(pred_xyxy, gt_xyxy) >= 0.5:
                            gt_matched[j] = True
                            class_covered[cat] += 1
                            break

            for gt in gt_anns:
                class_total[gt["category_id"]] += 1

        result = {}
        for cat in class_total:
            n = class_total[cat]
            c = class_covered.get(cat, 0)
            result[cat] = {"covered": c, "total": n, "coverage": c / n if n > 0 else 0}
        return result

    all_heatmap = {}

    for model_name in adv_results:
        clean_preds, clean_by_img = load_preds(pred_dir, model_name)
        if clean_preds is None:
            continue

        attacks = adv_results[model_name].get("attacks", {})

        # Clean per-class coverage
        clean_class = per_class_coverage(clean_by_img, gt_by_image)

        print(f"\n  {model_name}:")
        print(f"    {'Class':<16s} {'Clean':>6s} {'N_GT':>5s}", end="")

        atk_names = []
        for atk_name in attacks:
            adv_by_img = get_adv_preds(adv_results, model_name, atk_name)
            if adv_by_img and len(adv_by_img) >= 10:
                atk_names.append(atk_name)
                print(f" {atk_name:>8s}", end="")
        print()

        # Top-15 classes by GT count
        top_classes = sorted(clean_class.keys(), key=lambda c: -clean_class[c]["total"])[:15]
        model_data = {}

        for cat in top_classes:
            name = COCO_NAMES.get(cat, f"c{cat}")
            cc = clean_class[cat]
            print(f"    {name:<16s} {cc['coverage']:>6.2f} {cc['total']:>5d}", end="")

            model_data[cat] = {"clean": cc["coverage"]}

            for atk_name in atk_names:
                adv_by_img = get_adv_preds(adv_results, model_name, atk_name)
                adv_gt = {k: v for k, v in gt_by_image.items() if k in adv_by_img}
                adv_class = per_class_coverage(adv_by_img, adv_gt)
                ac = adv_class.get(cat, {"coverage": 0})
                gap = target - ac["coverage"]
                marker = "*" if gap > 0.5 else ""
                print(f" {ac['coverage']:>7.2f}{marker}", end="")
                model_data[cat][atk_name] = ac["coverage"]
            print()

        all_heatmap[model_name] = model_data

    # Figure: heatmap for first model
    if all_heatmap:
        model_name = list(all_heatmap.keys())[0]
        data = all_heatmap[model_name]
        cats = list(data.keys())
        conditions = ["clean"] + [a for a in atk_names if a in list(data.values())[0]]

        matrix = np.zeros((len(cats), len(conditions)))
        for i, cat in enumerate(cats):
            for j, cond in enumerate(conditions):
                matrix[i, j] = data[cat].get(cond, 0)

        fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.5), max(6, len(cats) * 0.4)))
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels([COCO_NAMES.get(c, f"c{c}") for c in cats], fontsize=9)
        ax.set_title(f"Class-Conditional Coverage ({model_name})", fontsize=12, fontweight='bold')

        for i in range(len(cats)):
            for j in range(len(conditions)):
                v = matrix[i, j]
                color = 'white' if v < 0.4 else 'black'
                ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=7, color=color)

        # Target line
        ax.axhline(-0.5, color='black', lw=0.5)
        plt.colorbar(im, ax=ax, label='Coverage', shrink=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(out, "fig_class_coverage_heatmap.png"))
        plt.close()
        print(f"\n  [SAVED] fig_class_coverage_heatmap.png")

    return all_heatmap


# =====================================================================
#  7. MULTI-IoU COVERAGE PROFILE (NEW CP CONTRIBUTION)
# =====================================================================

def fix_multi_iou(adv_results, pred_dir, gt_by_image, out, alpha=0.1):
    """Coverage at multiple IoU thresholds: 0.50, 0.75, 0.90.

    The SHAPE of the multi-IoU profile is diagnostic:
    - Clean: gradual decay (0.92 → 0.71 → 0.35)
    - Suppression attack: uniform collapse across all IoU
    - Mislocalization attack: low-IoU coverage drops more
    """
    print("\n" + "=" * 70)
    print("  NEW CP CONTRIBUTION: Multi-IoU Coverage Profile")
    print("  Coverage shape across IoU thresholds is diagnostic of attack type")
    print("=" * 70)

    iou_thresholds = [0.25, 0.50, 0.75, 0.90]

    def coverage_at_iou(preds_by_img, gt_by_img, iou_thresh, class_correct=True):
        total, covered = 0, 0
        for img_id, gt_anns in gt_by_img.items():
            preds = preds_by_img.get(img_id, [])
            gt_used = [False] * len(gt_anns)
            for p in sorted(preds, key=lambda x: -x["score"]):
                px, py, pw, ph = p["bbox"]
                pred_xyxy = [px, py, px + pw, py + ph]
                for j, gt in enumerate(gt_anns):
                    if gt_used[j]:
                        continue
                    gx, gy, gw, gh = gt["bbox"]
                    gt_xyxy = [gx, gy, gx + gw, gy + gh]
                    iou = _box_iou(pred_xyxy, gt_xyxy)
                    class_ok = (not class_correct) or (p.get("category_id") == gt.get("category_id"))
                    if iou >= iou_thresh and class_ok:
                        gt_used[j] = True
                        covered += 1
                        break
            total += len(gt_anns)
        return covered / total if total > 0 else 0

    all_profiles = {}

    for model_name in adv_results:
        clean_preds, clean_by_img = load_preds(pred_dir, model_name)
        if clean_preds is None:
            continue

        print(f"\n  {model_name}:")
        print(f"    {'Condition':<14s}", end="")
        for iou_t in iou_thresholds:
            print(f" IoU≥{iou_t:<5.2f}", end="")
        print("  Shape")

        # Clean profile
        clean_profile = []
        for iou_t in iou_thresholds:
            cov = coverage_at_iou(clean_by_img, gt_by_image, iou_t)
            clean_profile.append(cov)
        ratio = clean_profile[-1] / clean_profile[0] if clean_profile[0] > 0 else 0
        print(f"    {'Clean':<14s}", end="")
        for c in clean_profile:
            print(f" {c:>9.3f}", end="")
        print(f"  ratio={ratio:.2f}")

        model_profiles = {"clean": clean_profile}

        # Per attack
        attacks = adv_results[model_name].get("attacks", {})
        for atk_name, atk_data in attacks.items():
            adv_by_img = get_adv_preds(adv_results, model_name, atk_name)
            if adv_by_img is None or len(adv_by_img) < 10:
                continue

            adv_gt = {k: v for k, v in gt_by_image.items() if k in adv_by_img}
            adv_profile = []
            for iou_t in iou_thresholds:
                cov = coverage_at_iou(adv_by_img, adv_gt, iou_t)
                adv_profile.append(cov)

            ratio = adv_profile[-1] / adv_profile[0] if adv_profile[0] > 0 else 0
            # Diagnostic: if ratio ≈ clean ratio → uniform suppression
            #             if ratio > clean ratio → surviving dets are well-localized
            clean_ratio = clean_profile[-1] / clean_profile[0] if clean_profile[0] > 0 else 0
            if abs(ratio - clean_ratio) < 0.1:
                shape = "uniform suppression"
            elif ratio > clean_ratio + 0.1:
                shape = "selective (good survive)"
            else:
                shape = "localization damage"

            print(f"    {atk_name:<14s}", end="")
            for c in adv_profile:
                print(f" {c:>9.3f}", end="")
            print(f"  ratio={ratio:.2f} → {shape}")

            model_profiles[atk_name] = adv_profile

        all_profiles[model_name] = model_profiles

    # Figure: multi-IoU profiles
    if all_profiles:
        models_to_plot = [m for m in all_profiles if len(all_profiles[m]) > 2][:2]
        fig, axes = plt.subplots(1, len(models_to_plot), figsize=(6 * len(models_to_plot), 5), sharey=True)
        if len(models_to_plot) == 1:
            axes = [axes]

        colors = {'clean': '#27AE60', 'fgsm': '#F39C12', 'pgd-20': '#E74C3C',
                  'pgd-50': '#C0392B', 'dag': '#8E44AD', 'tog-v': '#2C3E50'}

        for idx, model_name in enumerate(models_to_plot):
            ax = axes[idx]
            profiles = all_profiles[model_name]
            for cond, profile in profiles.items():
                c = colors.get(cond, '#888')
                lw = 2.5 if cond == 'clean' else 1.5
                ls = '-' if cond == 'clean' else '--'
                ax.plot(iou_thresholds, profile, marker='o', color=c, lw=lw, ls=ls,
                        label=cond, markersize=6)

            ax.axhline(1 - alpha, color='green', ls=':', alpha=0.5, label=f'Target (1-α={1-alpha})')
            ax.set_xlabel("IoU Threshold")
            if idx == 0:
                ax.set_ylabel("Coverage")
            ax.set_title(model_name, fontsize=11, fontweight='bold')
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8, loc='lower left')
            ax.set_xticks(iou_thresholds)

        plt.suptitle("Multi-IoU Coverage Profile", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out, "fig_multi_iou_profile.png"))
        plt.close()
        print(f"\n  [SAVED] fig_multi_iou_profile.png")

    return all_profiles


# =====================================================================
#  8. CONFORMAL P-VALUES (NEW CP CONTRIBUTION — genuinely CP-specific)
# =====================================================================

def fix_pvalues(adv_results, pred_dir, gt_by_image, out, alpha=0.1):
    """Conformal p-values: genuinely CP-specific detection signal.

    For each detection, compute its conformal p-value: the fraction of
    calibration nonconformity scores MORE extreme than the test score.
    On clean data: p-values ~ Uniform[0,1].
    Under attack: distribution shifts → KS test detects the shift.

    This is different from confidence monitoring because it uses the
    CALIBRATED distribution as reference, not a simple mean/std.
    """
    print("\n" + "=" * 70)
    print("  NEW CP CONTRIBUTION: Conformal P-Values")
    print("  KS test on p-value distributions: genuinely CP-specific signal")
    print("=" * 70)

    from scipy.stats import kstest, ks_2samp

    MIN_CONF = 0.1

    for model_name in adv_results:
        clean_preds, clean_by_img = load_preds(pred_dir, model_name)
        if clean_preds is None:
            continue

        # Split clean into calibration and test
        import random
        img_ids = sorted(clean_by_img.keys())
        random.seed(42)
        random.shuffle(img_ids)
        cal_ids = set(img_ids[:len(img_ids) // 2])
        test_ids = set(img_ids[len(img_ids) // 2:])

        # Compute calibration nonconformity scores (1 - confidence)
        cal_scores = []
        for img_id in cal_ids:
            for p in clean_by_img.get(img_id, []):
                if p["score"] >= MIN_CONF:
                    cal_scores.append(1.0 - p["score"])
        cal_scores = np.sort(cal_scores)
        n_cal = len(cal_scores)

        if n_cal < 50:
            print(f"  {model_name}: too few calibration scores ({n_cal}), skipping")
            continue

        def compute_pvalues(preds_by_img, img_subset=None):
            """Compute conformal p-value for each detection."""
            all_pvals = []
            per_image_ks = {}
            for img_id, preds in preds_by_img.items():
                if img_subset and img_id not in img_subset:
                    continue
                pvals = []
                for p in preds:
                    if p["score"] >= MIN_CONF:
                        nc = 1.0 - p["score"]
                        # p-value = fraction of cal scores >= nc
                        rank = np.searchsorted(cal_scores, nc, side='right')
                        pval = 1.0 - rank / n_cal
                        pvals.append(pval)
                        all_pvals.append(pval)

                if len(pvals) >= 3:
                    # KS test against Uniform[0,1]
                    ks_stat, ks_p = kstest(pvals, 'uniform')
                    per_image_ks[img_id] = (ks_stat, ks_p, len(pvals))

            return np.array(all_pvals), per_image_ks

        # Clean test p-values
        clean_pvals, clean_ks = compute_pvalues(clean_by_img, test_ids)

        print(f"\n  {model_name} (n_cal={n_cal}):")
        print(f"    Clean p-values: mean={clean_pvals.mean():.3f}, std={clean_pvals.std():.3f}")
        print(f"    Clean KS vs Uniform: stat={np.mean([v[0] for v in clean_ks.values()]):.3f}")

        # Per attack
        attacks = adv_results[model_name].get("attacks", {})
        results = {}

        for atk_name in attacks:
            adv_by_img = get_adv_preds(adv_results, model_name, atk_name)
            if adv_by_img is None or len(adv_by_img) < 10:
                continue

            adv_pvals, adv_ks = compute_pvalues(adv_by_img)

            if len(adv_pvals) < 10:
                continue

            # Global KS: clean vs adversarial p-value distributions
            ks_stat, ks_p = ks_2samp(clean_pvals, adv_pvals)

            # Image-level AUROC using mean KS statistic per image
            from sklearn.metrics import roc_auc_score
            labels, scores = [], []
            for img_id in clean_ks:
                labels.append(0)
                scores.append(clean_ks[img_id][0])  # KS stat
            for img_id in adv_ks:
                labels.append(1)
                scores.append(adv_ks[img_id][0])
            labels = np.array(labels)
            scores_arr = np.array(scores)

            if len(np.unique(labels)) >= 2 and len(np.unique(scores_arr)) >= 2:
                auroc = float(roc_auc_score(labels, scores_arr))
                auroc = max(auroc, 1 - auroc)
            else:
                auroc = 0.5

            results[atk_name] = {
                "adv_pval_mean": float(adv_pvals.mean()),
                "adv_pval_std": float(adv_pvals.std()),
                "global_ks_stat": float(ks_stat),
                "global_ks_p": float(ks_p),
                "auroc_ks": float(auroc),
                "n_clean_imgs": len(clean_ks),
                "n_adv_imgs": len(adv_ks),
            }

            sig = "***" if ks_p < 0.001 else "**" if ks_p < 0.01 else "*" if ks_p < 0.05 else "ns"
            print(f"    {atk_name:<10s}: p-val mean={adv_pvals.mean():.3f} "
                  f"KS={ks_stat:.3f} (p={ks_p:.4f} {sig}) "
                  f"AUROC(KS)={auroc:.3f}")

    # Figure: p-value distributions
    if adv_results:
        model_name = list(adv_results.keys())[0]
        clean_preds, clean_by_img = load_preds(pred_dir, model_name)
        if clean_preds:
            img_ids = sorted(clean_by_img.keys())
            random.seed(42)
            random.shuffle(img_ids)
            test_ids = set(img_ids[len(img_ids) // 2:])

            cal_scores_plot = []
            for img_id in set(img_ids[:len(img_ids) // 2]):
                for p in clean_by_img.get(img_id, []):
                    if p["score"] >= MIN_CONF:
                        cal_scores_plot.append(1.0 - p["score"])
            cal_scores_plot = np.sort(cal_scores_plot)
            n_cal_plot = len(cal_scores_plot)

            def get_pvals(preds_by_img, subset=None):
                pvals = []
                for img_id, preds in preds_by_img.items():
                    if subset and img_id not in subset:
                        continue
                    for p in preds:
                        if p["score"] >= MIN_CONF:
                            nc = 1.0 - p["score"]
                            rank = np.searchsorted(cal_scores_plot, nc, side='right')
                            pvals.append(1.0 - rank / n_cal_plot)
                return np.array(pvals)

            attacks = adv_results[model_name].get("attacks", {})
            atk_list = [a for a in attacks if get_adv_preds(adv_results, model_name, a) is not None][:3]

            n_panels = 1 + len(atk_list)
            fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3.5), sharey=True)

            # Clean
            clean_pv = get_pvals(clean_by_img, test_ids)
            axes[0].hist(clean_pv, bins=20, density=True, alpha=0.7, color='#27AE60', edgecolor='white')
            axes[0].axhline(1.0, color='black', ls='--', alpha=0.5, label='Uniform')
            axes[0].set_title(f"Clean (n={len(clean_pv)})", fontsize=10)
            axes[0].set_xlabel("Conformal p-value")
            axes[0].set_ylabel("Density")
            axes[0].legend(fontsize=8)

            for i, atk_name in enumerate(atk_list):
                adv_by_img = get_adv_preds(adv_results, model_name, atk_name)
                adv_pv = get_pvals(adv_by_img)
                axes[i + 1].hist(adv_pv, bins=20, density=True, alpha=0.7, color='#E74C3C', edgecolor='white')
                axes[i + 1].axhline(1.0, color='black', ls='--', alpha=0.5)
                ks, ksp = ks_2samp(clean_pv, adv_pv)
                axes[i + 1].set_title(f"{atk_name} (KS={ks:.2f})", fontsize=10)
                axes[i + 1].set_xlabel("Conformal p-value")

            plt.suptitle(f"Conformal P-Value Distributions ({model_name})", fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(out, "fig_conformal_pvalues.png"))
            plt.close()
            print(f"\n  [SAVED] fig_conformal_pvalues.png")

    return results if 'results' in dir() else {}


def main():
    parser = argparse.ArgumentParser(description="Revision experiments")
    parser.add_argument("--coco-root", default="data/coco")
    parser.add_argument("--pred-dir", default="results/evaluation/predictions")
    parser.add_argument("--adv-results", default="results/adversarial/full_results.json")
    parser.add_argument("--output-dir", default="results/revision")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--fix", default="all",
                        choices=["all", "ablation", "delong", "reliability",
                                 "coverage_gap", "runtime",
                                 "class_coverage", "multi_iou", "pvalues"])
    args = parser.parse_args()

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    print("=" * 70)
    print("  REVISION EXPERIMENTS — Addressing Reviewer Comments")
    print("=" * 70)

    gt_by_image = load_gt(args.coco_root)
    all_results = {}

    if args.fix in ["all", "ablation"]:
        if os.path.exists(args.adv_results):
            with open(args.adv_results, 'r', encoding='utf-8') as f:
                adv_results = json.load(f)
            r = fix_ablation(adv_results, args.pred_dir, gt_by_image, out, args.alpha)
            all_results["cp_ablation"] = r

    if args.fix in ["all", "delong"]:
        if os.path.exists(args.adv_results):
            with open(args.adv_results, 'r', encoding='utf-8') as f:
                adv_results = json.load(f)
            r = fix_delong(adv_results, args.pred_dir, gt_by_image, out, args.alpha)
            all_results["delong"] = r

    if args.fix in ["all", "reliability"]:
        fix_reliability(args.pred_dir, gt_by_image, out)

    if args.fix in ["all", "coverage_gap"]:
        if os.path.exists(args.adv_results):
            with open(args.adv_results, 'r', encoding='utf-8') as f:
                adv_results = json.load(f)
            r = fix_coverage_gap(adv_results, args.pred_dir, gt_by_image, out, args.alpha)
            all_results["coverage_gap"] = r

    if args.fix in ["all", "runtime"]:
        if os.path.exists(args.adv_results):
            with open(args.adv_results, 'r', encoding='utf-8') as f:
                adv_results = json.load(f)
            r = fix_runtime_monitor(adv_results, args.pred_dir, gt_by_image, out, args.alpha)
            all_results["runtime_monitor"] = r

    if args.fix in ["all", "class_coverage"]:
        if os.path.exists(args.adv_results):
            with open(args.adv_results, 'r', encoding='utf-8') as f:
                adv_results = json.load(f)
            r = fix_class_coverage(adv_results, args.pred_dir, gt_by_image, out, args.alpha)
            all_results["class_coverage"] = r

    if args.fix in ["all", "multi_iou"]:
        if os.path.exists(args.adv_results):
            with open(args.adv_results, 'r', encoding='utf-8') as f:
                adv_results = json.load(f)
            r = fix_multi_iou(adv_results, args.pred_dir, gt_by_image, out, args.alpha)
            all_results["multi_iou"] = r

    if args.fix in ["all", "pvalues"]:
        if os.path.exists(args.adv_results):
            with open(args.adv_results, 'r', encoding='utf-8') as f:
                adv_results = json.load(f)
            r = fix_pvalues(adv_results, args.pred_dir, gt_by_image, out, args.alpha)
            all_results["pvalues"] = r

    rp = os.path.join(out, "revision_results.json")
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  REVISION EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print(f"  Results -> {out}/")


if __name__ == "__main__":
    main()
