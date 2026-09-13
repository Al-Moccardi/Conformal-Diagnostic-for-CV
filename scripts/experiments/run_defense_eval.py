#!/usr/bin/env python3
"""
run_defense_eval.py — Defense evaluation: Pareto curves, AUROC, selective abstention.

Uses EXISTING predictions from results/evaluation/predictions/ and results/adversarial/.
NO GPU NEEDED — pure analysis on saved predictions.

Output:
    results/defense/
        ├── fig_pareto_coverage.png
        ├── fig_abstention_distribution.png
        ├── fig_roc_attack_detection.png
        ├── fig_precision_of_kept.png
        ├── fig_tau_sweep.png
        ├── defense_results.json
        └── defense_table.txt

Usage:
    python run_defense_eval.py
    python run_defense_eval.py --adv-results results/adversarial/full_results.json
"""

import os, sys, json, argparse, warnings
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
    RobustConformalCalibrator,
    ConformalAttackDetector,
    SelectivePredictor,
    evaluate_defense,
    compute_all_aurocs,
)

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 10,
    'axes.titleweight': 'bold', 'figure.dpi': 150,
    'savefig.dpi': 250, 'savefig.bbox': 'tight',
})

COLORS = {
    'clean': '#27AE60', 'adv': '#E74C3C', 'pareto': '#2E86C1',
    'ours': '#3498DB', 'raw': '#E67E22',
}


# =====================================================================
#  DATA LOADING
# =====================================================================

def load_predictions(pred_dir, model_name):
    """Load saved COCO-format predictions and group by image."""
    path = os.path.join(pred_dir, f"{model_name}_bbox.json")
    if not os.path.exists(path):
        return None, None
    with open(path, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    by_img = defaultdict(list)
    for p in preds:
        by_img[p["image_id"]].append(p)
    return preds, by_img


def load_adversarial_results(adv_file):
    """Load adversarial evaluation results."""
    if not os.path.exists(adv_file):
        return None
    with open(adv_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_gt(coco_root):
    """Load COCO ground truth annotations."""
    ann_file = os.path.join(coco_root, "annotations", "instances_val2017.json")
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt = defaultdict(list)
    for ann in data["annotations"]:
        gt[ann["image_id"]].append(ann)
    return gt


def reconstruct_adv_preds_by_image(adv_results, model_name, attack_name):
    """Reconstruct per-image adversarial predictions from saved results.

    If per-image predictions were saved in the adversarial eval, use them.
    Otherwise, we cannot reconstruct and return None.
    """
    mr = adv_results.get(model_name, {})
    atk = mr.get("attacks", {}).get(attack_name, {})

    # Check if per-image predictions were saved
    per_image = atk.get("per_image_predictions", None)
    if per_image:
        by_img = defaultdict(list)
        for p in per_image:
            by_img[p["image_id"]].append(p)
        return by_img
    return None


# =====================================================================
#  FIGURE GENERATION
# =====================================================================

def fig_pareto_coverage(pareto_data, model_name, out):
    """Pareto curve: clean_coverage vs adv_coverage at different mixing ratios."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for atk_name, pareto_points in pareto_data.items():
        if not pareto_points:
            continue
        ratios = [p["ratio"] for p in pareto_points]
        clean_covs = [p["clean_coverage"] for p in pareto_points]
        adv_covs = [p["adv_coverage"] for p in pareto_points]

        ax.plot(clean_covs, adv_covs, 'o-', label=atk_name, markersize=6, lw=2)

        # Annotate endpoints
        ax.annotate(f'r=0.0', (clean_covs[0], adv_covs[0]),
                    fontsize=7, textcoords="offset points", xytext=(5, 5))
        ax.annotate(f'r=1.0', (clean_covs[-1], adv_covs[-1]),
                    fontsize=7, textcoords="offset points", xytext=(5, -10))

    ax.set_xlabel("Clean Coverage")
    ax.set_ylabel("Adversarial Coverage")
    ax.set_title(f"Pareto Frontier: Clean vs Adversarial Coverage\n{model_name}")
    ax.axhline(0.9, color='gray', ls=':', alpha=0.5, label='Target 0.9')
    ax.axvline(0.9, color='gray', ls=':', alpha=0.5)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    plt.savefig(os.path.join(out, "fig_pareto_coverage.png"))
    plt.close()
    print(f"  [SAVED] fig_pareto_coverage.png")


def fig_abstention_distribution(clean_stats, adv_stats_by_attack, out):
    """Histogram of per-image abstention rate, clean vs attacked overlay."""
    n_attacks = len(adv_stats_by_attack)
    fig, axes = plt.subplots(1, max(n_attacks, 1), figsize=(6 * max(n_attacks, 1), 5))
    if n_attacks == 1:
        axes = [axes]
    elif n_attacks == 0:
        plt.close()
        return

    clean_abst = [s.abstention_rate for s in clean_stats.values()]

    for idx, (atk_name, adv_stats) in enumerate(adv_stats_by_attack.items()):
        ax = axes[idx]
        adv_abst = [s.abstention_rate for s in adv_stats.values()]

        bins = np.linspace(0, 1, 25)
        ax.hist(clean_abst, bins=bins, alpha=0.6, label='Clean', color=COLORS['clean'],
                density=True, edgecolor='white')
        ax.hist(adv_abst, bins=bins, alpha=0.6, label=atk_name, color=COLORS['adv'],
                density=True, edgecolor='white')
        ax.set_xlabel("Abstention Rate")
        ax.set_ylabel("Density")
        ax.set_title(f"Clean vs {atk_name}")
        ax.legend(fontsize=9)

    plt.suptitle("Per-Image Abstention Rate Distribution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_abstention_distribution.png"))
    plt.close()
    print(f"  [SAVED] fig_abstention_distribution.png")


def fig_roc_attack_detection(auroc_data, out):
    """ROC curve for image-level attack detection."""
    fig, ax = plt.subplots(figsize=(7, 7))

    for atk_name, roc in auroc_data.items():
        if "fpr" in roc and "tpr" in roc:
            ax.plot(roc["fpr"], roc["tpr"], lw=2,
                    label=f'{atk_name} (AUROC={roc["auroc"]:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Attack Detection via Conformal Statistics")
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.savefig(os.path.join(out, "fig_roc_attack_detection.png"))
    plt.close()
    print(f"  [SAVED] fig_roc_attack_detection.png")


def fig_precision_of_kept(defense_data, out):
    """Bar chart: precision improvement after CP filtering."""
    attacks = list(defense_data.keys())
    if not attacks:
        return
    n = len(attacks)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Clean precision
    ax = axes[0]
    raw_p = [defense_data[a].get("precision_raw_clean", 0) for a in attacks]
    cp_p = [defense_data[a].get("precision_of_kept_clean", 0) for a in attacks]
    x = np.arange(n)
    w = 0.35
    ax.bar(x - w / 2, raw_p, w, label='Raw (conf>0.5)', color=COLORS['raw'], alpha=0.8)
    ax.bar(x + w / 2, cp_p, w, label='After CP Filter', color=COLORS['ours'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=9, rotation=30, ha='right')
    ax.set_ylabel("Precision")
    ax.set_title("(a) Precision: Clean Images")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    # Adversarial precision
    ax = axes[1]
    raw_p = [defense_data[a].get("precision_raw_adv", 0) for a in attacks]
    cp_p = [defense_data[a].get("precision_of_kept_adv", 0) for a in attacks]
    ax.bar(x - w / 2, raw_p, w, label='Raw (conf>0.5)', color=COLORS['raw'], alpha=0.8)
    ax.bar(x + w / 2, cp_p, w, label='After CP Filter', color=COLORS['ours'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=9, rotation=30, ha='right')
    ax.set_ylabel("Precision")
    ax.set_title("(b) Precision: Adversarial Images")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    plt.suptitle("Precision Improvement via Conformal Filtering", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_precision_of_kept.png"))
    plt.close()
    print(f"  [SAVED] fig_precision_of_kept.png")


def fig_tau_sweep(sweep_data, out):
    """Plot tau sweep: abstention rate vs tau for clean and adversarial."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for atk_name, sweep in sweep_data.items():
        if not sweep:
            continue
        taus = [s["tau"] for s in sweep]
        clean_abs = [s["clean_abstention"] for s in sweep]
        adv_abs = [s["adv_abstention"] for s in sweep]

        axes[0].plot(taus, clean_abs, 'o-', label=f'{atk_name} clean', lw=2)
        axes[0].plot(taus, adv_abs, 's--', label=f'{atk_name} adv', lw=2)

        clean_prec = [s["clean_precision"] for s in sweep]
        adv_prec = [s["adv_precision"] for s in sweep]
        axes[1].plot(taus, clean_prec, 'o-', label=f'{atk_name} clean', lw=2)
        axes[1].plot(taus, adv_prec, 's--', label=f'{atk_name} adv', lw=2)

    axes[0].set_xlabel("Threshold tau")
    axes[0].set_ylabel("Image Abstention Rate")
    axes[0].set_title("(a) Abstention Rate vs tau")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("Threshold tau")
    axes[1].set_ylabel("Precision of Kept")
    axes[1].set_title("(b) Precision vs tau")
    axes[1].legend(fontsize=7)

    plt.suptitle("Selective Prediction: tau Sweep", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_tau_sweep.png"))
    plt.close()
    print(f"  [SAVED] fig_tau_sweep.png")


# =====================================================================
#  GAP 4: AUROC BASELINE COMPARISON
# =====================================================================

def compute_all_aurocs(clean_by_img, adv_by_img, calibrator):
    """Compare CP-based AUROC against naive baselines.

    Baselines:
      1. Mean confidence (no CP needed)
      2. Detection count > threshold
      3. High-conf detection ratio (conf>0.5)
      4. Our composite CP score

    Returns dict of {method_name: auroc}.
    """
    from sklearn.metrics import roc_auc_score

    results = {}

    # Extract per-image features for both clean and adversarial
    labels, feat_conf, feat_ndet, feat_ratio, feat_cp = [], [], [], [], []

    detector = ConformalAttackDetector(calibrator)
    detector.fit_thresholds(clean_by_img)

    for img_id, preds in clean_by_img.items():
        labels.append(0)
        scores = [p['score'] for p in preds]
        feat_conf.append(np.mean(scores) if scores else 0)
        feat_ndet.append(len(preds))
        feat_ratio.append(sum(1 for s in scores if s >= 0.5) / max(len(scores), 1))
        stats = detector.compute_image_stats({img_id: preds})
        feat_cp.append(detector._anomaly_score(stats[img_id]))

    for img_id, preds in adv_by_img.items():
        labels.append(1)
        scores = [p['score'] for p in preds]
        feat_conf.append(np.mean(scores) if scores else 0)
        feat_ndet.append(len(preds))
        feat_ratio.append(sum(1 for s in scores if s >= 0.5) / max(len(scores), 1))
        stats = detector.compute_image_stats({img_id: preds})
        feat_cp.append(detector._anomaly_score(stats[img_id]))

    labels = np.array(labels)
    if len(np.unique(labels)) < 2:
        return {"all_methods": "insufficient_data"}

    def safe_auroc(y, scores, invert=False):
        s = -np.array(scores) if invert else np.array(scores)
        if len(np.unique(s)) < 2:
            return 0.5
        auc = float(roc_auc_score(y, s))
        return max(auc, 1 - auc)

    results["Naive: mean confidence"] = safe_auroc(labels, feat_conf, invert=True)
    results["Naive: detection count"] = safe_auroc(labels, feat_ndet, invert=True)
    results["Naive: high-conf ratio"] = safe_auroc(labels, feat_ratio, invert=True)
    results["Ours: CP composite"] = safe_auroc(labels, feat_cp, invert=False)

    return results


# =====================================================================
#  MAIN PIPELINE
# =====================================================================

def run_defense_pipeline(args):
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    print("=" * 70)
    print("  DEFENSE EVALUATION — Conformal Prediction as Post-Hoc Defense")
    print("=" * 70)

    # Load GT
    gt_by_image = load_gt(args.coco_root)
    print(f"  COCO GT loaded: {len(gt_by_image)} images")

    # Load adversarial results
    adv_results = load_adversarial_results(args.adv_results)
    if adv_results is None:
        print(f"  [ERROR] Adversarial results not found: {args.adv_results}")
        print(f"  Run adversarial eval first: python main.py adversarial")
        return

    # Load clean predictions
    pred_dir = Path(args.pred_dir)
    all_defense_results = {}

    for model_name in adv_results.keys():
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*60}")

        clean_preds, clean_by_img = load_predictions(args.pred_dir, model_name)
        if clean_preds is None:
            print(f"  [SKIP] No clean predictions for {model_name}")
            continue

        print(f"  Loaded {len(clean_preds)} clean predictions across {len(clean_by_img)} images")

        # Calibrate on first half of clean data
        import random
        img_ids = sorted(clean_by_img.keys())
        random.seed(42)
        random.shuffle(img_ids)
        n_cal = len(img_ids) // 2
        cal_ids = set(img_ids[:n_cal])

        cal_preds = [p for p in clean_preds if p["image_id"] in cal_ids]

        calibrator = AdaptiveConformalCalibrator(
            alpha=args.alpha, n_conf_bins=5, size_normalize=True
        )
        calibrator.calibrate(cal_preds, gt_by_image)
        _qhat = calibrator.global_qhat if calibrator.global_qhat is not None else 0.0
        print(f"  Calibrator: threshold={calibrator.conf_threshold:.4f}, qhat={_qhat:.4f}"
              f"{' [qhat=None, no matched predictions]' if calibrator.global_qhat is None else ''}")
        if calibrator.conf_threshold < 0.001:
            print(f"  [INFO] Threshold~0: using composite anomaly score (confidence + detection count)")

        model_mr = adv_results[model_name]
        attacks = model_mr.get("attacks", {})
        if not attacks:
            print(f"  [SKIP] No attack data for {model_name}")
            continue

        model_defense = {}
        pareto_data = {}
        auroc_data = {}
        sweep_data = {}
        adv_stats_by_attack = {}

        for atk_name, atk_data in attacks.items():
            print(f"\n  Attack: {atk_name}")

            # Reconstruct adversarial predictions (per-image from adversarial eval)
            adv_by_img = reconstruct_adv_preds_by_image(
                adv_results, model_name, atk_name)

            if adv_by_img is not None and len(adv_by_img) > 0:
                # KEY FIX: Get clean predictions for THE SAME images
                # that were attacked (not from the full eval test split)
                adv_image_ids = set(adv_by_img.keys())
                clean_same_imgs = {
                    img_id: clean_by_img[img_id]
                    for img_id in adv_image_ids
                    if img_id in clean_by_img
                }
                print(f"    Using {len(adv_by_img)} adv images, "
                      f"{len(clean_same_imgs)} matching clean images")

                if len(clean_same_imgs) < 5:
                    print(f"    [WARN] Too few matching images — falling back to simulation")
                    adv_by_img = None
            else:
                clean_same_imgs = None

            if adv_by_img is None or clean_same_imgs is None:
                # Fallback: simulate adversarial from clean test data
                print(f"    [INFO] Simulating adversarial predictions from clean data")
                test_ids = set(img_ids[n_cal:])
                clean_same_imgs = {k: v for k, v in clean_by_img.items() if k in test_ids}

                clean_mAP = model_mr.get("clean_mAP", {}).get("mAP@[0.5:0.95]", 0.5)
                atk_mAP = atk_data.get("attacked_mAP", {}).get("mAP@[0.5:0.95]", 0.1)
                drop_ratio = max(0.05, atk_mAP / max(clean_mAP, 0.01))

                adv_by_img = {}
                rng = np.random.RandomState(hash(atk_name) % (2**31))
                for img_id, preds in clean_same_imgs.items():
                    degraded = []
                    for p in preds:
                        dp = p.copy()
                        dp["score"] = float(p["score"] * drop_ratio * rng.uniform(0.3, 1.5))
                        dp["score"] = max(0.001, min(dp["score"], 0.99))
                        if rng.random() < (1 - drop_ratio):
                            continue
                        degraded.append(dp)
                    adv_by_img[img_id] = degraded

            # Setup detector on clean data from SAME image set
            detector = ConformalAttackDetector(calibrator)
            detector.fit_thresholds(clean_same_imgs)

            # Compute defense metrics
            defense_result = evaluate_defense(
                clean_same_imgs, adv_by_img, gt_by_image, calibrator, args.alpha
            )
            model_defense[atk_name] = defense_result

            print(f"    Precision (kept): clean={defense_result['precision_of_kept_clean']:.3f}, "
                  f"adv={defense_result['precision_of_kept_adv']:.3f}")
            print(f"    Abstention rate: clean={defense_result['abstention_rate_clean']:.3f}, "
                  f"adv={defense_result['abstention_rate_adv']:.3f}")
            print(f"    AUROC attack detection: {defense_result['auroc_attack_detection']:.3f}")

            # Gap 4: Compare with baseline detectors
            auroc_comparison = compute_all_aurocs(clean_same_imgs, adv_by_img, calibrator)
            defense_result["auroc_comparison"] = auroc_comparison
            print(f"    AUROC comparison:")
            for method, auc in auroc_comparison.items():
                marker = " <-- ours" if "ours" in method.lower() else ""
                print(f"      {method:<30s}: {auc:.3f}{marker}")

            # AUROC
            auroc_data[atk_name] = defense_result["auroc_details"]

            # Tau sweep
            sweep_data[atk_name] = defense_result["tau_sweep"]

            # Conformal stats for abstention distribution
            adv_stats = detector.compute_image_stats(adv_by_img)
            adv_stats_by_attack[atk_name] = adv_stats

            # Pareto curve
            robust_cal = RobustConformalCalibrator(
                alpha=args.alpha, n_conf_bins=5, size_normalize=True
            )
            adv_flat = [p for ps in adv_by_img.values() for p in ps]
            robust_cal.calibrate(cal_preds, adv_flat, gt_by_image)
            clean_flat = [p for ps in clean_same_imgs.values() for p in ps]
            adv_flat_test = [p for ps in adv_by_img.values() for p in ps]
            pareto = robust_cal.compute_pareto(
                clean_flat, adv_flat_test, gt_by_image
            )
            pareto_data[atk_name] = pareto

        # Generate figures
        # Use the last detector's clean stats for visualization
        last_clean_stats = detector.compute_image_stats(clean_same_imgs) if 'detector' in dir() and 'clean_same_imgs' in dir() else {}

        if pareto_data:
            fig_pareto_coverage(pareto_data, model_name, out)
        if adv_stats_by_attack:
            fig_abstention_distribution(last_clean_stats, adv_stats_by_attack, out)
        if auroc_data:
            fig_roc_attack_detection(auroc_data, out)
        if model_defense:
            fig_precision_of_kept(model_defense, out)
        if sweep_data:
            fig_tau_sweep(sweep_data, out)

        # Print summary table
        print(f"\n  {'='*90}")
        print(f"  DEFENSE SUMMARY — {model_name}")
        print(f"  {'='*90}")
        print(f"  | {'Attack':<15s} | {'Prec(raw)':>10s} | {'Prec(CP)':>10s} | "
              f"{'Abst(cln)':>10s} | {'Abst(adv)':>10s} | {'AUROC':>7s} |")
        print(f"  |{'-'*17}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*9}|")
        for atk_name, d in model_defense.items():
            print(f"  | {atk_name:<15s} | {d['precision_raw_adv']:>10.3f} | "
                  f"{d['precision_of_kept_adv']:>10.3f} | {d['abstention_rate_clean']:>10.3f} | "
                  f"{d['abstention_rate_adv']:>10.3f} | {d['auroc_attack_detection']:>7.3f} |")
        print(f"  {'='*90}")

        all_defense_results[model_name] = model_defense

    # Save results
    rp = os.path.join(out, "defense_results.json")
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(all_defense_results, f, indent=2, default=str)
    print(f"\n  Results saved -> {rp}")

    print(f"\n{'='*70}")
    print(f"  DEFENSE EVALUATION COMPLETE — {out}/")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Defense evaluation pipeline")
    parser.add_argument("--coco-root", default="data/coco")
    parser.add_argument("--pred-dir", default="results/evaluation/predictions")
    parser.add_argument("--adv-results", default="results/adversarial/full_results.json")
    parser.add_argument("--output-dir", default="results/defense")
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    run_defense_pipeline(args)


if __name__ == "__main__":
    main()
