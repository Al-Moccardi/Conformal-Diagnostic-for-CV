#!/usr/bin/env python3
"""
run_adversarial_eval.py — Full adversarial + conformal evaluation pipeline.

FIXES from v8:
  - Creates FILTERED annotation file for the image subset (fixes mAP=0.005 bug)
  - Saves adversarial images at ORIGINAL resolution (not 640x640)
  - Handles attack failures gracefully (no "No predictions" for working models)
"""

import os, sys, json, argparse, warnings, time, random, gc, tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import MODEL_ZOO
from src.calibration.adaptive_conformal import AdaptiveConformalCalibrator


# =====================================================================
#  DATA LOADING — with filtered annotation file creation
# =====================================================================

def load_coco_data(coco_root, max_images=None):
    ann_file = os.path.join(coco_root, "annotations", "instances_val2017.json")
    img_dir = os.path.join(coco_root, "val2017")
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    images = {img["id"]: img for img in data["images"]}
    gt_by_image = defaultdict(list)
    for ann in data["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)
    cat_ids = sorted({c["id"] for c in data["categories"]})

    candidates = []
    for img_id, img_info in images.items():
        if len(gt_by_image.get(img_id, [])) >= 2:
            path = os.path.join(img_dir, img_info["file_name"])
            if os.path.exists(path):
                candidates.append({"image_id": img_id, "file_path": path,
                                   "width": img_info["width"], "height": img_info["height"]})
    random.seed(42)
    random.shuffle(candidates)
    if max_images:
        candidates = candidates[:max_images]

    return candidates, gt_by_image, images, cat_ids, data


def create_subset_annotations(full_coco_data, image_ids, output_path):
    """Create a filtered COCO annotation file for a subset of images.
    This is CRITICAL — COCO eval divides by total images in the file,
    so using the full file with 25 images gives mAP / 200."""
    subset_ids = set(image_ids)
    subset = {
        "images": [img for img in full_coco_data["images"] if img["id"] in subset_ids],
        "annotations": [ann for ann in full_coco_data["annotations"] if ann["image_id"] in subset_ids],
        "categories": full_coco_data["categories"],
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(subset, f)
    return output_path


# =====================================================================
#  CONFORMAL EVALUATION
# =====================================================================

def _box_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0


def _compute_ece(confs, accs, n_bins=15):
    if len(confs) == 0:
        return 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (confs > edges[i]) & (confs <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(confs)) * abs(accs[m].mean() - confs[m].mean())
    return float(ece)


def evaluate_with_conformal(preds_coco, gt_by_image, calibrator, label=""):
    preds_by_img = defaultdict(list)
    for p in preds_coco:
        preds_by_img[p["image_id"]].append(p)

    total_gt, covered_gt = 0, 0
    all_margins, all_set_sizes = [], []
    n_kept, n_filtered = 0, 0
    confs_kept, correct_kept = [], []
    confs_all, correct_all = [], []

    for img_id, img_preds in preds_by_img.items():
        gt_anns = gt_by_image.get(img_id, [])
        total_gt += len(gt_anns)
        gt_covered = [False] * len(gt_anns)

        for pred in sorted(img_preds, key=lambda x: -x["score"]):
            px, py, pw, ph = pred["bbox"]
            pred_xyxy = [px, py, px + pw, py + ph]
            is_tp = False
            for gt in gt_anns:
                if gt["category_id"] != pred["category_id"]:
                    continue
                gx, gy, gw, gh = gt["bbox"]
                if _box_iou(pred_xyxy, [gx, gy, gx + gw, gy + gh]) >= 0.5:
                    is_tp = True
                    break

            confs_all.append(pred["score"])
            correct_all.append(1.0 if is_tp else 0.0)
            result = calibrator.predict(pred)

            if result.keep:
                n_kept += 1
                all_margins.append(result.box_delta_pixels)
                all_set_sizes.append(result.class_set_size)
                confs_kept.append(pred["score"])
                correct_kept.append(1.0 if is_tp else 0.0)
                for j, gt in enumerate(gt_anns):
                    if gt_covered[j] or gt["category_id"] != pred["category_id"]:
                        continue
                    gx, gy, gw, gh = gt["bbox"]
                    if _box_iou(pred_xyxy, [gx, gy, gx + gw, gy + gh]) >= 0.5:
                        gt_covered[j] = True
                        break
            else:
                n_filtered += 1

        covered_gt += sum(gt_covered)

    coverage = covered_gt / max(total_gt, 1)
    total_preds = n_kept + n_filtered
    ece_before = _compute_ece(np.array(confs_all), np.array(correct_all)) if confs_all else 0
    ece_after = _compute_ece(np.array(confs_kept), np.array(correct_kept)) if confs_kept else 0

    return {
        "label": label, "coverage": float(coverage),
        "total_gt": total_gt, "covered_gt": covered_gt,
        "n_kept": n_kept, "n_filtered": n_filtered, "total_preds": total_preds,
        "filter_rate": n_filtered / max(total_preds, 1),
        "mean_margin_px": float(np.mean(all_margins)) if all_margins else 0,
        "mean_set_size": float(np.mean(all_set_sizes)) if all_set_sizes else 0,
        "conf_threshold": float(calibrator.conf_threshold),
        "ece_before_cp": float(ece_before), "ece_after_cp": float(ece_after),
    }


# =====================================================================
#  MAIN PIPELINE
# =====================================================================

def run_full_pipeline(args):
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "attack_samples"), exist_ok=True)
    tmp_dir = os.path.join(out, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("=" * 70)
    print("  FULL ADVERSARIAL + CONFORMAL EVALUATION PIPELINE (v9 — fixed)")
    print("=" * 70)

    print(f"\n  Loading COCO data (max {args.max_images} images)...")
    items, gt_by_image, images_dict, cat_ids, full_coco_data = load_coco_data(
        args.coco_root, args.max_images)
    print(f"  Selected {len(items)} images")

    random.seed(42)
    random.shuffle(items)
    n_cal = len(items) // 2
    cal_items, test_items = items[:n_cal], items[n_cal:]
    print(f"  Calibration: {len(cal_items)} | Test: {len(test_items)}")

    # FIX: Create filtered annotation files for COCO eval on subset
    test_ids = [it["image_id"] for it in test_items]
    test_ann_file = os.path.join(tmp_dir, "test_subset_annotations.json")
    create_subset_annotations(full_coco_data, test_ids, test_ann_file)
    print(f"  Created subset annotation file: {len(test_ids)} test images")

    models_to_eval = args.models or ["yolov8x"]
    attack_names = args.attacks or ["fgsm", "pgd-20", "pgd-50", "dag", "tog-v"]

    print(f"\n  Models: {models_to_eval}")
    print(f"  Attacks: {attack_names}")
    print(f"  Alpha: {args.alpha}")

    all_results = {}

    for model_name in models_to_eval:
        if model_name not in MODEL_ZOO:
            print(f"\n  [SKIP] {model_name}")
            continue

        model_config = MODEL_ZOO[model_name]
        print(f"\n{'='*70}")
        print(f"  MODEL: {model_name} ({model_config['architecture']})")
        print(f"{'='*70}")

        from src.models.detector_zoo import build_detector
        detector = build_detector(model_name, model_config, device)
        detector.load()

        from src.attacks.real_attacks import (
            UltralyticsAttackWrapper, HuggingFaceAttackWrapper,
            build_attack, ATTACK_CATALOG,
        )

        if model_config["framework"] == "ultralytics":
            wrapper = UltralyticsAttackWrapper(detector.model, device)
        elif model_config["framework"] == "huggingface":
            wrapper = HuggingFaceAttackWrapper(detector.model, detector.processor, device)
        else:
            print(f"  [SKIP] Framework {model_config['framework']}")
            continue

        # ── PHASE 1: Clean inference ──
        print(f"\n  Phase 1: Clean inference...")
        from src.evaluation.metrics import format_predictions_coco, run_coco_eval

        cal_preds, test_preds_clean = [], []
        for item in tqdm(cal_items, desc="  Cal", ncols=80):
            r = detector.predict(item["file_path"], conf_threshold=0.01)
            cal_preds.extend(format_predictions_coco(item["image_id"], r, cat_ids, "bbox"))

        for item in tqdm(test_items, desc="  Test", ncols=80):
            r = detector.predict(item["file_path"], conf_threshold=0.01)
            test_preds_clean.extend(format_predictions_coco(item["image_id"], r, cat_ids, "bbox"))

        # FIX: Evaluate against SUBSET annotation file
        clean_map = run_coco_eval(test_preds_clean, test_ann_file, iou_type="bbox")
        print(f"  Clean mAP@0.5={clean_map['mAP@0.5']:.4f} | mAP={clean_map['mAP@[0.5:0.95]']:.4f}")

        # ── PHASE 2: Calibrate on clean ──
        print(f"\n  Phase 2: Conformal calibration (alpha={args.alpha})...")
        clean_cal = AdaptiveConformalCalibrator(alpha=args.alpha, n_conf_bins=5, size_normalize=True)
        clean_cal.calibrate(cal_preds, gt_by_image)
        print(f"    Threshold={clean_cal.conf_threshold:.4f} | qhat={clean_cal.global_qhat:.4f}")

        # ── PHASE 3: Clean + conformal ──
        print(f"\n  Phase 3: Clean + conformal...")
        clean_cf = evaluate_with_conformal(test_preds_clean, gt_by_image, clean_cal, "Clean+CleanCP")
        print(f"    Coverage={clean_cf['coverage']:.4f} | Margin={clean_cf['mean_margin_px']:.1f}px | "
              f"ECE={clean_cf['ece_after_cp']:.4f}")

        model_results = {
            "model": model_name, "architecture": model_config["architecture"],
            "alpha": args.alpha, "n_cal": len(cal_items), "n_test": len(test_items),
            "clean_mAP": clean_map, "clean_conformal": clean_cf,
            "calibration_stats": clean_cal.get_summary(), "attacks": {},
        }

        # ── PHASE 4-6: Each attack ──
        for atk_name in attack_names:
            if atk_name not in ATTACK_CATALOG:
                continue
            atk_info = ATTACK_CATALOG[atk_name]
            print(f"\n  {'~'*60}")
            print(f"  ATTACK: {atk_name} ({atk_info['type']}, {atk_info['norm']})")

            try:
                attack = build_attack(atk_name, wrapper)
            except Exception as e:
                print(f"  [ERROR] Build failed: {e}")
                continue

            # Phase 4a: Attack test images
            print(f"  Phase 4a: Attacking test images...")
            atk_test_preds = []
            per_image_preds = []  # NEW: save per-image for defense eval
            atk_times, l_infs, l_2s = [], [], []
            n_success, n_fail = 0, 0

            for idx, item in enumerate(tqdm(test_items, desc=f"  {atk_name} test", ncols=80)):
                try:
                    ar = attack(item["file_path"])
                    atk_times.append(ar.attack_time_ms)
                    l_infs.append(ar.l_inf)
                    l_2s.append(ar.l_2)

                    # Save adversarial image at ORIGINAL resolution and detect
                    adv_path = os.path.join(tmp_dir, f"adv_{idx}.png")
                    Image.fromarray(ar.adv_image).save(adv_path)
                    dr = detector.predict(adv_path, conf_threshold=0.01)
                    preds = format_predictions_coco(item["image_id"], dr, cat_ids, "bbox")
                    atk_test_preds.extend(preds)
                    per_image_preds.extend(preds)  # track per-image
                    n_success += 1

                    # Save ALL adversarial images for transfer attack experiments
                    sd = os.path.join(out, "attack_samples")
                    Image.fromarray(ar.adv_image).save(
                        os.path.join(sd, f"{model_name}_{atk_name}_adv_{idx}.png"))
                    if idx < 5:
                        Image.fromarray(ar.clean_image).save(
                            os.path.join(sd, f"{model_name}_{atk_name}_clean_{idx}.png"))

                    if os.path.exists(adv_path):
                        os.remove(adv_path)

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                    n_fail += 1
                    continue
                except Exception as e:
                    n_fail += 1
                    continue

            print(f"  Attack success: {n_success}/{n_success+n_fail}")

            # Phase 4b: Attack calibration images
            print(f"  Phase 4b: Attacking cal images...")
            atk_cal_preds = []
            for idx, item in enumerate(tqdm(cal_items, desc=f"  {atk_name} cal", ncols=80)):
                try:
                    ar = attack(item["file_path"])
                    adv_path = os.path.join(tmp_dir, f"adv_cal_{idx}.png")
                    Image.fromarray(ar.adv_image).save(adv_path)
                    dr = detector.predict(adv_path, conf_threshold=0.01)
                    atk_cal_preds.extend(format_predictions_coco(item["image_id"], dr, cat_ids, "bbox"))
                    if os.path.exists(adv_path):
                        os.remove(adv_path)
                except Exception:
                    continue

            if not atk_test_preds:
                print(f"  [WARN] No attack predictions — skipping")
                continue

            # FIX: Evaluate against SUBSET annotation file
            atk_map = run_coco_eval(atk_test_preds, test_ann_file, iou_type="bbox")
            dm = clean_map["mAP@[0.5:0.95]"] - atk_map.get("mAP@[0.5:0.95]", 0)
            print(f"\n  Attacked mAP={atk_map.get('mAP@[0.5:0.95]',0):.4f} (delta={dm:+.4f})")
            if atk_times:
                print(f"  L_inf={np.mean(l_infs):.4f} | L2={np.mean(l_2s):.4f} | Time={np.mean(atk_times):.0f}ms")

            # Phase 5: Clean calibrator on attacked preds
            print(f"  Phase 5: Attacked + CLEAN CP (naive)...")
            naive_cf = evaluate_with_conformal(atk_test_preds, gt_by_image, clean_cal, f"{atk_name}+CleanCP")
            print(f"    Coverage={naive_cf['coverage']:.4f} (drop={clean_cf['coverage']-naive_cf['coverage']:+.4f})")

            # Phase 6: Adversarial recalibration
            print(f"  Phase 6: Adversarial recalibration...")
            adv_cal = AdaptiveConformalCalibrator(alpha=args.alpha, n_conf_bins=5, size_normalize=True)
            adv_cal.calibrate(atk_cal_preds, gt_by_image)
            recal_cf = evaluate_with_conformal(atk_test_preds, gt_by_image, adv_cal, f"{atk_name}+AdvCP")
            print(f"    Coverage={recal_cf['coverage']:.4f} (from {naive_cf['coverage']:.4f})")
            print(f"    Margin={recal_cf['mean_margin_px']:.1f}px (was {clean_cf['mean_margin_px']:.1f}px)")

            # Summary table
            print(f"\n  +{'='*74}+")
            print(f"  | {'Setting':<30s} {'Cov':>6s} {'Margin':>8s} {'|C|':>5s} {'ECE':>7s} {'mAP':>8s} |")
            print(f"  +{'-'*74}+")
            print(f"  | {'Clean + Clean CP':<30s} {clean_cf['coverage']:>6.3f} {clean_cf['mean_margin_px']:>7.1f}px {clean_cf['mean_set_size']:>5.2f} {clean_cf['ece_after_cp']:>7.4f} {clean_map['mAP@[0.5:0.95]']:>8.4f} |")
            print(f"  | {atk_name+' + Clean CP':<30s} {naive_cf['coverage']:>6.3f} {naive_cf['mean_margin_px']:>7.1f}px {naive_cf['mean_set_size']:>5.2f} {naive_cf['ece_after_cp']:>7.4f} {atk_map.get('mAP@[0.5:0.95]',0):>8.4f} |")
            print(f"  | {atk_name+' + Adv. CP (recal)':<30s} {recal_cf['coverage']:>6.3f} {recal_cf['mean_margin_px']:>7.1f}px {recal_cf['mean_set_size']:>5.2f} {recal_cf['ece_after_cp']:>7.4f} {atk_map.get('mAP@[0.5:0.95]',0):>8.4f} |")
            print(f"  +{'='*74}+")

            model_results["attacks"][atk_name] = {
                "type": atk_info["type"], "norm": atk_info["norm"], "ref": atk_info["ref"],
                "attacked_mAP": atk_map, "delta_mAP": float(dm),
                "perturbation": {"mean_l_inf": float(np.mean(l_infs)) if l_infs else 0,
                                 "mean_l2": float(np.mean(l_2s)) if l_2s else 0,
                                 "mean_time_ms": float(np.mean(atk_times)) if atk_times else 0},
                "naive_conformal": naive_cf, "recalibrated_conformal": recal_cf,
                "recalibration_stats": adv_cal.get_summary(),
                "per_image_predictions": per_image_preds,  # NEW: for defense eval
            }

        all_results[model_name] = model_results
        del detector, wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Save — MERGE with existing results (FIX: don't overwrite previous runs)
    rp = os.path.join(out, "full_results.json")
    if os.path.exists(rp):
        try:
            with open(rp, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            # Merge: new results overwrite per-model, but preserve other models
            existing.update(all_results)
            all_results = existing
            print(f"\n  Merged with existing results ({len(existing)} total models)")
        except (json.JSONDecodeError, Exception):
            pass  # If file is corrupted, just overwrite

    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results -> {rp}")

    # Plots
    generate_paper_figures(all_results, out)

    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*70}\n  COMPLETE — {out}/\n{'='*70}")


def generate_paper_figures(all_results, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    C = {"clean": "#27AE60", "naive": "#E74C3C", "recal": "#2E86C1"}
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.grid': True,
        'grid.alpha': 0.3, 'font.size': 10, 'axes.titleweight': 'bold',
        'figure.dpi': 150, 'savefig.dpi': 250, 'savefig.bbox': 'tight'})

    for mn, mr in all_results.items():
        attacks = mr.get("attacks", {})
        if not attacks:
            continue
        an = list(attacks.keys())
        na = len(an)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        x = np.arange(na)
        w = 0.25
        cc = mr["clean_conformal"]["coverage"]
        nc = [attacks[a]["naive_conformal"]["coverage"] for a in an]
        rc = [attacks[a]["recalibrated_conformal"]["coverage"] for a in an]

        ax = axes[0]
        ax.bar(x - w, [cc] * na, w, label="Clean+CleanCP", color=C["clean"], alpha=0.8)
        ax.bar(x, nc, w, label="Attack+CleanCP", color=C["naive"], alpha=0.8)
        ax.bar(x + w, rc, w, label="Attack+AdvCP", color=C["recal"], alpha=0.8)
        ax.axhline(1 - mr["alpha"], color='black', ls='--', lw=1, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(an, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel("Coverage")
        ax.set_title("Coverage comparison")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.05)

        cm = mr["clean_conformal"]["mean_margin_px"]
        nm = [attacks[a]["naive_conformal"]["mean_margin_px"] for a in an]
        rm = [attacks[a]["recalibrated_conformal"]["mean_margin_px"] for a in an]
        ax = axes[1]
        ax.bar(x - w, [cm] * na, w, color=C["clean"], alpha=0.8, label="Clean CP")
        ax.bar(x, nm, w, color=C["naive"], alpha=0.8, label="Naive on attacked")
        ax.bar(x + w, rm, w, color=C["recal"], alpha=0.8, label="Adv CP")
        ax.set_xticks(x)
        ax.set_xticklabels(an, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel("Margin (px)")
        ax.set_title("Margin width")
        ax.legend(fontsize=7)

        ce = mr["clean_conformal"]["ece_after_cp"]
        ne = [attacks[a]["naive_conformal"]["ece_after_cp"] for a in an]
        re = [attacks[a]["recalibrated_conformal"]["ece_after_cp"] for a in an]
        ax = axes[2]
        ax.bar(x - w, [ce] * na, w, color=C["clean"], alpha=0.8, label="Clean CP")
        ax.bar(x, ne, w, color=C["naive"], alpha=0.8, label="Attacked naive")
        ax.bar(x + w, re, w, color=C["recal"], alpha=0.8, label="Adv CP")
        ax.set_xticks(x)
        ax.set_xticklabels(an, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel("ECE")
        ax.set_title("Calibration error")
        ax.legend(fontsize=7)

        fig.suptitle(f"{mn} — Conformal Under Attack (alpha={mr['alpha']})",
                     fontsize=14, fontweight='bold', y=1.03)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"fig_coverage_{mn}.png"))
        plt.close()
        print(f"  [SAVED] fig_coverage_{mn}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", default="data/coco")
    parser.add_argument("--output-dir", default="results/adversarial")
    parser.add_argument("--models", nargs="+", default=["yolov8x"])
    parser.add_argument("--attacks", nargs="+",
                        default=["fgsm", "pgd-20", "pgd-50", "dag", "tog-v"])
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-all-images", action="store_true", default=True,
                        help="Save ALL adversarial images for transfer attacks")
    args = parser.parse_args()
    run_full_pipeline(args)


if __name__ == "__main__":
    main()
