#!/usr/bin/env python3
"""
run_transfer_attacks.py — Transfer attack evaluation (black-box).

Loads adversarial images generated for model A, runs inference with model B.
Tests black-box robustness of the conformal defense.

Usage:
    python run_transfer_attacks.py
    python run_transfer_attacks.py --source yolov8x --targets yolov11x detr-resnet101
"""

import os, sys, json, argparse, warnings, gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from configs.model_config import MODEL_ZOO, ADVERSARIAL_MODELS
from src.models.detector_zoo import build_detector
from src.evaluation.metrics import format_predictions_coco, run_coco_eval
from src.calibration.adaptive_conformal import AdaptiveConformalCalibrator
from run_adversarial_eval import evaluate_with_conformal, create_subset_annotations


def find_attack_images(attack_dir, source_model, attack_name):
    """Find all saved adversarial images for a source model + attack."""
    images = []
    prefix = f"{source_model}_{attack_name}_adv_"
    if not os.path.isdir(attack_dir):
        return images
    for f in sorted(os.listdir(attack_dir)):
        if f.startswith(prefix) and f.endswith('.png'):
            idx_str = f[len(prefix):-4]
            try:
                idx = int(idx_str)
                images.append({
                    "path": os.path.join(attack_dir, f),
                    "index": idx,
                    "source_model": source_model,
                    "attack": attack_name,
                })
            except ValueError:
                continue
    return images


def run_transfer_pipeline(args):
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("=" * 70)
    print("  TRANSFER ATTACK EVALUATION (Black-Box)")
    print("=" * 70)

    attack_dir = os.path.join(args.adv_dir, "attack_samples")
    if not os.path.isdir(attack_dir):
        print(f"  [ERROR] Attack samples directory not found: {attack_dir}")
        print(f"  Run adversarial eval first with --save-all-images")
        return

    # Load GT for COCO eval
    ann_file = os.path.join(args.coco_root, "annotations", "instances_val2017.json")
    with open(ann_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    gt_by_image = defaultdict(list)
    for ann in coco_data["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)
    cat_ids = sorted({c["id"] for c in coco_data["categories"]})

    # Also load adversarial eval results for image ID mapping
    adv_results_file = os.path.join(args.adv_dir, "full_results.json")
    if not os.path.exists(adv_results_file):
        print(f"  [ERROR] {adv_results_file} not found")
        return
    with open(adv_results_file, 'r', encoding='utf-8') as f:
        adv_results = json.load(f)

    source_models = args.source or ADVERSARIAL_MODELS
    target_models = args.targets or [m for m in ADVERSARIAL_MODELS if m != source_models[0]]
    attacks = args.attacks or ["fgsm", "pgd-20", "dag"]

    all_results = {}

    for source in source_models:
        for atk_name in attacks:
            adv_images = find_attack_images(attack_dir, source, atk_name)
            if len(adv_images) < 5:
                print(f"\n  [SKIP] {source}/{atk_name}: only {len(adv_images)} images")
                continue

            print(f"\n{'='*60}")
            print(f"  SOURCE: {source} | ATTACK: {atk_name} | {len(adv_images)} images")
            print(f"{'='*60}")

            for target in target_models:
                if target == source:
                    continue
                if target not in MODEL_ZOO:
                    continue

                print(f"\n  Target model: {target}")
                try:
                    detector = build_detector(target, MODEL_ZOO[target], device)
                    detector.load()
                except Exception as e:
                    print(f"    [ERROR] Failed to load {target}: {e}")
                    continue

                # Run inference on adversarial images
                transfer_preds = []
                for item in tqdm(adv_images[:args.max_images],
                                 desc=f"    {target}", ncols=80):
                    try:
                        result = detector.predict(item["path"], conf_threshold=0.01)
                        # We don't have the correct image_id mapping for transfer,
                        # so we use the index as a synthetic ID
                        preds = format_predictions_coco(
                            item["index"], result, cat_ids, "bbox"
                        )
                        transfer_preds.extend(preds)
                    except Exception as e:
                        continue

                # Group transfer predictions by image
                transfer_by_img = defaultdict(list)
                for p in transfer_preds:
                    transfer_by_img[p["image_id"]].append(p)

                n_preds = len(transfer_preds)
                mean_conf = np.mean([p["score"] for p in transfer_preds]) if transfer_preds else 0

                # Load clean predictions for target model
                pred_path = os.path.join(args.pred_dir, f"{target}_bbox.json")
                target_cal = None
                auroc_transfer = 0.5
                clean_mean_conf = 0.0

                if os.path.exists(pred_path):
                    with open(pred_path, 'r', encoding='utf-8') as f:
                        clean_preds = json.load(f)
                    if len(clean_preds) > 100:
                        by_img = defaultdict(list)
                        for p in clean_preds:
                            by_img[p["image_id"]].append(p)
                        img_ids = sorted(by_img.keys())
                        cal_ids = set(img_ids[:len(img_ids) // 2])
                        test_ids = set(img_ids[len(img_ids) // 2:])
                        cal_preds_list = [p for p in clean_preds if p["image_id"] in cal_ids]
                        clean_test_by_img = {k: v for k, v in by_img.items() if k in test_ids}

                        target_cal = AdaptiveConformalCalibrator(
                            alpha=0.1, n_conf_bins=5, size_normalize=True
                        )
                        target_cal.calibrate(cal_preds_list, gt_by_image)

                        # Compute AUROC: clean test images vs transfer images
                        from src.calibration.conformal_defense import ConformalAttackDetector
                        det = ConformalAttackDetector(target_cal)
                        # Use a sample of clean test images (same count as transfer)
                        n_transfer = len(transfer_by_img)
                        clean_sample = dict(list(clean_test_by_img.items())[:n_transfer])
                        det.fit_thresholds(clean_sample)

                        try:
                            auroc_result = det.compute_auroc(clean_sample, transfer_by_img)
                            auroc_transfer = auroc_result["auroc"]
                        except Exception:
                            auroc_transfer = 0.5

                        clean_all_scores = [p["score"] for ps in clean_sample.values() for p in ps]
                        clean_mean_conf = np.mean(clean_all_scores) if clean_all_scores else 0

                key = f"{source}->{target}_{atk_name}"
                conf_drop = clean_mean_conf - mean_conf
                result = {
                    "source": source,
                    "target": target,
                    "attack": atk_name,
                    "n_adv_images": len(adv_images[:args.max_images]),
                    "n_predictions": n_preds,
                    "mean_confidence": float(mean_conf),
                    "clean_mean_confidence": float(clean_mean_conf),
                    "confidence_drop": float(conf_drop),
                    "auroc": float(auroc_transfer),
                }
                all_results[key] = result
                print(f"    Preds={n_preds}, MeanConf={mean_conf:.3f}, "
                      f"ConfDrop={conf_drop:+.3f}, AUROC={auroc_transfer:.3f}")

                del detector
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    # Save
    rp = os.path.join(out, "transfer_results.json")
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*85}")
    print(f"  TRANSFER ATTACK SUMMARY")
    print(f"{'='*85}")
    print(f"  | {'Transfer':<30s} | {'#Preds':>7s} | {'ConfDrop':>9s} | {'AUROC':>7s} |")
    print(f"  |{'-'*32}|{'-'*9}|{'-'*11}|{'-'*9}|")
    for key, r in all_results.items():
        print(f"  | {key:<30s} | {r['n_predictions']:>7d} | "
              f"{r.get('confidence_drop', 0):>+9.3f} | {r.get('auroc', 0.5):>7.3f} |")
    print(f"  {'='*85}")

    print(f"\n  Results -> {rp}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", default="data/coco")
    parser.add_argument("--pred-dir", default="results/evaluation/predictions")
    parser.add_argument("--adv-dir", default="results/adversarial")
    parser.add_argument("--output-dir", default="results/transfer")
    parser.add_argument("--source", nargs="+", default=None)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--attacks", nargs="+", default=None)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_transfer_pipeline(args)


if __name__ == "__main__":
    main()
