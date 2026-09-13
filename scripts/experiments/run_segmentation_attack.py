#!/usr/bin/env python3
"""
run_segmentation_attack.py — Adversarial attacks on segmentation models.

Runs the adversarial pipeline on YOLOv8x-seg and YOLOv11x-seg.
Compares bbox_AP drop vs mask_AP drop under the same attacks.
Key question: is segmentation more vulnerable than detection?

Usage:
    python run_segmentation_attack.py
    python run_segmentation_attack.py --models yolov8x-seg --max-images 50
"""

import os, sys, json, argparse, warnings, gc, random, tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from configs.model_config import MODEL_ZOO, SEGMENTATION_MODELS
from src.models.detector_zoo import build_detector
from src.evaluation.metrics import format_predictions_coco, run_coco_eval
from src.attacks.real_attacks import UltralyticsAttackWrapper, build_attack, ATTACK_CATALOG
from run_adversarial_eval import create_subset_annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.grid': True, 'grid.alpha': 0.3,
    'font.size': 10, 'axes.titleweight': 'bold',
    'savefig.dpi': 250, 'savefig.bbox': 'tight',
})


def run_seg_attack_pipeline(args):
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    tmp_dir = os.path.join(out, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("=" * 70)
    print("  SEGMENTATION UNDER ATTACK")
    print("=" * 70)

    # Load COCO
    ann_file = os.path.join(args.coco_root, "annotations", "instances_val2017.json")
    with open(ann_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    images = {img["id"]: img for img in coco_data["images"]}
    gt_by_image = defaultdict(list)
    for ann in coco_data["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)
    cat_ids = sorted({c["id"] for c in coco_data["categories"]})

    # Select images with masks
    img_dir = os.path.join(args.coco_root, "val2017")
    candidates = []
    for img_id, img_info in images.items():
        if len(gt_by_image.get(img_id, [])) >= 2:
            path = os.path.join(img_dir, img_info["file_name"])
            if os.path.exists(path):
                candidates.append({"image_id": img_id, "file_path": path})
    random.seed(42)
    random.shuffle(candidates)
    items = candidates[:args.max_images]
    print(f"  Selected {len(items)} images")

    # Create subset annotations
    test_ids = [it["image_id"] for it in items]
    subset_ann = os.path.join(tmp_dir, "seg_subset_annotations.json")
    create_subset_annotations(coco_data, test_ids, subset_ann)

    models = args.models or SEGMENTATION_MODELS
    attacks = args.attacks or ["fgsm", "pgd-20", "dag"]

    all_results = {}

    for model_name in models:
        if model_name not in MODEL_ZOO:
            print(f"  [SKIP] {model_name} not in MODEL_ZOO")
            continue
        cfg = MODEL_ZOO[model_name]
        if "instance_segmentation" not in cfg.get("tasks", []):
            print(f"  [SKIP] {model_name} is not a segmentation model")
            continue

        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*60}")

        try:
            detector = build_detector(model_name, cfg, device)
            detector.load()
        except Exception as e:
            print(f"  [ERROR] Failed to load {model_name}: {e}")
            continue

        wrapper = UltralyticsAttackWrapper(detector.model, device)

        # Clean inference
        print(f"  Clean inference...")
        clean_bbox_preds, clean_segm_preds = [], []
        for item in tqdm(items, desc="  Clean", ncols=80):
            try:
                r = detector.predict(item["file_path"], conf_threshold=0.01)
                bbox_p = format_predictions_coco(item["image_id"], r, cat_ids, "bbox")
                clean_bbox_preds.extend(bbox_p)
                if r.get("masks") is not None:
                    segm_p = format_predictions_coco(item["image_id"], r, cat_ids, "segm")
                    clean_segm_preds.extend(segm_p)
            except Exception:
                continue

        clean_bbox_map = run_coco_eval(clean_bbox_preds, subset_ann, "bbox")
        clean_segm_map = run_coco_eval(clean_segm_preds, subset_ann, "segm") if clean_segm_preds else {}

        print(f"  Clean bbox mAP={clean_bbox_map.get('mAP@[0.5:0.95]', 0):.4f}")
        print(f"  Clean mask mAP={clean_segm_map.get('mAP@[0.5:0.95]', 0):.4f}")

        model_results = {
            "model": model_name,
            "n_images": len(items),
            "clean_bbox_mAP": clean_bbox_map,
            "clean_mask_mAP": clean_segm_map,
            "attacks": {},
        }

        for atk_name in attacks:
            if atk_name not in ATTACK_CATALOG:
                continue

            print(f"\n  Attack: {atk_name}")
            try:
                attack = build_attack(atk_name, wrapper)
            except Exception as e:
                print(f"    [ERROR] Build failed: {e}")
                continue

            atk_bbox_preds, atk_segm_preds = [], []
            l_infs, n_success, n_fail = [], 0, 0
            first_error = None

            for item in tqdm(items, desc=f"    {atk_name}", ncols=80):
                try:
                    ar = attack(item["file_path"])
                    l_infs.append(ar.l_inf)

                    adv_path = os.path.join(tmp_dir, f"seg_adv.png")
                    Image.fromarray(ar.adv_image).save(adv_path)
                    r = detector.predict(adv_path, conf_threshold=0.01)

                    bbox_p = format_predictions_coco(item["image_id"], r, cat_ids, "bbox")
                    atk_bbox_preds.extend(bbox_p)
                    if r.get("masks") is not None:
                        segm_p = format_predictions_coco(item["image_id"], r, cat_ids, "segm")
                        atk_segm_preds.extend(segm_p)

                    n_success += 1
                    if os.path.exists(adv_path):
                        os.remove(adv_path)

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                    n_fail += 1
                    if first_error is None:
                        first_error = f"RuntimeError: {str(e)[:200]}"
                    continue
                except Exception as e:
                    n_fail += 1
                    if first_error is None:
                        first_error = f"{type(e).__name__}: {str(e)[:200]}"
                    continue

            print(f"    Success: {n_success}/{n_success+n_fail}")
            if first_error:
                print(f"    First error: {first_error}")

            if not atk_bbox_preds:
                print(f"    [WARN] No predictions — skipping")
                continue

            atk_bbox_map = run_coco_eval(atk_bbox_preds, subset_ann, "bbox")
            atk_segm_map = run_coco_eval(atk_segm_preds, subset_ann, "segm") if atk_segm_preds else {}

            bbox_drop = clean_bbox_map.get("mAP@[0.5:0.95]", 0) - atk_bbox_map.get("mAP@[0.5:0.95]", 0)
            mask_drop = clean_segm_map.get("mAP@[0.5:0.95]", 0) - atk_segm_map.get("mAP@[0.5:0.95]", 0)

            print(f"    bbox mAP: {atk_bbox_map.get('mAP@[0.5:0.95]', 0):.4f} (drop={bbox_drop:+.4f})")
            print(f"    mask mAP: {atk_segm_map.get('mAP@[0.5:0.95]', 0):.4f} (drop={mask_drop:+.4f})")
            print(f"    L_inf={np.mean(l_infs):.4f} | Success={n_success}/{len(items)}")

            model_results["attacks"][atk_name] = {
                "attacked_bbox_mAP": atk_bbox_map,
                "attacked_mask_mAP": atk_segm_map,
                "bbox_mAP_drop": float(bbox_drop),
                "mask_mAP_drop": float(mask_drop),
                "mean_l_inf": float(np.mean(l_infs)) if l_infs else 0,
                "n_success": n_success,
                "mask_more_vulnerable": float(mask_drop) > float(bbox_drop),
            }

        all_results[model_name] = model_results

        del detector, wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Generate comparison figure
    _generate_seg_figure(all_results, out)

    # Save
    rp = os.path.join(out, "segmentation_attack_results.json")
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Cleanup
    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*70}")
    print(f"  SEGMENTATION ATTACK COMPLETE — {out}/")
    print(f"{'='*70}")


def _generate_seg_figure(all_results, out):
    """Generate bbox vs mask vulnerability comparison figure."""
    for model_name, mr in all_results.items():
        attacks = mr.get("attacks", {})
        if not attacks:
            continue

        atk_names = list(attacks.keys())
        bbox_drops = [attacks[a]["bbox_mAP_drop"] for a in atk_names]
        mask_drops = [attacks[a]["mask_mAP_drop"] for a in atk_names]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(atk_names))
        w = 0.35
        ax.bar(x - w / 2, bbox_drops, w, label='Bbox mAP drop', color='#3498DB', alpha=0.8)
        ax.bar(x + w / 2, mask_drops, w, label='Mask mAP drop', color='#E74C3C', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(atk_names, fontsize=10)
        ax.set_ylabel("mAP Drop")
        ax.set_title(f"Detection vs Segmentation Vulnerability — {model_name}")
        ax.legend(fontsize=10)

        for i in range(len(atk_names)):
            if mask_drops[i] > bbox_drops[i]:
                ax.text(i, max(bbox_drops[i], mask_drops[i]) + 0.01,
                        "Mask more\nvulnerable", ha='center', fontsize=7, color='red')

        plt.tight_layout()
        plt.savefig(os.path.join(out, f"fig_seg_vulnerability_{model_name}.png"))
        plt.close()
        print(f"  [SAVED] fig_seg_vulnerability_{model_name}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", default="data/coco")
    parser.add_argument("--output-dir", default="results/segmentation")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--attacks", nargs="+", default=None)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_seg_attack_pipeline(args)


if __name__ == "__main__":
    main()
