#!/usr/bin/env python3
"""
run_natural_corruptions.py — GAP 1: Natural corruptions experiment.

Applies fog, Gaussian blur, Gaussian noise, JPEG compression at 3 severity
levels to COCO images, then evaluates mAP drop + conformal coverage drop.

This fills the title promise: "Adversarial AND Natural Corruptions".

Usage:
    python run_natural_corruptions.py
    python run_natural_corruptions.py --models yolov8x rtdetr-l --max-images 200
"""

import os, sys, json, argparse, warnings, gc, random, tempfile
from pathlib import Path
from collections import defaultdict
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from configs.model_config import MODEL_ZOO
from src.calibration.adaptive_conformal import AdaptiveConformalCalibrator
from src.calibration.conformal_defense import ConformalAttackDetector

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.grid': True, 'grid.alpha': 0.3,
    'font.size': 10, 'axes.titleweight': 'bold',
    'savefig.dpi': 250, 'savefig.bbox': 'tight',
})


# =====================================================================
#  CORRUPTION FUNCTIONS
# =====================================================================

def apply_gaussian_noise(img_array, severity):
    """Add Gaussian noise. severity in {1,2,3}."""
    sigma = {1: 15, 2: 35, 3: 65}[severity]
    noise = np.random.normal(0, sigma, img_array.shape).astype(np.float32)
    noisy = np.clip(img_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_gaussian_blur(img_array, severity):
    """Apply Gaussian blur. severity in {1,2,3}."""
    radius = {1: 2, 2: 4, 3: 7}[severity]
    img = Image.fromarray(img_array)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred)


def apply_jpeg_compression(img_array, severity):
    """Apply JPEG compression artifacts. severity in {1,2,3}."""
    quality = {1: 40, 2: 15, 3: 5}[severity]
    img = Image.fromarray(img_array)
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert('RGB')
    return np.array(compressed)


def apply_fog(img_array, severity):
    """Simulate fog by blending with white + blur. severity in {1,2,3}."""
    alpha = {1: 0.2, 2: 0.4, 3: 0.65}[severity]
    blur_r = {1: 3, 2: 6, 3: 10}[severity]
    h, w = img_array.shape[:2]
    fog = np.full_like(img_array, 240, dtype=np.uint8)
    blended = (img_array.astype(np.float32) * (1 - alpha) +
               fog.astype(np.float32) * alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    img = Image.fromarray(blended).filter(ImageFilter.GaussianBlur(radius=blur_r))
    return np.array(img)


def apply_brightness(img_array, severity):
    """Reduce brightness. severity in {1,2,3}."""
    factor = {1: 0.7, 2: 0.4, 3: 0.2}[severity]
    dark = (img_array.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
    return dark


CORRUPTIONS = {
    'gaussian_noise': apply_gaussian_noise,
    'gaussian_blur': apply_gaussian_blur,
    'jpeg_compression': apply_jpeg_compression,
    'fog': apply_fog,
    'brightness': apply_brightness,
}


# =====================================================================
#  MAIN PIPELINE
# =====================================================================

def run_corruption_pipeline(args):
    import torch
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    tmp_dir = os.path.join(out, '_tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    print('=' * 70)
    print('  NATURAL CORRUPTIONS EXPERIMENT')
    print('=' * 70)

    # Load COCO
    ann_file = os.path.join(args.coco_root, 'annotations', 'instances_val2017.json')
    img_dir = os.path.join(args.coco_root, 'val2017')
    with open(ann_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    images = {img['id']: img for img in coco_data['images']}
    gt_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        gt_by_image[ann['image_id']].append(ann)
    cat_ids = sorted({c['id'] for c in coco_data['categories']})

    # Select images
    candidates = []
    for img_id, img_info in images.items():
        if len(gt_by_image.get(img_id, [])) >= 2:
            path = os.path.join(img_dir, img_info['file_name'])
            if os.path.exists(path):
                candidates.append({'image_id': img_id, 'file_path': path})
    random.seed(42)
    random.shuffle(candidates)
    items = candidates[:args.max_images]

    # Split cal/test
    n_cal = len(items) // 2
    cal_items, test_items = items[:n_cal], items[n_cal:]
    print(f'  Images: {len(items)} (cal={len(cal_items)}, test={len(test_items)})')

    # Subset annotations
    from run_adversarial_eval import create_subset_annotations
    test_ids = [it['image_id'] for it in test_items]
    subset_ann = os.path.join(tmp_dir, 'corruption_subset_ann.json')
    create_subset_annotations(coco_data, test_ids, subset_ann)

    models_to_eval = args.models or ['yolov8x', 'yolov11x']
    corruption_names = args.corruptions or list(CORRUPTIONS.keys())
    severities = [1, 2, 3]

    all_results = {}

    for model_name in models_to_eval:
        if model_name not in MODEL_ZOO:
            continue
        print(f'\n{"="*60}')
        print(f'  MODEL: {model_name}')
        print(f'{"="*60}')

        from src.models.detector_zoo import build_detector
        from src.evaluation.metrics import format_predictions_coco, run_coco_eval

        detector = build_detector(model_name, MODEL_ZOO[model_name], device)
        detector.load()

        # Clean inference on cal + test
        cal_preds, test_preds_clean = [], []
        for item in tqdm(cal_items, desc='  Cal clean', ncols=80):
            r = detector.predict(item['file_path'], conf_threshold=0.01)
            cal_preds.extend(format_predictions_coco(item['image_id'], r, cat_ids, 'bbox'))
        for item in tqdm(test_items, desc='  Test clean', ncols=80):
            r = detector.predict(item['file_path'], conf_threshold=0.01)
            test_preds_clean.extend(format_predictions_coco(item['image_id'], r, cat_ids, 'bbox'))

        clean_map = run_coco_eval(test_preds_clean, subset_ann, 'bbox')
        print(f'  Clean mAP={clean_map["mAP@[0.5:0.95]"]:.4f}')

        # Calibrate
        calibrator = AdaptiveConformalCalibrator(alpha=args.alpha, n_conf_bins=5, size_normalize=True)
        calibrator.calibrate(cal_preds, gt_by_image)

        # Clean conformal stats for AUROC baseline
        clean_by_img = defaultdict(list)
        for p in test_preds_clean:
            clean_by_img[p['image_id']].append(p)
        atk_detector = ConformalAttackDetector(calibrator)
        atk_detector.fit_thresholds(clean_by_img)

        # Clean coverage
        from run_adversarial_eval import evaluate_with_conformal
        clean_cf = evaluate_with_conformal(test_preds_clean, gt_by_image, calibrator, 'Clean')

        model_results = {
            'model': model_name,
            'n_test': len(test_items),
            'clean_mAP': clean_map,
            'clean_coverage': clean_cf['coverage'],
            'corruptions': {},
        }

        for corr_name in corruption_names:
            corr_fn = CORRUPTIONS[corr_name]
            for sev in severities:
                key = f'{corr_name}_s{sev}'
                print(f'\n  Corruption: {key}')

                corr_preds = []
                corr_by_img = defaultdict(list)

                for item in tqdm(test_items, desc=f'    {key}', ncols=80):
                    try:
                        img = np.array(Image.open(item['file_path']).convert('RGB'))
                        corrupted = corr_fn(img, sev)
                        tmp_path = os.path.join(tmp_dir, 'corrupted.png')
                        Image.fromarray(corrupted).save(tmp_path)
                        r = detector.predict(tmp_path, conf_threshold=0.01)
                        preds = format_predictions_coco(item['image_id'], r, cat_ids, 'bbox')
                        corr_preds.extend(preds)
                        for p in preds:
                            corr_by_img[p['image_id']].append(p)
                    except Exception:
                        continue

                if not corr_preds:
                    continue

                corr_map = run_coco_eval(corr_preds, subset_ann, 'bbox')
                corr_cf = evaluate_with_conformal(corr_preds, gt_by_image, calibrator, key)

                # AUROC: can we detect the corruption?
                auroc_result = atk_detector.compute_auroc(clean_by_img, corr_by_img)

                mAP_drop = clean_map['mAP@[0.5:0.95]'] - corr_map.get('mAP@[0.5:0.95]', 0)
                cov_drop = clean_cf['coverage'] - corr_cf['coverage']

                print(f'    mAP={corr_map.get("mAP@[0.5:0.95]",0):.4f} (drop={mAP_drop:+.4f})')
                print(f'    Coverage={corr_cf["coverage"]:.4f} (drop={cov_drop:+.4f})')
                print(f'    AUROC={auroc_result["auroc"]:.3f}')

                model_results['corruptions'][key] = {
                    'corruption': corr_name,
                    'severity': sev,
                    'mAP': corr_map,
                    'mAP_drop': float(mAP_drop),
                    'coverage': corr_cf['coverage'],
                    'coverage_drop': float(cov_drop),
                    'auroc': auroc_result['auroc'],
                }

        all_results[model_name] = model_results

        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Generate figures
    _generate_corruption_figures(all_results, out)

    # Save
    rp = os.path.join(out, 'natural_corruption_results.json')
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Cleanup
    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Print summary table
    print(f'\n{"="*90}')
    print(f'  NATURAL CORRUPTION SUMMARY')
    print(f'{"="*90}')
    for mn, mr in all_results.items():
        print(f'\n  {mn}:')
        print(f'  | {"Corruption":<25s} | {"mAP":>6s} | {"Drop":>7s} | {"Cov":>6s} | {"AUROC":>6s} |')
        print(f'  |{"-"*27}|{"-"*8}|{"-"*9}|{"-"*8}|{"-"*8}|')
        print(f'  | {"Clean":<25s} | {mr["clean_mAP"]["mAP@[0.5:0.95]"]:>6.3f} | {"":>7s} | {mr["clean_coverage"]:>6.3f} | {"":>6s} |')
        for key, cr in mr.get('corruptions', {}).items():
            print(f'  | {key:<25s} | {cr["mAP"].get("mAP@[0.5:0.95]",0):>6.3f} | {cr["mAP_drop"]:>+7.3f} | {cr["coverage"]:>6.3f} | {cr["auroc"]:>6.3f} |')

    print(f'\n  Results -> {rp}')


def _generate_corruption_figures(all_results, out):
    """Generate corruption impact figures."""
    for model_name, mr in all_results.items():
        corrs = mr.get('corruptions', {})
        if not corrs:
            continue

        # Group by corruption type
        corr_types = sorted(set(c['corruption'] for c in corrs.values()))
        colors = {'gaussian_noise': '#E74C3C', 'gaussian_blur': '#3498DB',
                  'jpeg_compression': '#F39C12', 'fog': '#95A5A6', 'brightness': '#2ECC71'}

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        for ct in corr_types:
            sevs, maps, covs, aurocs = [], [], [], []
            for key, cr in sorted(corrs.items()):
                if cr['corruption'] == ct:
                    sevs.append(cr['severity'])
                    maps.append(cr['mAP'].get('mAP@[0.5:0.95]', 0))
                    covs.append(cr['coverage'])
                    aurocs.append(cr['auroc'])
            c = colors.get(ct, '#999')
            axes[0].plot(sevs, maps, 'o-', color=c, label=ct, lw=2, markersize=8)
            axes[1].plot(sevs, covs, 'o-', color=c, label=ct, lw=2, markersize=8)
            axes[2].plot(sevs, aurocs, 'o-', color=c, label=ct, lw=2, markersize=8)

        clean_mAP = mr['clean_mAP']['mAP@[0.5:0.95]']
        axes[0].axhline(clean_mAP, color='black', ls='--', alpha=0.5, label='Clean')
        axes[0].set_xlabel('Severity'); axes[0].set_ylabel('mAP')
        axes[0].set_title('(a) mAP under corruption'); axes[0].legend(fontsize=7)
        axes[0].set_xticks([1,2,3])

        axes[1].axhline(mr['clean_coverage'], color='black', ls='--', alpha=0.5)
        axes[1].set_xlabel('Severity'); axes[1].set_ylabel('Coverage')
        axes[1].set_title('(b) Conformal coverage under corruption')
        axes[1].set_xticks([1,2,3]); axes[1].set_ylim(0, 1.05)

        axes[2].axhline(0.5, color='gray', ls=':', alpha=0.5, label='Random')
        axes[2].set_xlabel('Severity'); axes[2].set_ylabel('AUROC')
        axes[2].set_title('(c) Corruption detection AUROC')
        axes[2].legend(fontsize=7); axes[2].set_xticks([1,2,3]); axes[2].set_ylim(0.3, 1.05)

        fig.suptitle(f'Natural Corruptions — {model_name}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f'fig_corruptions_{model_name}.png'))
        plt.close()
        print(f'  [SAVED] fig_corruptions_{model_name}.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-root', default='data/coco')
    parser.add_argument('--output-dir', default='results/corruptions')
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--corruptions', nargs='+', default=None)
    parser.add_argument('--max-images', type=int, default=200)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run_corruption_pipeline(args)


if __name__ == '__main__':
    main()
