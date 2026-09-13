#!/usr/bin/env python3
"""
run_cross_dataset_voc.py — GAP 5: Real cross-dataset transfer.

Calibrate conformal predictor on COCO val2017, test on VOC2012 with REAL inference.
Measures coverage degradation under genuine distribution shift.

Usage:
    python run_cross_dataset_voc.py
    python run_cross_dataset_voc.py --models yolov8x rtdetr-l
"""

import os, sys, json, argparse, warnings, gc, random
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET

import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from configs.model_config import MODEL_ZOO
from src.calibration.adaptive_conformal import AdaptiveConformalCalibrator

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.grid': True, 'grid.alpha': 0.3,
    'font.size': 10, 'axes.titleweight': 'bold',
    'savefig.dpi': 250, 'savefig.bbox': 'tight',
})

# VOC class name -> COCO category ID mapping (20 shared classes)
VOC_TO_COCO = {
    'person': 1, 'bicycle': 2, 'car': 3, 'motorbike': 4,
    'aeroplane': 5, 'bus': 6, 'train': 7, 'boat': 9,
    'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19,
    'sheep': 20, 'cow': 21, 'bottle': 44, 'chair': 62,
    'sofa': 63, 'pottedplant': 64, 'diningtable': 67, 'tvmonitor': 72,
}


def download_voc(data_root):
    """Download VOC2012 if not present."""
    voc_root = os.path.join(data_root, 'voc')
    devkit = os.path.join(voc_root, 'VOCdevkit', 'VOC2012')
    if os.path.isdir(os.path.join(devkit, 'JPEGImages')):
        print(f'  VOC2012 already at {devkit}')
        return devkit

    os.makedirs(voc_root, exist_ok=True)
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    tar_path = os.path.join(voc_root, 'voc2012.tar')

    if not os.path.exists(tar_path):
        print(f'  Downloading VOC2012 (~2GB)...')
        import urllib.request
        urllib.request.urlretrieve(url, tar_path)

    print(f'  Extracting...')
    import tarfile
    with tarfile.open(tar_path) as tf:
        tf.extractall(voc_root)

    if os.path.exists(tar_path):
        os.remove(tar_path)

    print(f'  VOC2012 ready at {devkit}')
    return devkit


def load_voc_annotations(voc_root, max_images=500):
    """Parse VOC XML annotations into COCO-like format."""
    ann_dir = os.path.join(voc_root, 'Annotations')
    img_dir = os.path.join(voc_root, 'JPEGImages')

    # Use val split
    val_file = os.path.join(voc_root, 'ImageSets', 'Main', 'val.txt')
    if os.path.exists(val_file):
        with open(val_file, 'r', encoding='utf-8') as f:
            val_ids = [line.strip() for line in f if line.strip()]
    else:
        val_ids = [f.replace('.xml', '') for f in sorted(os.listdir(ann_dir)) if f.endswith('.xml')]

    random.seed(42)
    random.shuffle(val_ids)
    val_ids = val_ids[:max_images]

    gt_by_image = {}
    image_items = []

    for img_name in val_ids:
        xml_path = os.path.join(ann_dir, f'{img_name}.xml')
        img_path = os.path.join(img_dir, f'{img_name}.jpg')
        if not os.path.exists(xml_path) or not os.path.exists(img_path):
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_id = hash(img_name) % (2**31)  # synthetic image_id

        anns = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in VOC_TO_COCO:
                continue
            cat_id = VOC_TO_COCO[name]
            bbox_el = obj.find('bndbox')
            x1 = float(bbox_el.find('xmin').text)
            y1 = float(bbox_el.find('ymin').text)
            x2 = float(bbox_el.find('xmax').text)
            y2 = float(bbox_el.find('ymax').text)
            anns.append({
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'category_id': cat_id,
            })

        if anns:
            gt_by_image[img_id] = anns
            image_items.append({'image_id': img_id, 'file_path': img_path})

    return image_items, gt_by_image


def _iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0


def compute_coverage(preds_list, gt_by_image, calibrator):
    """Compute conformal coverage on a dataset."""
    preds_by_img = defaultdict(list)
    for p in preds_list:
        preds_by_img[p['image_id']].append(p)

    total_gt, covered = 0, 0
    n_kept, n_filt = 0, 0
    margins = []

    for img_id, img_preds in preds_by_img.items():
        gt_anns = gt_by_image.get(img_id, [])
        total_gt += len(gt_anns)
        gt_cov = [False] * len(gt_anns)

        for pred in sorted(img_preds, key=lambda x: -x['score']):
            r = calibrator.predict(pred)
            if r.keep:
                n_kept += 1
                margins.append(r.box_delta_pixels)
                px, py, pw, ph = pred['bbox']
                pxy = [px, py, px+pw, py+ph]
                for j, gt in enumerate(gt_anns):
                    if gt_cov[j] or gt['category_id'] != pred['category_id']:
                        continue
                    gx, gy, gw, gh = gt['bbox']
                    if _iou(pxy, [gx, gy, gx+gw, gy+gh]) >= 0.5:
                        gt_cov[j] = True
                        break
            else:
                n_filt += 1
        covered += sum(gt_cov)

    return {
        'coverage': covered / max(total_gt, 1),
        'total_gt': total_gt, 'covered_gt': covered,
        'n_kept': n_kept, 'n_filtered': n_filt,
        'mean_margin_px': float(np.mean(margins)) if margins else 0,
    }


def run_cross_dataset(args):
    import torch
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    print('=' * 70)
    print('  CROSS-DATASET: Calibrate on COCO, Test on VOC2012')
    print('=' * 70)

    # Download/find VOC
    data_root = os.path.dirname(args.coco_root)
    voc_root = os.path.join(data_root, 'voc', 'VOCdevkit', 'VOC2012')
    if not os.path.isdir(os.path.join(voc_root, 'JPEGImages')):
        try:
            voc_root = download_voc(data_root)
        except Exception as e:
            print(f'  [ERROR] VOC download failed: {e}')
            print(f'  Download manually from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/')
            return

    # Load VOC annotations
    voc_items, voc_gt = load_voc_annotations(voc_root, max_images=args.max_voc_images)
    print(f'  VOC2012: {len(voc_items)} images, {sum(len(v) for v in voc_gt.values())} annotations')

    # Load COCO for calibration
    ann_file = os.path.join(args.coco_root, 'annotations', 'instances_val2017.json')
    with open(ann_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    coco_gt = defaultdict(list)
    for ann in coco_data['annotations']:
        coco_gt[ann['image_id']].append(ann)
    coco_cat_ids = sorted({c['id'] for c in coco_data['categories']})

    models_to_eval = args.models or ['yolov8x', 'yolov11x', 'rtdetr-l']
    all_results = {}

    for model_name in models_to_eval:
        if model_name not in MODEL_ZOO:
            continue
        print(f'\n{"="*60}')
        print(f'  MODEL: {model_name}')
        print(f'{"="*60}')

        from src.models.detector_zoo import build_detector
        from src.evaluation.metrics import format_predictions_coco

        detector = build_detector(model_name, MODEL_ZOO[model_name], device)
        detector.load()

        # --- Step 1: COCO calibration ---
        print(f'  Step 1: COCO calibration...')
        pred_path = os.path.join(args.pred_dir, f'{model_name}_bbox.json')
        if os.path.exists(pred_path):
            with open(pred_path, 'r', encoding='utf-8') as f:
                coco_preds = json.load(f)
            print(f'    Loaded {len(coco_preds)} existing predictions')
        else:
            print(f'    Running inference on COCO (first 1000 images)...')
            coco_img_dir = os.path.join(args.coco_root, 'val2017')
            coco_images = sorted(coco_data['images'], key=lambda x: x['id'])[:1000]
            coco_preds = []
            for img_info in tqdm(coco_images, desc='    COCO', ncols=80):
                path = os.path.join(coco_img_dir, img_info['file_name'])
                if not os.path.exists(path):
                    continue
                r = detector.predict(path, conf_threshold=0.01)
                coco_preds.extend(format_predictions_coco(img_info['id'], r, coco_cat_ids, 'bbox'))

        # Split and calibrate
        by_img = defaultdict(list)
        for p in coco_preds:
            by_img[p['image_id']].append(p)
        img_ids = sorted(by_img.keys())
        random.seed(42)
        random.shuffle(img_ids)
        cal_ids = set(img_ids[:len(img_ids)//2])
        test_ids = set(img_ids[len(img_ids)//2:])
        cal_preds = [p for p in coco_preds if p['image_id'] in cal_ids]
        test_preds = [p for p in coco_preds if p['image_id'] in test_ids]

        calibrator = AdaptiveConformalCalibrator(alpha=args.alpha, n_conf_bins=5, size_normalize=True)
        calibrator.calibrate(cal_preds, coco_gt)
        print(f'    Calibrator: threshold={calibrator.conf_threshold:.4f}')

        # COCO test coverage
        coco_cov = compute_coverage(test_preds, coco_gt, calibrator)
        print(f'    COCO coverage: {coco_cov["coverage"]:.4f} | margin: {coco_cov["mean_margin_px"]:.1f}px')

        # --- Step 2: VOC inference ---
        print(f'  Step 2: VOC2012 inference...')
        voc_preds = []
        for item in tqdm(voc_items, desc='    VOC', ncols=80):
            try:
                r = detector.predict(item['file_path'], conf_threshold=0.01)
                # Filter to VOC categories only
                for i in range(len(r['scores'])):
                    label = int(r['labels'][i])
                    # Ultralytics: 0-indexed -> COCO cat ID
                    coco_cat_ids_list = sorted(set(coco_cat_ids))
                    if label < len(coco_cat_ids_list):
                        cat_id = coco_cat_ids_list[label]
                    else:
                        continue
                    if cat_id not in VOC_TO_COCO.values():
                        continue
                    x1, y1, x2, y2 = r['boxes'][i]
                    voc_preds.append({
                        'image_id': item['image_id'],
                        'category_id': cat_id,
                        'score': float(r['scores'][i]),
                        'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    })
            except Exception:
                continue

        print(f'    VOC predictions: {len(voc_preds)}')

        # VOC coverage with COCO-calibrated CP
        voc_cov = compute_coverage(voc_preds, voc_gt, calibrator)
        print(f'    VOC coverage: {voc_cov["coverage"]:.4f} | margin: {voc_cov["mean_margin_px"]:.1f}px')

        gap = coco_cov['coverage'] - voc_cov['coverage']
        print(f'    Coverage GAP: {gap:+.4f} ({"SHIFT DETECTED" if gap > 0.05 else "ROBUST"})')

        all_results[model_name] = {
            'coco_coverage': coco_cov,
            'voc_coverage': voc_cov,
            'coverage_gap': float(gap),
            'n_coco_test': len(test_ids),
            'n_voc_test': len(voc_items),
        }

        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Figure
    _generate_cross_dataset_figure(all_results, out)

    # Save
    rp = os.path.join(out, 'cross_dataset_results.json')
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n  Results -> {rp}')


def _generate_cross_dataset_figure(all_results, out):
    models = list(all_results.keys())
    if not models:
        return
    n = len(models)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n)
    w = 0.35
    coco_covs = [all_results[m]['coco_coverage']['coverage'] for m in models]
    voc_covs = [all_results[m]['voc_coverage']['coverage'] for m in models]
    ax.bar(x - w/2, coco_covs, w, label='COCO (in-dist)', color='#27AE60', alpha=0.8)
    ax.bar(x + w/2, voc_covs, w, label='VOC2012 (OOD)', color='#E74C3C', alpha=0.8)
    ax.axhline(0.9, color='black', ls='--', alpha=0.5, label='Target 0.9')
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('Coverage'); ax.set_ylim(0, 1.1)
    ax.set_title('Cross-Dataset: COCO-calibrated CP tested on VOC2012')
    ax.legend(fontsize=9)
    for i in range(n):
        gap = coco_covs[i] - voc_covs[i]
        ax.text(i, max(coco_covs[i], voc_covs[i]) + 0.02, f'gap={gap:+.3f}',
                ha='center', fontsize=8, color='#C0392B')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'fig_cross_dataset_coco_voc.png'))
    plt.close()
    print(f'  [SAVED] fig_cross_dataset_coco_voc.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-root', default='data/coco')
    parser.add_argument('--pred-dir', default='results/evaluation/predictions')
    parser.add_argument('--output-dir', default='results/cross_dataset')
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--max-voc-images', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run_cross_dataset(args)


if __name__ == '__main__':
    main()
