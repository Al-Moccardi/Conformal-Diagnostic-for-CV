#!/usr/bin/env python3
"""
run_defense_baselines.py — GAP 2: Non-CP defense baselines.

Compares CP defense against input-preprocessing defenses:
  1. JPEG Compression (Dziugaite et al., 2016)
  2. Feature Squeezing (Xu et al., NDSS 2018)
  3. Spatial Smoothing (median filter)

Key point: these try to REMOVE perturbation (mAP recovery),
while CP tries to DETECT the attack (AUROC). Different goals.

Usage:
    python run_defense_baselines.py
"""

import os, sys, json, argparse, warnings, gc
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

from configs.model_config import MODEL_ZOO, ADVERSARIAL_MODELS

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.grid': True, 'grid.alpha': 0.3,
    'font.size': 10, 'axes.titleweight': 'bold',
    'savefig.dpi': 250, 'savefig.bbox': 'tight',
})


def jpeg_defense(img, quality=75):
    buf = BytesIO()
    Image.fromarray(img).save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))


def feature_squeeze(img, bit_depth=4):
    factor = 2 ** (8 - bit_depth)
    return np.clip((img // factor) * factor + factor // 2, 0, 255).astype(np.uint8)


def spatial_smooth(img, kernel_size=3):
    return np.array(Image.fromarray(img).filter(ImageFilter.MedianFilter(size=kernel_size)))


DEFENSES = {
    'none':      {'fn': lambda x: x,                     'label': 'No Defense'},
    'jpeg_75':   {'fn': lambda x: jpeg_defense(x, 75),   'label': 'JPEG q=75'},
    'jpeg_50':   {'fn': lambda x: jpeg_defense(x, 50),   'label': 'JPEG q=50'},
    'squeeze_4': {'fn': lambda x: feature_squeeze(x, 4), 'label': 'Squeeze 4-bit'},
    'smooth_3':  {'fn': lambda x: spatial_smooth(x, 3),  'label': 'Median 3x3'},
}


def run_defense_baselines(args):
    import torch
    from src.models.detector_zoo import build_detector
    from src.evaluation.metrics import format_predictions_coco

    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    tmp_dir = os.path.join(out, '_tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    print('=' * 70)
    print('  NON-CP DEFENSE BASELINES')
    print('=' * 70)

    ann_file = os.path.join(args.coco_root, 'annotations', 'instances_val2017.json')
    with open(ann_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    cat_ids = sorted({c['id'] for c in coco_data['categories']})

    attack_dir = os.path.join(args.adv_dir, 'attack_samples')
    if not os.path.isdir(attack_dir):
        print(f'  [ERROR] {attack_dir} not found — run adversarial eval first')
        return

    models = args.models or ADVERSARIAL_MODELS
    attacks = args.attacks or ['fgsm', 'pgd-20', 'dag']
    defense_names = list(DEFENSES.keys())
    all_results = {}

    for model_name in models:
        if model_name not in MODEL_ZOO:
            continue
        print(f"\n{'='*60}")
        print(f'  MODEL: {model_name}')
        print(f"{'='*60}")

        try:
            detector = build_detector(model_name, MODEL_ZOO[model_name], device)
            detector.load()
        except Exception as e:
            print(f'  [ERROR] {e}')
            continue

        model_results = {}
        for atk_name in attacks:
            adv_images = sorted([
                os.path.join(attack_dir, f)
                for f in os.listdir(attack_dir)
                if f.startswith(f'{model_name}_{atk_name}_adv_') and f.endswith('.png')
            ])[:args.max_images]

            if len(adv_images) < 5:
                print(f'  [SKIP] {atk_name}: {len(adv_images)} images')
                continue

            print(f'\n  Attack: {atk_name} ({len(adv_images)} images)')
            atk_results = {}

            for def_name, defense in DEFENSES.items():
                n_high, total_conf, n_preds = 0, 0.0, 0
                for img_path in tqdm(adv_images, desc=f'    {defense["label"]:15s}', ncols=80, leave=False):
                    try:
                        img = np.array(Image.open(img_path).convert('RGB'))
                        defended = defense['fn'](img)
                        tmp_path = os.path.join(tmp_dir, 'def.png')
                        Image.fromarray(defended).save(tmp_path)
                        r = detector.predict(tmp_path, conf_threshold=0.01)
                        scores = r['scores'] if 'scores' in r else np.array([])
                        n_preds += len(scores)
                        n_high += int((scores >= 0.5).sum())
                        total_conf += float(scores.sum())
                    except Exception:
                        continue

                mean_conf = total_conf / max(n_preds, 1)
                atk_results[def_name] = {
                    'label': defense['label'], 'n_preds': n_preds,
                    'n_high_conf': n_high, 'mean_conf': float(mean_conf),
                }
                print(f'    {defense["label"]:15s}: {n_high:4d} high-conf dets, mean_conf={mean_conf:.3f}')

            model_results[atk_name] = atk_results
        all_results[model_name] = model_results

        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Figure
    for mn, mr in all_results.items():
        atks = list(mr.keys())
        if not atks:
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(atks))
        defs = list(mr[atks[0]].keys())
        w = 0.8 / len(defs)
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
        for i, dn in enumerate(defs):
            vals = [mr[a].get(dn, {}).get('n_high_conf', 0) for a in atks]
            ax.bar(x + i*w - 0.4 + w/2, vals, w, label=mr[atks[0]][dn]['label'],
                   color=colors[i % len(colors)], alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(atks)
        ax.set_ylabel('High-confidence detections')
        ax.set_title(f'Defense Baselines — {mn}')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f'fig_defense_baselines_{mn}.png'))
        plt.close()
        print(f'  [SAVED] fig_defense_baselines_{mn}.png')

    # Table
    print(f"\n{'='*85}")
    print(f"  | {'Model':<13s} | {'Attack':<8s} | {'Defense':<15s} | {'#HighConf':>9s} | {'MeanConf':>8s} |")
    print(f"  |{'-'*15}|{'-'*10}|{'-'*17}|{'-'*11}|{'-'*10}|")
    for mn, mr in all_results.items():
        for an, ar in mr.items():
            for dn, dr in ar.items():
                print(f"  | {mn:<13s} | {an:<8s} | {dr['label']:<15s} | "
                      f"{dr['n_high_conf']:>9d} | {dr['mean_conf']:>8.3f} |")
    print(f"  {'='*85}")

    rp = os.path.join(out, 'defense_baselines.json')
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f'\n  Results -> {rp}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-root', default='data/coco')
    parser.add_argument('--adv-dir', default='results/adversarial')
    parser.add_argument('--output-dir', default='results/defense_baselines')
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--attacks', nargs='+', default=None)
    parser.add_argument('--max-images', type=int, default=200)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run_defense_baselines(args)


if __name__ == '__main__':
    main()
