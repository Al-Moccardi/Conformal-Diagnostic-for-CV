import sys, json
import numpy as np
from collections import defaultdict
sys.path.insert(0, '.')
from scripts.rebuttal.revision_experiments import compute_per_image_coverage, get_adv_preds

gt = defaultdict(list)
for a in json.load(open('data/coco/annotations/instances_val2017.json'))['annotations']:
    gt[a['image_id']].append(a)
adv = json.load(open('results/adversarial/full_results.json'))

PUB = {('yolov8x','fgsm'):0.601, ('yolov8x','pgd-20'):0.262, ('yolov8x','dag'):0.193,
       ('yolov8x','tog-v'):0.170, ('yolov11x','fgsm'):0.635, ('yolov11x','pgd-20'):0.366,
       ('yolov11x','dag'):0.469, ('yolov11x','tog-v'):0.354}
PUBCLEAN = {'yolov8x':0.923, 'yolov11x':0.921, 'detr-resnet101':0.884}

for m in ['yolov8x','yolov11x','detr-resnet101']:
    by = defaultdict(list)
    for p in json.load(open(f'results/evaluation/predictions/{m}_bbox.json')):
        by[p['image_id']].append(p)
    full = np.mean([c[2] for c in compute_per_image_coverage(by, gt).values()])
    print(f'\n{m}: clean macro over ALL {len(gt)} imgs = {full:.3f}  (published {PUBCLEAN[m]})')
    print(f"  {'attack':<9}{'n_img':>6}{'clean@same':>12}{'adv':>8}{'pub adv':>9}{'Δ pub':>8}{'Δ vs matched clean':>20}")
    for a in adv[m].get('attacks', {}):
        ab = get_adv_preds(adv, m, a)
        if not ab: continue
        sub = {k: v for k, v in gt.items() if k in ab}
        cm = np.mean([c[2] for c in compute_per_image_coverage(by, sub).values()])
        am = np.mean([c[2] for c in compute_per_image_coverage(ab, sub).values()])
        pub = PUB.get((m, a))
        d_pub = (0.90 - pub) if pub else float('nan')
        d_matched = cm - am
        print(f'  {a:<9}{len(sub):>6}{cm:>12.3f}{am:>8.3f}'
              f'{(f"{pub:.3f}" if pub else "   ---"):>9}'
              f'{(f"{d_pub:+.2f}" if pub else "  ---"):>8}{d_matched:>+20.2f}')
