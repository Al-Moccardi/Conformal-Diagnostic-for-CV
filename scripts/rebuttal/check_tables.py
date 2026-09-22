"""Check Tables 8 and 9: the CLEAN baselines are computed over all 5000 val
images, while the adversarial rows use only the ~100 attacked images.
Recompute the clean baselines restricted to the same images and compare."""
import sys, json
from collections import defaultdict
sys.path.insert(0, '.')

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (a[2]-a[0])*(a[3]-a[1]); a2 = (b[2]-b[0])*(b[3]-b[1])
    d = a1 + a2 - inter
    return inter/d if d > 0 else 0.0

def xyxy(b): return [b[0], b[1], b[0]+b[2], b[1]+b[3]]

def cov_at_iou(preds_by, gt_by, thr):
    tot = cov = 0
    for img, anns in gt_by.items():
        used = [False]*len(anns)
        for p in sorted(preds_by.get(img, []), key=lambda x: -x['score']):
            for j, g in enumerate(anns):
                if used[j] or p.get('category_id') != g.get('category_id'):
                    continue
                if iou(xyxy(p['bbox']), xyxy(g['bbox'])) >= thr:
                    used[j] = True; cov += 1; break
        tot += len(anns)
    return cov/tot if tot else 0.0

def per_class(preds_by, gt_by):
    c, t = defaultdict(int), defaultdict(int)
    for img, anns in gt_by.items():
        used = [False]*len(anns)
        for p in sorted(preds_by.get(img, []), key=lambda x: -x['score']):
            for j, g in enumerate(anns):
                if used[j] or p.get('category_id') != g.get('category_id'):
                    continue
                if iou(xyxy(p['bbox']), xyxy(g['bbox'])) >= 0.5:
                    used[j] = True; c[g['category_id']] += 1; break
        for g in anns: t[g['category_id']] += 1
    return {k: (c.get(k,0)/t[k], t[k]) for k in t}

gt = defaultdict(list)
for a in json.load(open('data/coco/annotations/instances_val2017.json'))['annotations']:
    gt[a['image_id']].append(a)
adv_results = json.load(open('results/adversarial/full_results.json'))

clean_by = defaultdict(list)
for p in json.load(open('results/evaluation/predictions/yolov8x_bbox.json')):
    clean_by[p['image_id']].append(p)

THR = [0.25, 0.50, 0.75, 0.90]
print('=== Table 9: multi-IoU clean profile (YOLOv8x) ===')
full = [cov_at_iou(clean_by, gt, t) for t in THR]
print(f'  clean over ALL {len(gt)} val images   : ' +
      ' '.join(f'{v:.3f}' for v in full) + f'   r={full[-1]/full[0]:.2f}   <- as published (0.922 0.881 0.735 0.411, r=0.45)')

for atk in ['tog-v', 'pgd-20']:
    ad = adv_results['yolov8x']['attacks'].get(atk)
    if not ad or not ad.get('per_image_predictions'): continue
    ids = {p['image_id'] for p in ad['per_image_predictions']}
    sub = {k: v for k, v in gt.items() if k in ids}
    prof = [cov_at_iou(clean_by, sub, t) for t in THR]
    print(f'  clean over the {len(sub)} {atk} images : ' +
          ' '.join(f'{v:.3f}' for v in prof) + f'   r={prof[-1]/prof[0]:.2f}')

print('\n=== Table 8: class-conditional clean coverage (YOLOv8x) ===')
NAMES = {1:'person', 3:'car', 44:'bottle', 10:'traffic light', 16:'bird', 31:'handbag'}
PUB = {1:0.95, 3:0.91, 44:0.85, 10:0.82, 16:0.78, 31:0.75}
ad = adv_results['yolov8x']['attacks']['pgd-20']
ids = {p['image_id'] for p in ad['per_image_predictions']}
sub = {k: v for k, v in gt.items() if k in ids}
pc_full, pc_sub = per_class(clean_by, gt), per_class(clean_by, sub)
print(f"  {'class':<15}{'published':>10}{'all 5000':>12}{'n':>7}{'100 imgs':>11}{'n':>6}")
for cat, nm in NAMES.items():
    f = pc_full.get(cat, (0, 0)); s = pc_sub.get(cat, (0, 0))
    print(f'  {nm:<15}{PUB[cat]:>10.2f}{f[0]:>12.2f}{f[1]:>7d}{s[0]:>11.2f}{s[1]:>6d}')
