"""Why tau = 0, and what tau(alpha) looks like under the two constructions:
   A. as coded   : cls_score = 1-s if class-correct else 1.0   (all IoU>=0.5 matches)
   B. as Eq.(11) : cls_score = 1-s over class-correct TPs only
"""
import sys, json, random
import numpy as np
from collections import defaultdict
sys.path.insert(0, '.')
from src.calibration.adaptive_conformal import AdaptiveConformalCalibrator

gt = defaultdict(list)
for a in json.load(open('data/coco/annotations/instances_val2017.json'))['annotations']:
    gt[a['image_id']].append(a)

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
out = {}
for model in ['yolov8x', 'yolov11x', 'rtdetr-l', 'detr-resnet101']:
    preds = json.load(open(f'results/evaluation/predictions/{model}_bbox.json'))
    by = defaultdict(list)
    for p in preds: by[p['image_id']].append(p)
    ids = sorted(by); random.seed(42); random.shuffle(ids)
    cal_ids = set(ids[:len(ids)//2])
    cal_preds = [p for p in preds if p['image_id'] in cal_ids]

    cal = AdaptiveConformalCalibrator(alpha=0.1, n_conf_bins=5, size_normalize=True)
    matched = cal._match_all(cal_preds, gt, 0.5)
    confs = np.array([m['conf'] for m in matched]); correct = np.array([m['correct'] for m in matched])
    n_all, n_tp = len(confs), int(correct.sum())
    tp_conf = confs[correct]

    print(f'\n=== {model} ===  IoU-matched={n_all}  class-correct TP={n_tp} ({100*n_tp/n_all:.1f}%)')
    print(f'  TP confidence: median={np.median(tp_conf):.3f}  '
          f'frac<0.1={np.mean(tp_conf<0.1):.3f}  frac<0.25={np.mean(tp_conf<0.25):.3f}  frac<0.5={np.mean(tp_conf<0.5):.3f}')
    print(f'  {"alpha":>6} | {"tau_A (coded)":>14} {"tau_B (Eq.11)":>14} | {"frac TP filtered @B":>20}')
    rows = {}
    for a in ALPHAS:
        # A: as coded
        sA = np.where(correct, 1.0 - confs, 1.0); n = len(sA)
        qA = min(np.ceil((n+1)*(1-a))/n, 1.0); tauA = max(0.0, 1.0 - np.quantile(sA, qA))
        # B: TP only (Eq. 11)
        sB = 1.0 - tp_conf; nB = len(sB)
        qB = min(np.ceil((nB+1)*(1-a))/nB, 1.0); tauB = max(0.0, 1.0 - np.quantile(sB, qB))
        fracB = float(np.mean(tp_conf < tauB))
        rows[a] = dict(tau_coded=round(float(tauA),4), tau_eq11=round(float(tauB),4), frac_tp_filtered_eq11=round(fracB,4))
        print(f'  {a:>6.2f} | {tauA:>14.4f} {tauB:>14.4f} | {fracB:>20.3f}')
    out[model] = dict(n_iou_matched=n_all, n_class_correct=n_tp, tp_conf_median=float(np.median(tp_conf)),
                      tp_frac_below_0p1=float(np.mean(tp_conf<0.1)), rows=rows)

json.dump(out, open('results/revision/alpha_sweep_tau.json','w'), indent=2)
print('\nwritten -> results/revision/alpha_sweep_tau.json')
