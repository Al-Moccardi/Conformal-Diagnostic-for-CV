"""Data for the graphical abstract: real YOLOv8x / PGD-20 diagnostics from the stored predictions.
(1) runtime-monitor stream exactly as revision_experiments builds it (40 clean / 40 adv / 40 clean, W=20, pre-threshold);
(2) conformal p-values (clean test split and adversarial) with the Table 12 convention;
(3) per-class coverage: from Table 10 of the manuscript (check_tables.py); multi-IoU: Table 11 values."""
import json, random, sys
import numpy as np
from collections import defaultdict
from scipy.stats import kstest
sys.path.insert(0, '.')
from scripts.rebuttal.revision_experiments import compute_per_image_coverage, get_adv_preds
MIN_CONF = 0.1
gt = defaultdict(list)
for a in json.load(open('data/coco/annotations/instances_val2017.json'))['annotations']: gt[a['image_id']].append(a)
adv = json.load(open('results/adversarial/full_results.json'))
clean = defaultdict(list)
for p in json.load(open('results/evaluation/predictions/yolov8x_bbox.json')): clean[p['image_id']].append(p)
ab = get_adv_preds(adv, 'yolov8x', 'pgd-20')
# (1) stream
common = sorted(set(clean) & set(ab) & set(gt)); n_phase = min(len(common) // 3, 40)
np.random.seed(42); sh = np.random.permutation(common).tolist()
ph = [sh[:n_phase], sh[n_phase:2*n_phase], sh[2*n_phase:3*n_phase]]
seq, lab = [], []
for k, ids in enumerate(ph):
    src = ab if k == 1 else clean
    for i in ids:
        c = compute_per_image_coverage({i: src[i]}, {i: gt[i]})
        if i in c: seq.append(c[i][2]); lab.append(1 if k == 1 else 0)
W = 20; win = [float(np.mean(seq[i-W:i])) for i in range(W, len(seq)+1)]; pos = list(range(W, len(seq)+1))
gaps = [0.9 - w for w in win]; alerts = [i for i, g in enumerate(gaps) if g > 0.15]
first_alert_pos = pos[alerts[0]] if alerts else None
adv_start = len(ph[0]); adv_end = len(ph[0]) + len([1 for l in lab if l == 1])
# (2) p-values
ids = sorted(clean.keys()); random.seed(42); random.shuffle(ids)
cal_ids = set(ids[:len(ids)//2]); test_ids = set(ids[len(ids)//2:])
cal = np.sort([1.0 - p['score'] for i in cal_ids for p in clean[i] if p['score'] >= MIN_CONF]); n = len(cal)
def pv(by, subset=None):
    out = []
    for i, ps in by.items():
        if subset is not None and i not in subset: continue
        for p in ps:
            if p['score'] >= MIN_CONF: out.append(1.0 - np.searchsorted(cal, 1.0 - p['score'], side='right') / n)
    return np.array(out)
cp = pv(clean, test_ids); ap = pv(ab)
hist_c, edges = np.histogram(cp, bins=10, range=(0, 1)); hist_a, _ = np.histogram(ap, bins=10, range=(0, 1))
out = dict(stream=dict(seq=seq, labels=lab, window=W, windowed=win, positions=pos, adv_start=adv_start, adv_end=adv_end,
                       first_alert_pos=first_alert_pos, alert_delay=(first_alert_pos - adv_start) if first_alert_pos else None,
                       clean_mean=float(np.mean([w for w, p in zip(win, pos) if p <= adv_start])),
                       adv_mean=float(np.mean([w for w, p in zip(win, pos) if adv_start + W <= p <= adv_end]))),
           pvalues=dict(clean_hist=hist_c.tolist(), adv_hist=hist_a.tolist(), edges=edges.tolist(),
                        D_clean=float(kstest(cp, 'uniform')[0]), D_adv=float(kstest(ap, 'uniform')[0]),
                        n_clean=int(len(cp)), n_adv=int(len(ap))),
           class_cov=dict(names=["person", "car", "bottle", "traffic light", "bird", "handbag"],
                          clean=[0.93, 0.89, 0.85, 0.82, 0.78, 0.75], pgd20=[0.19, 0.15, 0.07, 0.04, 0.00, 0.00]),
           multi_iou=dict(th=[0.25, 0.5, 0.75, 0.9], clean=[0.922, 0.881, 0.735, 0.411], tog_v=[0.138, 0.104, 0.078, 0.034],
                          v11x_pgd20=[0.149, 0.138, 0.113, 0.066]),
           gap=dict(clean=0.04, pgd20=0.69), abstention=dict(clean_rej=0.05, adv_rej=0.81, tau_A=0.31))
json.dump(out, open('results/revision/ga_diagnostics.json', 'w'))
print('stream len', len(seq), 'phases', [len(p) for p in ph], 'first alert at', first_alert_pos, 'delay', out['stream']['alert_delay'],
      'clean/adv window means', round(out['stream']['clean_mean'], 3), round(out['stream']['adv_mean'], 3))
print('pvals: D_clean', round(out['pvalues']['D_clean'], 3), 'D_adv', round(out['pvalues']['D_adv'], 3), 'n', len(cp), len(ap))
