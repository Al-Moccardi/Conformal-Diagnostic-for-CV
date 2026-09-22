"""Table 4 re-evaluated with the conformal filter ACTIVE (Eq. 11 construction).
Cov_pre: tau=0 (what the paper currently reports). Cov_post: tau_B at alpha=0.10.
Macro = mean of per-image coverage (paper's convention); micro = objects covered / objects."""
import sys, json, random
import numpy as np
from collections import defaultdict
sys.path.insert(0, '.')
from src.calibration.adaptive_conformal import AdaptiveConformalCalibrator
from scripts.rebuttal.revision_experiments import compute_per_image_coverage, get_adv_preds

ALPHA = 0.10
gt = defaultdict(list)
for a in json.load(open('data/coco/annotations/instances_val2017.json'))['annotations']:
    gt[a['image_id']].append(a)
adv = json.load(open('results/adversarial/full_results.json'))

def tau_eq11(preds, gt_by, alpha):
    cal = AdaptiveConformalCalibrator(alpha=alpha, n_conf_bins=5, size_normalize=True)
    m = cal._match_all(preds, gt_by, 0.5)
    tp = np.array([x['conf'] for x in m if x['correct']])
    s = 1.0 - tp; n = len(s)
    q = min(np.ceil((n+1)*(1-alpha))/n, 1.0)
    return max(0.0, 1.0 - float(np.quantile(s, q))), n

def cov(preds_by, gt_by, thr):
    r = compute_per_image_coverage(preds_by, gt_by, conf_threshold=thr)
    macro = float(np.mean([v[2] for v in r.values()])) if r else 0.0
    c = sum(v[0] for v in r.values()); t = sum(v[1] for v in r.values())
    return macro, (c/t if t else 0.0)

def frac_filtered(preds_by, thr):
    n = sum(len(v) for v in preds_by.values()); k = sum(1 for v in preds_by.values() for p in v if p['score'] < thr)
    return k/n if n else 0.0

out = {}
for model in ['yolov8x', 'yolov11x', 'detr-resnet101']:
    preds = json.load(open(f'results/evaluation/predictions/{model}_bbox.json'))
    by = defaultdict(list)
    for p in preds: by[p['image_id']].append(p)
    ids = sorted(by); random.seed(42); random.shuffle(ids)
    cal_ids = set(ids[:len(ids)//2]); test_ids = [i for i in ids if i not in cal_ids]
    cal_preds = [p for p in preds if p['image_id'] in cal_ids]
    tau, n_tp = tau_eq11(cal_preds, gt, ALPHA)

    test_by = {i: by[i] for i in test_ids if i in gt}
    test_gt = {i: gt[i] for i in test_by}
    pre_ma, pre_mi = cov(test_by, test_gt, 0.0)
    post_ma, post_mi = cov(test_by, test_gt, tau)
    print(f'\n=== {model}  tau(alpha=0.10, Eq.11) = {tau:.4f}   n_TP_cal = {n_tp} ===')
    print(f'  {"condition":<10}{"n_img":>6} {"filtered%":>10} | {"Cov_pre":>8} {"Cov_post":>9} {"Delta_pre":>10} {"Delta_post":>11} |  micro pre/post')
    print(f'  {"clean":<10}{len(test_by):>6} {100*frac_filtered(test_by,tau):>9.1f}% | {pre_ma:>8.3f} {post_ma:>9.3f} {0.9-pre_ma:>+10.3f} {0.9-post_ma:>+11.3f} |  {pre_mi:.3f} / {post_mi:.3f}')
    rows = {'tau': tau, 'n_tp_cal': n_tp,
            'clean': dict(n=len(test_by), pre_macro=pre_ma, post_macro=post_ma, pre_micro=pre_mi, post_micro=post_mi,
                          filtered=frac_filtered(test_by,tau))}
    for atk in adv[model].get('attacks', {}):
        ab = get_adv_preds(adv, model, atk)
        if not ab: continue
        sub = {k: v for k, v in gt.items() if k in ab}
        a_pre_ma, a_pre_mi = cov(ab, sub, 0.0); a_post_ma, a_post_mi = cov(ab, sub, tau)
        print(f'  {atk:<10}{len(sub):>6} {100*frac_filtered(ab,tau):>9.1f}% | {a_pre_ma:>8.3f} {a_post_ma:>9.3f} {0.9-a_pre_ma:>+10.3f} {0.9-a_post_ma:>+11.3f} |  {a_pre_mi:.3f} / {a_post_mi:.3f}')
        rows[atk] = dict(n=len(sub), pre_macro=a_pre_ma, post_macro=a_post_ma, pre_micro=a_pre_mi, post_micro=a_post_mi,
                         filtered=frac_filtered(ab,tau))
    out[model] = rows
json.dump(out, open('results/revision/table4_active_filter.json','w'), indent=2)
print('\nwritten -> results/revision/table4_active_filter.json')
