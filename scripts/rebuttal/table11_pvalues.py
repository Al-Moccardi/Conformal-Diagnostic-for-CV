"""Table 11 (conformal p-values) recomputed exactly as Eq. (21)-(22) describe:
one-sample KS test of the test p-values against Uniform[0,1].
Calibration scores V = 1 - s over detections with s >= 0.1 on the calibration
split (same convention as revision_experiments.fix_pvalues); p_i = |{j: V_j >= V_test}| / n.
"""
import json, random, sys, os
import numpy as np
from collections import defaultdict
from scipy.stats import kstest
from sklearn.metrics import roc_auc_score

MIN_CONF = 0.1
adv = json.load(open('results/adversarial/full_results.json'))

def load_preds(model):
    preds = json.load(open(f'results/evaluation/predictions/{model}_bbox.json'))
    by = defaultdict(list)
    for p in preds: by[p['image_id']].append(p)
    return by

def adv_preds(model, atk):
    per = adv.get(model, {}).get('attacks', {}).get(atk, {}).get('per_image_predictions')
    if not per: return None
    by = defaultdict(list)
    for p in per: by[p['image_id']].append(p)
    return by

out = {}
for model in ['yolov8x', 'yolov11x']:
    clean = load_preds(model)
    ids = sorted(clean.keys()); random.seed(42); random.shuffle(ids)
    cal_ids = set(ids[:len(ids)//2]); test_ids = set(ids[len(ids)//2:])
    cal = np.sort([1.0 - p['score'] for i in cal_ids for p in clean[i] if p['score'] >= MIN_CONF])
    n = len(cal)
    def pvals(by, subset=None):
        allp, per_img = [], {}
        for i, ps in by.items():
            if subset is not None and i not in subset: continue
            pv = []
            for p in ps:
                if p['score'] >= MIN_CONF:
                    nc = 1.0 - p['score']
                    pv.append(1.0 - np.searchsorted(cal, nc, side='right') / n)
            allp += pv
            if len(pv) >= 3: per_img[i] = kstest(pv, 'uniform')[0]
        return np.array(allp), per_img
    cp, cks = pvals(clean, test_ids)
    D_cln, p_cln = kstest(cp, 'uniform')
    row = {'n_cal': int(n), 'p_mean_clean': float(cp.mean()), 'D_clean_onesample': float(D_cln), 'ksp_clean': float(p_cln)}
    for atk in ['pgd-20']:
        ab = adv_preds(model, atk)
        if ab is None: continue
        ap, aks = pvals(ab)
        D, pv = kstest(ap, 'uniform')
        lab = [0]*len(cks) + [1]*len(aks); sc = list(cks.values()) + list(aks.values())
        au = roc_auc_score(lab, sc)
        row[atk] = {'n_adv_imgs': len(ab), 'n_adv_dets': int(len(ap)), 'p_mean_adv': float(ap.mean()),
                    'D_adv_onesample': float(D), 'ksp_adv': float(pv), 'auroc_image_ks': float(au), 'n_imgs_with_ge3_dets_clean': len(cks), 'n_imgs_with_ge3_dets_adv': len(aks)}
    out[model] = row
    print(model, json.dumps(row, indent=1))
json.dump(out, open('results/revision/table11_pvalues_onesample.json', 'w'), indent=2)
