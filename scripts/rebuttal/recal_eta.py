"""Proposition 2 evidence with the conformal filter ACTIVE (Eq. 11 calibration).

For each model--attack pair on the saved adversarial predictions:
  AR        : class-correct recall at IoU>=0.5 of the raw output (tau = 0), per-image mean
  Cov_clean : post-threshold coverage with tau calibrated on CLEAN data (Table 4)
  Cov_recal : post-threshold coverage with tau recalibrated on the ADVERSARIAL true positives
              (in-sample: the most favourable case for recalibration)
  Cov_recal_oos : same, split-half out-of-sample (50 cal / 50 test, 20 random splits)
  eta       : fraction of ground-truth objects matched only after expanding every raw box by
              the localisation quantile qhat (existential matching, correct class, IoU>=0.5)
All quantities are per-image means over the images of the pair (paper convention)."""
import sys, json, random
import numpy as np
from collections import defaultdict
sys.path.insert(0, '.')
from src.calibration.adaptive_conformal import AdaptiveConformalCalibrator
from scripts.rebuttal.revision_experiments import compute_per_image_coverage, get_adv_preds
from scripts.rebuttal.active_filter_eval import tau_eq11, cov  # reuses clean tau computation

ALPHA = 0.10
gt = defaultdict(list)
for a in json.load(open('data/coco/annotations/instances_val2017.json'))['annotations']:
    gt[a['image_id']].append(a)
adv = json.load(open('results/adversarial/full_results.json'))
t4 = json.load(open('results/revision/table4_active_filter.json'))

def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1]); x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2-x1)*max(0, y2-y1); a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if a1+a2-inter > 0 else 0.0

def xyxy(b, q=0.0):
    x, y, w, h = b
    return [x - q*w, y - q*h, x + w + q*w, y + h + q*h]

def existential_cov(preds_by, gt_by, q=0.0, thr=0.0):
    """per-image fraction of GT objects matched by SOME detection (correct class, IoU>=0.5)"""
    out = {}
    for img, anns in gt_by.items():
        if not anns: continue
        ps = [p for p in preds_by.get(img, []) if p['score'] >= thr]
        hit = 0
        for g in anns:
            gb = xyxy(g['bbox'])
            if any(p['category_id'] == g['category_id'] and iou(xyxy(p['bbox'], q), gb) >= 0.5 for p in ps):
                hit += 1
        out[img] = hit/len(anns)
    return out

def qhat_loc(preds, gt_by, alpha):
    cal = AdaptiveConformalCalibrator(alpha=alpha, n_conf_bins=5, size_normalize=True)
    cal.calibrate(preds, gt_by, 0.5)
    return float(cal.global_qhat) if cal.global_qhat is not None else 0.0

out = {}
for model in ['yolov8x', 'yolov11x', 'detr-resnet101']:
    tau_clean = t4[model]['tau']
    out[model] = {'tau_clean': tau_clean}
    for atk in adv[model].get('attacks', {}):
        ab = get_adv_preds(adv, model, atk)
        if not ab: continue
        sub = {k: v for k, v in gt.items() if k in ab}
        flat = [p for v in ab.values() for p in v]
        # in-sample recalibration on adversarial TPs
        tau_adv, n_tp = tau_eq11(flat, sub, ALPHA)
        q_adv = qhat_loc(flat, sub, ALPHA)
        AR, _ = cov(ab, sub, 0.0)
        C_clean, _ = cov(ab, sub, tau_clean)
        C_recal, _ = cov(ab, sub, tau_adv)
        # existential versions (Proposition 2 definitions)
        A_ex = existential_cov(ab, sub, 0.0, 0.0)
        B_ex = existential_cov(ab, sub, q_adv, 0.0)
        eta = float(np.mean([max(0.0, B_ex[i] - A_ex[i]) for i in A_ex]))
        B_minus_A = float(np.mean([B_ex[i] - A_ex[i] for i in A_ex]))
        # out-of-sample split-half recalibration
        ids = sorted(sub); oos = []
        for seed in range(20):
            random.seed(seed); random.shuffle(ids)
            c_ids, t_ids = set(ids[:len(ids)//2]), ids[len(ids)//2:]
            c_flat = [p for i in c_ids for p in ab.get(i, [])]
            c_gt = {i: sub[i] for i in c_ids}
            if not c_flat: continue
            t_o, _ = tau_eq11(c_flat, c_gt, ALPHA)
            t_by = {i: ab.get(i, []) for i in t_ids}; t_gt = {i: sub[i] for i in t_ids}
            oos.append(cov(t_by, t_gt, t_o)[0])
        row = dict(n=len(sub), n_tp_adv=n_tp, tau_adv=tau_adv, qhat_adv=q_adv,
                   AR=AR, AR_existential=float(np.mean(list(A_ex.values()))),
                   cov_cleanCP=C_clean, cov_recalCP=C_recal,
                   cov_recalCP_oos_mean=float(np.mean(oos)) if oos else None,
                   eta=eta, B_minus_A_mean=B_minus_A,
                   bound_ok=bool(C_recal <= AR + eta + 1e-12))
        out[model][atk] = row
        print(f"{model:15s} {atk:8s} n={len(sub):3d} tau_clean={tau_clean:.3f} tau_adv={tau_adv:.3f} qhat_adv={q_adv:.3f} | AR={AR:.3f} Cov(cleanCP)={C_clean:.3f} Cov(recalCP)={C_recal:.3f} oos={row['cov_recalCP_oos_mean']:.3f} | eta={eta:.4f} (B-A={B_minus_A:+.4f})  bound_ok={row['bound_ok']}")
    # clean row: recalibration = calibration, eta on clean test split
json.dump(out, open('results/revision/prop2_recalibration_eta.json', 'w'), indent=2)
print('written -> results/revision/prop2_recalibration_eta.json')
