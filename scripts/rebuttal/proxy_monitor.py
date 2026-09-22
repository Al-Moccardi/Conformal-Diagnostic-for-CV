"""Table 7b — runtime monitor under the label-free proxy, Eq. (15) + Eq. (16b).
Replicates the protocol of fix_runtime_monitor() exactly (seed 42, same phases),
replacing per-image conformal coverage with the normalised detection count."""
import sys, json, random
import numpy as np
from collections import defaultdict
sys.path.insert(0, '.')

ALPHA, W_DEFAULT, MIN_CONF = 0.10, 20, 0.1
SCEN = [('yolov8x', 'pgd-20'), ('yolov8x', 'tog-v'), ('yolov8x', 'fgsm'),
        ('detr-resnet101', 'pgd-20')]

gt = defaultdict(list)
for a in json.load(open('data/coco/annotations/instances_val2017.json'))['annotations']:
    gt[a['image_id']].append(a)
adv_results = json.load(open('results/adversarial/full_results.json'))

def load_clean(m):
    preds = json.load(open(f'results/evaluation/predictions/{m}_bbox.json'))
    by = defaultdict(list)
    for p in preds:
        by[p['image_id']].append(p)
    return preds, by

def n_det(preds):
    return sum(1 for p in preds if p['score'] >= MIN_CONF)

def windows(seq, W):
    return [(i, float(np.mean(seq[i - W:i]))) for i in range(W, len(seq) + 1)]

out = {}
for model, atk in SCEN:
    clean_preds, clean_by = load_clean(model)
    ad = adv_results.get(model, {}).get('attacks', {}).get(atk)
    if not ad or not ad.get('per_image_predictions'):
        print(f'skip {model}/{atk}'); continue
    adv_by = defaultdict(list)
    for p in ad['per_image_predictions']:
        adv_by[p['image_id']].append(p)

    # N_cal: mean detection count on the clean CALIBRATION half (seed 42, as in run_defense_eval)
    ids = sorted(clean_by.keys()); random.seed(42); random.shuffle(ids)
    cal_ids = ids[:len(ids)//2]
    N_cal = float(np.mean([n_det(clean_by[i]) for i in cal_ids]))

    common = sorted(set(clean_by) & set(adv_by) & set(gt))
    n_phase = min(len(common)//3, 40)
    np.random.seed(42)
    sh = np.random.permutation(common).tolist()
    p1, p2, p3 = sh[:n_phase], sh[n_phase:2*n_phase], sh[2*n_phase:3*n_phase]

    seq = ([n_det(clean_by[i])/N_cal for i in p1] +
           [n_det(adv_by[i])/N_cal   for i in p2] +
           [n_det(clean_by[i])/N_cal for i in p3])
    adv_end = len(p1) + len(p2)

    # pure-clean control stream of the same length, for gamma_p and false alarms
    used = set(p1 + p2 + p3)
    ctrl_ids = [i for i in sorted(clean_by) if i not in used]
    np.random.seed(7)
    ctrl_ids = np.random.permutation(ctrl_ids).tolist()[:max(len(seq), 200)]
    ctrl = [n_det(clean_by[i])/N_cal for i in ctrl_ids]

    rows = {}
    for W in (10, 15, 20, 30):
        wc_clean = windows(ctrl, W)
        # one-sided: the alert fires only on a DOWNWARD excursion, so calibrate
        # on downward clean deviations (upward count spikes are irrelevant)
        gamma_p = 1.5 * max(max(0.0, 1.0 - v) for _, v in wc_clean)   # Eq. (16b)
        thr = 1.0 - gamma_p
        wc = windows(seq, W)
        alert = next((pos for pos, v in wc if v < thr), None)
        delay = (alert - len(p1)) if alert is not None else None
        rec = next((pos for pos, v in wc if pos > adv_end and v >= thr), None)
        fa = sum(1 for _, v in wc_clean if v < thr)
        # Variant B: quantile-calibrated threshold (order statistic of the clean
        # windowed proxy), the direct analogue of the Proposition 1 construction.
        cvals = sorted(v for _, v in wc_clean)
        thrQ = cvals[max(0, int(np.floor(ALPHA * len(cvals))) - 1)]
        alertQ = next((pos for pos, v in wc if v < thrQ), None)
        delayQ = (alertQ - len(p1)) if alertQ is not None else None
        faQ = sum(1 for _, v in wc_clean if v < thrQ)

        rows[W] = dict(
            gamma_p=round(gamma_p, 3), thr=round(thr, 3),
            clean=round(float(np.mean(seq[:len(p1)])), 3),
            adv=round(float(np.mean(seq[len(p1):adv_end])), 3),
            rec=round(float(np.mean(seq[adv_end:])), 3),
            delay=delay, recovery_delay=(rec - adv_end) if rec is not None else None,
            false_alarms=fa,
            thr_q=round(thrQ, 3), delay_q=delayQ, false_alarms_q=faQ,
            n_clean_windows=len(cvals))
    out[f'{model}/{atk}'] = dict(N_cal=round(N_cal, 2), rows=rows)

    r = rows[W_DEFAULT]
    print(f"{model}/{atk:8s} N_cal={N_cal:5.2f} clean={r['clean']:.3f} adv={r['adv']:.3f} rec={r['rec']:.3f}")
    print(f"    A) 1.5x rule : thr={r['thr']:.3f}  delay={r['delay']}  recov={r['recovery_delay']}  FA={r['false_alarms']}/{r['n_clean_windows']}")
    print(f"    B) quantile  : thr={r['thr_q']:.3f}  delay={r['delay_q']}  FA={r['false_alarms_q']}/{r['n_clean_windows']}")

print('\nW sensitivity (delay / false alarms):')
for k, v in out.items():
    print(' ', k, {W: (v['rows'][W]['delay'], v['rows'][W]['false_alarms']) for W in (10,15,20,30)})

json.dump(out, open('results/revision/table7b_proxy_monitor.json', 'w'), indent=2)
print('\nwritten -> results/revision/table7b_proxy_monitor.json')
