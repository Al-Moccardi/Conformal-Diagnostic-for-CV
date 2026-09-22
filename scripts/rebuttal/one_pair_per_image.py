"""Section 3.3 check: pair-level conformal coverage Cov^CP on the clean test split
(a) pooled calibration (all matched TPs, the paper's construction) and
(b) one random true positive per calibration image, repeated 100x (finite-sample valid, Dunn et al.)."""
import sys, json, random
import numpy as np
from collections import defaultdict
sys.path.insert(0, '.')
from src.calibration.adaptive_conformal import AdaptiveConformalCalibrator
ALPHA = 0.10
gt = defaultdict(list)
for a in json.load(open('data/coco/annotations/instances_val2017.json'))['annotations']:
    gt[a['image_id']].append(a)
out = {}
for model in ['yolov8x', 'yolov11x', 'rtdetr-l', 'detr-resnet101']:
    try: preds = json.load(open(f'results/evaluation/predictions/{model}_bbox.json'))
    except FileNotFoundError: continue
    by = defaultdict(list)
    for p in preds: by[p['image_id']].append(p)
    ids = sorted(by); random.seed(42); random.shuffle(ids)
    cal_ids = set(ids[:len(ids)//2]); test_ids = [i for i in ids if i not in cal_ids]
    cal = AdaptiveConformalCalibrator(alpha=ALPHA)
    m_cal = [m for m in cal._match_all([p for p in preds if p['image_id'] in cal_ids], gt, 0.5) if m['correct']]
    m_test = [m for m in cal._match_all([p for p in preds if p['image_id'] in set(test_ids)], gt, 0.5) if m['correct']]
    def tau_from(scores):
        s = np.array(scores); n = len(s); q = min(np.ceil((n+1)*(1-ALPHA))/n, 1.0)
        return 1.0 - float(np.quantile(s, q))
    tau_pool = tau_from([1-m['conf'] for m in m_cal])
    test_conf = np.array([m['conf'] for m in m_test])
    cov_pool = float(np.mean(test_conf >= tau_pool))
    # one pair per image
    cal_by = defaultdict(list); test_by = defaultdict(list)
    for m in m_cal: cal_by[m['image_id']].append(m['conf'])
    for m in m_test: test_by[m['image_id']].append(m['conf'])
    covs_pooledtest, covs_subtest, taus = [], [], []
    for r in range(100):
        rng = random.Random(r)
        sc = [1 - rng.choice(v) for v in cal_by.values()]
        t = tau_from(sc); taus.append(t)
        covs_pooledtest.append(float(np.mean(test_conf >= t)))
        st = np.array([rng.choice(v) for v in test_by.values()])
        covs_subtest.append(float(np.mean(st >= t)))
    out[model] = dict(n_cal_pairs=len(m_cal), n_cal_images=len(cal_by), n_test_pairs=len(m_test), n_test_images=len(test_by),
                      tau_pooled=tau_pool, cov_cp_pooled=cov_pool,
                      tau_onepair_mean=float(np.mean(taus)), tau_onepair_sd=float(np.std(taus)),
                      cov_cp_onepair_cal_pooled_test=float(np.mean(covs_pooledtest)),
                      cov_cp_onepair_cal_onepair_test=float(np.mean(covs_subtest)), cov_cp_onepair_sd=float(np.std(covs_subtest)))
    print(model, {k:(round(v,4) if isinstance(v,float) else v) for k,v in out[model].items()})
json.dump(out, open('results/revision/one_pair_per_image.json','w'), indent=2)
