"""Cohen's d, Welch p and image-level bootstrap 95% CIs for post-threshold coverage (filter active)."""
import sys, json, random
import numpy as np
from collections import defaultdict
from scipy import stats
sys.path.insert(0, '.')
from scripts.rebuttal.revision_experiments import compute_per_image_coverage, get_adv_preds
gt = defaultdict(list)
for a in json.load(open('data/coco/annotations/instances_val2017.json'))['annotations']:
    gt[a['image_id']].append(a)
adv = json.load(open('results/adversarial/full_results.json'))
t4 = json.load(open('results/revision/table4_active_filter.json'))
rng = np.random.default_rng(0)
def boot(x, B=10000):
    x = np.asarray(x); m = x.mean()
    bs = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(B)])
    return float(m), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
out = {}
for model in ['yolov8x', 'yolov11x', 'detr-resnet101']:
    tau = t4[model]['tau']
    preds = json.load(open(f'results/evaluation/predictions/{model}_bbox.json'))
    by = defaultdict(list)
    for p in preds: by[p['image_id']].append(p)
    ids = sorted(by); random.seed(42); random.shuffle(ids)
    cal_ids = set(ids[:len(ids)//2]); test_ids = [i for i in ids if i not in cal_ids]
    test_by = {i: by[i] for i in test_ids if i in gt}
    clean_cov = np.array([v[2] for v in compute_per_image_coverage(test_by, {i: gt[i] for i in test_by}, tau).values()])
    m, lo, hi = boot(clean_cov)
    out[model] = {'clean': dict(n=len(clean_cov), mean=m, ci=[lo, hi], gap=0.9-m, gap_ci=[0.9-hi, 0.9-lo])}
    print(f"{model} clean n={len(clean_cov)} cov={m:.3f} [{lo:.3f},{hi:.3f}]")
    for atk in adv[model].get('attacks', {}):
        ab = get_adv_preds(adv, model, atk)
        if not ab: continue
        sub = {k: gt[k] for k in ab if k in gt}
        adv_cov = np.array([v[2] for v in compute_per_image_coverage(ab, sub, tau).values()])
        # clean coverage on the SAME images (paired comparison) and vs full clean test split
        same = {i: by[i] for i in sub if i in by}
        clean_same = np.array([v[2] for v in compute_per_image_coverage(same, {i: gt[i] for i in same}, tau).values()])
        sp = np.sqrt(((len(clean_same)-1)*clean_same.var(ddof=1) + (len(adv_cov)-1)*adv_cov.var(ddof=1)) / (len(clean_same)+len(adv_cov)-2))
        d = float(abs(clean_same.mean() - adv_cov.mean()) / sp) if sp > 0 else float('inf')
        p = float(stats.ttest_ind(clean_same, adv_cov, equal_var=False).pvalue)
        m, lo, hi = boot(adv_cov)
        out[model][atk] = dict(n=len(adv_cov), mean=m, ci=[lo, hi], gap=0.9-m, gap_ci=[0.9-hi, 0.9-lo],
                               clean_same_mean=float(clean_same.mean()), cohens_d=d, welch_p=p)
        print(f"  {atk:8s} n={len(adv_cov):3d} cov={m:.3f} [{lo:.3f},{hi:.3f}] gap={0.9-m:+.3f}  clean_same={clean_same.mean():.3f}  d={d:.2f} p={p:.2e}")
json.dump(out, open('results/revision/effect_sizes_active_filter.json', 'w'), indent=2)
print('written -> results/revision/effect_sizes_active_filter.json')
