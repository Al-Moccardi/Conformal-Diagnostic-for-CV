"""Effect sizes, image-level bootstrap CIs and Table 5 (AR / Cov / recalibrated Cov / eta)
over ALL 100 attacked test images.  An attacked image with no detection at conf 0.01 contributes
coverage 0 (and no true positive, so it does not move the recalibrated threshold either); the
per-image means of the previous JSONs are therefore rescaled exactly by n_present/100.
Clean reference for d: post-threshold coverage of the same 100 images."""
import sys, json
import numpy as np
from collections import defaultdict
sys.path.insert(0, '.')
from scripts.rebuttal.revision_experiments import compute_per_image_coverage, get_adv_preds
from scipy import stats
rng = np.random.default_rng(42)
gt = defaultdict(list)
for a in json.load(open('data/coco/annotations/instances_val2017.json'))['annotations']:
    gt[a['image_id']].append(a)
adv = json.load(open('results/adversarial/full_results.json'))
t4 = json.load(open('results/revision/table4_active_filter.json'))
t5 = json.load(open('results/revision/prop2_recalibration_eta.json'))
test_ids = sorted(get_adv_preds(adv, 'yolov8x', 'fgsm').keys()); assert len(test_ids) == 100
sub_gt = {i: gt[i] for i in test_ids}
def per_img(preds_by, thr):
    r = compute_per_image_coverage(preds_by, sub_gt, conf_threshold=thr)
    return np.array([r[i][2] if i in r else 0.0 for i in test_ids])
def boot(x, B=10000):
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(B)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
out = {}
print(json.dumps({k: list(v.keys())[:3] for k, v in t5.items()})[:300])
for model in ['yolov8x', 'yolov11x', 'detr-resnet101']:
    tau = t4[model]['tau']
    preds = json.load(open(f'results/evaluation/predictions/{model}_bbox.json'))
    by = defaultdict(list)
    for p in preds: by[p['image_id']].append(p)
    clean = per_img({i: by[i] for i in test_ids}, tau)
    ci = boot(clean)
    out[model] = {'clean': {'n': 100, 'mean': float(clean.mean()), 'ci': ci, 'gap': 0.9 - float(clean.mean()),
                            'gap_ci': [0.9 - ci[1], 0.9 - ci[0]]}}
    for atk in adv[model]['attacks']:
        ab = get_adv_preds(adv, model, atk); npres = len(ab)
        pre = per_img(ab, 0.0); post = per_img(ab, tau)
        sp = np.sqrt((clean.var(ddof=1) + post.var(ddof=1)) / 2)
        d = abs(clean.mean() - post.mean()) / sp
        p = stats.ttest_ind(clean, post, equal_var=False).pvalue
        ci = boot(post)
        row = {'n': 100, 'n_present': npres, 'mean': float(post.mean()), 'ci': ci, 'gap': 0.9 - float(post.mean()),
               'gap_ci': [0.9 - ci[1], 0.9 - ci[0]], 'clean_same_mean': float(clean.mean()),
               'cohens_d': float(d), 'welch_p': float(p), 'AR100': float(pre.mean())}
        # Table 5 rescaled
        key = None
        for k in t5: 
            if k.startswith(model) and atk in k: key = k
        if key is None and model in t5 and atk in t5[model]: key = (model, atk)
        src = t5[key] if isinstance(key, str) else (t5[model][atk] if key else None)
        if src:
            f = npres / 100.0
            row['table5'] = {k2: (float(v) * f if isinstance(v, (int, float)) and k2 in ('AR', 'cov_cleanCP', 'cov_recalCP', 'eta') else v)
                             for k2, v in src.items()}
        out[model][atk] = row
        t5s = row.get('table5', {})
        print(f"{model:15s} {atk:7s} present={npres:3d} AR={pre.mean():.3f} Cov={post.mean():.3f} CI=[{ci[0]:.3f},{ci[1]:.3f}] "
              f"d={d:.2f} | T5: AR={t5s.get('AR', float('nan')):.3f} cln={t5s.get('cov_cleanCP', float('nan')):.3f} "
              f"adv={t5s.get('cov_recalCP', float('nan')):.3f} eta={t5s.get('eta', float('nan')):.3f}")
json.dump(out, open('results/revision/effect_sizes_all100.json', 'w'), indent=2)
print('written results/revision/effect_sizes_all100.json')
