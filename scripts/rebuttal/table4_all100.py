"""Table 4 / effect sizes recomputed over ALL 100 attacked test images:
an attacked image with no detection at conf 0.01 counts as coverage 0 (it is the attack succeeding,
not a failed attack).  Clean reference = the same 100 images, post-threshold."""
import sys, json
import numpy as np
from collections import defaultdict
sys.path.insert(0, '.')
from scripts.rebuttal.revision_experiments import compute_per_image_coverage, get_adv_preds
from scipy import stats
gt = defaultdict(list)
for a in json.load(open('data/coco/annotations/instances_val2017.json'))['annotations']:
    gt[a['image_id']].append(a)
adv = json.load(open('results/adversarial/full_results.json'))
t4 = json.load(open('results/revision/table4_active_filter.json'))
test_ids = sorted(get_adv_preds(adv, 'yolov8x', 'fgsm').keys()); assert len(test_ids) == 100
sub_gt = {i: gt[i] for i in test_ids}
def per_img(preds_by, thr):
    r = compute_per_image_coverage(preds_by, sub_gt, conf_threshold=thr)
    return np.array([r[i][2] if i in r else 0.0 for i in test_ids])
out = {}
for model in ['yolov8x', 'yolov11x', 'detr-resnet101']:
    tau = t4[model]['tau']
    preds = json.load(open(f'results/evaluation/predictions/{model}_bbox.json'))
    by = defaultdict(list)
    for p in preds: by[p['image_id']].append(p)
    clean = per_img({i: by[i] for i in test_ids}, tau)
    out[model] = {'tau': tau, 'clean100_post': float(clean.mean())}
    print(f'\n{model}  tau={tau:.3f}  clean(100 imgs, post)={clean.mean():.3f}')
    for atk in adv[model]['attacks']:
        ab = get_adv_preds(adv, model, atk)
        n_present = len(ab)
        pre = per_img(ab, 0.0); post = per_img(ab, tau)
        sp = np.sqrt(((clean.var(ddof=1) + post.var(ddof=1)) / 2))
        d = abs(clean.mean() - post.mean()) / sp
        p = stats.ttest_ind(clean, post, equal_var=False).pvalue
        old = t4[model][atk]['post_macro']
        print(f'  {atk:7s} n_present={n_present:3d}  AR(100)={pre.mean():.3f}  Cov_post(100)={post.mean():.3f}  '
              f'(was {old:.3f} over n)  Delta={0.9-post.mean():+.2f}  d={d:.2f}  p={p:.1e}')
        out[model][atk] = dict(n_present=n_present, AR100=float(pre.mean()), post100=float(post.mean()),
                               delta=float(0.9-post.mean()), cohens_d=float(d), welch_p=float(p), post_over_n=old)
json.dump(out, open('results/revision/table4_all100.json', 'w'), indent=2)
print('\nwritten results/revision/table4_all100.json')
