"""v6 / fix A3 -- selective predictor and label-free monitor with DISJOINT clean sets.

Replaces the in-sample protocol behind Table 7 (tau_A) and Table 9 (Rule A / Rule B false alarms)
of the v5 manuscript, and produces the numbers behind the W-sweep / re-split sentences.

Protocol (Proposition 1 needs three disjoint clean sets):
  B  baseline set   : clean predictions of the conformal calibration split (seed-42 half of val2017)
                      minus the 200 images of the adversarial subset          -> (mu_k, sigma_k) of Eq. (13)
  K  threshold set  : the 100 clean CALIBRATION images of the adversarial subset
                      -> tau_A = A_(k), k = ceil((1-alpha)(n+1)) = 91          (order statistic of Prop. 1)
  T  test set       : the 100 attacked TEST images, clean and attacked versions
                      -> Rej_cln, Rej_adv, Cov_cln, Cov_adv at tau_A and on the grid {0.1, 0.2, 0.3, 0.5}
  The in-sample rejection on K (which is 9/100 by construction) is reported as a sanity check.

Label-free monitor (Table 9): Rule A (1.5x max clean deviation) and Rule B (alpha-quantile) thresholds are
calibrated on a clean control stream C1 (200 images, seed 7, as in proxy_monitor.py) and the false alarms
are counted on a DISJOINT clean stream C2 (200 images, seed 8).  Delays are measured on the
clean -> adversarial -> clean sequence of fix_runtime_monitor (seed 42), exactly as before.

Extras: labelled-monitor W sweep with held-out false alarms (gamma = 0.15, stream C2), and five re-splits
of the conformal calibration/test partition (tau and post-threshold coverage, clean and PGD-20).

Inputs  (repository root):
  results/evaluation/predictions/{model}_bbox.json      clean predictions (all 5,000 val2017 images)
  data/coco/annotations/instances_val2017.json
  results/adversarial/full_results.json                 saved adversarial predictions
  results/revision/test_ids_order.json                  the 100 attacked test images, in attack order
Outputs:
  results/revision/oos_abstention_monitor.json
  paper/generated/numbers_v6.tex                        LaTeX macros read by paper_revision_v6_*.tex

Run from the repository root:   python scripts/rebuttal/oos_abstention_monitor.py
Only numpy is required (scipy is optional).  Runtime: a few minutes on a laptop CPU.
"""
import sys, os, json, random, math
from collections import defaultdict

import numpy as np

sys.path.insert(0, '.')
from src.calibration.adaptive_conformal import AdaptiveConformalCalibrator          # noqa: E402
from src.calibration.conformal_defense import (ConformalAttackDetector,              # noqa: E402
                                               SelectivePredictor, _match_preds_to_gt)
from scripts.rebuttal.revision_experiments import compute_per_image_coverage, get_adv_preds  # noqa: E402

ALPHA = 0.10
MODEL = 'yolov8x'
GRID = [0.1, 0.2, 0.3, 0.5]
ATTACKS_T7 = ['pgd-20', 'tog-v', 'fgsm']
SCEN_MONITOR = [('yolov8x', 'pgd-20'), ('yolov8x', 'tog-v'), ('yolov8x', 'fgsm'), ('detr-resnet101', 'pgd-20')]
MIN_CONF = 0.1          # screening threshold of z_hc and of the proxy count
GAMMA_LABELLED = 0.15   # Eq. (17)
W_DEFAULT = 20
RESPLIT_SEEDS = [42, 1, 2, 3, 4]
ANN = 'data/coco/annotations/instances_val2017.json'
ADV = 'results/adversarial/full_results.json'
OUT_JSON = 'results/revision/oos_abstention_monitor.json'
OUT_TEX = 'paper/generated/numbers_v6.tex'


# ---------------------------------------------------------------- helpers
def load_clean(model):
    preds = json.load(open(f'results/evaluation/predictions/{model}_bbox.json'))
    by = defaultdict(list)
    for p in preds:
        by[p['image_id']].append(p)
    return preds, by


def conformal_split(by, seed=42):
    """Seed-42 half split of the images with predictions (active_filter_eval.py convention)."""
    ids = sorted(by)
    random.seed(seed)
    random.shuffle(ids)
    cal = ids[:len(ids) // 2]
    test = ids[len(ids) // 2:]
    return cal, test


def tau_eq11(preds, gt_by, alpha):
    """Conformal confidence threshold of Eq. (11): (1-alpha)-quantile of 1-s over matched true positives."""
    cal = AdaptiveConformalCalibrator(alpha=alpha, n_conf_bins=5, size_normalize=True)
    m = cal._match_all(preds, gt_by, 0.5)
    tp = np.array([x['conf'] for x in m if x['correct']])
    s = 1.0 - tp
    n = len(s)
    q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return max(0.0, 1.0 - float(np.quantile(s, q))), n


def adversarial_subset(coco, val_dir='data/coco/val2017', check_files=True):
    """Replicates load_coco_data() + the 100/100 split of run_adversarial_eval.py (seed 42).
    Returns (cal_ids, test_ids) of the 200-image adversarial subset.  With check_files the
    candidate list is restricted to images present on disk, as in the original run."""
    gt_count = defaultdict(int)
    for a in coco['annotations']:
        gt_count[a['image_id']] += 1
    use_dir = check_files and os.path.isdir(val_dir)
    cands = []
    for img in coco['images']:                     # same order as load_coco_data (dict insertion order)
        if gt_count.get(img['id'], 0) >= 2:
            if use_dir and not os.path.exists(os.path.join(val_dir, img['file_name'])):
                continue
            cands.append(img['id'])
    random.seed(42)
    random.shuffle(cands)
    items = cands[:200]
    random.seed(42)
    random.shuffle(items)
    return items[:100], items[100:]


def clopper_pearson(k, n, conf=0.95):
    """Exact binomial interval (no scipy needed): bisection on the binomial CDF."""
    def cdf(x, p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, x + 1))
    a = (1 - conf) / 2
    lo, hi = 0.0, 1.0
    if k > 0:
        l, h = 0.0, 1.0
        for _ in range(60):
            m = (l + h) / 2
            if 1 - cdf(k - 1, m) < a: l = m
            else: h = m
        lo = (l + h) / 2
    if k < n:
        l, h = 0.0, 1.0
        for _ in range(60):
            m = (l + h) / 2
            if cdf(k, m) < a: h = m
            else: l = m
        hi = (l + h) / 2
    return lo, hi


def evaluate_fixed(sp, preds_by_image, gt_by_image, all_ids=None):
    """Abstention rate and object-pooled coverage of the retained images (ge2_table6_coverage_fix.py).
    If all_ids is given, images absent from preds_by_image count as abstained (score 1)."""
    ids = list(all_ids) if all_ids is not None else list(preds_by_image.keys())
    n_img = n_abst = 0
    gt_all = gt_ret = cov_ret = 0
    for img_id in ids:
        n_img += 1
        r = sp.predict_image(preds_by_image.get(img_id, []))
        anns = gt_by_image.get(img_id, [])
        gt_all += len(anns)
        if r['abstain']:
            n_abst += 1
            continue
        gt_ret += len(anns)
        _, n_cov = _match_preds_to_gt(r['kept_preds'], anns)
        cov_ret += n_cov
    return {'abst': n_abst / max(n_img, 1),
            'cov_retained': cov_ret / max(gt_ret, 1),
            'cov_selective': cov_ret / max(gt_all, 1),
            'n_img': n_img, 'n_abst': n_abst}


def windows(seq, W):
    return [(i, float(np.mean(seq[i - W:i]))) for i in range(W, len(seq) + 1)]


def n_det(preds):
    return sum(1 for p in preds if p['score'] >= MIN_CONF)


def fmt(x, nd=3):
    return '---' if x is None else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))


# ---------------------------------------------------------------- data
coco = json.load(open(ANN))
gt = defaultdict(list)
for a in coco['annotations']:
    gt[a['image_id']].append(a)
adv = json.load(open(ADV))
T_ids = json.load(open('results/revision/test_ids_order.json'))['test_ids']
K_ids, T_check = adversarial_subset(coco)
subset_ok = set(T_check) == set(T_ids)
if not subset_ok:                                   # e.g. incomplete val2017 folder: retry without the file check
    K_ids, T_check = adversarial_subset(coco, check_files=False)
    subset_ok = set(T_check) == set(T_ids)
if not subset_ok:
    print('[WARN] could not reproduce the adversarial subset from the annotation file; '
          'using 100 random calibration-split images disjoint from the test images as K')
out = {'alpha': ALPHA, 'subset_reproduced': subset_ok}

# ================================================================ 1. selective predictor (Table 7)
clean_preds, clean_by = load_clean(MODEL)
cal_ids, test_ids = conformal_split(clean_by)
tau, n_tp = tau_eq11([p for p in clean_preds if p['image_id'] in set(cal_ids)], gt, ALPHA)
calib = AdaptiveConformalCalibrator(alpha=ALPHA, n_conf_bins=5, size_normalize=True)
calib.calibrate([p for p in clean_preds if p['image_id'] in set(cal_ids)], gt)
print(f'{MODEL}: tau(Eq.11) = {tau:.4f} (calibrator {calib.conf_threshold:.4f}), n_TP_cal = {n_tp}')

T_set = set(T_ids)
if not subset_ok:
    pool = [i for i in cal_ids if i not in T_set]
    random.seed(42)
    K_ids = random.sample(pool, 100)
K_set = set(K_ids)
B_ids = [i for i in cal_ids if i not in T_set and i not in K_set]
print(f'disjoint clean sets: |B| = {len(B_ids)}  |K| = {len(K_ids)}  |T| = {len(T_ids)}')

det = ConformalAttackDetector(calib)
det.fit_thresholds({i: clean_by[i] for i in B_ids if i in clean_by})
statsK = det.compute_image_stats({i: clean_by.get(i, []) for i in K_ids})
A_K = np.array([det._anomaly_score(statsK[i]) for i in K_ids])
n = len(A_K)
k = int(math.ceil((1 - ALPHA) * (n + 1)))
tau_A = float(np.sort(A_K)[k - 1])
in_sample_rej_K = float(np.mean(A_K > tau_A))
print(f'tau_A = A_({k}) of {n} = {tau_A:.4f};  in-sample rejection on K = {in_sample_rej_K:.2f}')

clean_T = {i: clean_by.get(i, []) for i in T_ids}
t7 = {'tau': tau, 'tau_A': tau_A, 'k': k, 'n_K': n, 'n_B': len(B_ids),
      'in_sample_rej_K': in_sample_rej_K, 'rows': {}}
sp_none = SelectivePredictor(calib, tau=1.01, detector=det)
t7['cov_cln_no_abstention'] = evaluate_fixed(sp_none, clean_T, gt, T_ids)['cov_retained']
for atk in ATTACKS_T7:
    ab = get_adv_preds(adv, MODEL, atk)
    if not ab:
        continue
    present = [i for i in T_ids if i in ab]
    rows = []
    for t in GRID + [tau_A]:
        sp = SelectivePredictor(calib, tau=t, detector=det)
        c = evaluate_fixed(sp, clean_T, gt, T_ids)
        a = evaluate_fixed(sp, ab, gt, present)          # images with surviving output (Table 7 convention)
        a100 = evaluate_fixed(sp, ab, gt, T_ids)         # all 100 (an image without output abstains)
        row = {'tau': t, 'is_tau_A': t == tau_A,
               'rej_clean': c['abst'], 'rej_adv': a['abst'], 'rej_adv_all100': a100['abst'],
               'cov_clean_retained': c['cov_retained'], 'cov_adv_retained': a['cov_retained'],
               'cov_clean_selective': c['cov_selective'], 'cov_adv_selective': a['cov_selective'],
               'n_adv_present': len(present), 'n_adv_retained': len(present) - a['n_abst']}
        if t == tau_A:
            lo, hi = clopper_pearson(c['n_abst'], c['n_img'])
            row['rej_clean_ci95'] = [lo, hi]
        rows.append(row)
        print(f'  {atk:7s} tau_A={t:.3f}{"*" if t == tau_A else " "}  rej_cln={c["abst"]:.2f} rej_adv={a["abst"]:.2f} '
              f'cov_cln={c["cov_retained"]:.3f} cov_adv={a["cov_retained"]:.3f} (retained adv imgs {len(present) - a["n_abst"]})')
    t7['rows'][atk] = rows
out['selective_predictor'] = t7

# ================================================================ 2. label-free monitor, held-out (Table 9)
mon = {}
for model, atk in SCEN_MONITOR:
    preds_m, by_m = load_clean(model)
    ab = get_adv_preds(adv, model, atk)
    if not ab:
        print(f'skip {model}/{atk}')
        continue
    cal_m, _ = conformal_split(by_m)
    N_cal = float(np.mean([n_det(by_m[i]) for i in cal_m]))
    common = sorted(set(by_m) & set(ab) & set(gt))
    n_phase = min(len(common) // 3, 40)
    np.random.seed(42)
    sh = np.random.permutation(common).tolist()
    p1, p2, p3 = sh[:n_phase], sh[n_phase:2 * n_phase], sh[2 * n_phase:3 * n_phase]
    seq = ([n_det(by_m[i]) / N_cal for i in p1] + [n_det(ab[i]) / N_cal for i in p2] +
           [n_det(by_m[i]) / N_cal for i in p3])
    adv_end = len(p1) + len(p2)
    used = set(p1 + p2 + p3)
    rest = [i for i in sorted(by_m) if i not in used]
    L = max(len(seq), 200)
    np.random.seed(7)
    c1 = np.random.permutation(rest).tolist()[:L]                       # calibration stream (as before)
    rest2 = [i for i in rest if i not in set(c1)]
    np.random.seed(8)
    c2 = np.random.permutation(rest2).tolist()[:L]                      # held-out stream (new)
    ctrl1 = [n_det(by_m[i]) / N_cal for i in c1]
    ctrl2 = [n_det(by_m[i]) / N_cal for i in c2]
    # labelled coverage of the same streams, for the W sweep of the labelled monitor
    def lab_cov(ids, src):
        r = compute_per_image_coverage({i: src[i] for i in ids}, {i: gt[i] for i in ids})
        return [r[i][2] for i in ids if i in r]
    seq_lab = lab_cov(p1, by_m) + lab_cov(p2, ab) + lab_cov(p3, by_m)
    ctrl2_lab = lab_cov([i for i in c2 if i in gt], by_m)
    rows = {}
    for W in (10, 15, 20, 30):
        w1, w2, wseq = windows(ctrl1, W), windows(ctrl2, W), windows(seq, W)
        # Rule A: 1.5x the largest downward clean deviation, calibrated on C1
        gamma_p = 1.5 * max(max(0.0, 1.0 - v) for _, v in w1)
        thrA = 1.0 - gamma_p
        # Rule B: floor(alpha m)-th smallest clean window of C1
        c1v = sorted(v for _, v in w1)
        thrB = c1v[max(0, int(np.floor(ALPHA * len(c1v))) - 1)]
        def first_alert(thr):
            # delay = first alerting window that ends after the onset; windows entirely inside the
            # clean phase that breach the threshold are counted separately as pre-onset false alarms
            pos = next((p for p, v in wseq if p > len(p1) and v < thr), None)
            return None if pos is None else pos - len(p1)
        def pre_onset(thr):
            return sum(1 for p, v in wseq if p <= len(p1) and v < thr)
        def recov(thr):
            pos = next((p for p, v in wseq if p > adv_end and v >= thr), None)
            return None if pos is None else pos - adv_end
        faA_in, faB_in = sum(v < thrA for _, v in w1), sum(v < thrB for _, v in w1)
        faA, faB = sum(v < thrA for _, v in w2), sum(v < thrB for _, v in w2)
        # labelled monitor, Eq. (17), gamma = 0.15: delay on the sequence, false alarms on C2
        wl = windows(seq_lab, W)
        posL = next((p for p, v in wl if p > len(p1) and (1 - ALPHA) - v > GAMMA_LABELLED), None)
        preL = sum(1 for p, v in wl if p <= len(p1) and (1 - ALPHA) - v > GAMMA_LABELLED)
        delayL = None if posL is None else posL - len(p1)
        faL = sum((1 - ALPHA) - v > GAMMA_LABELLED for _, v in windows(ctrl2_lab, W))
        rows[W] = dict(gamma_p=round(gamma_p, 3), thr_A=round(thrA, 3), thr_B=round(thrB, 3),
                       clean=round(float(np.mean(seq[:len(p1)])), 3), adv=round(float(np.mean(seq[len(p1):adv_end])), 3),
                       rec=round(float(np.mean(seq[adv_end:])), 3),
                       delay_A=first_alert(thrA), recovery_A=recov(thrA), fa_A_heldout=faA, fa_A_insample=faA_in,
                       pre_onset_alerts_A=pre_onset(thrA),
                       delay_B=first_alert(thrB), recovery_B=recov(thrB), fa_B_heldout=faB, fa_B_insample=faB_in,
                       pre_onset_alerts_B=pre_onset(thrB),
                       n_windows_cal=len(w1), n_windows_heldout=len(w2),
                       labelled_delay=delayL, labelled_fa_heldout=faL, labelled_n_windows=len(windows(ctrl2_lab, W)),
                       labelled_pre_onset_alerts=preL,
                       labelled_clean=round(float(np.mean(seq_lab[:len(p1)])), 3),
                       labelled_adv=round(float(np.mean(seq_lab[len(p1):adv_end])), 3),
                       labelled_rec=round(float(np.mean(seq_lab[adv_end:])), 3))
    mon[f'{model}/{atk}'] = dict(N_cal=round(N_cal, 2), n_phase=n_phase, rows=rows)
    r = rows[W_DEFAULT]
    print(f'{model}/{atk}: proxy clean={r["clean"]:.3f} adv={r["adv"]:.3f} | A: delay={r["delay_A"]} FA={r["fa_A_heldout"]}/{r["n_windows_heldout"]} '
          f'| B: delay={r["delay_B"]} FA={r["fa_B_heldout"]}/{r["n_windows_heldout"]} (in-sample {r["fa_B_insample"]}) '
          f'| labelled: delay={r["labelled_delay"]} FA={r["labelled_fa_heldout"]}/{r["labelled_n_windows"]}')
out['monitor'] = mon

# ================================================================ 3. re-splits of the conformal partition
rs = []
ab_pgd = get_adv_preds(adv, MODEL, 'pgd-20')
for seed in RESPLIT_SEEDS:
    cal_s, test_s = conformal_split(clean_by, seed)
    tau_s, _ = tau_eq11([p for p in clean_preds if p['image_id'] in set(cal_s)], gt, ALPHA)
    test_by = {i: clean_by[i] for i in test_s if i in gt}
    r = compute_per_image_coverage(test_by, {i: gt[i] for i in test_by}, conf_threshold=tau_s)
    cov_clean = float(np.mean([v[2] for v in r.values()]))
    ra = compute_per_image_coverage(ab_pgd, {i: gt[i] for i in T_ids}, conf_threshold=tau_s)
    cov_adv = float(np.mean([ra[i][2] if i in ra else 0.0 for i in T_ids]))
    rs.append({'seed': seed, 'tau': tau_s, 'cov_clean_post': cov_clean, 'cov_pgd20_post': cov_adv})
    print(f'resplit seed {seed}: tau={tau_s:.4f} clean post={cov_clean:.4f} pgd-20 post={cov_adv:.4f}')
out['resplits'] = {'rows': rs,
                   'tau_range': [min(r['tau'] for r in rs), max(r['tau'] for r in rs)],
                   'sd_cov_clean': float(np.std([r['cov_clean_post'] for r in rs])),
                   'sd_cov_pgd20': float(np.std([r['cov_pgd20_post'] for r in rs]))}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump(out, open(OUT_JSON, 'w'), indent=2)
print('written ->', OUT_JSON)

# ================================================================ 4. LaTeX macros for the manuscript
def pct(x):
    return str(int(round(100 * x)))

rows_pgd = {('A' if r['is_tau_A'] else 'G' + 'abcd'[GRID.index(r["tau"])]): r for r in t7['rows']['pgd-20']}
rA = rows_pgd['A']
lo, hi = rA['rej_clean_ci95']
L = []
L.append('%% Generated by scripts/rebuttal/oos_abstention_monitor.py -- do not edit by hand.')
L.append(f'%% subset_reproduced={subset_ok}  |B|={len(B_ids)}  |K|={n}  |T|={len(T_ids)}')
L.append('\\newcommand{\\vSixProvisional}{}')   # the fallback in the .tex prints a visible warning; generated numbers silence it
L.append(f'\\newcommand{{\\vTauA}}{{{tau_A:.2f}}}')
L.append(f'\\newcommand{{\\vTauAThree}}{{{tau_A:.3f}}}')
L.append(f'\\newcommand{{\\vRejClnA}}{{{pct(rA["rej_clean"])}}}')
L.append(f'\\newcommand{{\\vRetClnA}}{{{100 - int(pct(rA["rej_clean"]))}}}')
L.append(f'\\newcommand{{\\vRejAdvA}}{{{pct(rA["rej_adv"])}}}')
L.append(f'\\newcommand{{\\vCovClnA}}{{{rA["cov_clean_retained"]:.3f}}}')
L.append(f'\\newcommand{{\\vCovAdvA}}{{{rA["cov_adv_retained"]:.3f}}}')
L.append(f'\\newcommand{{\\vRejClnCI}}{{{100 * lo:.0f}--{100 * hi:.0f}}}')
L.append(f'\\newcommand{{\\vInSampleRejK}}{{{pct(in_sample_rej_K)}}}')
L.append(f'\\newcommand{{\\vBaselineN}}{{{len(B_ids)}}}')
L.append(f'\\newcommand{{\\vCovClnNoAbst}}{{{t7["cov_cln_no_abstention"]:.3f}}}')
for g in ('Ga', 'Gb', 'Gc', 'Gd'):
    r = rows_pgd[g]
    L.append(f'\\newcommand{{\\vRejCln{g}}}{{{pct(r["rej_clean"])}}}')
    L.append(f'\\newcommand{{\\vRejAdv{g}}}{{{pct(r["rej_adv"])}}}')
    L.append(f'\\newcommand{{\\vCovCln{g}}}{{{r["cov_clean_retained"]:.3f}}}')
    L.append(f'\\newcommand{{\\vCovAdv{g}}}{{{r["cov_adv_retained"]:.3f}}}')
    L.append(f'\\newcommand{{\\vRetAdv{g}}}{{{rows_pgd[g]["n_adv_retained"]}}}')
L.append(f'\\newcommand{{\\vRetClnGd}}{{{100 - int(pct(rows_pgd["Gd"]["rej_clean"]))}}}')
L.append(f'\\newcommand{{\\vPassAdvGd}}{{{100 - int(pct(rows_pgd["Gd"]["rej_adv"]))}}}')
# monitor macros
tags = {'yolov8x/pgd-20': 'PGD', 'yolov8x/tog-v': 'TOG', 'yolov8x/fgsm': 'FGSM', 'detr-resnet101/pgd-20': 'DETR'}
m_heldout = None
for key, tag in tags.items():
    if key not in mon:
        continue
    r = mon[key]['rows'][W_DEFAULT]
    m_heldout = r['n_windows_heldout']
    L.append(f'\\newcommand{{\\vRuleAdelay{tag}}}{{{fmt(r["delay_A"])}}}')
    L.append(f'\\newcommand{{\\vRuleAfa{tag}}}{{{r["fa_A_heldout"]}}}')
    L.append(f'\\newcommand{{\\vRuleBdelay{tag}}}{{{fmt(r["delay_B"])}}}')
    L.append(f'\\newcommand{{\\vRuleBfa{tag}}}{{{r["fa_B_heldout"]}}}')
    L.append(f'\\newcommand{{\\vRuleBfaPct{tag}}}{{{100 * r["fa_B_heldout"] / r["n_windows_heldout"]:.1f}}}')
    L.append(f'\\newcommand{{\\vRuleAfaPct{tag}}}{{{100 * r["fa_A_heldout"] / r["n_windows_heldout"]:.1f}}}')
L.append(f'\\newcommand{{\\vHeldoutWindows}}{{{m_heldout}}}')
rp = mon.get('yolov8x/pgd-20', {}).get('rows', {}).get(W_DEFAULT)
if rp and rp['fa_B_heldout'] > 0:
    L.append(f'\\newcommand{{\\vRuleBfaPerWin}}{{{rp["n_windows_heldout"] / rp["fa_B_heldout"]:.0f}}}')
else:
    L.append('\\newcommand{\\vRuleBfaPerWin}{---}')
# W-sweep sentence (labelled monitor, strong attacks) and re-split sentence
strong = [mon[k]['rows'] for k in ('yolov8x/pgd-20', 'yolov8x/tog-v') if k in mon]
delays = [rw[W]['labelled_delay'] for rw in strong for W in (10, 15, 20, 30) if rw[W]['labelled_delay'] is not None]
fas = [rw[W]['labelled_fa_heldout'] for rw in strong for W in (10, 15, 20, 30)]
nwin = [rw[W]['labelled_n_windows'] for rw in strong for W in (10, 15, 20, 30)]
if delays:
    L.append('\\newcommand{\\WsweepSentence}{Varying $W \\in \\{10, 15, 20, 30\\}$ changes the detection delay of the labelled monitor '
             f'under PGD-20 and TOG-V from {min(delays)} to {max(delays)} images, with {min(fas)}--{max(fas)} false alarms over '
             f'{min(nwin)}--{max(nwin)} windows of a held-out clean stream ($\\gamma = 0.15$).}}')
else:
    L.append('\\newcommand{\\WsweepSentence}{}')
r = out['resplits']
L.append('\\newcommand{\\ResplitSentence}{Five random re-splits of the calibration/test partition move $\\tau$ within '
         f'{r["tau_range"][0]:.3f}--{r["tau_range"][1]:.3f} and change post-threshold coverage by $\\pm{r["sd_cov_clean"]:.3f}$ (clean) '
         f'and $\\pm{r["sd_cov_pgd20"]:.3f}$ (YOLOv8x, PGD-20) across seeds, so aggregate results are not artefacts of a particular partition.}}')
os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
open(OUT_TEX, 'w').write('\n'.join(L) + '\n')
print('written ->', OUT_TEX)
