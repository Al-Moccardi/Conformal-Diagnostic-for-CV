import sys, json, random
from collections import defaultdict
sys.path.insert(0, '.')
from src.calibration.adaptive_conformal import AdaptiveConformalCalibrator
from src.calibration.conformal_defense import (
    ConformalAttackDetector, SelectivePredictor, _match_preds_to_gt)

MODEL = 'yolov8x'
ALPHA = 0.1

gt_by_image = defaultdict(list)
for ann in json.load(open('data/coco/annotations/instances_val2017.json'))['annotations']:
    gt_by_image[ann['image_id']].append(ann)

clean_preds = json.load(open(f'results/evaluation/predictions/{MODEL}_bbox.json'))
clean_by_img = defaultdict(list)
for p in clean_preds:
    clean_by_img[p['image_id']].append(p)

adv_results = json.load(open('results/adversarial/full_results.json'))

img_ids = sorted(clean_by_img.keys())
random.seed(42); random.shuffle(img_ids)
cal_ids = set(img_ids[:len(img_ids)//2])
cal_preds = [p for p in clean_preds if p['image_id'] in cal_ids]

calib = AdaptiveConformalCalibrator(alpha=ALPHA, n_conf_bins=5, size_normalize=True)
calib.calibrate(cal_preds, gt_by_image)
print(f'calibrator conf_threshold = {calib.conf_threshold:.6f}  '
      f'-> uses_anomaly_score = {calib.conf_threshold < 0.001}')

def evaluate_fixed(sp, preds_by_image, gt_by_image):
    """Both coverage definitions, side by side."""
    n_img = n_abst = 0
    gt_all = gt_ret = cov_ret = 0
    for img_id, img_preds in preds_by_image.items():
        n_img += 1
        r = sp.predict_image(img_preds)
        gt_anns = gt_by_image.get(img_id, [])
        gt_all += len(gt_anns)
        if r['abstain']:
            n_abst += 1
            continue
        gt_ret += len(gt_anns)
        _, n_cov = _match_preds_to_gt(r['kept_preds'], gt_anns)
        cov_ret += n_cov
    return {
        'abst': n_abst / max(n_img, 1),
        'cov_retained': cov_ret / max(gt_ret, 1),   # coverage ON RETAINED images
        'cov_selective': cov_ret / max(gt_all, 1),  # what the paper reports
        'n_img': n_img,
    }

TAUS = [0.1, 0.2, 0.3, 0.5]
for atk in ['pgd-20', 'tog-v', 'fgsm']:
    atk_data = adv_results[MODEL]['attacks'].get(atk)
    if not atk_data or not atk_data.get('per_image_predictions'):
        continue
    adv_by_img = defaultdict(list)
    for p in atk_data['per_image_predictions']:
        adv_by_img[p['image_id']].append(p)
    clean_same = {i: clean_by_img[i] for i in adv_by_img if i in clean_by_img}

    det = ConformalAttackDetector(calib); det.fit_thresholds(clean_same)
    print(f'\n=== {MODEL} / {atk}  (n_clean={len(clean_same)}, n_adv={len(adv_by_img)}) ===')
    print(f'{"tau":>5} | {"Rej_cln":>8} {"Rej_adv":>8} | '
          f'{"Cov_RETAINED":>13} {"Cov_selective":>14} | {"Cov_adv_ret":>12}')
    for t in TAUS:
        sp = SelectivePredictor(calib, tau=t, detector=det)
        c = evaluate_fixed(sp, clean_same, gt_by_image)
        a = evaluate_fixed(sp, adv_by_img, gt_by_image)
        print(f'{t:>5} | {c["abst"]*100:7.1f}% {a["abst"]*100:7.1f}% | '
              f'{c["cov_retained"]:13.3f} {c["cov_selective"]:14.3f} | {a["cov_retained"]:12.3f}')

# ---------------------------------------------------------------
# Persist the corrected Table 6 for the revision.
# ---------------------------------------------------------------
import os
os.makedirs('results/revision', exist_ok=True)
out = {}
for atk in ['pgd-20', 'tog-v', 'fgsm']:
    ad = adv_results[MODEL]['attacks'].get(atk)
    if not ad or not ad.get('per_image_predictions'):
        continue
    abi = defaultdict(list)
    for p in ad['per_image_predictions']:
        abi[p['image_id']].append(p)
    cs = {i: clean_by_img[i] for i in abi if i in clean_by_img}
    dt = ConformalAttackDetector(calib); dt.fit_thresholds(cs)
    rows = []
    for t in TAUS:
        sp = SelectivePredictor(calib, tau=t, detector=dt)
        c = evaluate_fixed(sp, cs, gt_by_image)
        a = evaluate_fixed(sp, abi, gt_by_image)
        rows.append({'tau': t,
                     'rej_clean': c['abst'], 'rej_adv': a['abst'],
                     'cov_clean_retained': c['cov_retained'],
                     'cov_clean_selective': c['cov_selective'],
                     'cov_adv_retained': a['cov_retained'],
                     'cov_adv_selective': a['cov_selective']})
    out[atk] = rows
json.dump(out, open('results/revision/table6_corrected.json', 'w'), indent=2)
print('\nwritten -> results/revision/table6_corrected.json')
