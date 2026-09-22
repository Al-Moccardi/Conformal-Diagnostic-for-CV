"""v6 / fix A1 -- realised L_inf budget of the saved adversarial images.

The attacks of run_adversarial_eval.py enforce ||delta||_inf <= eps = 8/255 by projection in the detector's
INPUT space (640-px letterbox for YOLO; the processor's resized, ImageNet-normalised tensor for DETR).  Each
attacked image is then written as an 8-bit PNG at the ORIGINAL resolution and the detector is re-run on it.
full_results.json stores, per pair, the mean over images of the L_inf distance between that saved image and the
clean image ("mean_l_inf": 0.107--0.110 for the YOLO pairs, 0.35 for DETR-R101), which includes the resampling
and quantisation error of the round trip and is not the budget the projection enforced.

This script separates the two.  For every saved image results/adversarial/attack_samples/{model}_{atk}_adv_{idx}.png
(idx = position of the image in results/revision/test_ids_order.json) it measures
  (a) image-space L_inf   : max|adv - clean| / 255 at the original resolution (reproduces mean_l_inf);
  (b) input-space L_inf   : the same distance after the detector's own preprocessing of BOTH images
                            (YOLO: 640-px letterbox; DETR: shortest side 800, longest <= 1333, bilinear);
  (c) resampling floor    : (a) and (b) for the clean image sent through the SAME round trip with delta = 0
                            (letterbox -> crop -> resize back -> uint8), i.e. the non-adversarial part of (a).
Per pair it reports mean / median / max of (a), (b), (c) and the share of images that the YOLO letterbox does not
resample (long side already 640 px).

Inputs : results/adversarial/attack_samples/*.png, data/coco/val2017/*.jpg, data/coco/annotations/instances_val2017.json,
         results/revision/test_ids_order.json
Output : results/revision/realised_linf.json
Run from the repository root:  python scripts/rebuttal/realised_budget.py     (PIL + numpy only)
"""
import os, sys, json, glob
import numpy as np
from PIL import Image

EPS = 8 / 255
IMGSZ = 640
PAIRS = [('yolov8x', ['fgsm', 'pgd-20', 'pgd-50', 'dag', 'tog-v']),
         ('yolov11x', ['fgsm', 'pgd-20', 'pgd-50', 'dag', 'tog-v']),
         ('detr-resnet101', ['fgsm', 'pgd-20'])]
SAMPLES = 'results/adversarial/attack_samples'
VAL = 'data/coco/val2017'
OUT = 'results/revision/realised_linf.json'


def letterbox(img, imgsz=IMGSZ):
    """UltralyticsAttackWrapper.preprocess: scale to fit, pad with 114 to imgsz x imgsz. Returns (array, meta)."""
    w, h = img.size
    scale = min(imgsz / w, imgsz / h)
    nw, nh = int(w * scale), int(h * scale)
    canvas = Image.new('RGB', (imgsz, imgsz), (114, 114, 114))
    px, py = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    return np.asarray(canvas).astype(np.float32) / 255.0, dict(scale=scale, px=px, py=py, nw=nw, nh=nh, w=w, h=h)


def unletterbox(arr, meta):
    """UltralyticsAttackWrapper.tensor_to_image: crop the content, resize back to the original size, uint8."""
    c = arr[meta['py']:meta['py'] + meta['nh'], meta['px']:meta['px'] + meta['nw']]
    c8 = (np.clip(c, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(c8).resize((meta['w'], meta['h']), Image.BILINEAR)


def detr_resize(img, shortest=800, longest=1333):
    """DetrImageProcessor default resize (shortest edge 800, longest edge <= 1333, bilinear)."""
    w, h = img.size
    s = shortest / min(w, h)
    if max(w, h) * s > longest:
        s = longest / max(w, h)
    return np.asarray(img.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.BILINEAR)).astype(np.float32) / 255.0


def linf(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).max())


def main():
    coco = json.load(open('data/coco/annotations/instances_val2017.json'))
    fname = {im['id']: im['file_name'] for im in coco['images']}
    test_ids = json.load(open('results/revision/test_ids_order.json'))['test_ids']
    out = {'eps': EPS, 'pairs': {}}
    print(f'{"pair":24s} {"n":>3s} {"img-space":>10s} {"input-space":>12s} {"floor img":>10s} {"floor in":>9s} {"unresized":>10s}')
    for model, attacks in PAIRS:
        for atk in attacks:
            files = sorted(glob.glob(os.path.join(SAMPLES, f'{model}_{atk}_adv_*.png')))
            if not files:
                print(f'{model}/{atk}: no saved images'); continue
            rows = []
            for f in files:
                idx = int(os.path.basename(f).rsplit('_', 1)[1].split('.')[0])
                if idx >= len(test_ids):
                    continue
                img_id = test_ids[idx]
                clean = Image.open(os.path.join(VAL, fname[img_id])).convert('RGB')
                adv = Image.open(f).convert('RGB')
                if adv.size != clean.size:
                    adv = adv.resize(clean.size, Image.BILINEAR)
                c8, a8 = np.asarray(clean), np.asarray(adv)
                r = {'image_id': img_id, 'idx': idx, 'linf_image': linf(a8, c8) / 255.0}
                if model.startswith('yolo'):
                    lc, meta = letterbox(clean)
                    la, _ = letterbox(adv)
                    rt = unletterbox(lc, meta)                       # clean round trip, delta = 0
                    lrt, _ = letterbox(rt)
                    r.update(linf_input=linf(la, lc), floor_image=linf(np.asarray(rt), c8) / 255.0,
                             floor_input=linf(lrt, lc), unresized=bool(meta['scale'] == 1.0))
                else:
                    dc, da = detr_resize(clean), detr_resize(adv)
                    rt = Image.fromarray((np.clip(dc, 0, 1) * 255).astype(np.uint8)).resize(clean.size, Image.BILINEAR)
                    r.update(linf_input=linf(da, dc), floor_image=linf(np.asarray(rt), c8) / 255.0,
                             floor_input=linf(detr_resize(rt), dc), unresized=False)
                rows.append(r)
            if not rows:
                continue
            summ = {k: dict(mean=float(np.mean([x[k] for x in rows])), median=float(np.median([x[k] for x in rows])),
                            max=float(np.max([x[k] for x in rows]))) for k in ('linf_image', 'linf_input', 'floor_image', 'floor_input')}
            summ['share_unresized'] = float(np.mean([x['unresized'] for x in rows]))
            summ['n'] = len(rows)
            summ['share_input_within_eps'] = float(np.mean([x['linf_input'] <= EPS + 1.5 / 255 for x in rows]))
            out['pairs'][f'{model}/{atk}'] = {'summary': summ, 'per_image': rows}
            print(f'{model + "/" + atk:24s} {len(rows):3d} {summ["linf_image"]["mean"]:10.3f} {summ["linf_input"]["mean"]:12.3f} '
                  f'{summ["floor_image"]["mean"]:10.3f} {summ["floor_input"]["mean"]:9.3f} {summ["share_unresized"]:10.2f}')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    print('written ->', OUT, f'(eps = {EPS:.4f}; input-space values within eps + quantisation are the enforced budget)')


if __name__ == '__main__':
    main()
