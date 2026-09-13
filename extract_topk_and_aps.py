#!/usr/bin/env python3
"""
extract_topk_and_aps.py — Beyond Top-1: Genuine APS with Full Class Distributions

Tests whether CP-specific signals improve adversarial detection when the
detector exposes full class probability distributions (not just top-1).

Usage:
    # Clean baseline only (~10 min)
    python extract_topk_and_aps.py --n-images 100 --attack none

    # Clean vs PGD-20 (~30 min)
    python extract_topk_and_aps.py --n-images 100 --attack pgd --steps 20

    # Quick test (~2 min)
    python extract_topk_and_aps.py --n-images 10 --attack pgd --steps 5
"""

import os, json, argparse, warnings
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.environ["HF_HUB_OFFLINE"] = "1"  # Use cached model, no network

VALID_COCO_IDS = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,
                  27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,
                  51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,
                  76,77,78,79,80,81,82,84,85,86,87,88,89,90}


def load_detr(device='cuda'):
    """Load DETR from cache (offline)."""
    from transformers import DetrForObjectDetection, DetrImageProcessor
    try:
        processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-101", local_files_only=True)
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-101", local_files_only=True)
    except Exception:
        print("  Cache miss — downloading DETR (first time only)...")
        os.environ.pop("HF_HUB_OFFLINE", None)
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")
    model = model.to(device).eval()
    return model, processor


def extract_preds_with_probs(logits, boxes, pil_img, image_id, conf=0.05):
    """Extract detections with FULL class distributions from DETR output."""
    probs = logits.softmax(-1)
    obj_probs = probs[:, :-1]  # [100, 91] remove no-object
    max_probs, max_classes = obj_probs.max(dim=-1)
    img_w, img_h = pil_img.size

    preds = []
    for i in range(logits.shape[0]):
        if max_probs[i] < conf:
            continue
        coco_id = int(max_classes[i])
        if coco_id not in VALID_COCO_IDS:
            continue

        cx, cy, w, h = boxes[i].cpu().numpy()
        full_dist = obj_probs[i].cpu().numpy()  # 91 class probs

        preds.append({
            "image_id": image_id,
            "bbox": [float((cx-w/2)*img_w), float((cy-h/2)*img_h),
                     float(w*img_w), float(h*img_h)],
            "category_id": coco_id,
            "score": float(max_probs[i]),
            "full_probs": full_dist,  # keep as numpy for speed
        })
    return preds


def compute_aps(probs, alpha=0.1):
    """Genuine APS: accumulate sorted probs until sum >= 1-alpha."""
    sorted_p = np.sort(probs)[::-1]
    cumsum = np.cumsum(sorted_p)
    set_size = int(np.searchsorted(cumsum, 1 - alpha) + 1)
    set_size = min(set_size, len(probs))
    entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
    return set_size, entropy


def per_image_features(preds_dict, alpha=0.1):
    """Compute per-image APS features."""
    feats = {}
    for img_id, preds in preds_dict.items():
        confs = [p["score"] for p in preds]
        sizes, ents = [], []
        for p in preds:
            if p["full_probs"] is not None:
                ss, ent = compute_aps(p["full_probs"], alpha)
                sizes.append(ss)
                ents.append(ent)

        feats[img_id] = {
            "n_dets": len(preds),
            "mean_conf": float(np.mean(confs)) if confs else 0,
            "max_conf": float(np.max(confs)) if confs else 0,
            "mean_set_size": float(np.mean(sizes)) if sizes else 0,
            "max_set_size": int(np.max(sizes)) if sizes else 0,
            "mean_entropy": float(np.mean(ents)) if ents else 0,
            "frac_large_sets": float(np.mean([s > 3 for s in sizes])) if sizes else 0,
        }
    return feats


# ================================================================
#  PGD ATTACK
# ================================================================

def pgd_detr(model, pixel_values, eps_per_ch, steps=20):
    """Simple PGD on DETR: minimize total object confidence."""
    x_adv = pixel_values.clone().detach()
    alpha = 2.0 * eps_per_ch / steps

    for _ in range(steps):
        x_adv.requires_grad_(True)
        out = model(x_adv)
        obj_probs = out.logits.softmax(-1)[..., :-1]
        loss = obj_probs.max(dim=-1).values.sum()
        loss.backward()
        g = x_adv.grad.detach()
        with torch.no_grad():
            x_adv = x_adv - alpha * g.sign()
            delta = torch.max(torch.min(x_adv - pixel_values, eps_per_ch), -eps_per_ch)
            x_adv = (pixel_values + delta).detach()
    return x_adv


# ================================================================
#  MAIN EXPERIMENT
# ================================================================

def run_experiment(coco_root, n_images, attack, steps, alpha, output_dir, device):
    from PIL import Image
    import glob

    img_dir = os.path.join(coco_root, "val2017")
    ann_file = os.path.join(coco_root, "annotations", "instances_val2017.json")

    print("=" * 70)
    print("  BEYOND TOP-1: Genuine APS with Full Class Distributions")
    print(f"  Images: {n_images}, Attack: {attack}, Steps: {steps}")
    print("=" * 70)

    # Load
    print("\n  Loading DETR-ResNet101...")
    model, processor = load_detr(device)

    eps = 8/255
    eps_per_ch = torch.tensor([eps/s for s in [0.229, 0.224, 0.225]]).view(3,1,1).to(device)

    with open(ann_file, 'r') as f:
        coco = json.load(f)
    fname_to_id = {img["file_name"]: img["id"] for img in coco["images"]}
    gt_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)

    images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:n_images]
    print(f"  Processing {len(images)} images...")

    clean_preds = {}
    adv_preds = {} if attack != "none" else None

    for img_path in tqdm(images, desc="  DETR APS"):
        fname = os.path.basename(img_path)
        image_id = fname_to_id.get(fname)
        if image_id is None:
            continue

        pil_img = Image.open(img_path).convert("RGB")
        inputs = processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        # Clean inference
        with torch.no_grad():
            out = model(pixel_values)
            clean_preds[image_id] = extract_preds_with_probs(
                out.logits[0], out.pred_boxes[0], pil_img, image_id)

        # Adversarial
        if attack == "pgd":
            x_adv = pgd_detr(model, pixel_values, eps_per_ch, steps)
            with torch.no_grad():
                adv_out = model(x_adv)
                adv_preds[image_id] = extract_preds_with_probs(
                    adv_out.logits[0], adv_out.pred_boxes[0], pil_img, image_id)

    # ============================================================
    #  ANALYSIS
    # ============================================================
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    clean_feats = per_image_features(clean_preds, alpha)

    # --- Clean baseline stats ---
    vals = [f for f in clean_feats.values() if f["n_dets"] > 0]
    print(f"\n  CLEAN BASELINE ({len(vals)} images with detections):")
    print(f"    Detections/image:     {np.mean([v['n_dets'] for v in vals]):.1f}")
    print(f"    Mean confidence:      {np.mean([v['mean_conf'] for v in vals]):.3f}")
    print(f"    Mean APS set size:    {np.mean([v['mean_set_size'] for v in vals]):.2f}")
    print(f"    Mean entropy:         {np.mean([v['mean_entropy'] for v in vals]):.3f}")
    print(f"    Frac large sets (>3): {np.mean([v['frac_large_sets'] for v in vals]):.3f}")

    # Per-detection stats
    all_clean_sizes = []
    for img_id, preds in clean_preds.items():
        for p in preds:
            if p["full_probs"] is not None:
                ss, _ = compute_aps(p["full_probs"], alpha)
                all_clean_sizes.append(ss)
    if all_clean_sizes:
        print(f"\n    Per-detection APS set size distribution (clean):")
        for s in [1, 2, 3, 5, 10, 20]:
            frac = np.mean([x <= s for x in all_clean_sizes])
            print(f"      set_size <= {s:>2d}: {frac:.1%}")

    if adv_preds is None:
        print("\n  No attack requested. Clean baseline only.")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "aps_clean_baseline.json"), 'w') as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "full_probs"}
                       for k, v in clean_feats.items()}, f, indent=2, default=str)
        return

    # --- Adversarial stats ---
    adv_feats = per_image_features(adv_preds, alpha)
    adv_vals = [f for f in adv_feats.values() if f["n_dets"] > 0]

    print(f"\n  ADVERSARIAL PGD-{steps} ({len(adv_vals)} images with detections):")
    print(f"    Detections/image:     {np.mean([v['n_dets'] for v in adv_vals]):.1f}")
    print(f"    Mean confidence:      {np.mean([v['mean_conf'] for v in adv_vals]):.3f}")
    print(f"    Mean APS set size:    {np.mean([v['mean_set_size'] for v in adv_vals]):.2f}")
    print(f"    Mean entropy:         {np.mean([v['mean_entropy'] for v in adv_vals]):.3f}")
    print(f"    Frac large sets (>3): {np.mean([v['frac_large_sets'] for v in adv_vals]):.3f}")

    all_adv_sizes = []
    for img_id, preds in adv_preds.items():
        for p in preds:
            if p["full_probs"] is not None:
                ss, _ = compute_aps(p["full_probs"], alpha)
                all_adv_sizes.append(ss)
    if all_adv_sizes:
        print(f"\n    Per-detection APS set size distribution (adversarial):")
        for s in [1, 2, 3, 5, 10, 20]:
            frac = np.mean([x <= s for x in all_adv_sizes])
            print(f"      set_size <= {s:>2d}: {frac:.1%}")

    # ============================================================
    #  ABLATION: Do APS signals improve AUROC?
    # ============================================================
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    print(f"\n{'='*70}")
    print(f"  APS SIGNAL ABLATION")
    print(f"{'='*70}")

    common = set(clean_feats.keys()) & set(adv_feats.keys())
    print(f"  Common images: {len(common)}")

    feature_keys = ["n_dets", "mean_conf", "max_conf",
                    "mean_set_size", "max_set_size", "mean_entropy", "frac_large_sets"]

    X_c = np.array([[clean_feats[i][k] for k in feature_keys] for i in common])
    X_a = np.array([[adv_feats[i][k] for k in feature_keys] for i in common])
    X = np.vstack([X_c, X_a])
    y = np.array([0]*len(X_c) + [1]*len(X_a))

    # Feature comparison
    print(f"\n  {'Feature':<22s} {'Clean':>10s} {'Adv':>10s} {'Ratio':>8s}")
    print(f"  {'-'*50}")
    for j, k in enumerate(feature_keys):
        cm, am = X_c[:,j].mean(), X_a[:,j].mean()
        r = am/cm if cm > 0 else 0
        flag = " ***" if abs(r-1) > 0.3 else ""
        print(f"  {k:<22s} {cm:>10.3f} {am:>10.3f} {r:>8.2f}{flag}")

    # AUROC per feature and combinations
    def auroc_combo(feat_indices, X, y):
        X_sub = X[:, feat_indices]
        sc = StandardScaler().fit(X_sub[:len(X_c)])
        z = np.abs(sc.transform(X_sub)).mean(axis=1)
        try:
            a = float(roc_auc_score(y, z))
            return max(a, 1-a)
        except:
            return 0.5

    configs = [
        ("n_dets only",                [0]),
        ("n_dets + mean_conf",         [0, 1]),
        ("n_dets + mean_conf + max_conf (OUTPUT BASE)", [0, 1, 2]),
        ("  + mean_set_size (APS)",    [0, 1, 2, 3]),
        ("  + max_set_size (APS)",     [0, 1, 2, 4]),
        ("  + mean_entropy (APS)",     [0, 1, 2, 5]),
        ("  + frac_large_sets (APS)",  [0, 1, 2, 6]),
        ("  + ALL APS signals",        [0, 1, 2, 3, 4, 5, 6]),
        ("APS-only (no output-level)", [3, 4, 5, 6]),
    ]

    print(f"\n  {'Configuration':<45s} {'AUROC':>7s} {'Delta':>8s}")
    print(f"  {'-'*60}")
    base = 0
    results = {}
    for name, idx in configs:
        a = auroc_combo(idx, X, y)
        if "OUTPUT BASE" in name:
            base = a
            d = "  (base)"
        elif base > 0:
            d = f"  {a-base:+.3f}" + (" ***" if abs(a-base) > 0.02 else "")
        else:
            d = ""
        print(f"  {name:<45s} {a:>7.3f}{d}")
        results[name] = a

    # Also test individual APS features as standalone
    print(f"\n  Individual APS feature AUROC:")
    for j, k in enumerate(feature_keys[3:], start=3):
        a = auroc_combo([j], X, y)
        print(f"    {k:<22s} {a:.3f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "n_images": len(common),
        "attack": f"pgd-{steps}",
        "clean_mean_set_size": float(np.mean([v['mean_set_size'] for v in vals])),
        "adv_mean_set_size": float(np.mean([v['mean_set_size'] for v in adv_vals])),
        "clean_mean_entropy": float(np.mean([v['mean_entropy'] for v in vals])),
        "adv_mean_entropy": float(np.mean([v['mean_entropy'] for v in adv_vals])),
        "auroc_results": results,
    }
    with open(os.path.join(output_dir, "aps_ablation_results.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved to {output_dir}/aps_ablation_results.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-root", default="data/coco")
    p.add_argument("--n-images", type=int, default=50)
    p.add_argument("--attack", default="pgd", choices=["pgd", "none"])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--output-dir", default="results/revision")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    run_experiment(a.coco_root, a.n_images, a.attack, a.steps, a.alpha, a.output_dir, a.device)


if __name__ == "__main__":
    main()