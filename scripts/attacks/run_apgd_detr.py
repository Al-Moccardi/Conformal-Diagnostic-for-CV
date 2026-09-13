#!/usr/bin/env python3
"""
run_apgd_detr.py — Auto-PGD attack for DETR with per-channel L∞ constraints.
FIXED: loss sign, DETR→COCO category mapping, PIL size order.

Usage:
    python run_apgd_detr.py --n-images 100 --steps 100 --restarts 5
    python run_apgd_detr.py --n-images 5 --steps 20 --restarts 2  # quick test
"""

import os, sys, json, argparse, warnings, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings('ignore')

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# HuggingFace DETR uses COCO category IDs directly as labels.
# logits shape: [B, 100, 92] where class 0 = N/A, classes 1-91 = COCO IDs 1-91
# After removing last class: obj_probs[..., i] corresponds to DETR label i
# DETR label i = COCO category ID i (they are identical)
# Invalid COCO IDs (12, 26, 29, 30, 45, 66, 68, 69, 71, 83) have no GT and are harmless.
VALID_COCO_IDS = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,
                  27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,
                  51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,
                  76,77,78,79,80,81,82,84,85,86,87,88,89,90}


def apgd_detection_loss(model, x_adv, original_outputs):
    """Detection-aware loss: MINIMIZE total object confidence.
    
    loss = +sum(max_obj_conf_per_query)
    Gradient descent on this MINIMIZES confidence = SUPPRESSES detections.
    """
    outputs = model(x_adv)
    
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
        pred_probs = logits.softmax(-1)
        obj_probs = pred_probs[..., :-1]  # remove no-object class
        max_obj_per_query = obj_probs.max(dim=-1).values
        # POSITIVE: gradient descent minimizes this = suppresses detections
        loss = max_obj_per_query.sum()
    elif isinstance(outputs, dict) and 'pred_logits' in outputs:
        logits = outputs['pred_logits']
        pred_probs = logits.softmax(-1)
        obj_probs = pred_probs[..., :-1]
        max_obj_per_query = obj_probs.max(dim=-1).values
        loss = max_obj_per_query.sum()
    else:
        raise ValueError(f"Unknown output format: {type(outputs)}")
    
    return loss, outputs


def apgd_attack(model, images, eps_per_channel, n_steps=100, n_restarts=5,
                rho=0.75, device='cuda', verbose=True):
    """Auto-PGD: adaptive step size, per-channel L-inf, multiple restarts."""
    model.eval()
    images = images.to(device)
    eps = eps_per_channel.view(3, 1, 1).to(device)
    
    with torch.no_grad():
        clean_outputs = model(images)
    
    best_adv = images.clone()
    best_loss = torch.tensor(float('inf')).to(device)
    
    for restart in range(n_restarts):
        delta = torch.zeros_like(images).uniform_(-1, 1) * eps
        x_adv = (images + delta).clone().detach().requires_grad_(True)
        
        step_size = 2.0 * eps
        W = max(1, int(n_steps * 0.22))
        loss_history = []
        
        for step in range(n_steps):
            x_adv = x_adv.clone().detach().requires_grad_(True)
            
            loss, outputs = apgd_detection_loss(model, x_adv, clean_outputs)
            loss.backward()
            
            grad = x_adv.grad.detach()
            
            with torch.no_grad():
                # Gradient DESCENT: minimize positive loss = suppress detections
                x_adv_new = x_adv - step_size * grad.sign()
                
                # Project to per-channel L-inf ball
                delta_new = x_adv_new - images
                delta_new = torch.max(torch.min(delta_new, eps), -eps)
                x_adv = images + delta_new
            
            loss_val = loss.item()
            loss_history.append(loss_val)
            
            # Adaptive step size reduction
            if step > 0 and step % W == 0 and len(loss_history) >= W:
                recent = loss_history[-W:]
                if len(loss_history) > W and min(recent) >= min(loss_history[:-W]) * (1 - 1e-3):
                    step_size = step_size * 0.5
        
        with torch.no_grad():
            final_loss, _ = apgd_detection_loss(model, x_adv.detach(), clean_outputs)
            if final_loss < best_loss:
                best_loss = final_loss
                best_adv = x_adv.detach().clone()
        
        if verbose and n_restarts > 1:
            print(f"    Restart {restart+1}/{n_restarts}: loss={final_loss.item():.2f}")
    
    return best_adv


def extract_preds(logits_softmax, boxes, pil_img, image_id, conf_thresh=0.05):
    """Extract predictions. DETR labels = COCO IDs directly."""
    probs = logits_softmax[0, :, :-1]  # [100, 91], index i = DETR label i
    max_probs, max_classes = probs.max(dim=-1)
    
    img_w, img_h = pil_img.size  # PIL: (width, height)
    
    preds = []
    n_det = 0
    for i in range(probs.shape[0]):
        if max_probs[i] > conf_thresh:
            # max_classes[i] IS the COCO category ID (DETR uses COCO IDs directly)
            coco_id = int(max_classes[i])
            if coco_id not in VALID_COCO_IDS:
                continue
            
            box = boxes[0, i].cpu().numpy()
            cx, cy, w, h = box
            x = (cx - w/2) * img_w
            y = (cy - h/2) * img_h
            bw = w * img_w
            bh = h * img_h
            
            preds.append({
                "image_id": image_id,
                "category_id": coco_id,
                "score": float(max_probs[i]),
                "bbox": [float(x), float(y), float(bw), float(bh)],
            })
            n_det += 1
    
    return preds, n_det


def run_apgd_detr(coco_root, n_images=200, n_steps=100, n_restarts=5,
                   eps=8/255, output_dir='results/revision', device='cuda'):
    print("=" * 70)
    print("  APGD Attack on DETR-ResNet101 (FIXED v2)")
    print(f"  Steps={n_steps}, Restarts={n_restarts}, eps={eps:.4f}")
    print("=" * 70)
    
    from transformers import DetrForObjectDetection, DetrImageProcessor
    from PIL import Image
    import glob
    
    print("  Loading DETR-ResNet101...")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")
    model = model.to(device).eval()
    
    eps_per_channel = torch.tensor([eps / s for s in [0.229, 0.224, 0.225]])
    print(f"  Per-channel eps (normalized): {eps_per_channel.tolist()}")
    
    img_dir = os.path.join(coco_root, "val2017")
    ann_file = os.path.join(coco_root, "annotations", "instances_val2017.json")
    
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    fname_to_id = {img["file_name"]: img["id"] for img in coco_data["images"]}
    gt_by_image = defaultdict(list)
    for ann in coco_data["annotations"]:
        gt_by_image[ann["image_id"]].append(ann)
    
    images_list = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:n_images]
    print(f"  Processing {len(images_list)} images")
    
    # Verify DETR→COCO mapping on first image
    pil_test = Image.open(images_list[0]).convert("RGB")
    inputs_test = processor(images=pil_test, return_tensors="pt")
    with torch.no_grad():
        out_test = model(inputs_test["pixel_values"].to(device))
        probs_test = out_test.logits.softmax(-1)[0, :, :-1]
        mp, mc = probs_test.max(dim=-1)
        top5 = mp.topk(5)
        print("  Verification (top-5 clean detections):")
        for idx in range(5):
            qi = top5.indices[idx].item()
            coco_id = int(mc[qi])  # DETR label = COCO ID directly
            name = model.config.id2label.get(coco_id, "?")
            valid = "✓" if coco_id in VALID_COCO_IDS else "✗"
            print(f"    query={qi} conf={top5.values[idx]:.3f} "
                  f"coco_id={coco_id} name={name} {valid}")
    
    clean_preds = defaultdict(list)
    adv_preds = defaultdict(list)
    clean_det_counts = []
    adv_det_counts = []
    
    for img_path in tqdm(images_list, desc="  APGD"):
        fname = os.path.basename(img_path)
        image_id = fname_to_id.get(fname)
        if image_id is None:
            continue
        
        pil_img = Image.open(img_path).convert("RGB")
        inputs = processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        
        # Clean inference
        with torch.no_grad():
            clean_out = model(pixel_values)
            clean_sm = clean_out.logits.softmax(-1)
            preds_c, n_c = extract_preds(clean_sm, clean_out.pred_boxes, pil_img, image_id)
            clean_preds[image_id] = preds_c
            clean_det_counts.append(n_c)
        
        # APGD attack
        adv_images = apgd_attack(
            model, pixel_values, eps_per_channel,
            n_steps=n_steps, n_restarts=n_restarts,
            device=device, verbose=False
        )
        
        # Adversarial inference
        with torch.no_grad():
            adv_out = model(adv_images)
            adv_sm = adv_out.logits.softmax(-1)
            preds_a, n_a = extract_preds(adv_sm, adv_out.pred_boxes, pil_img, image_id)
            adv_preds[image_id] = preds_a
            adv_det_counts.append(n_a)
    
    # Results
    print(f"\n  {'='*60}")
    print(f"  APGD RESULTS (DETR-R101) — FIXED v2")
    print(f"  {'='*60}")
    print(f"  Steps: {n_steps}, Restarts: {n_restarts}")
    print(f"  Images: {len(images_list)}")
    print(f"  Clean dets/img (conf>0.05): {np.mean(clean_det_counts):.1f}")
    print(f"  Adv dets/img (conf>0.05):   {np.mean(adv_det_counts):.1f}")
    det_red = 1 - np.mean(adv_det_counts)/max(np.mean(clean_det_counts), 1)
    print(f"  Detection reduction: {det_red:.1%}")
    
    # Inline coverage computation (no external dependency)
    def _iou(boxA, boxB):
        """IoU between two [x,y,w,h] boxes."""
        ax, ay, aw, ah = boxA
        bx, by, bw, bh = boxB
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix = max(0, min(ax2, bx2) - max(ax, bx))
        iy = max(0, min(ay2, by2) - max(ay, by))
        inter = ix * iy
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0

    # Filter GT to only processed images
    processed_ids = set(clean_preds.keys())
    gt_subset = {k: v for k, v in gt_by_image.items() if k in processed_ids}
    print(f"  Processed images: {len(processed_ids)}, GT images: {len(gt_subset)}")

    # Count total GT objects
    total_gt = sum(len(anns) for anns in gt_subset.values())
    print(f"  Total GT objects in subset: {total_gt}")

    def compute_coverage(preds_dict, gt_dict):
        total, covered = 0, 0
        for img_id, anns in gt_dict.items():
            preds = preds_dict.get(img_id, [])
            preds_sorted = sorted(preds, key=lambda p: -p["score"])
            gt_matched = [False] * len(anns)
            for p in preds_sorted:
                for j, gt in enumerate(anns):
                    if gt_matched[j]:
                        continue
                    if gt.get("iscrowd", 0):
                        continue
                    if p["category_id"] != gt["category_id"]:
                        continue
                    iou = _iou(p["bbox"], gt["bbox"])
                    if iou >= 0.5:
                        gt_matched[j] = True
                        break
            covered += sum(gt_matched)
            total += sum(1 for g in anns if not g.get("iscrowd", 0))
        return covered / total if total > 0 else 0

    clean_mean = compute_coverage(dict(clean_preds), gt_subset)
    adv_mean = compute_coverage(dict(adv_preds), gt_subset)
    gap = 0.9 - adv_mean

    # Debug: show sample predictions vs GT for first image
    first_id = list(processed_ids)[0]
    print(f"\n  DEBUG (image {first_id}):")
    print(f"    Clean preds: {len(clean_preds[first_id])}")
    if clean_preds[first_id]:
        p = clean_preds[first_id][0]
        print(f"    First pred: cat={p['category_id']} conf={p['score']:.3f} bbox={[round(b,1) for b in p['bbox']]}")
    print(f"    GT anns: {len(gt_subset.get(first_id, []))}")
    if gt_subset.get(first_id):
        g = gt_subset[first_id][0]
        print(f"    First GT:   cat={g['category_id']} bbox={[round(b,1) for b in g['bbox']]}")
    common = True  # we always have results now

    print(f"\n  Clean coverage: {clean_mean:.3f}")
    print(f"  APGD coverage:  {adv_mean:.3f}")
    print(f"  Coverage gap:   {gap:+.3f}")
    print(f"  Coverage drop:  {clean_mean - adv_mean:.3f}")

    print(f"\n  COMPARISON with PGD-20 (no restarts):")
    print(f"    PGD-20:  coverage ~0.845, gap ~+0.06")
    print(f"    APGD:    coverage {adv_mean:.3f}, gap {gap:+.3f}")

    if gap < 0.15:
        verdict = "CONFIRMED: DETR robustness holds under APGD"
    elif gap < 0.30:
        verdict = "MODERATE: APGD degrades DETR more than PGD-20"
    else:
        verdict = "BROKEN: DETR robustness was an artifact of weak attack"
    print(f"\n  >>> {verdict}")
    
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "attack": "APGD_v2_fixed",
        "model": "detr-resnet101",
        "steps": n_steps,
        "restarts": n_restarts,
        "eps": eps,
        "n_images": len(images_list),
        "clean_mean_cov": float(clean_mean),
        "adv_mean_cov": float(adv_mean),
        "coverage_gap": float(gap),
        "clean_dets_per_img": float(np.mean(clean_det_counts)),
        "adv_dets_per_img": float(np.mean(adv_det_counts)),
        "detection_reduction": float(det_red),
    }
    
    out_path = os.path.join(output_dir, "apgd_detr_results_v2.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", default="data/coco")
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--eps", type=float, default=8/255)
    parser.add_argument("--output-dir", default="results/revision")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_apgd_detr(args.coco_root, args.n_images, args.steps, args.restarts,
                  args.eps, args.output_dir, args.device)


if __name__ == "__main__":
    main()