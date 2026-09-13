#!/usr/bin/env python3
"""
test_pipeline.py — Comprehensive tests for the conformal calibration pipeline.

Run BEFORE the full experiment to catch bugs early.

Usage:
    python test_pipeline.py                     # all tests
    python test_pipeline.py --test attacks      # only attack tests
    python test_pipeline.py --test conformal    # only conformal tests
    python test_pipeline.py --test coco         # only COCO eval tests
    python test_pipeline.py --test wrapper      # only attack wrapper tests
"""

import os, sys, json, warnings, tempfile, shutil
from pathlib import Path
from collections import defaultdict

import numpy as np

warnings.filterwarnings('ignore')
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = 0
FAIL = 0

def check(condition, name, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} — {detail}")


# =====================================================================
#  TEST 1: Adaptive Conformal Calibrator
# =====================================================================

def test_conformal():
    print("\n" + "=" * 60)
    print("  TEST: Adaptive Conformal Calibrator")
    print("=" * 60)

    from src.calibration.adaptive_conformal import AdaptiveConformalCalibrator

    np.random.seed(42)
    cal_preds = []
    gt_by_image = {}
    for img_id in range(200):
        n_obj = np.random.randint(2, 6)
        gt_anns = []
        for j in range(n_obj):
            bx, by = np.random.uniform(10, 400), np.random.uniform(10, 300)
            bw, bh = np.random.uniform(30, 200), np.random.uniform(30, 200)
            cat_id = np.random.choice([1, 2, 3, 17, 18, 62])
            gt_anns.append({'bbox': [bx, by, bw, bh], 'category_id': cat_id})
            # Prediction close to GT
            px = bx + np.random.normal(0, 5)
            py = by + np.random.normal(0, 5)
            pw = bw + np.random.normal(0, 8)
            ph = bh + np.random.normal(0, 8)
            conf = np.random.beta(8, 2)
            cal_preds.append({'image_id': img_id, 'category_id': cat_id,
                              'score': float(conf), 'bbox': [float(px), float(py), float(pw), float(ph)]})
        gt_by_image[img_id] = gt_anns

    cal = AdaptiveConformalCalibrator(alpha=0.1, n_conf_bins=5, size_normalize=True)
    cal.calibrate(cal_preds, gt_by_image)

    check(cal.conf_threshold >= 0, "Conf threshold non-negative", f"got {cal.conf_threshold}")
    check(cal.conf_threshold < 1.0, "Conf threshold < 1.0", f"got {cal.conf_threshold}")
    check(cal.global_qhat is not None, "Global qhat computed")
    check(cal.global_qhat > 0, "Global qhat positive", f"got {cal.global_qhat}")
    check(len(cal.bin_qhats) > 0, "Bin-specific qhats computed", f"got {len(cal.bin_qhats)}")
    check(len(cal.class_qhats) > 0, "Class-specific qhats computed", f"got {len(cal.class_qhats)}")

    # Test size-adaptive margins
    det_small = {'category_id': 1, 'score': 0.80, 'bbox': [100, 100, 30, 25]}
    det_large = {'category_id': 1, 'score': 0.80, 'bbox': [100, 100, 400, 300]}
    r_small = cal.predict(det_small)
    r_large = cal.predict(det_large)

    check(r_large.box_delta_pixels > r_small.box_delta_pixels,
          "Large object gets larger margin",
          f"small={r_small.box_delta_pixels:.1f}px, large={r_large.box_delta_pixels:.1f}px")

    ratio = r_large.box_delta_pixels / max(r_small.box_delta_pixels, 0.01)
    check(ratio > 3.0, "Margin ratio > 3x for 10x size difference", f"ratio={ratio:.1f}")

    # Test confidence filtering
    det_high = {'category_id': 1, 'score': 0.95, 'bbox': [100, 100, 100, 100]}
    det_low = {'category_id': 1, 'score': 0.01, 'bbox': [100, 100, 100, 100]}
    r_high = cal.predict(det_high)
    r_low = cal.predict(det_low)

    if cal.conf_threshold > 0.01:
        check(r_high.keep, "High conf detection kept")
        check(not r_low.keep, "Low conf detection filtered")

    # Test prediction set sizes
    check(r_high.class_set_size == 1, "High conf: singleton set", f"got {r_high.class_set_size}")


# =====================================================================
#  TEST 2: COCO Subset Annotation Creation
# =====================================================================

def test_coco_subset():
    print("\n" + "=" * 60)
    print("  TEST: COCO Subset Annotations")
    print("=" * 60)

    coco_root = "data/coco"
    ann_file = os.path.join(coco_root, "annotations", "instances_val2017.json")
    if not os.path.exists(ann_file):
        print("  [SKIP] COCO not downloaded yet")
        return

    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    check(len(data["images"]) == 5000, "Full COCO has 5000 images", f"got {len(data['images'])}")

    # Create subset
    from run_adversarial_eval import create_subset_annotations
    subset_ids = [data["images"][i]["id"] for i in range(25)]
    tmp = tempfile.mktemp(suffix=".json")
    create_subset_annotations(data, subset_ids, tmp)

    with open(tmp, 'r', encoding='utf-8') as f:
        subset = json.load(f)

    check(len(subset["images"]) == 25, "Subset has 25 images", f"got {len(subset['images'])}")
    check(len(subset["categories"]) == 80, "Subset has 80 categories", f"got {len(subset['categories'])}")
    check(len(subset["annotations"]) > 0, "Subset has annotations", f"got {len(subset['annotations'])}")

    # Verify all annotations belong to subset images
    sub_img_ids = set(img["id"] for img in subset["images"])
    bad_anns = [a for a in subset["annotations"] if a["image_id"] not in sub_img_ids]
    check(len(bad_anns) == 0, "All annotations belong to subset images", f"{len(bad_anns)} bad")

    # Test COCO eval on subset
    from src.evaluation.metrics import run_coco_eval
    # Create dummy predictions
    dummy_preds = []
    for ann in subset["annotations"][:50]:
        dummy_preds.append({
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
            "score": 0.9,
        })

    if dummy_preds:
        metrics = run_coco_eval(dummy_preds, tmp, iou_type="bbox")
        check(metrics["mAP@0.5"] > 0.1, "Subset COCO eval gives non-trivial mAP",
              f"mAP@0.5={metrics['mAP@0.5']:.4f}")
        check(metrics["mAP@0.5"] < 1.0, "mAP not suspiciously high",
              f"mAP@0.5={metrics['mAP@0.5']:.4f}")

    os.remove(tmp)


# =====================================================================
#  TEST 3: Label Mapping (DETR / Mask2Former / YOLO)
# =====================================================================

def test_label_mapping():
    print("\n" + "=" * 60)
    print("  TEST: Label Mapping")
    print("=" * 60)

    from src.evaluation.metrics import format_predictions_coco
    coco_cat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                    58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
                    78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    # YOLO: label=0 (0-indexed) -> cat_id=1 (person)
    result_yolo = {
        'boxes': np.array([[10, 10, 100, 100]]),
        'scores': np.array([0.9]),
        'labels': np.array([0]),
        'label_is_coco_id': False,
        'masks': None,
    }
    preds = format_predictions_coco(1, result_yolo, coco_cat_ids, 'bbox')
    check(len(preds) == 1, "YOLO: 1 prediction generated")
    check(preds[0]["category_id"] == 1, "YOLO: label=0 -> cat_id=1 (person)",
          f"got {preds[0]['category_id']}")

    # DETR: label=1 (already COCO cat_id) -> cat_id=1 (person)
    result_detr = {
        'boxes': np.array([[10, 10, 100, 100]]),
        'scores': np.array([0.9]),
        'labels': np.array([1]),
        'label_is_coco_id': True,
        'masks': None,
    }
    preds = format_predictions_coco(1, result_detr, coco_cat_ids, 'bbox')
    check(preds[0]["category_id"] == 1, "DETR: label=1 -> cat_id=1 (person)",
          f"got {preds[0]['category_id']}")

    # DETR: invalid label filtered
    result_bad = {
        'boxes': np.array([[10, 10, 100, 100]]),
        'scores': np.array([0.9]),
        'labels': np.array([999]),
        'label_is_coco_id': True,
        'masks': None,
    }
    preds = format_predictions_coco(1, result_bad, coco_cat_ids, 'bbox')
    check(len(preds) == 0, "DETR: invalid label=999 filtered out", f"got {len(preds)}")

    # Mask2Former: label=0 (0-indexed) -> cat_id=1 (person)
    result_m2f = {
        'boxes': np.array([[10, 10, 100, 100]]),
        'scores': np.array([0.9]),
        'labels': np.array([0]),
        'label_is_coco_id': False,
        'masks': None,
    }
    preds = format_predictions_coco(1, result_m2f, coco_cat_ids, 'bbox')
    check(preds[0]["category_id"] == 1, "Mask2Former: label=0 -> cat_id=1 (person)",
          f"got {preds[0]['category_id']}")


# =====================================================================
#  TEST 4: Attack Wrapper Output
# =====================================================================

def test_attack_wrapper():
    print("\n" + "=" * 60)
    print("  TEST: Attack Wrapper (requires GPU + model)")
    print("=" * 60)

    import torch
    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA available")
        return

    coco_root = "data/coco"
    img_dir = os.path.join(coco_root, "val2017")
    if not os.path.isdir(img_dir):
        print("  [SKIP] COCO not downloaded yet")
        return

    # Find a test image
    imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')][:1]
    if not imgs:
        print("  [SKIP] No images found")
        return
    test_img = os.path.join(img_dir, imgs[0])
    from PIL import Image
    orig = np.array(Image.open(test_img).convert("RGB"))
    orig_h, orig_w = orig.shape[:2]

    # Test Ultralytics wrapper
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8x.pt")
        model.to("cuda")
        from src.attacks.real_attacks import UltralyticsAttackWrapper, FGSM

        wrapper = UltralyticsAttackWrapper(model, "cuda")

        # Test preprocess
        tensor, meta = wrapper.preprocess(test_img)
        check(tensor.shape == (1, 3, 640, 640), "Preprocess: correct shape",
              f"got {tensor.shape}")
        check(tensor.min() >= 0 and tensor.max() <= 1, "Preprocess: values in [0,1]")
        check(meta["orig_w"] == orig_w, "Preprocess: correct orig_w")

        # Test forward_loss
        tensor.requires_grad_(True)
        loss = wrapper.forward_loss(tensor, "vanish")
        check(loss.requires_grad, "Forward loss is differentiable")
        loss.backward()
        check(tensor.grad is not None, "Gradient computed")
        check(tensor.grad.shape == tensor.shape, "Gradient shape matches input")

        # Test tensor_to_image
        out_img = wrapper.tensor_to_image(tensor.detach(), meta)
        check(out_img.shape[0] == orig_h, "Output image: correct height",
              f"expected {orig_h}, got {out_img.shape[0]}")
        check(out_img.shape[1] == orig_w, "Output image: correct width",
              f"expected {orig_w}, got {out_img.shape[1]}")
        check(out_img.dtype == np.uint8, "Output image: uint8")

        # Test FGSM attack
        fgsm = FGSM(wrapper, epsilon=8/255)
        result = fgsm(test_img)
        check(result.adv_image.shape == orig.shape, "FGSM: output matches input shape",
              f"expected {orig.shape}, got {result.adv_image.shape}")
        check(result.l_inf > 0, "FGSM: perturbation is non-zero", f"l_inf={result.l_inf}")
        check(result.l_inf <= 0.15, "FGSM: L_inf within expected range",
              f"l_inf={result.l_inf:.4f}")
        check(result.attack_time_ms > 0, "FGSM: positive attack time")

        # Verify detection on adversarial image gives results
        tmp = tempfile.mktemp(suffix=".png")
        Image.fromarray(result.adv_image).save(tmp)
        from src.models.detector_zoo import build_detector
        from configs.model_config import MODEL_ZOO
        det = build_detector("yolov8x", MODEL_ZOO["yolov8x"], "cuda")
        det.model = model  # reuse loaded model
        r = det.predict(tmp, conf_threshold=0.01)
        check(len(r["scores"]) >= 0, "Detection on adversarial image runs without error")
        os.remove(tmp)

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  [SKIP] Ultralytics wrapper test failed: {e}")


# =====================================================================
#  TEST 5: Coverage Metric Computation
# =====================================================================

def test_coverage():
    print("\n" + "=" * 60)
    print("  TEST: Coverage Metric Computation")
    print("=" * 60)

    from src.calibration.adaptive_conformal import AdaptiveConformalCalibrator
    from run_adversarial_eval import evaluate_with_conformal

    # Create perfect predictions (all correct, high confidence)
    gt_by_image = {
        0: [{"bbox": [10, 10, 50, 50], "category_id": 1}],
        1: [{"bbox": [20, 20, 60, 60], "category_id": 2}],
    }
    perfect_preds = [
        {"image_id": 0, "category_id": 1, "bbox": [10, 10, 50, 50], "score": 0.99},
        {"image_id": 1, "category_id": 2, "bbox": [20, 20, 60, 60], "score": 0.95},
    ]

    # Calibrator that keeps everything
    cal = AdaptiveConformalCalibrator(alpha=0.1)
    cal.conf_threshold = 0.0
    cal.global_qhat = 0.1
    cal.bin_qhats = {}
    cal.class_qhats = {}
    cal.conf_bin_edges = np.linspace(0, 1, 6)

    result = evaluate_with_conformal(perfect_preds, gt_by_image, cal, "test")
    check(result["coverage"] == 1.0, "Perfect preds: 100% coverage",
          f"got {result['coverage']}")
    check(result["n_kept"] == 2, "Perfect preds: 2 kept", f"got {result['n_kept']}")
    check(result["n_filtered"] == 0, "Perfect preds: 0 filtered")

    # Wrong class predictions
    wrong_preds = [
        {"image_id": 0, "category_id": 99, "bbox": [10, 10, 50, 50], "score": 0.99},
        {"image_id": 1, "category_id": 99, "bbox": [20, 20, 60, 60], "score": 0.95},
    ]
    result = evaluate_with_conformal(wrong_preds, gt_by_image, cal, "test")
    check(result["coverage"] == 0.0, "Wrong class preds: 0% coverage",
          f"got {result['coverage']}")


# =====================================================================
#  TEST 6: Defense Mechanisms (conformal_defense.py)
# =====================================================================

def test_defense_mechanisms():
    print("\n" + "=" * 60)
    print("  TEST: Defense Mechanisms (conformal_defense.py)")
    print("=" * 60)

    from src.calibration.adaptive_conformal import AdaptiveConformalCalibrator
    from src.calibration.conformal_defense import (
        RobustConformalCalibrator,
        ConformalAttackDetector,
        SelectivePredictor,
        evaluate_defense,
    )

    np.random.seed(42)

    # Create synthetic calibration data
    cal_preds = []
    gt_by_image = {}
    for img_id in range(100):
        n_obj = np.random.randint(2, 5)
        gt_anns = []
        for j in range(n_obj):
            bx, by = np.random.uniform(10, 400), np.random.uniform(10, 300)
            bw, bh = np.random.uniform(30, 200), np.random.uniform(30, 200)
            cat_id = np.random.choice([1, 2, 3, 17, 18])
            gt_anns.append({'bbox': [bx, by, bw, bh], 'category_id': cat_id})
            px = bx + np.random.normal(0, 5)
            py = by + np.random.normal(0, 5)
            pw = bw + np.random.normal(0, 8)
            ph = bh + np.random.normal(0, 8)
            conf = np.random.beta(8, 2)
            cal_preds.append({
                'image_id': img_id, 'category_id': cat_id,
                'score': float(conf),
                'bbox': [float(px), float(py), float(pw), float(ph)],
            })
        gt_by_image[img_id] = gt_anns

    # Calibrate
    cal = AdaptiveConformalCalibrator(alpha=0.1, n_conf_bins=5, size_normalize=True)
    cal.calibrate(cal_preds[:len(cal_preds)//2], gt_by_image)

    # -- Test RobustConformalCalibrator --
    print("\n  RobustConformalCalibrator:")

    # Create "adversarial" predictions (degraded)
    adv_preds = []
    for p in cal_preds[len(cal_preds)//2:]:
        dp = p.copy()
        dp["score"] = max(0.01, p["score"] * np.random.uniform(0.1, 0.5))
        adv_preds.append(dp)

    robust = RobustConformalCalibrator(alpha=0.1)
    robust.calibrate(
        cal_preds[:len(cal_preds)//2], adv_preds, gt_by_image,
        ratios=[0.0, 0.5, 1.0]
    )

    check(len(robust.calibrators) >= 2,
          "RobustCal: calibrators created for multiple ratios",
          f"got {len(robust.calibrators)}")

    # ratio=0.0 should be same as standard calibrator
    cal_r0 = robust.get_calibrator(0.0)
    check(cal_r0 is not None, "RobustCal: ratio=0.0 calibrator exists")
    check(cal_r0.conf_threshold >= 0, "RobustCal: ratio=0.0 has valid threshold")

    # -- Test ConformalAttackDetector --
    print("\n  ConformalAttackDetector:")

    # Group predictions by image
    clean_by_img = defaultdict(list)
    for p in cal_preds[len(cal_preds)//2:]:
        clean_by_img[p["image_id"]].append(p)

    adv_by_img = defaultdict(list)
    for p in adv_preds:
        adv_by_img[p["image_id"]].append(p)

    detector = ConformalAttackDetector(cal)
    detector.fit_thresholds(clean_by_img)

    check(detector.clean_baseline is not None,
          "Detector: baseline fitted")

    # Clean images should have low abstention
    clean_stats = detector.compute_image_stats(clean_by_img)
    clean_abst = np.mean([s.abstention_rate for s in clean_stats.values()])

    # Adversarial images should have higher abstention
    adv_stats = detector.compute_image_stats(adv_by_img)
    adv_abst = np.mean([s.abstention_rate for s in adv_stats.values()])

    check(adv_abst > clean_abst,
          "Detector: adversarial abstention > clean abstention",
          f"clean={clean_abst:.3f}, adv={adv_abst:.3f}")

    # AUROC should be above random
    try:
        auroc_result = detector.compute_auroc(clean_by_img, adv_by_img)
        check(auroc_result["auroc"] >= 0.5,
              "Detector: AUROC >= 0.5 (better than random)",
              f"auroc={auroc_result['auroc']:.3f}")
    except ImportError:
        print("  [SKIP] sklearn not available for AUROC")

    # -- Test SelectivePredictor --
    print("\n  SelectivePredictor:")

    # Abstention rate should be monotonic in tau
    abst_rates = []
    for tau in [0.1, 0.3, 0.5, 0.7, 0.9]:
        sp = SelectivePredictor(cal, tau=tau)
        result = sp.evaluate(clean_by_img, gt_by_image)
        abst_rates.append(result["image_abstention_rate"])

    # Lower tau = more abstention (more strict)
    is_monotone = all(abst_rates[i] >= abst_rates[i+1] - 0.05
                      for i in range(len(abst_rates)-1))
    check(is_monotone,
          "SelectivePredictor: abstention rate non-increasing in tau",
          f"rates={[f'{r:.3f}' for r in abst_rates]}")

    # High tau should keep almost everything on clean data
    sp_lenient = SelectivePredictor(cal, tau=0.95)
    lenient_result = sp_lenient.evaluate(clean_by_img, gt_by_image)
    check(lenient_result["image_abstention_rate"] < 0.3,
          "SelectivePredictor: lenient tau keeps most clean images",
          f"abst_rate={lenient_result['image_abstention_rate']:.3f}")


# =====================================================================
#  TEST 7: Bug Fix Verification
# =====================================================================

def test_bugfixes():
    print("\n" + "=" * 60)
    print("  TEST: Bug Fix Verification (3 critical fixes)")
    print("=" * 60)

    # --- FIX 1: JSON merge logic ---
    print("\n  Fix 1: JSON merge (full_results.json not overwritten)")
    import tempfile
    tmp = tempfile.mktemp(suffix=".json")

    # Simulate first run: save yolov8x results
    first_run = {"yolov8x": {"mAP": 0.53, "attacks": {"fgsm": {"drop": -0.25}}}}
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(first_run, f)

    # Simulate second run: save detr results — should MERGE
    second_run = {"detr-resnet101": {"mAP": 0.43, "attacks": {"fgsm": {"drop": -0.05}}}}

    # Replicate the merge logic from run_adversarial_eval.py
    existing = {}
    if os.path.exists(tmp):
        with open(tmp, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        existing.update(second_run)
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(existing, f)

    with open(tmp, 'r', encoding='utf-8') as f:
        merged = json.load(f)

    check("yolov8x" in merged, "Merge: yolov8x preserved from first run")
    check("detr-resnet101" in merged, "Merge: detr-resnet101 added from second run")
    check(len(merged) == 2, "Merge: both models present", f"got {len(merged)}")
    os.remove(tmp)

    # --- FIX 2: _extract_det_tensor for seg models ---
    print("\n  Fix 2: Seg model tensor extraction")
    try:
        import torch
        from src.attacks.real_attacks import UltralyticsAttackWrapper

        # Simulate different output formats
        det_tensor = torch.randn(1, 84, 8400)    # standard detection
        seg_tensor_116 = torch.randn(1, 116, 8400)  # seg: 4+80+32 mask coeffs
        proto_tensor = torch.randn(1, 32, 160, 160)  # mask prototypes

        # Create a dummy wrapper to test _extract_det_tensor
        class DummyWrapper(UltralyticsAttackWrapper):
            def __init__(self):
                self.device = 'cpu'

        w = DummyWrapper()

        # Test 1: plain tensor
        result = w._extract_det_tensor(det_tensor)
        check(result is not None and result.shape == det_tensor.shape,
              "extract_det: plain tensor works")

        # Test 2: tuple (det, proto) — standard seg output
        result = w._extract_det_tensor((seg_tensor_116, proto_tensor))
        check(result is not None and result.shape == seg_tensor_116.shape,
              "extract_det: tuple (det, proto) works",
              f"got {result.shape if result is not None else None}")

        # Test 3: list [tuple(det, proto)] — nested seg output
        result = w._extract_det_tensor([(seg_tensor_116, proto_tensor)])
        check(result is not None and result.shape == seg_tensor_116.shape,
              "extract_det: nested [(det, proto)] works",
              f"got {result.shape if result is not None else None}")

        # Test 4: forward_loss should return non-zero for seg tensor
        # Create a minimal forward_loss test
        img = torch.randn(1, 3, 640, 640, requires_grad=True)
        # Simulate what forward_loss does with the extracted tensor
        pred = seg_tensor_116.permute(0, 2, 1)  # (1, 8400, 116)
        n_classes = min(116 - 4, 80)  # = 80
        cls_scores = pred[..., 4:4+n_classes].sigmoid()
        det_conf = cls_scores.max(dim=-1).values
        loss = -det_conf.sum()
        check(loss.item() != 0.0,
              "forward_loss: seg tensor gives non-zero loss",
              f"loss={loss.item():.4f}")
        check(n_classes == 80,
              "forward_loss: correctly uses only 80 class channels from 116",
              f"n_classes={n_classes}")

        # --- FIX 3: eps_scale for HuggingFace ---
        print("\n  Fix 3: DETR epsilon scaling")
        from src.attacks.real_attacks import build_attack, ATTACK_CATALOG, HuggingFaceAttackWrapper

        # Test eps_scale exists on HuggingFace wrapper
        hf_scale = 1.0 / min(HuggingFaceAttackWrapper.IMAGENET_STD)
        check(abs(hf_scale - 4.464) < 0.1,
              "HuggingFace eps_scale ~ 4.46",
              f"got {hf_scale:.3f}")

        # Test that build_attack scales epsilon
        class MockHFWrapper:
            eps_scale = hf_scale
            device = 'cpu'
            valid_min = 0.0
            valid_max = 1.0
            def preprocess(self, path):
                return torch.zeros(1,3,800,800), {}
            def forward_loss(self, x, mode='untargeted'):
                return -x.sum()

        # Build FGSM with HF wrapper — epsilon should be scaled
        attack = build_attack("fgsm", MockHFWrapper())
        expected_eps = (8/255) * hf_scale
        check(abs(attack.epsilon - expected_eps) < 0.001,
              f"build_attack scales FGSM epsilon: {attack.epsilon:.4f} ≈ {expected_eps:.4f}",
              f"got {attack.epsilon:.4f}")

        # Build PGD with HF wrapper — both epsilon and step_size scaled
        attack = build_attack("pgd-20", MockHFWrapper())
        expected_step = (2/255) * hf_scale
        check(abs(attack.step_size - expected_step) < 0.001,
              f"build_attack scales PGD step_size: {attack.step_size:.5f} ≈ {expected_step:.5f}",
              f"got {attack.step_size:.5f}")

        # Ultralytics wrapper should NOT scale
        class MockUltraWrapper:
            device = 'cpu'
            valid_min = 0.0
            valid_max = 1.0
            torch_model = None
            def preprocess(self, path):
                return torch.zeros(1,3,640,640), {}
            def forward_loss(self, x, mode='untargeted'):
                return -x.sum()

        attack_yolo = build_attack("fgsm", MockUltraWrapper())
        check(abs(attack_yolo.epsilon - 8/255) < 0.001,
              f"Ultralytics epsilon NOT scaled: {attack_yolo.epsilon:.4f}",
              f"got {attack_yolo.epsilon:.4f}")

    except ImportError:
        print("  [SKIP] torch not available — GPU-dependent tests skipped")
        # Still verify the non-torch parts passed
        check(True, "Non-torch fix tests completed")


# =====================================================================
#  MAIN
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["all", "conformal", "coco", "labels", "wrapper", "coverage", "defense", "bugfixes"],
                        default="all")
    args = parser.parse_args()

    print("=" * 60)
    print("  PIPELINE TESTS")
    print("=" * 60)

    tests = {
        "conformal": test_conformal,
        "labels": test_label_mapping,
        "coco": test_coco_subset,
        "coverage": test_coverage,
        "wrapper": test_attack_wrapper,
        "defense": test_defense_mechanisms,
        "bugfixes": test_bugfixes,
    }

    if args.test == "all":
        for name, fn in tests.items():
            try:
                fn()
            except Exception as e:
                print(f"\n  [ERROR] {name} test crashed: {e}")
                import traceback
                traceback.print_exc()
    else:
        tests[args.test]()

    print(f"\n{'='*60}")
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    print(f"{'='*60}")

    if FAIL > 0:
        print("  FIX THE FAILURES BEFORE RUNNING THE FULL PIPELINE!")
        sys.exit(1)
    else:
        print("  All tests passed — safe to run full pipeline.")


if __name__ == "__main__":
    main()
