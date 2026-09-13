"""
adaptive_conformal.py — Adaptive conformal prediction for object detection.

Unlike fixed-margin approaches, this module implements:

1. CONFIDENCE-ADAPTIVE MARGINS (CQR-style):
   Score = residual / sigma(confidence)
   Objects with high confidence -> tight margins
   Objects with low confidence -> wide margins (or filtered)

2. SIZE-NORMALIZED MARGINS:
   Residuals are normalized by object size (w, h)
   A 5px error on a 300px car is different from 5px on a 30px bottle

3. CLASS-CONDITIONAL CALIBRATION:
   Per-class quantiles (frequent classes get their own threshold)
   Rare classes fall back to the global quantile

4. ADAPTIVE PREDICTION SETS (APS):
   For each detection, return a SET of plausible classes
   High-confidence detections: {car} (singleton)
   Low-confidence detections: {car, truck, bus} (larger set)

5. COMPOSITE CONFORMAL PREDICTOR:
   Combines class filtering + adaptive box margins + prediction sets

References:
  - Romano et al., "Conformalized Quantile Regression", NeurIPS 2019
  - Timans et al., "Two-Step CP for OD", ECCV 2024
  - Angelopoulos et al., "Uncertainty Sets for Image Classifiers", ICLR 2021
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ConformalResult:
    """Result of conformal prediction for a single detection."""
    keep: bool                     # Whether this detection passes the conformal filter
    conf_threshold: float          # Confidence threshold used
    box_delta: np.ndarray          # (4,) adaptive margin [dx, dy, dw, dh]
    box_delta_pixels: float        # Mean margin in pixels
    class_set: List[int]           # Prediction set of plausible class IDs
    class_set_size: int            # |C_alpha|
    confidence: float              # Original confidence
    normalized_score: float        # Nonconformity score used


class AdaptiveConformalCalibrator:
    """
    Full adaptive conformal prediction for object detection.

    Calibration computes:
      - Confidence threshold (tau): detections below tau are filtered
      - Per-confidence-bin box margins: high conf -> small margin
      - Class-conditional quantiles: per-class calibration when possible
      - Prediction set threshold: cumulative softmax threshold for class sets
    """

    def __init__(self, alpha: float = 0.1, n_conf_bins: int = 5,
                 min_class_samples: int = 20, size_normalize: bool = True):
        self.alpha = alpha
        self.n_conf_bins = n_conf_bins
        self.min_class_samples = min_class_samples
        self.size_normalize = size_normalize

        # Calibrated parameters
        self.conf_threshold = 0.0
        self.global_qhat = None           # fallback quantile
        self.bin_qhats = {}               # {bin_idx: qhat} confidence-adaptive
        self.class_qhats = {}             # {cat_id: qhat} class-conditional
        self.conf_bin_edges = None
        self.cal_stats = {}

    def calibrate(self, cal_preds: List[Dict], gt_by_image: Dict,
                  iou_threshold: float = 0.5) -> 'AdaptiveConformalCalibrator':
        """
        Calibrate on prediction-GT matched pairs.

        cal_preds: list of {image_id, category_id, score, bbox=[x,y,w,h]}
        gt_by_image: {image_id: [list of GT annotations]}
        """
        matched = self._match_all(cal_preds, gt_by_image, iou_threshold)
        if not matched:
            print("  [WARN] No matched predictions for calibration")
            return self

        # --- Step 1: Confidence threshold (APS-style) ---
        confs = np.array([m["conf"] for m in matched])
        correct = np.array([m["correct"] for m in matched])
        # Nonconformity score for classification
        cls_scores = np.where(correct, 1.0 - confs, 1.0)
        n = len(cls_scores)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.conf_threshold = max(0.0, 1.0 - np.quantile(cls_scores, q_level))

        # --- Step 2: Size-normalized box residuals ---
        residuals_raw = np.array([m["box_residual"] for m in matched])  # (N, 4)
        box_sizes = np.array([m["box_size"] for m in matched])          # (N, 2) = [w, h]
        confs_matched = np.array([m["conf"] for m in matched])

        if self.size_normalize:
            # Normalize residuals by object size: relative error
            norm_factors = np.column_stack([
                np.maximum(box_sizes[:, 0], 1),  # w for dx, dw
                np.maximum(box_sizes[:, 1], 1),  # h for dy, dh
                np.maximum(box_sizes[:, 0], 1),
                np.maximum(box_sizes[:, 1], 1),
            ])
            residuals_norm = residuals_raw / norm_factors
        else:
            residuals_norm = residuals_raw

        # --- Step 3: Global quantile (fallback) ---
        global_scores = residuals_norm.max(axis=1)  # worst-case across 4 coords
        self.global_qhat = float(np.quantile(global_scores, q_level))

        # --- Step 4: Confidence-adaptive quantiles (CQR-style) ---
        self.conf_bin_edges = np.linspace(0, 1, self.n_conf_bins + 1)
        for b in range(self.n_conf_bins):
            lo, hi = self.conf_bin_edges[b], self.conf_bin_edges[b + 1]
            mask = (confs_matched >= lo) & (confs_matched < hi)
            if mask.sum() >= 10:
                bin_scores = residuals_norm[mask].max(axis=1)
                self.bin_qhats[b] = float(np.quantile(bin_scores, q_level))

        # --- Step 5: Class-conditional quantiles ---
        class_residuals = defaultdict(list)
        for m in matched:
            if m["correct"]:
                cat_id = m["category_id"]
                r = m["box_residual"]
                sz = m["box_size"]
                if self.size_normalize:
                    r = r / np.array([max(sz[0],1), max(sz[1],1), max(sz[0],1), max(sz[1],1)])
                class_residuals[cat_id].append(r.max())

        for cat_id, scores in class_residuals.items():
            if len(scores) >= self.min_class_samples:
                self.class_qhats[cat_id] = float(np.quantile(scores, q_level))

        # --- Stats ---
        self.cal_stats = {
            "n_matched": len(matched),
            "n_correct": int(correct.sum()),
            "conf_threshold": float(self.conf_threshold),
            "global_qhat": float(self.global_qhat),
            "n_conf_bins_calibrated": len(self.bin_qhats),
            "n_class_specific": len(self.class_qhats),
            "mean_conf": float(confs.mean()),
            "alpha": self.alpha,
        }

        return self

    def predict(self, detection: Dict) -> ConformalResult:
        """
        Apply conformal prediction to a single detection.

        detection: {category_id, score, bbox=[x,y,w,h]}
        Returns: ConformalResult with adaptive margin
        """
        conf = detection["score"]
        cat_id = detection["category_id"]
        bx, by, bw, bh = detection["bbox"]

        # --- Filter by confidence threshold ---
        keep = conf >= self.conf_threshold

        # --- Get adaptive quantile ---
        # Priority: class-specific > confidence-bin > global
        qhat = self.global_qhat or 0.1

        # Class-conditional
        if cat_id in self.class_qhats:
            qhat = self.class_qhats[cat_id]

        # Confidence-adaptive (CQR): override with bin-specific quantile
        if self.conf_bin_edges is not None:
            bin_idx = np.searchsorted(self.conf_bin_edges[1:], conf)
            bin_idx = min(bin_idx, self.n_conf_bins - 1)
            if bin_idx in self.bin_qhats:
                qhat = self.bin_qhats[bin_idx]

        # --- Compute adaptive margin ---
        if self.size_normalize:
            # De-normalize: margin in pixels = qhat * object_size
            delta = np.array([
                qhat * max(bw, 1),  # dx
                qhat * max(bh, 1),  # dy
                qhat * max(bw, 1),  # dw
                qhat * max(bh, 1),  # dh
            ])
        else:
            delta = np.full(4, qhat)

        # --- Clamp margins to reasonable range ---
        # Never more than 50% of the box size, never less than 2px
        max_margin = np.array([bw * 0.5, bh * 0.5, bw * 0.5, bh * 0.5])
        delta = np.clip(delta, 2.0, np.maximum(max_margin, 2.0))

        # --- Prediction set (simplified APS using confidence) ---
        # With full softmax we'd include classes until cumsum >= 1-alpha
        # With only top-1 confidence, we estimate set size from confidence
        if conf >= 0.9:
            class_set = [cat_id]
        elif conf >= 0.7:
            class_set = [cat_id]  # singleton but less certain
        elif conf >= 0.5:
            class_set = [cat_id, -1]  # placeholder for "other possible class"
        else:
            class_set = [cat_id, -1, -2]  # larger uncertainty set

        # Nonconformity score for this detection
        score = 1.0 - conf if keep else 1.0

        return ConformalResult(
            keep=keep,
            conf_threshold=self.conf_threshold,
            box_delta=delta,
            box_delta_pixels=float(delta.mean()),
            class_set=class_set,
            class_set_size=len(class_set),
            confidence=conf,
            normalized_score=score,
        )

    def predict_batch(self, detections: List[Dict]) -> List[ConformalResult]:
        """Apply conformal prediction to a batch of detections."""
        return [self.predict(d) for d in detections]

    def get_summary(self) -> Dict:
        """Return calibration summary for reporting."""
        return {
            **self.cal_stats,
            "bin_qhats": {str(k): f"{v:.4f}" for k, v in self.bin_qhats.items()},
            "class_qhats_sample": {str(k): f"{v:.4f}" for k, v in
                                   list(self.class_qhats.items())[:10]},
        }

    # --- Internal matching ---

    def _match_all(self, preds, gt_by_image, iou_thresh):
        """Match predictions to GT via greedy IoU matching."""
        matched = []
        preds_by_img = defaultdict(list)
        for p in preds:
            preds_by_img[p["image_id"]].append(p)

        for img_id, img_preds in preds_by_img.items():
            gt_anns = gt_by_image.get(img_id, [])
            gt_used = [False] * len(gt_anns)

            for pred in sorted(img_preds, key=lambda x: -x["score"]):
                px, py, pw, ph = pred["bbox"]
                pred_xyxy = [px, py, px + pw, py + ph]
                best_iou, best_j = 0, -1

                for j, gt in enumerate(gt_anns):
                    if gt_used[j] or gt["category_id"] != pred["category_id"]:
                        continue
                    gx, gy, gw, gh = gt["bbox"]
                    gt_xyxy = [gx, gy, gx + gw, gy + gh]
                    iou = self._iou(pred_xyxy, gt_xyxy)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j

                is_correct = best_iou >= iou_thresh and best_j >= 0
                box_residual = np.zeros(4)
                box_size = np.array([max(pw, 1), max(ph, 1)])

                if is_correct:
                    gx, gy, gw, gh = gt_anns[best_j]["bbox"]
                    box_residual = np.abs(np.array([px, py, pw, ph]) - np.array([gx, gy, gw, gh]))
                    box_size = np.array([max(gw, 1), max(gh, 1)])
                    gt_used[best_j] = True

                matched.append({
                    "conf": pred["score"],
                    "correct": is_correct,
                    "iou": best_iou,
                    "box_residual": box_residual,
                    "box_size": box_size,
                    "category_id": pred["category_id"],
                    "image_id": pred["image_id"],
                })
        return matched

    @staticmethod
    def _iou(b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0
