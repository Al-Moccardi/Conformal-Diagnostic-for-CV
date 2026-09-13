"""
conformal.py — Post-hoc conformal calibration methods for OD & segmentation.

Classification:  APS, RAPS, LAC, Temperature Scaling
Localization:    Box-Std, Box-CQR, Box-Ensemble
Segmentation:    Dilation-based margin, Pixel-wise LAC, CRC
Robust:          Adversarial calibration, Quantile inflation, RSCP
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.ndimage import binary_dilation


@dataclass
class ConformalConfig:
    alpha: float = 0.10           # miscoverage rate
    cal_ratio: float = 0.5        # calibration/test split
    class_conditional: bool = True
    seed: int = 42
    # Robust calibration
    adversarial_calibration: bool = False
    quantile_inflation: float = 0.0  # extra inflation for robustness
    smoothing_sigma: float = 0.0     # RSCP smoothing


# ═══════════════════════════════════════════════════════════════
#  CLASSIFICATION — Prediction Sets
# ═══════════════════════════════════════════════════════════════

class APS:
    """Adaptive Prediction Sets (Romano, Sesia & Candès, 2020)."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.qhat = None

    def calibrate(self, softmax_scores: np.ndarray, true_labels: np.ndarray):
        """
        Calibrate on held-out data.
        softmax_scores: (N, C) softmax probabilities
        true_labels: (N,) true class indices
        """
        n = len(true_labels)
        scores = []
        for i in range(n):
            probs = softmax_scores[i]
            sorted_idx = np.argsort(-probs)
            cumsum = np.cumsum(probs[sorted_idx])
            # Find position of true label in sorted order
            true_rank = np.where(sorted_idx == true_labels[i])[0][0]
            score = cumsum[true_rank]  # cumulative prob up to true label
            scores.append(score)

        scores = np.array(scores)
        self.qhat = np.quantile(scores, np.ceil((n + 1) * (1 - self.alpha)) / n,
                                interpolation='higher')
        return self

    def predict(self, softmax_scores: np.ndarray) -> List[List[int]]:
        """Return prediction sets for each sample."""
        prediction_sets = []
        for probs in softmax_scores:
            sorted_idx = np.argsort(-probs)
            cumsum = np.cumsum(probs[sorted_idx])
            # Include classes until cumulative prob >= qhat
            n_include = np.searchsorted(cumsum, self.qhat) + 1
            pred_set = sorted_idx[:n_include].tolist()
            prediction_sets.append(pred_set)
        return prediction_sets


class RAPS:
    """Regularized APS — penalizes large prediction sets."""

    def __init__(self, alpha: float = 0.1, lam: float = 0.01, k_reg: int = 3):
        self.alpha = alpha
        self.lam = lam
        self.k_reg = k_reg
        self.qhat = None

    def calibrate(self, softmax_scores: np.ndarray, true_labels: np.ndarray):
        n = len(true_labels)
        scores = []
        for i in range(n):
            probs = softmax_scores[i]
            sorted_idx = np.argsort(-probs)
            cumsum = np.cumsum(probs[sorted_idx])
            true_rank = np.where(sorted_idx == true_labels[i])[0][0]
            # Regularization penalty
            penalty = self.lam * max(0, true_rank + 1 - self.k_reg)
            score = cumsum[true_rank] + penalty
            scores.append(score)

        scores = np.array(scores)
        self.qhat = np.quantile(scores, np.ceil((n + 1) * (1 - self.alpha)) / n,
                                interpolation='higher')
        return self

    def predict(self, softmax_scores: np.ndarray) -> List[List[int]]:
        prediction_sets = []
        for probs in softmax_scores:
            sorted_idx = np.argsort(-probs)
            cumsum = np.cumsum(probs[sorted_idx])
            penalties = np.array([self.lam * max(0, j + 1 - self.k_reg) for j in range(len(probs))])
            reg_cumsum = cumsum + penalties
            n_include = np.searchsorted(reg_cumsum, self.qhat) + 1
            pred_set = sorted_idx[:n_include].tolist()
            prediction_sets.append(pred_set)
        return prediction_sets


class TemperatureScaling:
    """Temperature scaling for confidence calibration."""

    def __init__(self):
        self.temperature = 1.0

    def calibrate(self, logits: np.ndarray, true_labels: np.ndarray, lr=0.01, max_iter=100):
        """Optimize temperature via NLL minimization."""
        T = 1.5
        for _ in range(max_iter):
            scaled = logits / T
            exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            nll = -np.mean(np.log(probs[np.arange(len(true_labels)), true_labels] + 1e-8))
            # Gradient
            grad = np.mean(
                (probs[np.arange(len(true_labels)), true_labels] - 1) *
                logits[np.arange(len(true_labels)), true_labels] / (T ** 2)
            )
            T -= lr * grad
            T = max(0.1, min(10.0, T))
        self.temperature = T
        return self

    def scale(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)


# ═══════════════════════════════════════════════════════════════
#  LOCALIZATION — Bounding Box Intervals
# ═══════════════════════════════════════════════════════════════

class BoxStd:
    """Standard conformal regression on box coordinates."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.qhat = None  # (4,) quantile per coordinate

    def calibrate(self, pred_boxes: np.ndarray, gt_boxes: np.ndarray):
        """
        pred_boxes, gt_boxes: (N, 4) in [x1, y1, x2, y2]
        """
        residuals = np.abs(pred_boxes - gt_boxes)  # (N, 4)
        n = len(residuals)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.qhat = np.quantile(residuals, q_level, axis=0)  # (4,)
        return self

    def predict(self, pred_boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower_bounds, upper_bounds) for each box."""
        lower = pred_boxes - self.qhat
        upper = pred_boxes + self.qhat
        return lower, upper


class BoxCQR:
    """Conformalized Quantile Regression for adaptive box intervals."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.qhat = None

    def calibrate(self, pred_boxes: np.ndarray, gt_boxes: np.ndarray,
                  pred_lower: np.ndarray, pred_upper: np.ndarray):
        """CQR requires quantile regression predictions."""
        scores = np.maximum(pred_lower - gt_boxes, gt_boxes - pred_upper)  # (N, 4)
        scores = scores.max(axis=1)  # (N,) worst-case across coords
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.qhat = np.quantile(scores, q_level)
        return self

    def predict(self, pred_lower: np.ndarray, pred_upper: np.ndarray) -> Tuple:
        return pred_lower - self.qhat, pred_upper + self.qhat


class TwoStepCP:
    """Two-Step Conformal Prediction (Timans et al., ECCV 2024).
    Step 1: Conformal label set (class)
    Step 2: Conformal box interval conditioned on label set
    """

    def __init__(self, alpha_label: float = 0.05, alpha_box: float = 0.1):
        self.alpha_label = alpha_label
        self.alpha_box = alpha_box
        self.aps = APS(alpha_label)
        self.box_std = {}  # per-class box calibration

    def calibrate(self, softmax_scores, true_labels, pred_boxes, gt_boxes):
        # Step 1: calibrate label sets
        self.aps.calibrate(softmax_scores, true_labels)
        # Step 2: per-class box calibration
        unique_classes = np.unique(true_labels)
        for c in unique_classes:
            mask = true_labels == c
            if mask.sum() < 5:
                continue
            box_cal = BoxStd(self.alpha_box)
            box_cal.calibrate(pred_boxes[mask], gt_boxes[mask])
            self.box_std[c] = box_cal
        # Fallback (all classes)
        self.box_std_all = BoxStd(self.alpha_box)
        self.box_std_all.calibrate(pred_boxes, gt_boxes)
        return self

    def predict(self, softmax_scores, pred_boxes, pred_labels):
        label_sets = self.aps.predict(softmax_scores)
        box_intervals = []
        for i, (label_set, box, pred_label) in enumerate(zip(label_sets, pred_boxes, pred_labels)):
            cal = self.box_std.get(pred_label, self.box_std_all)
            lower, upper = cal.predict(box.reshape(1, 4))
            box_intervals.append({"lower": lower[0], "upper": upper[0], "label_set": label_set})
        return label_sets, box_intervals


# ═══════════════════════════════════════════════════════════════
#  SEGMENTATION — Mask Margins
# ═══════════════════════════════════════════════════════════════

class DilationMarginCP:
    """Conformal mask margin via morphological dilation."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.margin_px = None

    def calibrate(self, pred_masks: List[np.ndarray], gt_masks: List[np.ndarray],
                  max_margin: int = 30):
        """Find minimal margin such that coverage >= 1 - alpha."""
        n = len(pred_masks)
        for margin in range(0, max_margin + 1):
            struct = np.ones((2*margin+1, 2*margin+1))
            covered = 0
            for pm, gm in zip(pred_masks, gt_masks):
                dilated = binary_dilation(pm, struct).astype(np.uint8)
                if np.logical_and(dilated, gm).sum() / max(gm.sum(), 1) >= 0.95:
                    covered += 1
            coverage = covered / n
            if coverage >= 1 - self.alpha:
                self.margin_px = margin
                return self
        self.margin_px = max_margin
        return self

    def predict(self, pred_mask: np.ndarray) -> np.ndarray:
        struct = np.ones((2*self.margin_px+1, 2*self.margin_px+1))
        return binary_dilation(pred_mask, struct).astype(np.uint8)


class PixelwiseLAC:
    """Pixel-wise Least Ambiguous Set-Valued Classifier with CRC."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.threshold = None

    def calibrate(self, pixel_softmax: np.ndarray, pixel_labels: np.ndarray):
        """
        pixel_softmax: (N_pixels, C) softmax per pixel
        pixel_labels: (N_pixels,) true class per pixel
        """
        true_probs = pixel_softmax[np.arange(len(pixel_labels)), pixel_labels]
        scores = 1.0 - true_probs
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.threshold = 1.0 - np.quantile(scores, q_level)
        return self

    def predict(self, pixel_softmax: np.ndarray) -> np.ndarray:
        """Return prediction sets (binary mask per class): (N_pixels, C)."""
        return (pixel_softmax >= self.threshold).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
#  ROBUST CALIBRATION (under adversarial attack)
# ═══════════════════════════════════════════════════════════════

class AdversarialCalibration:
    """Calibrate using adversarially perturbed calibration data."""

    def __init__(self, base_method, attack_fn, alpha: float = 0.1):
        self.base = base_method
        self.attack_fn = attack_fn
        self.alpha = alpha

    def calibrate(self, images, labels, model):
        # Generate adversarial calibration data
        adv_images = self.attack_fn(images, {"labels": labels})
        # Get model predictions on adversarial data
        adv_outputs = model(adv_images)
        # Calibrate on adversarial predictions
        self.base.calibrate(adv_outputs.softmax, labels)
        return self


class QuantileInflation:
    """Inflate conformal quantile to maintain coverage under perturbation."""

    def __init__(self, base_method, epsilon: float, lipschitz_bound: float = 1.0):
        self.base = base_method
        self.epsilon = epsilon
        self.lipschitz = lipschitz_bound

    def calibrate(self, *args, **kwargs):
        self.base.calibrate(*args, **kwargs)
        # Inflate quantile
        inflation = self.lipschitz * self.epsilon
        if hasattr(self.base, 'qhat'):
            if isinstance(self.base.qhat, np.ndarray):
                self.base.qhat += inflation
            else:
                self.base.qhat += inflation
        return self


# ═══════════════════════════════════════════════════════════════
#  EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════

def compute_coverage(prediction_sets: List[List[int]], true_labels: np.ndarray) -> float:
    """Compute empirical coverage."""
    covered = sum(1 for ps, tl in zip(prediction_sets, true_labels) if tl in ps)
    return covered / len(true_labels) if len(true_labels) > 0 else 0.0

def compute_set_size(prediction_sets: List[List[int]]) -> float:
    """Mean prediction set size."""
    return np.mean([len(ps) for ps in prediction_sets])

def compute_box_coverage(box_intervals: List[Dict], gt_boxes: np.ndarray) -> float:
    """Coverage for box intervals."""
    covered = 0
    for bi, gt in zip(box_intervals, gt_boxes):
        if (gt >= bi["lower"]).all() and (gt <= bi["upper"]).all():
            covered += 1
    return covered / len(gt_boxes) if len(gt_boxes) > 0 else 0.0

def compute_box_interval_width(box_intervals: List[Dict]) -> float:
    """Mean interval width."""
    widths = [np.mean(bi["upper"] - bi["lower"]) for bi in box_intervals]
    return np.mean(widths)
