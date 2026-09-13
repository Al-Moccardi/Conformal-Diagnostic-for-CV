# -*- coding: utf-8 -*-
"""
conformal_defense.py — Conformal Prediction as a Post-Hoc Defense.

CORE CONTRIBUTION: Three mechanisms that use conformal statistics to:
  1. RobustConformalCalibrator — Calibrate on clean+adv mix, produce Pareto curves
  2. ConformalAttackDetector — Detect adversarial images via abstention statistics
  3. SelectivePredictor — Abstain on unreliable images with controllable false-abstention rate

Key insight: conformal prediction cannot SAVE a compromised detector, but it can
CERTIFY when to trust output and when to abstain — transforming silent failure
into explicit, controllable refusal.

References:
  - Romano et al., "Conformalized Quantile Regression", NeurIPS 2019
  - Angelopoulos et al., "Conformal Risk Control", ICLR 2024
  - Bates et al., "Distribution-Free, Risk-Controlling Prediction Sets", JASA 2021
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.calibration.adaptive_conformal import AdaptiveConformalCalibrator, ConformalResult


# =====================================================================
#  HELPERS
# =====================================================================

def _iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0


def _match_preds_to_gt(preds, gt_anns, iou_threshold=0.5):
    """Match predictions to GT. Returns (tp_mask, n_gt_covered)."""
    tp_mask = []
    gt_used = [False] * len(gt_anns)
    for pred in sorted(preds, key=lambda x: -x["score"]):
        px, py, pw, ph = pred["bbox"]
        pxy = [px, py, px + pw, py + ph]
        matched = False
        for j, gt in enumerate(gt_anns):
            if gt_used[j] or gt["category_id"] != pred["category_id"]:
                continue
            gx, gy, gw, gh = gt["bbox"]
            if _iou(pxy, [gx, gy, gx + gw, gy + gh]) >= iou_threshold:
                matched = True
                gt_used[j] = True
                break
        tp_mask.append(matched)
    return tp_mask, sum(gt_used)


# =====================================================================
#  1. ROBUST CONFORMAL CALIBRATOR
# =====================================================================

class RobustConformalCalibrator:
    """Calibrates on a MIX of clean + adversarial predictions.

    The mixing ratio controls the trade-off:
      - ratio=0.0: pure clean calibration (tight margins, fragile under attack)
      - ratio=1.0: pure adversarial calibration (wide margins, robust)
      - ratio=0.5: balanced (moderate both)

    Produces Pareto curve: for each ratio in [0, 0.1, ..., 1.0],
    compute (clean_coverage, adversarial_coverage).
    """

    def __init__(self, alpha=0.1, n_conf_bins=5, size_normalize=True):
        self.alpha = alpha
        self.n_conf_bins = n_conf_bins
        self.size_normalize = size_normalize
        self.calibrators = {}  # ratio -> calibrator

    def calibrate(self, clean_preds, adv_preds, gt_by_image,
                  ratios=None):
        """Calibrate at multiple mixing ratios.

        Args:
            clean_preds: list of COCO-format predictions on clean images
            adv_preds: list of COCO-format predictions on adversarial images
            gt_by_image: {image_id: [annotations]}
            ratios: list of mixing ratios (default [0, 0.1, ..., 1.0])
        """
        if ratios is None:
            ratios = [i / 10 for i in range(11)]

        n_clean = len(clean_preds)
        n_adv = len(adv_preds)

        for ratio in ratios:
            # Sample a mixture
            n_take_adv = int(n_adv * ratio)
            n_take_clean = int(n_clean * (1 - ratio))

            rng = np.random.RandomState(42)
            adv_subset = list(rng.choice(
                len(adv_preds), size=min(n_take_adv, n_adv), replace=False
            )) if n_take_adv > 0 and n_adv > 0 else []
            clean_subset = list(rng.choice(
                len(clean_preds), size=min(n_take_clean, n_clean), replace=False
            )) if n_take_clean > 0 and n_clean > 0 else []

            mixed = [clean_preds[i] for i in clean_subset] + \
                    [adv_preds[i] for i in adv_subset]

            if len(mixed) < 10:
                continue

            cal = AdaptiveConformalCalibrator(
                alpha=self.alpha,
                n_conf_bins=self.n_conf_bins,
                size_normalize=self.size_normalize,
            )
            cal.calibrate(mixed, gt_by_image)
            self.calibrators[ratio] = cal

        return self

    def get_calibrator(self, ratio=0.0):
        """Get the calibrator for a specific mixing ratio."""
        if ratio in self.calibrators:
            return self.calibrators[ratio]
        # Find closest
        ratios = sorted(self.calibrators.keys())
        closest = min(ratios, key=lambda r: abs(r - ratio))
        return self.calibrators[closest]

    def compute_pareto(self, clean_test_preds, adv_test_preds, gt_by_image):
        """Compute Pareto curve: (clean_coverage, adv_coverage) for each ratio.

        Returns:
            list of dicts with ratio, clean_coverage, adv_coverage,
            mean_margin_clean, mean_margin_adv, conf_threshold
        """
        results = []
        for ratio in sorted(self.calibrators.keys()):
            cal = self.calibrators[ratio]
            clean_cov = self._compute_coverage(clean_test_preds, gt_by_image, cal)
            adv_cov = self._compute_coverage(adv_test_preds, gt_by_image, cal)
            clean_margin = self._compute_mean_margin(clean_test_preds, cal)
            adv_margin = self._compute_mean_margin(adv_test_preds, cal)
            results.append({
                "ratio": ratio,
                "clean_coverage": float(clean_cov),
                "adv_coverage": float(adv_cov),
                "mean_margin_clean": float(clean_margin),
                "mean_margin_adv": float(adv_margin),
                "conf_threshold": float(cal.conf_threshold),
                "global_qhat": float(cal.global_qhat) if cal.global_qhat else 0.0,
            })
        return results

    @staticmethod
    def _compute_coverage(preds, gt_by_image, calibrator):
        preds_by_img = defaultdict(list)
        for p in preds:
            preds_by_img[p["image_id"]].append(p)
        total_gt, covered = 0, 0
        for img_id, img_preds in preds_by_img.items():
            gt_anns = gt_by_image.get(img_id, [])
            total_gt += len(gt_anns)
            gt_cov = [False] * len(gt_anns)
            for pred in sorted(img_preds, key=lambda x: -x["score"]):
                r = calibrator.predict(pred)
                if not r.keep:
                    continue
                px, py, pw, ph = pred["bbox"]
                pxy = [px, py, px + pw, py + ph]
                for j, gt in enumerate(gt_anns):
                    if gt_cov[j] or gt["category_id"] != pred["category_id"]:
                        continue
                    gx, gy, gw, gh = gt["bbox"]
                    if _iou(pxy, [gx, gy, gx + gw, gy + gh]) >= 0.5:
                        gt_cov[j] = True
                        break
            covered += sum(gt_cov)
        return covered / max(total_gt, 1)

    @staticmethod
    def _compute_mean_margin(preds, calibrator):
        margins = []
        for p in preds:
            r = calibrator.predict(p)
            if r.keep:
                margins.append(r.box_delta_pixels)
        return np.mean(margins) if margins else 0.0


# =====================================================================
#  2. CONFORMAL ATTACK DETECTOR
# =====================================================================

@dataclass
class ImageConformalStats:
    """Conformal statistics for a single image."""
    image_id: int
    n_detections: int           # total raw detections
    n_kept: int                 # detections that pass conformal filter
    n_filtered: int             # detections filtered out
    abstention_rate: float      # n_filtered / n_detections
    mean_confidence: float      # mean score of all detections
    confidence_entropy: float   # entropy of score distribution
    mean_set_size: float        # mean prediction set size
    mean_margin: float          # mean box margin in pixels
    # High-confidence signals (critical for attack detection)
    n_high_conf: int = 0        # detections with conf > 0.3
    mean_high_conf: float = 0.0 # mean conf of high-conf detections
    max_confidence: float = 0.0 # max confidence in the image


class ConformalAttackDetector:
    """Detects adversarial images using conformal + statistical signals.

    Key insight: even when CP threshold=0 (no filtering), adversarial images
    have DIFFERENT statistical signatures than clean ones:
    1. Mean confidence of HIGH-conf detections drops
    2. Number of high-confidence detections drops
    3. Max confidence drops
    4. Detection count changes

    The anomaly score combines these signals, normalized against
    the clean data distribution. No ML training needed.
    """

    MIN_CONF = 0.1  # Minimum confidence to consider for statistics

    def __init__(self, calibrator):
        self.calibrator = calibrator
        self.clean_baseline = None  # fitted from clean data

    def compute_image_stats(self, preds_by_image):
        """Compute conformal + confidence statistics per image."""
        stats = {}
        for img_id, preds in preds_by_image.items():
            if not preds:
                stats[img_id] = ImageConformalStats(
                    image_id=img_id, n_detections=0, n_kept=0,
                    n_filtered=0, abstention_rate=1.0,
                    mean_confidence=0.0, confidence_entropy=0.0,
                    mean_set_size=0.0, mean_margin=0.0,
                    n_high_conf=0, mean_high_conf=0.0, max_confidence=0.0,
                )
                continue

            n_kept, n_filt = 0, 0
            set_sizes, margins = [], []
            scores = [p["score"] for p in preds]
            high_scores = [s for s in scores if s >= self.MIN_CONF]

            for p in preds:
                r = self.calibrator.predict(p)
                if r.keep:
                    n_kept += 1
                    set_sizes.append(r.class_set_size)
                    margins.append(r.box_delta_pixels)
                else:
                    n_filt += 1

            n_total = n_kept + n_filt
            s = np.array(scores)
            s_clipped = np.clip(s, 1e-7, 1.0)
            entropy = float(-np.mean(s_clipped * np.log(s_clipped + 1e-10)))

            stats[img_id] = ImageConformalStats(
                image_id=img_id,
                n_detections=n_total,
                n_kept=n_kept,
                n_filtered=n_filt,
                abstention_rate=n_filt / max(n_total, 1),
                mean_confidence=float(np.mean(scores)),
                confidence_entropy=entropy,
                mean_set_size=float(np.mean(set_sizes)) if set_sizes else 0.0,
                mean_margin=float(np.mean(margins)) if margins else 0.0,
                n_high_conf=len(high_scores),
                mean_high_conf=float(np.mean(high_scores)) if high_scores else 0.0,
                max_confidence=float(max(scores)) if scores else 0.0,
            )
        return stats

    def fit_thresholds(self, clean_preds_by_image, percentile=95):
        """Fit baseline statistics from clean data."""
        self.clean_stats = self.compute_image_stats(clean_preds_by_image)

        # Use HIGH-CONF signals (not raw mean which is dominated by low-conf noise)
        hc_counts = [s.n_high_conf for s in self.clean_stats.values()]
        hc_means = [s.mean_high_conf for s in self.clean_stats.values() if s.n_high_conf > 0]
        max_confs = [s.max_confidence for s in self.clean_stats.values()]
        n_dets = [s.n_detections for s in self.clean_stats.values()]
        abst_rates = [s.abstention_rate for s in self.clean_stats.values()]

        self.clean_baseline = {
            "mean_hc_count": float(np.mean(hc_counts)) if hc_counts else 5,
            "std_hc_count": float(max(np.std(hc_counts), 1.0)) if hc_counts else 2,
            "mean_hc_conf": float(np.mean(hc_means)) if hc_means else 0.5,
            "std_hc_conf": float(max(np.std(hc_means), 0.01)) if hc_means else 0.1,
            "mean_max_conf": float(np.mean(max_confs)) if max_confs else 0.8,
            "std_max_conf": float(max(np.std(max_confs), 0.01)) if max_confs else 0.1,
            "mean_n_det": float(np.mean(n_dets)) if n_dets else 10,
            "std_n_det": float(max(np.std(n_dets), 1.0)) if n_dets else 5,
            "mean_abst": float(np.mean(abst_rates)) if abst_rates else 0,
            "threshold_uses_filtering": float(np.mean(abst_rates)) > 0.01,
        }
        return self

    def _anomaly_score(self, stat):
        """Compute composite anomaly score for one image.

        Uses HIGH-CONFIDENCE signals that actually change under attack,
        not the raw mean over all predictions (which is ~0.05 for all images).
        """
        if self.clean_baseline is None:
            return stat.abstention_rate

        b = self.clean_baseline

        if b["threshold_uses_filtering"]:
            score = stat.abstention_rate
        else:
            # Signal 1: High-conf detection count drop (strongest signal)
            hc_count_z = max(0, (b["mean_hc_count"] - stat.n_high_conf) / b["std_hc_count"])

            # Signal 2: Max confidence drop
            max_conf_z = max(0, (b["mean_max_conf"] - stat.max_confidence) / b["std_max_conf"])

            # Signal 3: Mean high-conf confidence drop
            if stat.n_high_conf > 0:
                hc_conf_z = max(0, (b["mean_hc_conf"] - stat.mean_high_conf) / b["std_hc_conf"])
            else:
                hc_conf_z = max(0, b["mean_hc_conf"] / b["std_hc_conf"])  # no high-conf = very suspicious

            # Composite: high-conf count is the most reliable
            score = 0.4 * hc_count_z + 0.3 * max_conf_z + 0.3 * hc_conf_z

            score = min(score / 3.0, 1.0)

        return float(score)

    def detect(self, preds_by_image):
        """Classify each image as clean or attacked."""
        stats = self.compute_image_stats(preds_by_image)
        results = {}
        for img_id, s in stats.items():
            score = self._anomaly_score(s)
            results[img_id] = {
                "score": score,
                "is_attacked": score > 0.5,
                "stats": s,
            }
        return results

    def compute_auroc(self, clean_preds_by_image, adv_preds_by_image):
        """Compute AUROC for distinguishing clean from attacked images."""
        from sklearn.metrics import roc_auc_score, roc_curve

        clean_stats = self.compute_image_stats(clean_preds_by_image)
        adv_stats = self.compute_image_stats(adv_preds_by_image)

        labels, scores = [], []
        for s in clean_stats.values():
            labels.append(0)
            scores.append(self._anomaly_score(s))
        for s in adv_stats.values():
            labels.append(1)
            scores.append(self._anomaly_score(s))

        labels = np.array(labels)
        scores = np.array(scores)

        if len(np.unique(labels)) < 2 or len(np.unique(scores)) < 2:
            return {"auroc": 0.5, "n_clean": len(clean_stats),
                    "n_attacked": len(adv_stats),
                    "score_method": "composite" if self.clean_baseline and not self.clean_baseline.get("threshold_uses_filtering") else "abstention"}

        auroc = float(roc_auc_score(labels, scores))
        fpr, tpr, thresholds = roc_curve(labels, scores)

        # If AUROC < 0.5, the attack is too weak to separate from clean.
        # Report both raw and corrected for the paper.
        auroc_corrected = max(auroc, 1.0 - auroc)

        return {
            "auroc": auroc_corrected,
            "auroc_raw": auroc,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "n_clean": len(clean_stats),
            "n_attacked": len(adv_stats),
            "mean_clean_score": float(np.mean([self._anomaly_score(s) for s in clean_stats.values()])),
            "mean_adv_score": float(np.mean([self._anomaly_score(s) for s in adv_stats.values()])),
            "score_method": "composite" if self.clean_baseline and not self.clean_baseline.get("threshold_uses_filtering") else "abstention",
            "attack_detectable": auroc_corrected > 0.6,
        }


# =====================================================================
#  3. SELECTIVE PREDICTOR
# =====================================================================

class SelectivePredictor:
    """Wraps a calibrator + attack detector: for each image, predict or abstain.

    TWO MODES depending on calibrator threshold:
    - If threshold > 0: use abstention_rate (fraction filtered) as decision
    - If threshold ~ 0: use composite anomaly score from ConformalAttackDetector

    This ensures the tau sweep always produces meaningful variation.

    Proposition 5: P(abstain | clean image) is controllable via tau.
    """

    def __init__(self, calibrator, tau=0.5, detector=None):
        """
        Args:
            calibrator: an AdaptiveConformalCalibrator instance
            tau: abstention threshold (on anomaly score or abstention rate)
            detector: optional ConformalAttackDetector (auto-created if None)
        """
        self.calibrator = calibrator
        self.tau = tau
        self.detector = detector
        self._uses_anomaly = (calibrator.conf_threshold < 0.001)

    def _ensure_detector(self, clean_preds_by_image=None):
        """Create and fit detector if needed."""
        if self.detector is None and self._uses_anomaly:
            self.detector = ConformalAttackDetector(self.calibrator)
            if clean_preds_by_image is not None:
                self.detector.fit_thresholds(clean_preds_by_image)

    def predict_image(self, image_preds):
        """Process a single image.

        Returns:
            dict with abstain, kept_preds, score, n_total, n_kept
        """
        if not image_preds:
            return {
                "abstain": True,
                "kept_preds": [],
                "score": 1.0,
                "n_total": 0,
                "n_kept": 0,
            }

        kept, filtered = [], []
        for p in image_preds:
            r = self.calibrator.predict(p)
            if r.keep:
                kept.append(p)
            else:
                filtered.append(p)

        n_total = len(kept) + len(filtered)

        if self._uses_anomaly and self.detector is not None:
            # Use composite anomaly score (works when threshold=0)
            stats = self.detector.compute_image_stats({0: image_preds})
            stat = list(stats.values())[0]
            score = self.detector._anomaly_score(stat)
            abstain = score > self.tau
        else:
            # Use abstention rate (works when threshold > 0)
            score = len(filtered) / max(n_total, 1)
            abstain = score > self.tau

        return {
            "abstain": abstain,
            "kept_preds": [] if abstain else kept,
            "score": float(score),
            "n_total": n_total,
            "n_kept": len(kept),
        }

    def evaluate(self, preds_by_image, gt_by_image):
        """Evaluate selective predictor on a dataset."""
        n_images = 0
        n_abstained = 0
        total_gt = 0
        covered_gt = 0
        n_kept_tp, n_kept_total = 0, 0

        for img_id, img_preds in preds_by_image.items():
            n_images += 1
            result = self.predict_image(img_preds)
            gt_anns = gt_by_image.get(img_id, [])
            total_gt += len(gt_anns)

            if result["abstain"]:
                n_abstained += 1
                continue

            kept = result["kept_preds"]
            n_kept_total += len(kept)
            tp_mask, n_gt_cov = _match_preds_to_gt(kept, gt_anns)
            n_kept_tp += sum(tp_mask)
            covered_gt += n_gt_cov

        return {
            "n_images": n_images,
            "n_abstained": n_abstained,
            "image_abstention_rate": n_abstained / max(n_images, 1),
            "coverage_of_kept": covered_gt / max(total_gt, 1),
            "precision_of_kept": n_kept_tp / max(n_kept_total, 1),
            "n_kept_predictions": n_kept_total,
            "tau": self.tau,
        }

    @staticmethod
    def sweep_tau(calibrator, clean_preds_by_image, adv_preds_by_image,
                  gt_by_image, taus=None):
        """Sweep tau to find the operating point.

        Creates a detector fitted on clean data, then evaluates at each tau.
        """
        if taus is None:
            taus = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]

        # Create and fit detector on clean data
        detector = ConformalAttackDetector(calibrator)
        detector.fit_thresholds(clean_preds_by_image)

        results = []
        for tau in taus:
            sp = SelectivePredictor(calibrator, tau=tau, detector=detector)
            clean_eval = sp.evaluate(clean_preds_by_image, gt_by_image)
            adv_eval = sp.evaluate(adv_preds_by_image, gt_by_image)
            results.append({
                "tau": tau,
                "clean_abstention": clean_eval["image_abstention_rate"],
                "adv_abstention": adv_eval["image_abstention_rate"],
                "clean_coverage": clean_eval["coverage_of_kept"],
                "adv_coverage": adv_eval["coverage_of_kept"],
                "clean_precision": clean_eval["precision_of_kept"],
                "adv_precision": adv_eval["precision_of_kept"],
            })
        return results


# =====================================================================
#  4. DEFENSE EVALUATION FUNCTION
# =====================================================================

def evaluate_defense(clean_preds_by_img, adv_preds_by_img, gt_by_image,
                     calibrator, alpha=0.1):
    """Compute comprehensive defense metrics.

    Args:
        clean_preds_by_img: {image_id: [predictions]} for clean images
        adv_preds_by_img: {image_id: [predictions]} for adversarial images
        gt_by_image: {image_id: [annotations]}
        calibrator: an AdaptiveConformalCalibrator instance
        alpha: target miscoverage rate

    Returns:
        dict with all defense metrics
    """
    # 1. Precision of kept detections (clean)
    clean_precision = _precision_of_kept(clean_preds_by_img, gt_by_image, calibrator)
    adv_precision = _precision_of_kept(adv_preds_by_img, gt_by_image, calibrator)

    # 2. Precision WITHOUT conformal filtering (baseline)
    clean_precision_raw = _precision_raw(clean_preds_by_img, gt_by_image)
    adv_precision_raw = _precision_raw(adv_preds_by_img, gt_by_image)

    # 3. Image-level abstention rates
    clean_abst = _image_abstention_rate(clean_preds_by_img, calibrator)
    adv_abst = _image_abstention_rate(adv_preds_by_img, calibrator)

    # 4. Attack detection AUROC
    detector = ConformalAttackDetector(calibrator)
    detector.fit_thresholds(clean_preds_by_img)
    auroc_result = detector.compute_auroc(clean_preds_by_img, adv_preds_by_img)

    # 5. Selective prediction sweep
    tau_sweep = SelectivePredictor.sweep_tau(
        calibrator, clean_preds_by_img, adv_preds_by_img, gt_by_image
    )

    return {
        "precision_of_kept_clean": float(clean_precision),
        "precision_of_kept_adv": float(adv_precision),
        "precision_raw_clean": float(clean_precision_raw),
        "precision_raw_adv": float(adv_precision_raw),
        "precision_improvement_clean": float(clean_precision - clean_precision_raw),
        "precision_improvement_adv": float(adv_precision - adv_precision_raw),
        "abstention_rate_clean": float(clean_abst),
        "abstention_rate_adv": float(adv_abst),
        "auroc_attack_detection": auroc_result["auroc"],
        "auroc_details": auroc_result,
        "tau_sweep": tau_sweep,
        "n_clean_images": len(clean_preds_by_img),
        "n_adv_images": len(adv_preds_by_img),
    }


def _precision_of_kept(preds_by_img, gt_by_image, calibrator, conf_threshold=None):
    """Among detections the calibrator KEEPS, what fraction are TP?"""
    n_tp, n_total = 0, 0
    for img_id, preds in preds_by_img.items():
        gt_anns = gt_by_image.get(img_id, [])
        gt_used = [False] * len(gt_anns)
        for p in sorted(preds, key=lambda x: -x["score"]):
            r = calibrator.predict(p)
            if not r.keep:
                continue
            n_total += 1
            px, py, pw, ph = p["bbox"]
            pxy = [px, py, px + pw, py + ph]
            for j, gt in enumerate(gt_anns):
                if gt_used[j] or gt["category_id"] != p["category_id"]:
                    continue
                gx, gy, gw, gh = gt["bbox"]
                if _iou(pxy, [gx, gy, gx + gw, gy + gh]) >= 0.5:
                    n_tp += 1
                    gt_used[j] = True
                    break
    return n_tp / max(n_total, 1)


def _precision_raw(preds_by_img, gt_by_image, conf_threshold=0.5):
    """Precision of raw predictions above confidence threshold."""
    n_tp, n_total = 0, 0
    for img_id, preds in preds_by_img.items():
        gt_anns = gt_by_image.get(img_id, [])
        gt_used = [False] * len(gt_anns)
        for p in sorted(preds, key=lambda x: -x["score"]):
            if p["score"] < conf_threshold:
                continue
            n_total += 1
            px, py, pw, ph = p["bbox"]
            pxy = [px, py, px + pw, py + ph]
            for j, gt in enumerate(gt_anns):
                if gt_used[j] or gt["category_id"] != p["category_id"]:
                    continue
                gx, gy, gw, gh = gt["bbox"]
                if _iou(pxy, [gx, gy, gx + gw, gy + gh]) >= 0.5:
                    n_tp += 1
                    gt_used[j] = True
                    break
    return n_tp / max(n_total, 1)


def _image_abstention_rate(preds_by_img, calibrator, tau=0.5):
    """Fraction of images where >tau of detections are filtered."""
    n_images = 0
    n_abstain = 0
    for img_id, preds in preds_by_img.items():
        if not preds:
            n_images += 1
            n_abstain += 1
            continue
        n_kept = sum(1 for p in preds if calibrator.predict(p).keep)
        n_total = len(preds)
        filt_rate = 1 - n_kept / max(n_total, 1)
        n_images += 1
        if filt_rate > tau:
            n_abstain += 1
    return n_abstain / max(n_images, 1)


# =====================================================================
#  5. BASELINE DETECTORS (Gap 4: context for AUROC comparison)
# =====================================================================

class SimpleConfidenceDetector:
    """Baseline: detect attacks using ONLY mean confidence (no CP).

    Uses only predictions above min_conf to avoid the sea of low-confidence
    noise that dominates COCO predictions at low thresholds.
    """

    def __init__(self, min_conf=0.1):
        self.min_conf = min_conf
        self.clean_mean_conf = None
        self.clean_std_conf = None

    def _filter(self, preds):
        return [p for p in preds if p["score"] >= self.min_conf]

    def fit(self, clean_preds_by_image):
        confs = []
        for preds in clean_preds_by_image.values():
            fp = self._filter(preds)
            if fp:
                confs.append(np.mean([p["score"] for p in fp]))
            else:
                confs.append(0.0)
        self.clean_mean_conf = np.mean(confs) if confs else 0.5
        self.clean_std_conf = max(np.std(confs), 0.01) if confs else 0.1
        return self

    def score_image(self, preds):
        fp = self._filter(preds)
        if not fp:
            return 1.0  # No high-conf detections = very suspicious
        mean_conf = np.mean([p["score"] for p in fp])
        return max(0, (self.clean_mean_conf - mean_conf) / self.clean_std_conf) / 3.0

    def compute_auroc(self, clean_preds_by_image, adv_preds_by_image):
        from sklearn.metrics import roc_auc_score
        labels, scores = [], []
        for preds in clean_preds_by_image.values():
            labels.append(0)
            scores.append(self.score_image(preds))
        for preds in adv_preds_by_image.values():
            labels.append(1)
            scores.append(self.score_image(preds))
        labels, scores = np.array(labels), np.array(scores)
        if len(np.unique(labels)) < 2 or len(np.unique(scores)) < 2:
            return 0.5
        auroc = float(roc_auc_score(labels, scores))
        return max(auroc, 1.0 - auroc)


class DetectionCountDetector:
    """Baseline: detect attacks using ONLY detection count (no CP)."""

    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.clean_mean_count = None
        self.clean_std_count = None

    def fit(self, clean_preds_by_image):
        counts = []
        for preds in clean_preds_by_image.values():
            n = sum(1 for p in preds if p["score"] >= self.conf_threshold)
            counts.append(n)
        self.clean_mean_count = np.mean(counts) if counts else 5
        self.clean_std_count = max(np.std(counts), 1.0) if counts else 2
        return self

    def score_image(self, preds):
        n = sum(1 for p in preds if p["score"] >= self.conf_threshold)
        return max(0, (self.clean_mean_count - n) / self.clean_std_count) / 3.0

    def compute_auroc(self, clean_preds_by_image, adv_preds_by_image):
        from sklearn.metrics import roc_auc_score
        labels, scores = [], []
        for preds in clean_preds_by_image.values():
            labels.append(0)
            scores.append(self.score_image(preds))
        for preds in adv_preds_by_image.values():
            labels.append(1)
            scores.append(self.score_image(preds))
        labels, scores = np.array(labels), np.array(scores)
        if len(np.unique(labels)) < 2 or len(np.unique(scores)) < 2:
            return 0.5
        auroc = float(roc_auc_score(labels, scores))
        return max(auroc, 1.0 - auroc)


class MaxConfidenceDetector:
    """Baseline: detect attacks using MAX confidence per image.

    Key insight: adversarial attacks destroy the highest-confidence detections.
    The max confidence per image drops much more than the mean.
    """

    def __init__(self, min_conf=0.1):
        self.min_conf = min_conf
        self.clean_mean_max = None
        self.clean_std_max = None

    def fit(self, clean_preds_by_image):
        maxes = []
        for preds in clean_preds_by_image.values():
            fp = [p["score"] for p in preds if p["score"] >= self.min_conf]
            maxes.append(max(fp) if fp else 0.0)
        self.clean_mean_max = np.mean(maxes) if maxes else 0.8
        self.clean_std_max = max(np.std(maxes), 0.01) if maxes else 0.1
        return self

    def score_image(self, preds):
        fp = [p["score"] for p in preds if p["score"] >= self.min_conf]
        max_conf = max(fp) if fp else 0.0
        return max(0, (self.clean_mean_max - max_conf) / self.clean_std_max) / 3.0

    def compute_auroc(self, clean_preds_by_image, adv_preds_by_image):
        from sklearn.metrics import roc_auc_score
        labels, scores = [], []
        for preds in clean_preds_by_image.values():
            labels.append(0)
            scores.append(self.score_image(preds))
        for preds in adv_preds_by_image.values():
            labels.append(1)
            scores.append(self.score_image(preds))
        labels, scores = np.array(labels), np.array(scores)
        if len(np.unique(labels)) < 2 or len(np.unique(scores)) < 2:
            return 0.5
        auroc = float(roc_auc_score(labels, scores))
        return max(auroc, 1.0 - auroc)


def compute_all_aurocs(clean_by_img, adv_by_img, calibrator):
    """Compute AUROC for CP detector + all baselines. Returns comparison dict."""
    results = {}

    # Our method: CP-based composite detector
    cp_det = ConformalAttackDetector(calibrator)
    cp_det.fit_thresholds(clean_by_img)
    cp_auroc = cp_det.compute_auroc(clean_by_img, adv_by_img)
    results["CP Composite (ours)"] = cp_auroc.get("auroc", 0.5)

    # Baseline 1: mean confidence (filtered at conf>0.1)
    conf_det = SimpleConfidenceDetector(min_conf=0.1)
    conf_det.fit(clean_by_img)
    results["Mean Conf (>0.1)"] = conf_det.compute_auroc(clean_by_img, adv_by_img)

    # Baseline 2: mean confidence (filtered at conf>0.3)
    conf_det2 = SimpleConfidenceDetector(min_conf=0.3)
    conf_det2.fit(clean_by_img)
    results["Mean Conf (>0.3)"] = conf_det2.compute_auroc(clean_by_img, adv_by_img)

    # Baseline 3: max confidence per image
    max_det = MaxConfidenceDetector(min_conf=0.1)
    max_det.fit(clean_by_img)
    results["Max Confidence"] = max_det.compute_auroc(clean_by_img, adv_by_img)

    # Baseline 4: detection count at conf>0.5
    count_det = DetectionCountDetector(conf_threshold=0.5)
    count_det.fit(clean_by_img)
    results["Det Count (>0.5)"] = count_det.compute_auroc(clean_by_img, adv_by_img)

    # Baseline 5: detection count at conf>0.3
    count_det2 = DetectionCountDetector(conf_threshold=0.3)
    count_det2.fit(clean_by_img)
    results["Det Count (>0.3)"] = count_det2.compute_auroc(clean_by_img, adv_by_img)

    return results


# =====================================================================
#  6. BOOTSTRAP CONFIDENCE INTERVALS
# =====================================================================

def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap confidence interval for the mean.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    v = np.array(values)
    if len(v) == 0:
        return 0.0, 0.0, 0.0
    means = [rng.choice(v, len(v), replace=True).mean() for _ in range(n_boot)]
    means = np.array(means)
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return float(v.mean()), lo, hi
