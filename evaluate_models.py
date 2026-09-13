#!/usr/bin/env python3
"""
evaluate_models.py — Step 2: Model evaluation pipeline.

Loads all SOTA detectors/segmenters from the model zoo, runs inference on
COCO val2017, and computes comprehensive metrics for both OD and segmentation.

Usage:
    # Evaluate all models
    python evaluate_models.py

    # Evaluate specific models
    python evaluate_models.py --models yolov8x yolov8x-seg mask-rcnn-r101

    # Detection only
    python evaluate_models.py --task detection

    # Segmentation only
    python evaluate_models.py --task segmentation

    # Limit images (for quick testing)
    python evaluate_models.py --max-images 100

Output:
    results/evaluation/
    ├── detection_results.json         # Raw COCO-format predictions per model
    ├── segmentation_results.json
    ├── metrics_summary.json           # All metrics for all models
    ├── metrics_table.txt              # Pretty-printed comparison table
    ├── plots/
    │   ├── mAP_comparison.png
    │   ├── reliability_diagrams.png
    │   ├── size_stratified_ap.png
    │   ├── inference_time.png
    │   ├── mask_ap_comparison.png
    │   └── calibration_comparison.png
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import MODEL_ZOO, DETECTION_METRICS, SEGMENTATION_METRICS
from src.models.detector_zoo import build_detector, build_all_detectors
from src.evaluation.metrics import (
    run_coco_eval, format_predictions_coco,
    compute_calibration_metrics, compute_f1_at_threshold,
    compute_segmentation_metrics,
)


# ═══════════════════════════════════════════════════════════════
#  COCO DATA LOADER
# ═══════════════════════════════════════════════════════════════

class COCODataLoader:
    """Loads COCO val2017 annotations and provides image paths."""

    def __init__(self, coco_root: str):
        self.coco_root = coco_root
        self.ann_file = os.path.join(coco_root, "annotations", "instances_val2017.json")
        self.img_dir = os.path.join(coco_root, "val2017")

        assert os.path.exists(self.ann_file), f"Annotation file not found: {self.ann_file}"
        assert os.path.isdir(self.img_dir), f"Image directory not found: {self.img_dir}"

        with open(self.ann_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)

        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.categories = {cat["id"]: cat for cat in self.coco_data["categories"]}
        self.cat_ids = sorted(self.categories.keys())
        self.cat_id_to_idx = {cid: i for i, cid in enumerate(self.cat_ids)}

        # Group annotations by image
        self.gt_by_image = defaultdict(list)
        for ann in self.coco_data["annotations"]:
            self.gt_by_image[ann["image_id"]].append(ann)

        print(f"  COCO val2017: {len(self.images)} images, "
              f"{len(self.coco_data['annotations'])} annotations, "
              f"{len(self.categories)} categories")

    def get_image_paths(self, max_images: Optional[int] = None) -> List[dict]:
        """Return list of {image_id, file_path, width, height}."""
        items = []
        for img_id, img_info in self.images.items():
            path = os.path.join(self.img_dir, img_info["file_name"])
            if os.path.exists(path):
                items.append({
                    "image_id": img_id,
                    "file_path": path,
                    "width": img_info["width"],
                    "height": img_info["height"],
                })
        if max_images:
            items = items[:max_images]
        return items


# ═══════════════════════════════════════════════════════════════
#  EVALUATION ENGINE
# ═══════════════════════════════════════════════════════════════

class EvaluationEngine:
    """Runs all models and computes metrics."""

    def __init__(self, coco_loader: COCODataLoader, output_dir: str, device: str = "cuda"):
        self.loader = coco_loader
        self.output_dir = output_dir
        self.device = device
        self.results = {}

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)

    def evaluate_model(self, model_name: str, model_config: dict,
                       image_items: List[dict]) -> Dict:
        """Evaluate a single model on all images."""
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_name}")
        print(f"  Architecture: {model_config['architecture']}")
        print(f"  Tasks: {model_config['tasks']}")
        print(f"  Images: {len(image_items)}")
        print(f"{'='*60}")

        # Build and load model
        detector = build_detector(model_name, model_config, self.device)
        detector.load()

        # Run inference
        all_bbox_preds = []
        all_segm_preds = []
        inference_times = []

        for item in tqdm(image_items, desc=f"  {model_name}", ncols=80):
            result = detector.predict(item["file_path"], conf_threshold=0.001)
            inference_times.append(result["inference_time_ms"])

            # Format for COCO eval — bounding boxes
            bbox_preds = format_predictions_coco(
                item["image_id"], result, self.loader.cat_ids, iou_type="bbox"
            )
            all_bbox_preds.extend(bbox_preds)

            # Format for COCO eval — masks (if available)
            if result["masks"] is not None and "instance_segmentation" in model_config["tasks"]:
                segm_preds = format_predictions_coco(
                    item["image_id"], result, self.loader.cat_ids, iou_type="segm"
                )
                all_segm_preds.extend(segm_preds)

        # ── COCO mAP (bbox) ──
        print(f"\n  Computing detection metrics...")
        det_metrics = run_coco_eval(all_bbox_preds, self.loader.ann_file, iou_type="bbox")

        # ── COCO mAP (mask) ──
        seg_metrics = {}
        if all_segm_preds:
            print(f"  Computing segmentation metrics...")
            seg_metrics = run_coco_eval(all_segm_preds, self.loader.ann_file, iou_type="segm")
            # Rename keys for clarity
            seg_metrics = {f"mask_{k}": v for k, v in seg_metrics.items()}

        # ── Calibration metrics ──
        print(f"  Computing calibration metrics...")
        cal_metrics = compute_calibration_metrics(
            all_bbox_preds, dict(self.loader.gt_by_image), iou_threshold=0.5
        )

        # ── F1 ──
        f1_metrics = compute_f1_at_threshold(
            all_bbox_preds, dict(self.loader.gt_by_image),
            iou_threshold=0.5, conf_threshold=0.5
        )

        # ── Inference time ──
        time_metrics = {
            "inference_time_ms_mean": float(np.mean(inference_times)),
            "inference_time_ms_std": float(np.std(inference_times)),
            "inference_time_ms_p50": float(np.median(inference_times)),
            "inference_time_ms_p95": float(np.percentile(inference_times, 95)),
            "fps": float(1000.0 / np.mean(inference_times)) if np.mean(inference_times) > 0 else 0,
        }

        # Combine all metrics
        all_metrics = {
            "model": model_name,
            "architecture": model_config["architecture"],
            "family": model_config["family"],
            "tasks": model_config["tasks"],
            "n_images": len(image_items),
            "n_bbox_predictions": len(all_bbox_preds),
            "n_segm_predictions": len(all_segm_preds),
            **det_metrics,
            **seg_metrics,
            **{f"cal_{k}": v for k, v in cal_metrics.items() if k != "bin_data"},
            "calibration_bins": cal_metrics.get("bin_data", []),
            **f1_metrics,
            **time_metrics,
        }

        # Save per-model predictions
        pred_path = os.path.join(self.output_dir, "predictions", f"{model_name}_bbox.json")
        with open(pred_path, 'w', encoding='utf-8') as f:
            json.dump(all_bbox_preds, f)
        print(f"  Predictions saved → {pred_path}")

        # Print summary
        print(f"\n  ┌─ {model_name} Results ──────────────────────")
        print(f"  │ mAP@0.5         = {det_metrics.get('mAP@0.5', 0):.4f}")
        print(f"  │ mAP@[0.5:0.95]  = {det_metrics.get('mAP@[0.5:0.95]', 0):.4f}")
        print(f"  │ mAP_small       = {det_metrics.get('mAP_small', 0):.4f}")
        print(f"  │ mAP_medium      = {det_metrics.get('mAP_medium', 0):.4f}")
        print(f"  │ mAP_large       = {det_metrics.get('mAP_large', 0):.4f}")
        print(f"  │ AR@100          = {det_metrics.get('AR@100', 0):.4f}")
        if seg_metrics:
            print(f"  │ mask_mAP@0.5    = {seg_metrics.get('mask_mAP@0.5', 0):.4f}")
            print(f"  │ mask_mAP@[0.5:0.95] = {seg_metrics.get('mask_mAP@[0.5:0.95]', 0):.4f}")
        print(f"  │ ECE             = {cal_metrics.get('ECE', 0):.4f}")
        print(f"  │ Brier           = {cal_metrics.get('Brier', 0):.4f}")
        print(f"  │ F1@0.5          = {f1_metrics.get('F1@0.5', 0):.4f}")
        print(f"  │ Inference (ms)  = {time_metrics['inference_time_ms_mean']:.1f} ± {time_metrics['inference_time_ms_std']:.1f}")
        print(f"  │ FPS             = {time_metrics['fps']:.1f}")
        print(f"  └───────────────────────────────────────")

        # Free model from GPU before next evaluation
        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return all_metrics

    def evaluate_all(self, model_names: Optional[List[str]] = None,
                     max_images: Optional[int] = None,
                     filter_tasks: Optional[List[str]] = None) -> Dict:
        """Evaluate all (or selected) models."""
        image_items = self.loader.get_image_paths(max_images)
        print(f"\n  Total images to evaluate: {len(image_items)}")

        models_to_eval = {}
        for name, cfg in MODEL_ZOO.items():
            if model_names and name not in model_names:
                continue
            if filter_tasks and not any(t in cfg["tasks"] for t in filter_tasks):
                continue
            models_to_eval[name] = cfg

        print(f"  Models to evaluate: {list(models_to_eval.keys())}")

        all_results = {}
        for model_name, model_config in models_to_eval.items():
            try:
                metrics = self.evaluate_model(model_name, model_config, image_items)
                all_results[model_name] = metrics
            except Exception as e:
                print(f"\n  [ERROR] {model_name}: {e}")
                all_results[model_name] = {"model": model_name, "error": str(e)}
            finally:
                # Free VRAM between models
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()

        # Save summary
        summary_path = os.path.join(self.output_dir, "metrics_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Metrics summary → {summary_path}")

        # Generate comparison table
        self._print_comparison_table(all_results)

        # Generate plots
        self._generate_plots(all_results)

        return all_results

    def _print_comparison_table(self, results: Dict):
        """Print and save a comparison table."""
        from tabulate import tabulate

        # Detection table
        det_headers = ["Model", "Family", "mAP@0.5", "mAP@[.5:.95]", "mAP_S", "mAP_M", "mAP_L",
                       "AR@100", "ECE", "Brier", "F1@0.5", "FPS"]
        det_rows = []
        for name, m in results.items():
            if "error" in m:
                continue
            det_rows.append([
                name, m.get("family", ""),
                f"{m.get('mAP@0.5',0):.3f}", f"{m.get('mAP@[0.5:0.95]',0):.3f}",
                f"{m.get('mAP_small',0):.3f}", f"{m.get('mAP_medium',0):.3f}", f"{m.get('mAP_large',0):.3f}",
                f"{m.get('AR@100',0):.3f}",
                f"{m.get('cal_ECE',0):.4f}", f"{m.get('cal_Brier',0):.4f}",
                f"{m.get('F1@0.5',0):.3f}", f"{m.get('fps',0):.1f}",
            ])

        det_table = tabulate(det_rows, headers=det_headers, tablefmt="grid")
        print(f"\n{'='*100}")
        print("  OBJECT DETECTION RESULTS")
        print(f"{'='*100}")
        print(det_table)

        # Segmentation table (only models with mask predictions)
        seg_rows = []
        for name, m in results.items():
            if "error" in m or not m.get("mask_mAP@0.5"):
                continue
            seg_rows.append([
                name, m.get("family", ""),
                f"{m.get('mask_mAP@0.5',0):.3f}", f"{m.get('mask_mAP@[0.5:0.95]',0):.3f}",
                f"{m.get('mask_mAP_small',0):.3f}", f"{m.get('mask_mAP_medium',0):.3f}",
                f"{m.get('mask_mAP_large',0):.3f}", f"{m.get('mask_AR@100',0):.3f}",
            ])

        if seg_rows:
            seg_headers = ["Model", "Family", "mask_AP@0.5", "mask_AP@[.5:.95]",
                           "mask_AP_S", "mask_AP_M", "mask_AP_L", "mask_AR@100"]
            seg_table = tabulate(seg_rows, headers=seg_headers, tablefmt="grid")
            print(f"\n{'='*100}")
            print("  INSTANCE SEGMENTATION RESULTS")
            print(f"{'='*100}")
            print(seg_table)

        # Save tables to file
        table_path = os.path.join(self.output_dir, "metrics_table.txt")
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write("OBJECT DETECTION RESULTS\n")
            f.write("=" * 100 + "\n")
            f.write(det_table + "\n\n")
            if seg_rows:
                f.write("INSTANCE SEGMENTATION RESULTS\n")
                f.write("=" * 100 + "\n")
                f.write(seg_table + "\n")
        print(f"\n  Table saved → {table_path}")

    def _generate_plots(self, results: Dict):
        """Generate all evaluation plots."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        plot_dir = os.path.join(self.output_dir, "plots")
        PAL = {"primary": "#1B4F72", "secondary": "#2E86C1", "accent1": "#E74C3C",
               "accent2": "#27AE60", "accent3": "#F39C12", "accent4": "#8E44AD",
               "bg": "#FAFBFC", "text": "#2C3E50"}

        plt.rcParams.update({
            'figure.facecolor': PAL["bg"], 'axes.facecolor': '#FFFFFF',
            'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 10,
            'axes.titleweight': 'bold', 'figure.dpi': 150, 'savefig.dpi': 200,
        })

        valid = {k: v for k, v in results.items() if "error" not in v}
        if not valid:
            return

        names = list(valid.keys())
        families = [valid[n].get("family", "") for n in names]
        family_colors = {"YOLO": "#3498DB", "DETR": "#E74C3C", "R-CNN": "#27AE60",
                         "Mask2Former": "#8E44AD"}
        colors = [family_colors.get(f, "#999") for f in families]

        # ── Plot 1: mAP comparison ──
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax = axes[0]
        map50 = [valid[n].get("mAP@0.5", 0) for n in names]
        map5095 = [valid[n].get("mAP@[0.5:0.95]", 0) for n in names]
        x = np.arange(len(names)); w = 0.35
        ax.bar(x - w/2, map50, w, label="mAP@0.5", color=colors, alpha=0.8)
        ax.bar(x + w/2, map5095, w, label="mAP@[0.5:0.95]", color=colors, alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("mAP"); ax.set_title("Detection mAP comparison"); ax.legend()
        for i, (v1, v2) in enumerate(zip(map50, map5095)):
            ax.text(i - w/2, v1 + 0.01, f"{v1:.3f}", ha='center', fontsize=6)
            ax.text(i + w/2, v2 + 0.01, f"{v2:.3f}", ha='center', fontsize=6)

        # mAP vs FPS scatter
        ax = axes[1]
        fps_vals = [valid[n].get("fps", 0) for n in names]
        for i, n in enumerate(names):
            ax.scatter(fps_vals[i], map5095[i], c=colors[i], s=120, zorder=5, edgecolors='white', linewidth=1.5)
            ax.annotate(n, (fps_vals[i], map5095[i]), fontsize=7, textcoords="offset points",
                       xytext=(5, 5))
        ax.set_xlabel("FPS"); ax.set_ylabel("mAP@[0.5:0.95]")
        ax.set_title("Accuracy vs Speed trade-off")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "mAP_comparison.png")); plt.close()

        # ── Plot 2: Size-stratified AP ──
        fig, ax = plt.subplots(figsize=(12, 6))
        sizes = ["mAP_small", "mAP_medium", "mAP_large"]
        x = np.arange(len(names)); w = 0.25
        for i, sz in enumerate(sizes):
            vals = [valid[n].get(sz, 0) for n in names]
            ax.bar(x + i * w - w, vals, w, label=sz.replace("mAP_", "").capitalize(), alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("AP"); ax.set_title("Size-stratified AP (S / M / L)"); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "size_stratified_ap.png")); plt.close()

        # ── Plot 3: Calibration comparison ──
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        cal_metrics = ["cal_ECE", "cal_Brier", "cal_NLL"]
        cal_labels = ["ECE (↓)", "Brier Score (↓)", "NLL (↓)"]
        for ax, metric, label in zip(axes, cal_metrics, cal_labels):
            vals = [valid[n].get(metric, 0) for n in names]
            bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.8)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel(label); ax.set_title(label)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width()/2, v + 0.002, f"{v:.4f}",
                       ha='center', fontsize=7)
        plt.suptitle("Calibration metrics (pre-conformal, lower is better)",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "calibration_comparison.png")); plt.close()

        # ── Plot 4: Reliability diagrams ──
        n_models = len(valid)
        cols = min(4, n_models)
        rows = (n_models + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        for idx, (name, m) in enumerate(valid.items()):
            ax = axes[idx // cols, idx % cols]
            bins = m.get("calibration_bins", [])
            if bins:
                confs = [b["avg_conf"] for b in bins if b["count"] > 0]
                accs = [b["avg_acc"] for b in bins if b["count"] > 0]
                gaps = [b["gap"] for b in bins if b["count"] > 0]
                ax.bar(confs, accs, width=0.055, color=PAL["secondary"], alpha=0.7, label='Accuracy')
                ax.bar(confs, gaps, bottom=accs, width=0.055, color=PAL["accent1"], alpha=0.4, label='Gap')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f"{name}\nECE={m.get('cal_ECE',0):.4f}", fontsize=9)
            ax.set_xlabel("Confidence", fontsize=8); ax.set_ylabel("Accuracy", fontsize=8)
            if idx == 0:
                ax.legend(fontsize=7)

        # Hide empty axes
        for idx in range(n_models, rows * cols):
            axes[idx // cols, idx % cols].axis('off')

        plt.suptitle("Reliability Diagrams (pre-calibration)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "reliability_diagrams.png")); plt.close()

        # ── Plot 5: Mask AP (segmentation models only) ──
        seg_models = {n: m for n, m in valid.items() if m.get("mask_mAP@0.5")}
        if seg_models:
            fig, ax = plt.subplots(figsize=(10, 5))
            snames = list(seg_models.keys())
            mask50 = [seg_models[n].get("mask_mAP@0.5", 0) for n in snames]
            mask5095 = [seg_models[n].get("mask_mAP@[0.5:0.95]", 0) for n in snames]
            x = np.arange(len(snames)); w = 0.35
            scolors = [family_colors.get(valid[n].get("family", ""), "#999") for n in snames]
            ax.bar(x - w/2, mask50, w, label="mask_AP@0.5", color=scolors, alpha=0.8)
            ax.bar(x + w/2, mask5095, w, label="mask_AP@[0.5:0.95]", color=scolors, alpha=0.5)
            ax.set_xticks(x); ax.set_xticklabels(snames, rotation=45, ha='right')
            ax.set_ylabel("Mask AP"); ax.set_title("Instance Segmentation — Mask AP"); ax.legend()
            for i, (v1, v2) in enumerate(zip(mask50, mask5095)):
                ax.text(i - w/2, v1 + 0.005, f"{v1:.3f}", ha='center', fontsize=7)
                ax.text(i + w/2, v2 + 0.005, f"{v2:.3f}", ha='center', fontsize=7)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "mask_ap_comparison.png")); plt.close()

        # ── Plot 6: Inference time ──
        fig, ax = plt.subplots(figsize=(10, 5))
        times = [valid[n].get("inference_time_ms_mean", 0) for n in names]
        stds = [valid[n].get("inference_time_ms_std", 0) for n in names]
        bars = ax.bar(range(len(names)), times, yerr=stds, color=colors, alpha=0.8,
                      capsize=3, edgecolor='white')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Inference time (ms)")
        ax.set_title("Inference time per image (mean ± std)")
        for b, t, fps in zip(bars, times, fps_vals):
            ax.text(b.get_x() + b.get_width()/2, t + 2, f"{fps:.0f} FPS",
                   ha='center', fontsize=7, color=PAL["text"])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "inference_time.png")); plt.close()

        print(f"\n  All plots saved → {plot_dir}/")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate SOTA models on COCO val2017")
    parser.add_argument("--coco-root", type=str, default="data/coco",
                        help="Path to COCO dataset root")
    parser.add_argument("--output-dir", type=str, default="results/evaluation",
                        help="Output directory for results")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to evaluate (default: all)")
    parser.add_argument("--task", choices=["detection", "segmentation", "all"], default="all",
                        help="Filter by task")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images (for testing)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--conf-threshold", type=float, default=0.001,
                        help="Confidence threshold for predictions")
    args = parser.parse_args()

    print("=" * 65)
    print("  CONFORMAL CALIBRATION PIPELINE — Step 2: Model Evaluation")
    print("=" * 65)
    print(f"  Device: {args.device}")
    print(f"  COCO root: {args.coco_root}")
    print(f"  Output: {args.output_dir}")
    print()

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("  [WARN] CUDA not available, falling back to CPU")
        args.device = "cpu"
    if args.device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    print(f"\n  Loading COCO val2017...")
    loader = COCODataLoader(args.coco_root)

    # Task filter
    filter_tasks = None
    if args.task == "detection":
        filter_tasks = ["detection"]
    elif args.task == "segmentation":
        filter_tasks = ["instance_segmentation"]

    # Run evaluation
    engine = EvaluationEngine(loader, args.output_dir, args.device)
    results = engine.evaluate_all(
        model_names=args.models,
        max_images=args.max_images,
        filter_tasks=filter_tasks
    )

    print(f"\n{'='*65}")
    print(f"  EVALUATION COMPLETE")
    print(f"  Results: {args.output_dir}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
