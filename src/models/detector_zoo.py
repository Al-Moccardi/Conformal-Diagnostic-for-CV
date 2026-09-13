"""
detector_zoo.py — Unified wrappers for all detection/segmentation models.

Each wrapper normalizes output to a common format:
{
  "boxes":       np.ndarray (N, 4) in [x1, y1, x2, y2] absolute coords,
  "scores":      np.ndarray (N,)   confidence scores,
  "labels":      np.ndarray (N,)   class IDs (COCO 0-79),
  "class_logits": np.ndarray (N, C) raw logits (for calibration),
  "masks":       np.ndarray (N, H, W) binary masks (if segmentation),
  "mask_scores": np.ndarray (N, H, W) soft mask probabilities (if available),
}
"""

import time
import warnings
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Optional, Any

warnings.filterwarnings("ignore")


class BaseDetector:
    """Base class for all detectors."""

    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = device
        self.model = None
        self.tasks = []

    def load(self):
        raise NotImplementedError

    def predict(self, image_path: str, conf_threshold: float = 0.01) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def predict_batch(self, image_paths: List[str], conf_threshold: float = 0.01) -> List[Dict]:
        return [self.predict(p, conf_threshold) for p in image_paths]

    @staticmethod
    def _empty_result():
        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "scores": np.empty(0, dtype=np.float32),
            "labels": np.empty(0, dtype=np.int64),
            "label_is_coco_id": False,
            "class_logits": None,
            "masks": None,
            "mask_scores": None,
            "inference_time_ms": 0.0,
        }


# ═══════════════════════════════════════════════════════════════
#  ULTRALYTICS (YOLOv8, YOLOv11, RT-DETR)
# ═══════════════════════════════════════════════════════════════

class UltralyticsDetector(BaseDetector):
    """Wrapper for Ultralytics models (YOLO, RT-DETR)."""

    def __init__(self, name: str, weight_key: str, tasks: List[str], device: str = "cuda"):
        super().__init__(name, device)
        self.weight_key = weight_key
        self.tasks = tasks

    def load(self):
        from ultralytics import YOLO
        print(f"  Loading {self.name} ({self.weight_key})...")
        self.model = YOLO(self.weight_key)
        self.model.to(self.device)
        print(f"  [OK] {self.name} loaded on {self.device}")

    def predict(self, image_path: str, conf_threshold: float = 0.01) -> Dict:
        t0 = time.perf_counter()
        results = self.model(image_path, conf=conf_threshold, verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000

        boxes = results.boxes
        out = {
            "boxes": boxes.xyxy.cpu().numpy().astype(np.float32) if len(boxes) else np.empty((0, 4), dtype=np.float32),
            "scores": boxes.conf.cpu().numpy().astype(np.float32) if len(boxes) else np.empty(0, dtype=np.float32),
            "labels": boxes.cls.cpu().numpy().astype(np.int64) if len(boxes) else np.empty(0, dtype=np.int64),
            "class_logits": None,
            "masks": None,
            "mask_scores": None,
            "inference_time_ms": dt,
        }

        # Instance segmentation masks — resize to original image size
        if results.masks is not None and "instance_segmentation" in self.tasks:
            masks_raw = results.masks.data.cpu().numpy()  # (N, mask_h, mask_w) e.g. 160x160
            orig_h, orig_w = results.orig_shape  # original image dimensions
            n = masks_raw.shape[0]

            if n > 0:
                mask_h, mask_w = masks_raw.shape[1], masks_raw.shape[2]
                if mask_h != orig_h or mask_w != orig_w:
                    # Resize each mask to original image size using torch interpolate
                    masks_tensor = torch.from_numpy(masks_raw).unsqueeze(1).float()  # (N, 1, mh, mw)
                    masks_resized = torch.nn.functional.interpolate(
                        masks_tensor, size=(orig_h, orig_w), mode='bilinear', align_corners=False
                    ).squeeze(1).numpy()  # (N, orig_h, orig_w)
                else:
                    masks_resized = masks_raw

                out["masks"] = (masks_resized > 0.5).astype(np.uint8)
                out["mask_scores"] = masks_resized.astype(np.float32)

        return out


# ═══════════════════════════════════════════════════════════════
#  DETECTRON2 (Faster R-CNN, Mask R-CNN)
# ═══════════════════════════════════════════════════════════════

class Detectron2Detector(BaseDetector):
    """Wrapper for Detectron2 models."""

    def __init__(self, name: str, config_file: str, weight_key: str,
                 tasks: List[str], device: str = "cuda"):
        super().__init__(name, device)
        self.config_file = config_file
        self.weight_key = weight_key
        self.tasks = tasks

    def load(self):
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor

        print(f"  Loading {self.name} ({self.config_file})...")
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config_file))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.config_file)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
        cfg.MODEL.DEVICE = self.device
        self.model = DefaultPredictor(cfg)
        self.cfg = cfg
        print(f"  [OK] {self.name} loaded on {self.device}")

    def predict(self, image_path: str, conf_threshold: float = 0.01) -> Dict:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return self._empty_result()

        t0 = time.perf_counter()
        outputs = self.model(img)
        dt = (time.perf_counter() - t0) * 1000

        instances = outputs["instances"].to("cpu")
        mask_filt = instances.scores >= conf_threshold
        instances = instances[mask_filt]

        out = {
            "boxes": instances.pred_boxes.tensor.numpy().astype(np.float32) if len(instances) else np.empty((0, 4), dtype=np.float32),
            "scores": instances.scores.numpy().astype(np.float32) if len(instances) else np.empty(0, dtype=np.float32),
            "labels": instances.pred_classes.numpy().astype(np.int64) if len(instances) else np.empty(0, dtype=np.int64),
            "class_logits": None,
            "masks": None,
            "mask_scores": None,
            "inference_time_ms": dt,
        }

        if instances.has("pred_masks") and "instance_segmentation" in self.tasks:
            masks = instances.pred_masks.numpy()  # (N, H, W) bool
            out["masks"] = masks.astype(np.uint8)

        return out


# ═══════════════════════════════════════════════════════════════
#  HUGGINGFACE TRANSFORMERS (DETR, Mask2Former)
# ═══════════════════════════════════════════════════════════════

class HuggingFaceDetector(BaseDetector):
    """Wrapper for HuggingFace detection/segmentation models."""

    def __init__(self, name: str, model_id: str, tasks: List[str], device: str = "cuda"):
        super().__init__(name, device)
        self.model_id = model_id
        self.tasks = tasks

    def load(self):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        print(f"  Loading {self.name} ({self.model_id})...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)

        if "instance_segmentation" in self.tasks:
            from transformers import Mask2FormerForUniversalSegmentation
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_id)
        else:
            self.model = AutoModelForObjectDetection.from_pretrained(self.model_id)

        self.model.to(self.device)
        self.model.eval()

        # Build label mapping: HuggingFace internal ID -> COCO category ID
        # DETR/Mask2Former use id2label where keys are internal IDs
        self.hf_id_to_coco_catid = {}
        if hasattr(self.model.config, 'id2label'):
            # The HuggingFace DETR model's labels ARE COCO category IDs directly
            # (the post_process_object_detection returns them as-is)
            pass

        print(f"  [OK] {self.name} loaded on {self.device}")

    def predict(self, image_path: str, conf_threshold: float = 0.01) -> Dict:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        t0 = time.perf_counter()
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            try:
                outputs = self.model(**inputs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    return self._empty_result()
                raise

        dt = (time.perf_counter() - t0) * 1000

        # Free GPU memory after inference
        del inputs
        torch.cuda.empty_cache()

        if "instance_segmentation" in self.tasks:
            return self._process_mask2former(outputs, image, h, w, conf_threshold, dt)
        else:
            return self._process_detr(outputs, h, w, conf_threshold, dt)

    def _process_detr(self, outputs, h, w, conf_threshold, dt):
        target_sizes = torch.tensor([[h, w]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_threshold
        )[0]

        logits = outputs.logits.softmax(-1)[0].cpu().numpy()  # (N_queries, C+1)

        boxes = results["boxes"].cpu().numpy().astype(np.float32)
        scores = results["scores"].cpu().numpy().astype(np.float32)
        labels = results["labels"].cpu().numpy().astype(np.int64)

        # DETR labels are already COCO category IDs (1-90 with gaps), NOT 0-indexed
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "label_is_coco_id": True,  # signal to format_predictions_coco
            "class_logits": logits[:len(scores)] if len(scores) > 0 else None,
            "masks": None,
            "mask_scores": None,
            "inference_time_ms": dt,
        }

    def _process_mask2former(self, outputs, image, h, w, conf_threshold, dt):
        results = self.processor.post_process_instance_segmentation(
            outputs, target_sizes=[(h, w)], threshold=conf_threshold
        )[0]

        # Free outputs from GPU
        del outputs
        torch.cuda.empty_cache()

        segments = results["segments_info"]
        seg_map = results["segmentation"].cpu().numpy()  # (H, W) with segment IDs

        n = len(segments)
        if n == 0:
            return self._empty_result()

        boxes = np.empty((n, 4), dtype=np.float32)
        scores = np.empty(n, dtype=np.float32)
        labels = np.empty(n, dtype=np.int64)
        masks = np.empty((n, h, w), dtype=np.uint8)

        for i, seg in enumerate(segments):
            mask = (seg_map == seg["id"]).astype(np.uint8)
            masks[i] = mask
            ys, xs = np.where(mask)
            if len(ys) > 0:
                boxes[i] = [xs.min(), ys.min(), xs.max(), ys.max()]
            else:
                boxes[i] = [0, 0, 0, 0]
            scores[i] = seg["score"]
            labels[i] = seg["label_id"]

        # Mask2Former labels from post_process_instance_segmentation are 0-indexed
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "label_is_coco_id": False,
            "class_logits": None,
            "masks": masks,
            "mask_scores": None,
            "inference_time_ms": dt,
        }


# ═══════════════════════════════════════════════════════════════
#  FACTORY
# ═══════════════════════════════════════════════════════════════

def build_detector(name: str, config: dict, device: str = "cuda") -> BaseDetector:
    """Build a detector from config."""
    fw = config["framework"]
    tasks = config["tasks"]

    if fw == "ultralytics":
        return UltralyticsDetector(name, config["weight_key"], tasks, device)
    elif fw == "detectron2":
        return Detectron2Detector(name, config["config_file"], config["weight_key"], tasks, device)
    elif fw == "huggingface":
        return HuggingFaceDetector(name, config["model_id"], tasks, device)
    else:
        raise ValueError(f"Unknown framework: {fw}")


def build_all_detectors(model_zoo: dict, device: str = "cuda",
                        filter_tasks: Optional[List[str]] = None) -> Dict[str, BaseDetector]:
    """Build all detectors from the model zoo config."""
    detectors = {}
    for name, cfg in model_zoo.items():
        if filter_tasks:
            if not any(t in cfg["tasks"] for t in filter_tasks):
                continue
        detectors[name] = build_detector(name, cfg, device)
    return detectors
