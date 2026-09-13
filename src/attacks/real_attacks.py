"""
real_attacks.py — Real gradient-based adversarial attacks for object detection.

FIXES from v8:
  - YOLOv8/v11 output is (B, 4+C, N) NOT (B, N, 5+C) — no objectness channel
  - Seg models: ignore mask prototypes, use detection head only
  - RT-DETR via Ultralytics: uses different head, handle gracefully
  - HuggingFace: perturbation in normalized space, correct Linf measurement
  - All attacks: save ORIGINAL-RESOLUTION adversarial image (not 640x640)

White-box attacks:
  FGSM        — Goodfellow et al., ICLR 2015
  PGD-Linf    — Madry et al., ICLR 2018
  PGD-L2      — Madry et al., ICLR 2018
  C&W-Det     — Carlini & Wagner, IEEE S&P 2017
  DAG         — Wei et al., CVPR 2019
  TOG-V/M     — Chow et al., ECCV 2020
"""

import time
import warnings
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings('ignore')


@dataclass
class AttackResult:
    clean_image: np.ndarray       # (H, W, 3) uint8 — ORIGINAL resolution
    adv_image: np.ndarray         # (H, W, 3) uint8 — ORIGINAL resolution
    perturbation: np.ndarray      # (H, W, 3) float32 in [0,1] scale
    l_inf: float
    l_2: float
    attack_time_ms: float
    attack_name: str
    epsilon: float
    n_steps: int = 0


# =====================================================================
#  ULTRALYTICS WRAPPER — handles YOLOv8, YOLOv11, RT-DETR, -seg models
# =====================================================================

class UltralyticsAttackWrapper:
    """
    Wraps Ultralytics models for gradient attacks.

    Key: YOLOv8/v11 raw output is (B, 4+num_classes, N_anchors).
    There is NO separate objectness score (anchor-free design).
    Class scores at indices 4: are the detection confidences.
    """

    def __init__(self, model, device='cuda'):
        self.yolo = model
        self.device = device
        # Get the underlying torch model
        self.torch_model = model.model
        self.torch_model.eval()
        # Valid clamping range: Ultralytics uses pixel [0,1] space
        self.valid_min = 0.0
        self.valid_max = 1.0

    def preprocess(self, image_path: str, imgsz: int = 640) -> Tuple[torch.Tensor, dict]:
        """Load image, letterbox to imgsz, return tensor + metadata for unpadding."""
        img_orig = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img_orig.size

        scale = min(imgsz / orig_w, imgsz / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img_resized = img_orig.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
        pad_x, pad_y = (imgsz - new_w) // 2, (imgsz - new_h) // 2
        canvas.paste(img_resized, (pad_x, pad_y))

        img_tensor = torch.from_numpy(np.array(canvas)).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        meta = {"orig_img": np.array(img_orig), "orig_w": orig_w, "orig_h": orig_h,
                "scale": scale, "pad_x": pad_x, "pad_y": pad_y, "imgsz": imgsz,
                "new_w": new_w, "new_h": new_h}
        return img_tensor, meta

    def forward_loss(self, img_tensor: torch.Tensor, attack_mode: str = "untargeted") -> torch.Tensor:
        """
        Differentiable forward pass returning scalar loss.

        Handles YOLOv8/v11 output format: (B, 4+C, N) where C=80 for COCO.
        No objectness channel — class max is the detection confidence.

        FIX v10: Seg models return nested structures like [(det, proto)] or
        ((det, proto),). We recursively unwrap until we find the detection tensor.
        For seg models, det shape is (B, 116, N) = 4+80+32 mask coefficients.
        We only use the first 84 channels (4 bbox + 80 class).
        """
        with torch.enable_grad():
            raw = self.torch_model(img_tensor)

        # Recursively extract the detection tensor from nested output
        pred = self._extract_det_tensor(raw)

        if pred is None:
            # Fallback: use input norm as loss to get non-zero gradient
            return -(img_tensor ** 2).mean()

        # YOLOv8/v11: shape is (B, 4+C, N) — transpose to (B, N, 4+C)
        if pred.dim() == 3 and pred.shape[1] < pred.shape[2]:
            pred = pred.permute(0, 2, 1)  # (B, N, 4+C)

        n_feats = pred.shape[-1]

        if n_feats > 4:
            # For seg models: n_feats=116 (4+80+32). Only use class scores [4:84].
            # For det models: n_feats=84 (4+80). [4:] is already correct.
            n_classes = min(n_feats - 4, 80)
            cls_scores = pred[..., 4:4 + n_classes].sigmoid()
            det_conf = cls_scores.max(dim=-1).values  # (B, N)

            if attack_mode == "vanish":
                loss = -det_conf.sum()
            elif attack_mode == "mislabel":
                entropy = -(cls_scores * (cls_scores + 1e-8).log()).sum(dim=-1)
                loss = -entropy.mean()
            elif attack_mode == "untargeted":
                loss = -det_conf.sum() - 0.3 * (cls_scores * (cls_scores + 1e-8).log()).sum()
            else:
                loss = -det_conf.sum()
        else:
            loss = -pred.abs().mean()

        return loss

    def _extract_det_tensor(self, raw):
        """Recursively unwrap nested output to find the detection tensor.

        Seg models may return: (det, proto), [(det, proto)], ((det, proto),), etc.
        Detection tensor is a 3D tensor with shape (B, C, N) where C in [84, 116].
        """
        if isinstance(raw, torch.Tensor):
            if raw.dim() == 3:
                return raw
            return None

        if isinstance(raw, (list, tuple)):
            # Try each element — the detection tensor is the 3D one
            for item in raw:
                result = self._extract_det_tensor(item)
                if result is not None:
                    return result

        return None

    def tensor_to_image(self, img_tensor: torch.Tensor, meta: dict) -> np.ndarray:
        """Convert 640x640 model tensor back to ORIGINAL resolution image."""
        # Extract the content region (remove letterbox padding)
        t = img_tensor[0].detach().cpu().permute(1, 2, 0).numpy()  # (640, 640, 3)
        px, py = meta["pad_x"], meta["pad_y"]
        nw, nh = meta["new_w"], meta["new_h"]
        content = t[py:py+nh, px:px+nw]  # cropped content

        # Resize back to original resolution
        content_uint8 = (np.clip(content, 0, 1) * 255).astype(np.uint8)
        orig_img = Image.fromarray(content_uint8).resize(
            (meta["orig_w"], meta["orig_h"]), Image.BILINEAR)
        return np.array(orig_img)


# =====================================================================
#  HUGGINGFACE WRAPPER — DETR, with proper normalization handling
# =====================================================================

class HuggingFaceAttackWrapper:
    """
    Wraps HuggingFace DETR for gradient attacks.

    FIX v10: Provides valid_min/valid_max for correct clamping in normalized space.
    Clamping to [0,1] is WRONG for normalized tensors — it creates huge pixel-space
    perturbations. Instead, clamp to the range that maps back to [0,1] in pixel space.
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, model, processor, device='cuda'):
        self.model = model
        self.processor = processor
        self.device = device
        # Valid range in normalized space: (0 - mean)/std to (1 - mean)/std
        _mean = torch.tensor(self.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        _std = torch.tensor(self.IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.valid_min = (0.0 - _mean) / _std
        self.valid_max = (1.0 - _mean) / _std
        # FIX v10: Epsilon scaling factor.
        # Attacks define epsilon in pixel [0,1] space. For normalized-space models,
        # we must scale: eps_normalized = eps_pixel / min(std)
        # This ensures comparable pixel-space perturbation budgets across models.
        self.eps_scale = 1.0 / float(min(self.IMAGENET_STD))  # ~4.46

    def preprocess(self, image_path: str) -> Tuple[torch.Tensor, dict]:
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        meta = {"orig_img": np.array(img), "orig_w": orig_w, "orig_h": orig_h}
        return pixel_values, meta

    def forward_loss(self, pixel_values: torch.Tensor, attack_mode: str = "vanish") -> torch.Tensor:
        with torch.enable_grad():
            outputs = self.model(pixel_values=pixel_values)

        logits = outputs.logits
        probs = logits.softmax(-1)
        obj_probs = 1.0 - probs[..., -1]

        if attack_mode == "vanish":
            loss = -obj_probs.sum()
        elif attack_mode == "mislabel":
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
            loss = -entropy.mean()
        elif attack_mode == "untargeted":
            loss = -obj_probs.sum() - 0.3 * (probs * (probs + 1e-8).log()).sum()
        else:
            loss = -obj_probs.sum()

        return loss

    def tensor_to_image(self, pixel_values: torch.Tensor, meta: dict) -> np.ndarray:
        """Denormalize and resize back to original resolution."""
        t = pixel_values[0].detach().cpu().permute(1, 2, 0).numpy()
        # Denormalize
        img = t * self.IMAGENET_STD + self.IMAGENET_MEAN
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        # Resize to original
        orig = Image.fromarray(img).resize((meta["orig_w"], meta["orig_h"]), Image.BILINEAR)
        return np.array(orig)

    def pixel_epsilon(self, epsilon_normalized: float) -> float:
        """Convert epsilon from normalized space to pixel [0,1] space."""
        return epsilon_normalized * min(self.IMAGENET_STD)


# =====================================================================
#  ATTACK IMPLEMENTATIONS
# =====================================================================

class FGSM:
    """Fast Gradient Sign Method (Goodfellow et al., ICLR 2015)."""
    def __init__(self, wrapper, epsilon=8/255, attack_mode="untargeted"):
        self.w = wrapper
        self.epsilon = epsilon
        self.attack_mode = attack_mode

    def __call__(self, image_path: str) -> AttackResult:
        t0 = time.perf_counter()
        x, meta = self.w.preprocess(image_path)
        x.requires_grad_(True)
        loss = self.w.forward_loss(x, self.attack_mode)
        loss.backward()

        # FIX: If gradient is None (graph disconnected), use random noise
        if x.grad is not None:
            adv = x + self.epsilon * x.grad.sign()
        else:
            adv = x + self.epsilon * torch.sign(torch.randn_like(x))
        adv = torch.clamp(adv, self.w.valid_min, self.w.valid_max)

        clean_img = meta["orig_img"]
        adv_img = self.w.tensor_to_image(adv, meta)
        dt = (time.perf_counter() - t0) * 1000

        # L_inf always measured in pixel [0,1] space (denormalized images)
        pert = (adv_img.astype(np.float32) - clean_img.astype(np.float32)) / 255.0
        return AttackResult(
            clean_image=clean_img, adv_image=adv_img, perturbation=pert,
            l_inf=float(np.abs(pert).max()), l_2=float(np.sqrt((pert**2).mean())),
            attack_time_ms=dt, attack_name="FGSM", epsilon=self.epsilon, n_steps=1)


class PGD_Linf:
    """Projected Gradient Descent, L-inf (Madry et al., ICLR 2018)."""
    def __init__(self, wrapper, epsilon=8/255, step_size=2/255, n_steps=20,
                 attack_mode="untargeted", random_start=True):
        self.w = wrapper
        self.epsilon = epsilon
        self.step_size = step_size
        self.n_steps = n_steps
        self.attack_mode = attack_mode
        self.random_start = random_start

    def __call__(self, image_path: str) -> AttackResult:
        t0 = time.perf_counter()
        x, meta = self.w.preprocess(image_path)
        x_clean = x.detach().clone()
        adv = x_clean.clone()

        if self.random_start:
            adv = adv + torch.empty_like(adv).uniform_(-self.epsilon, self.epsilon)
            adv = torch.clamp(adv, self.w.valid_min, self.w.valid_max)

        for step in range(self.n_steps):
            adv = adv.detach().requires_grad_(True)
            try:
                loss = self.w.forward_loss(adv, self.attack_mode)
                loss.backward()
            except RuntimeError:
                # OOM during backward — use current adv and stop
                torch.cuda.empty_cache()
                break

            with torch.no_grad():
                # FIX: Handle None gradient (graph disconnected)
                if adv.grad is not None:
                    grad_sign = adv.grad.sign()
                else:
                    grad_sign = torch.sign(torch.randn_like(adv))
                adv = adv + self.step_size * grad_sign
                delta = torch.clamp(adv - x_clean, -self.epsilon, self.epsilon)
                adv = torch.clamp(x_clean + delta, self.w.valid_min, self.w.valid_max)

            # FIX: Free memory between steps (critical for DETR on 8GB VRAM)
            if step % 5 == 4 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        clean_img = meta["orig_img"]
        adv_img = self.w.tensor_to_image(adv.detach(), meta)
        dt = (time.perf_counter() - t0) * 1000

        # L_inf always measured in pixel [0,1] space
        pert = (adv_img.astype(np.float32) - clean_img.astype(np.float32)) / 255.0
        return AttackResult(
            clean_image=clean_img, adv_image=adv_img, perturbation=pert,
            l_inf=float(np.abs(pert).max()), l_2=float(np.sqrt((pert**2).mean())),
            attack_time_ms=dt, attack_name=f"PGD-{self.n_steps}-Linf",
            epsilon=self.epsilon, n_steps=self.n_steps)


class PGD_L2:
    """Projected Gradient Descent, L2 norm (Madry et al., ICLR 2018)."""
    def __init__(self, wrapper, epsilon=1.0, step_size=0.2, n_steps=20,
                 attack_mode="untargeted", random_start=True):
        self.w = wrapper
        self.epsilon = epsilon
        self.step_size = step_size
        self.n_steps = n_steps
        self.attack_mode = attack_mode
        self.random_start = random_start

    def __call__(self, image_path: str) -> AttackResult:
        t0 = time.perf_counter()
        x, meta = self.w.preprocess(image_path)
        x_clean = x.detach().clone()
        adv = x_clean.clone()

        if self.random_start:
            noise = torch.randn_like(adv)
            nf = noise.flatten(1)
            nn = nf.norm(2, dim=1, keepdim=True).clamp(min=1e-8)
            noise = (nf / nn * self.epsilon * torch.rand(1, device=adv.device)).view_as(adv)
            adv = torch.clamp(adv + noise, self.w.valid_min, self.w.valid_max)

        for step in range(self.n_steps):
            adv = adv.detach().requires_grad_(True)
            try:
                loss = self.w.forward_loss(adv, self.attack_mode)
                loss.backward()
            except RuntimeError:
                torch.cuda.empty_cache()
                break
            with torch.no_grad():
                if adv.grad is not None:
                    g = adv.grad.flatten(1)
                else:
                    g = torch.randn_like(adv).flatten(1)
                gn = g.norm(2, dim=1, keepdim=True).clamp(min=1e-8)
                adv = adv + self.step_size * (g / gn).view_as(adv)
                delta = (adv - x_clean).flatten(1)
                dn = delta.norm(2, dim=1, keepdim=True).clamp(min=1e-8)
                factor = torch.min(torch.ones_like(dn), self.epsilon / dn)
                adv = torch.clamp(x_clean + (delta * factor).view_as(adv), self.w.valid_min, self.w.valid_max)
            if step % 5 == 4 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        clean_img = meta["orig_img"]
        adv_img = self.w.tensor_to_image(adv.detach(), meta)
        dt = (time.perf_counter() - t0) * 1000

        pert = (adv_img.astype(np.float32) - clean_img.astype(np.float32)) / 255.0
        return AttackResult(
            clean_image=clean_img, adv_image=adv_img, perturbation=pert,
            l_inf=float(np.abs(pert).max()), l_2=float(np.sqrt((pert**2).mean())),
            attack_time_ms=dt, attack_name=f"PGD-{self.n_steps}-L2",
            epsilon=self.epsilon, n_steps=self.n_steps)


class CW_Detection:
    """C&W adapted for detection (Carlini & Wagner, IEEE S&P 2017)."""
    def __init__(self, wrapper, confidence=0.0, lr=0.01, n_steps=100,
                 binary_search_steps=3, attack_mode="vanish"):
        self.w = wrapper
        self.confidence = confidence
        self.lr = lr
        self.n_steps = n_steps
        self.binary_search_steps = binary_search_steps
        self.attack_mode = attack_mode

    def __call__(self, image_path: str) -> AttackResult:
        t0 = time.perf_counter()
        x, meta = self.w.preprocess(image_path)
        x_clean = x.detach().clone()

        x_tanh = torch.atanh(2 * x_clean.clamp(1e-6, 1 - 1e-6) - 1)
        w = x_tanh.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=self.lr)
        best_adv, best_l2 = x_clean.clone(), float('inf')
        c = 10.0

        for _ in range(self.binary_search_steps):
            for _ in range(self.n_steps):
                optimizer.zero_grad()
                adv = (torch.tanh(w) + 1) / 2
                l2_dist = ((adv - x_clean) ** 2).sum()
                det_loss = self.w.forward_loss(adv, self.attack_mode)
                loss = l2_dist + c * det_loss
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    if l2_dist.item() < best_l2:
                        best_l2 = l2_dist.item()
                        best_adv = ((torch.tanh(w) + 1) / 2).clone()
            c *= 2

        clean_img = meta["orig_img"]
        adv_img = self.w.tensor_to_image(best_adv.detach(), meta)
        dt = (time.perf_counter() - t0) * 1000

        pert = (adv_img.astype(np.float32) - clean_img.astype(np.float32)) / 255.0
        return AttackResult(
            clean_image=clean_img, adv_image=adv_img, perturbation=pert,
            l_inf=float(np.abs(pert).max()), l_2=float(np.sqrt((pert**2).mean())),
            attack_time_ms=dt, attack_name="C&W-Det",
            epsilon=float(np.sqrt(best_l2)), n_steps=self.n_steps)


class DAG:
    """Dense Adversary Generation (Wei et al., CVPR 2019)."""
    def __init__(self, wrapper, epsilon=8/255, step_size=1/255, n_steps=50, gamma=0.5):
        self.w = wrapper
        self.epsilon = epsilon
        self.step_size = step_size
        self.n_steps = n_steps
        self.gamma = gamma

    def __call__(self, image_path: str) -> AttackResult:
        t0 = time.perf_counter()
        x, meta = self.w.preprocess(image_path)
        x_clean = x.detach().clone()
        adv = x_clean.clone()

        for step in range(self.n_steps):
            adv = adv.detach().requires_grad_(True)
            try:
                lv = self.w.forward_loss(adv, "vanish")
                lm = self.w.forward_loss(adv, "mislabel")
                loss = self.gamma * lv + (1 - self.gamma) * lm
                loss.backward()
            except RuntimeError:
                torch.cuda.empty_cache()
                break
            with torch.no_grad():
                if adv.grad is not None:
                    step_dir = adv.grad.sign()
                else:
                    step_dir = torch.sign(torch.randn_like(adv))
                adv = adv + self.step_size * step_dir
                delta = torch.clamp(adv - x_clean, -self.epsilon, self.epsilon)
                adv = torch.clamp(x_clean + delta, self.w.valid_min, self.w.valid_max)
            if step % 5 == 4 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        clean_img = meta["orig_img"]
        adv_img = self.w.tensor_to_image(adv.detach(), meta)
        dt = (time.perf_counter() - t0) * 1000

        pert = (adv_img.astype(np.float32) - clean_img.astype(np.float32)) / 255.0
        return AttackResult(
            clean_image=clean_img, adv_image=adv_img, perturbation=pert,
            l_inf=float(np.abs(pert).max()), l_2=float(np.sqrt((pert**2).mean())),
            attack_time_ms=dt, attack_name="DAG",
            epsilon=self.epsilon, n_steps=self.n_steps)


class TOG:
    """Targeted Objectness Gradient (Chow et al., ECCV 2020)."""
    def __init__(self, wrapper, epsilon=8/255, step_size=1/255, n_steps=30, mode="vanish"):
        self.w = wrapper
        self.epsilon = epsilon
        self.step_size = step_size
        self.n_steps = n_steps
        self.mode = mode

    def __call__(self, image_path: str) -> AttackResult:
        t0 = time.perf_counter()
        x, meta = self.w.preprocess(image_path)
        x_clean = x.detach().clone()
        adv = x_clean.clone()

        for step in range(self.n_steps):
            adv = adv.detach().requires_grad_(True)
            try:
                loss = self.w.forward_loss(adv, self.mode)
                loss.backward()
            except RuntimeError:
                torch.cuda.empty_cache()
                break
            with torch.no_grad():
                if adv.grad is not None:
                    g = adv.grad
                else:
                    g = torch.randn_like(adv)
                gn = g.abs().amax(dim=(0, 1), keepdim=True).clamp(min=1e-8)
                adv = adv + self.step_size * (g / gn).sign()
                delta = torch.clamp(adv - x_clean, -self.epsilon, self.epsilon)
                adv = torch.clamp(x_clean + delta, self.w.valid_min, self.w.valid_max)
            if step % 5 == 4 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        clean_img = meta["orig_img"]
        adv_img = self.w.tensor_to_image(adv.detach(), meta)
        dt = (time.perf_counter() - t0) * 1000

        pert = (adv_img.astype(np.float32) - clean_img.astype(np.float32)) / 255.0
        name = f"TOG-{self.mode[0].upper()}"
        return AttackResult(
            clean_image=clean_img, adv_image=adv_img, perturbation=pert,
            l_inf=float(np.abs(pert).max()), l_2=float(np.sqrt((pert**2).mean())),
            attack_time_ms=dt, attack_name=name,
            epsilon=self.epsilon, n_steps=self.n_steps)


# =====================================================================
#  CATALOG & FACTORY
# =====================================================================

ATTACK_CATALOG = {
    "fgsm":   {"class": FGSM,     "type": "white-box", "norm": "L-inf",
               "ref": "Goodfellow et al., ICLR 2015",
               "defaults": {"epsilon": 8/255, "attack_mode": "untargeted"}},
    "pgd-10": {"class": PGD_Linf, "type": "white-box", "norm": "L-inf",
               "ref": "Madry et al., ICLR 2018",
               "defaults": {"epsilon": 8/255, "step_size": 2/255, "n_steps": 10}},
    "pgd-20": {"class": PGD_Linf, "type": "white-box", "norm": "L-inf",
               "ref": "Madry et al., ICLR 2018",
               "defaults": {"epsilon": 8/255, "step_size": 2/255, "n_steps": 20}},
    "pgd-50": {"class": PGD_Linf, "type": "white-box", "norm": "L-inf",
               "ref": "Madry et al., ICLR 2018",
               "defaults": {"epsilon": 8/255, "step_size": 1/255, "n_steps": 50}},
    "pgd-l2": {"class": PGD_L2,   "type": "white-box", "norm": "L2",
               "ref": "Madry et al., ICLR 2018",
               "defaults": {"epsilon": 1.0, "step_size": 0.2, "n_steps": 20}},
    "cw":     {"class": CW_Detection, "type": "white-box", "norm": "L2",
               "ref": "Carlini & Wagner, IEEE S&P 2017",
               "defaults": {"confidence": 0.0, "lr": 0.01, "n_steps": 50}},
    "dag":    {"class": DAG,      "type": "white-box", "norm": "L-inf",
               "ref": "Wei et al., CVPR 2019",
               "defaults": {"epsilon": 8/255, "step_size": 1/255, "n_steps": 30, "gamma": 0.5}},
    "tog-v":  {"class": TOG,      "type": "white-box", "norm": "L-inf",
               "ref": "Chow et al., ECCV 2020",
               "defaults": {"epsilon": 8/255, "step_size": 1/255, "n_steps": 20, "mode": "vanish"}},
    "tog-m":  {"class": TOG,      "type": "white-box", "norm": "L-inf",
               "ref": "Chow et al., ECCV 2020",
               "defaults": {"epsilon": 8/255, "step_size": 1/255, "n_steps": 20, "mode": "mislabel"}},
}


def build_attack(attack_name: str, wrapper, **kwargs):
    if attack_name not in ATTACK_CATALOG:
        raise ValueError(f"Unknown attack: {attack_name}. Available: {list(ATTACK_CATALOG.keys())}")
    info = ATTACK_CATALOG[attack_name]
    params = {**info["defaults"], **kwargs}

    # FIX v10: Auto-scale epsilon/step_size for normalized-space models (DETR).
    # Attack catalog defines epsilon in pixel [0,1] space. For models that operate
    # in ImageNet-normalized space, we scale so pixel-space perturbation is comparable.
    eps_scale = getattr(wrapper, 'eps_scale', 1.0)
    if eps_scale != 1.0:
        if "epsilon" in params and info["norm"] == "L-inf":
            params["epsilon"] = params["epsilon"] * eps_scale
        if "step_size" in params and info["norm"] == "L-inf":
            params["step_size"] = params["step_size"] * eps_scale

    return info["class"](wrapper, **params)


def build_all_attacks(wrapper, attack_names=None, **kwargs):
    if attack_names is None:
        attack_names = list(ATTACK_CATALOG.keys())
    return {n: build_attack(n, wrapper, **kwargs) for n in attack_names
            if n in ATTACK_CATALOG}


def get_attack_info():
    return {n: {"type": c["type"], "norm": c["norm"], "reference": c["ref"]}
            for n, c in ATTACK_CATALOG.items()}
