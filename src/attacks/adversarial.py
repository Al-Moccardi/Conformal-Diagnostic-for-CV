"""
adversarial.py — Adversarial attack implementations for object detection.

White-box: FGSM, PGD, C&W (adapted for detection loss)
Black-box: Square Attack, Transfer Attack
Common corruptions: Hendrycks benchmark

All attacks produce perturbation tensors that can be applied to images.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class AttackConfig:
    name: str = "pgd"
    norm: str = "linf"          # linf, l2, l1
    epsilon: float = 8/255      # perturbation budget
    step_size: float = 2/255    # PGD step
    n_steps: int = 20           # iterations
    targeted: bool = False
    attack_cls: bool = True     # attack classification head
    attack_box: bool = True     # attack localization head
    attack_obj: bool = True     # attack objectness head
    random_start: bool = True


# ═══════════════════════════════════════════════════════════════
#  WHITE-BOX ATTACKS
# ═══════════════════════════════════════════════════════════════

class FGSM:
    """Fast Gradient Sign Method (single-step)."""

    def __init__(self, model, epsilon=8/255, attack_cls=True, attack_box=True):
        self.model = model
        self.epsilon = epsilon
        self.attack_cls = attack_cls
        self.attack_box = attack_box

    def __call__(self, images: torch.Tensor, targets: Dict) -> torch.Tensor:
        images = images.clone().detach().requires_grad_(True)
        loss = self._compute_loss(images, targets)
        loss.backward()
        perturbation = self.epsilon * images.grad.sign()
        adv_images = torch.clamp(images + perturbation, 0, 1)
        return adv_images.detach()

    def _compute_loss(self, images, targets):
        outputs = self.model(images)
        loss = torch.tensor(0.0, device=images.device)
        if self.attack_cls and hasattr(outputs, 'logits'):
            loss += F.cross_entropy(outputs.logits, targets.get('labels', None))
        if self.attack_box and hasattr(outputs, 'pred_boxes'):
            loss += F.l1_loss(outputs.pred_boxes, targets.get('boxes', None))
        return loss


class PGD:
    """Projected Gradient Descent — iterative white-box attack."""

    def __init__(self, model, config: AttackConfig):
        self.model = model
        self.config = config

    def __call__(self, images: torch.Tensor, targets: Dict) -> torch.Tensor:
        c = self.config
        adv = images.clone().detach()

        if c.random_start:
            adv = adv + torch.empty_like(adv).uniform_(-c.epsilon, c.epsilon)
            adv = torch.clamp(adv, 0, 1)

        for step in range(c.n_steps):
            adv.requires_grad_(True)
            loss = self._compute_detection_loss(adv, targets)
            grad = torch.autograd.grad(loss, adv, retain_graph=False)[0]

            if c.norm == "linf":
                adv = adv.detach() + c.step_size * grad.sign()
                delta = torch.clamp(adv - images, -c.epsilon, c.epsilon)
                adv = torch.clamp(images + delta, 0, 1)
            elif c.norm == "l2":
                grad_norm = grad.flatten(1).norm(2, dim=1).view(-1,1,1,1).clamp(min=1e-8)
                adv = adv.detach() + c.step_size * grad / grad_norm
                delta = adv - images
                delta_norm = delta.flatten(1).norm(2, dim=1).view(-1,1,1,1).clamp(min=1e-8)
                delta = delta * torch.min(torch.ones_like(delta_norm), c.epsilon / delta_norm)
                adv = torch.clamp(images + delta, 0, 1)

        return adv.detach()

    def _compute_detection_loss(self, images, targets):
        outputs = self.model(images)
        loss = torch.tensor(0.0, device=images.device, requires_grad=True)
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = loss + outputs.loss
        elif hasattr(outputs, 'loss_dict'):
            for k, v in outputs.loss_dict.items():
                loss = loss + v
        return loss


class CarliniWagner:
    """Carlini & Wagner L2 attack adapted for detection."""

    def __init__(self, model, confidence=0.0, lr=0.01, n_steps=100, binary_search_steps=5):
        self.model = model
        self.confidence = confidence
        self.lr = lr
        self.n_steps = n_steps
        self.binary_search_steps = binary_search_steps

    def __call__(self, images: torch.Tensor, targets: Dict) -> torch.Tensor:
        w = torch.atanh(2 * images.clone().detach() - 1)
        w.requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=self.lr)
        c = 1.0
        best_adv = images.clone()
        best_l2 = float('inf')

        for _ in range(self.binary_search_steps):
            for step in range(self.n_steps):
                adv = (torch.tanh(w) + 1) / 2
                l2_dist = ((adv - images) ** 2).sum()
                det_loss = self._detection_loss(adv, targets)
                loss = l2_dist + c * det_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    curr_l2 = l2_dist.item()
                    if curr_l2 < best_l2 and det_loss.item() < 0:
                        best_l2 = curr_l2
                        best_adv = adv.clone()

            c *= 2 if best_l2 == float('inf') else c / 2

        return best_adv.detach()

    def _detection_loss(self, images, targets):
        outputs = self.model(images)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            labels = targets.get('labels')
            if labels is not None:
                real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
                other = (logits - 1e4 * F.one_hot(labels, logits.size(-1))).max(1)[0]
                return (other - real + self.confidence).clamp(min=0).mean()
        return torch.tensor(0.0, device=images.device)


# ═══════════════════════════════════════════════════════════════
#  BLACK-BOX ATTACKS
# ═══════════════════════════════════════════════════════════════

class SquareAttack:
    """Score-based black-box attack (Andriushchenko et al., 2020)."""

    def __init__(self, model, epsilon=8/255, n_queries=5000, norm="linf"):
        self.model = model
        self.epsilon = epsilon
        self.n_queries = n_queries
        self.norm = norm

    def __call__(self, images: torch.Tensor, targets: Dict) -> torch.Tensor:
        B, C, H, W = images.shape
        adv = images.clone()
        best_loss = self._eval_loss(adv, targets)

        for q in range(self.n_queries):
            # Random square perturbation
            s = max(1, int(np.sqrt(q + 1) / np.sqrt(self.n_queries) * min(H, W)))
            x0 = np.random.randint(0, W - s + 1)
            y0 = np.random.randint(0, H - s + 1)

            delta = torch.zeros_like(images)
            delta[:, :, y0:y0+s, x0:x0+s] = (2 * torch.randint(0, 2, (B, C, s, s),
                device=images.device).float() - 1) * self.epsilon

            candidate = torch.clamp(images + delta, 0, 1)
            cand_loss = self._eval_loss(candidate, targets)

            if cand_loss > best_loss:
                adv = candidate
                best_loss = cand_loss

        return adv.detach()

    def _eval_loss(self, images, targets):
        with torch.no_grad():
            outputs = self.model(images)
            if hasattr(outputs, 'loss'):
                return outputs.loss.item()
        return 0.0


class TransferAttack:
    """Transfer attack: generate adversarial on surrogate, test on victim."""

    def __init__(self, surrogate_model, epsilon=8/255, n_steps=50, step_size=2/255):
        self.surrogate = surrogate_model
        self.pgd = PGD(surrogate_model, AttackConfig(
            epsilon=epsilon, n_steps=n_steps, step_size=step_size
        ))

    def __call__(self, images: torch.Tensor, targets: Dict) -> torch.Tensor:
        return self.pgd(images, targets)


# ═══════════════════════════════════════════════════════════════
#  COMMON CORRUPTIONS (Hendrycks & Dietterich, 2019)
# ═══════════════════════════════════════════════════════════════

def apply_corruption(image: np.ndarray, corruption_type: str, severity: int = 3) -> np.ndarray:
    """Apply a common corruption to an image (H,W,3 uint8)."""
    img = image.astype(np.float32) / 255.0

    if corruption_type == "gaussian_noise":
        sigma = [0.04, 0.06, 0.08, 0.10, 0.12][severity - 1]
        img = img + np.random.normal(0, sigma, img.shape)
    elif corruption_type == "shot_noise":
        lam = [40, 25, 15, 10, 5][severity - 1]
        img = np.random.poisson(img * lam) / lam
    elif corruption_type == "impulse_noise":
        p = [0.01, 0.02, 0.05, 0.08, 0.15][severity - 1]
        mask = np.random.random(img.shape[:2]) < p
        img[mask] = np.random.choice([0.0, 1.0], size=mask.sum())
    elif corruption_type == "gaussian_blur":
        from scipy.ndimage import gaussian_filter
        sigma = [0.5, 0.75, 1.0, 1.5, 2.0][severity - 1]
        img = gaussian_filter(img, sigma=[sigma, sigma, 0])
    elif corruption_type == "fog":
        t = [0.9, 0.8, 0.65, 0.5, 0.35][severity - 1]
        img = img * t + (1 - t) * 0.8
    elif corruption_type == "brightness":
        factor = [1.1, 1.2, 1.3, 1.5, 1.8][severity - 1]
        img = img * factor
    elif corruption_type == "contrast":
        factor = [0.8, 0.65, 0.5, 0.35, 0.2][severity - 1]
        mean = img.mean()
        img = (img - mean) * factor + mean
    elif corruption_type == "jpeg_compression":
        import io
        from PIL import Image
        quality = [80, 65, 50, 35, 15][severity - 1]
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        img = np.array(Image.open(buf)).astype(np.float32) / 255.0
    elif corruption_type == "pixelate":
        factor = [0.9, 0.7, 0.5, 0.35, 0.2][severity - 1]
        h, w = img.shape[:2]
        sh, sw = max(1, int(h*factor)), max(1, int(w*factor))
        from PIL import Image
        pil = Image.fromarray((np.clip(img, 0, 1)*255).astype(np.uint8))
        small = pil.resize((sw, sh), Image.NEAREST)
        img = np.array(small.resize((w, h), Image.NEAREST)).astype(np.float32) / 255.0

    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


CORRUPTION_TYPES = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["gaussian_blur"],
    "weather": ["fog", "brightness"],
    "digital": ["contrast", "jpeg_compression", "pixelate"],
}

ATTACK_REGISTRY = {
    "fgsm":     {"type": "white-box", "norm": "L∞", "steps": 1},
    "pgd-20":   {"type": "white-box", "norm": "L∞", "steps": 20},
    "pgd-50":   {"type": "white-box", "norm": "L∞", "steps": 50},
    "cw":       {"type": "white-box", "norm": "L2", "steps": 100},
    "square":   {"type": "black-box", "norm": "L∞", "queries": 5000},
    "transfer": {"type": "black-box", "norm": "L∞", "surrogate": "yolov8x"},
}
