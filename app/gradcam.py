"""Grad-CAM heatmap overlays for the demo (design.md → app/gradcam.py).

Grad-CAM on `model.features[-1]` (the last conv block of EfficientNet-B0) via
pytorch-grad-cam, for the top-1 class, overlaid on the uploaded image. Everything
happens in memory and returns PNG bytes; the server base64-encodes them into the
`heatmap_png_base64` field of the /predict contract.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn

MAX_OVERLAY_SIDE = 512  # phone photos are 12 MP; the overlay is for a phone screen, not a poster


class HeatmapGenerator:
    """Built once at startup; `overlay_png()` per request (the caller serialises access)."""

    def __init__(self, model: nn.Module):
        try:
            target_layer = model.features[-1]
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError("Grad-CAM needs a torchvision EfficientNet with `.features`") from exc
        self.model = model
        self.cam = GradCAM(model=model, target_layers=[target_layer])

    def heatmap(self, input_tensor: torch.Tensor, class_index: int) -> np.ndarray:
        """Grayscale CAM in [0, 1] at input resolution (H, W) for `class_index`."""
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError(f"expected one image as (1, 3, H, W), got {tuple(input_tensor.shape)}")
        with torch.enable_grad():  # Grad-CAM needs the backward pass even in eval mode
            cam = self.cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(int(class_index))])
        self.model.zero_grad(set_to_none=True)
        return cam[0].astype(np.float32)

    def overlay_png(self, input_tensor: torch.Tensor, class_index: int, original: Image.Image) -> bytes:
        """Heatmap for `class_index` blended onto `original` (downscaled to ≤ 512 px) → PNG bytes."""
        cam = self.heatmap(input_tensor, class_index)
        base = original.convert("RGB")
        base.thumbnail((MAX_OVERLAY_SIDE, MAX_OVERLAY_SIDE))  # keeps aspect ratio, never upsamples
        cam_img = Image.fromarray(np.clip(cam * 255, 0, 255).astype(np.uint8)).resize(base.size, Image.Resampling.BILINEAR)
        mask = np.asarray(cam_img, dtype=np.float32) / 255.0
        rgb = np.asarray(base, dtype=np.float32) / 255.0
        overlay = show_cam_on_image(rgb, mask, use_rgb=True, image_weight=0.5)
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG", optimize=True)
        return buf.getvalue()


def png_to_base64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")
