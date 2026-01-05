# src/predict.py
# Image inference utilities for Satellite Image Classification (ResNet9)

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ---------------------------
# Model definition (must match training)
# ---------------------------

def conv_block(in_ch: int, out_ch: int, pool: bool = False) -> nn.Sequential:
    """Conv-BN-ReLU (+ optional MaxPool)."""
    layers: List[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    """Small residual CNN for 64x64 images."""
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.conv1 = conv_block(in_ch, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        return self.head(x)


# ---------------------------
# Inference helpers
# ---------------------------

@dataclass
class ModelBundle:
    model: nn.Module
    classes: List[str]
    image_size: int
    device: torch.device


def _strip_data_uri_prefix(s: str) -> str:
    # Supports strings like: "data:image/png;base64,AAAA..."
    if "," in s and s.strip().lower().startswith("data:"):
        return s.split(",", 1)[1]
    return s


def decode_base64_image(image_b64: str) -> Image.Image:
    """Decode base64 string into a PIL RGB image."""
    image_b64 = _strip_data_uri_prefix(image_b64)
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except Exception:
        # Some encoders include newlines/spaces; be lenient
        raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def build_infer_transform(image_size: int) -> transforms.Compose:
    """Inference transform (must match training's basic preprocessing)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def load_model_bundle(model_path: str = "model/model.pth") -> ModelBundle:
    """Load the trained model checkpoint and return an inference-ready bundle."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device)
    classes = ckpt.get("classes")
    image_size = int(ckpt.get("image_size", 64))

    if not classes or not isinstance(classes, (list, tuple)):
        raise ValueError("Checkpoint does not contain 'classes'. Re-train and save with classes.")

    model = ResNet9(in_ch=3, num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return ModelBundle(model=model, classes=list(classes), image_size=image_size, device=device)


@torch.no_grad()
def predict_image(bundle: ModelBundle, image: Image.Image) -> Tuple[str, Dict[str, float]]:
    """Return predicted class and probability distribution."""
    tfm = build_infer_transform(bundle.image_size)
    x = tfm(image).unsqueeze(0).to(bundle.device)  # [1,3,H,W]

    logits = bundle.model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

    prob_dict = {bundle.classes[i]: float(probs[i]) for i in range(len(bundle.classes))}
    pred_idx = int(probs.argmax())
    pred_class = bundle.classes[pred_idx]
    return pred_class, prob_dict
