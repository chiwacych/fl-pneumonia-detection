"""
Azure ML Online Endpoint — Scoring Script
FL-Pneumonia-Detection

This script is loaded by the Azure ML managed endpoint.
It receives a chest X-ray image (base64-encoded) and returns
a pneumonia/normal prediction with confidence scores.
"""

import os
import io
import json
import base64
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from opacus.validators import ModuleValidator

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────
CLASSES     = ["NORMAL", "PNEUMONIA"]
IMAGE_SIZE  = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_model = None


def _build_model():
    """Recreate the exact same architecture used during training."""
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_features, 2),
    )
    # Must apply ModuleValidator.fix() — same as training (BatchNorm→GroupNorm)
    model = ModuleValidator.fix(model)
    return model


def init():
    """Called once when the endpoint container starts."""
    global _model

    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR", "."),
        "fl_dp_eps10_iid_best.pth",
    )
    logger.info(f"Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")

    _model = _build_model()
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()

    logger.info(
        f"Model loaded. "
        f"Best training accuracy: {checkpoint.get('best_test_accuracy', 'N/A')}"
    )


def run(raw_data):
    """
    Called for each inference request.

    Expected input (JSON):
        {"image": "<base64-encoded JPEG/PNG>"}

    Returns (JSON):
        {
            "prediction": "PNEUMONIA" | "NORMAL",
            "confidence": 0.97,
            "scores": {"NORMAL": 0.03, "PNEUMONIA": 0.97}
        }
    """
    try:
        data = json.loads(raw_data)
        img_b64 = data["image"]

        # Decode base64 → PIL Image
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Preprocess
        tensor = _transform(img).unsqueeze(0)   # [1, 3, 224, 224]

        # Inference
        with torch.no_grad():
            logits = _model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze()

        pred_idx    = int(probs.argmax())
        prediction  = CLASSES[pred_idx]
        confidence  = float(probs[pred_idx])
        scores      = {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}

        return json.dumps({
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "scores":     {k: round(v, 4) for k, v in scores.items()},
        })

    except Exception as exc:
        logger.error(f"Inference error: {exc}")
        return json.dumps({"error": str(exc)})
