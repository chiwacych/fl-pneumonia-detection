"""
HuggingFace Spaces — Gradio inference app for FL Pneumonia Detection.

Secrets to set in Space Settings → Repository secrets:
  WANDB_API_KEY   your W&B API key (model download + prediction logging)

Model source (tried in order):
  1. ./model_cache/global_model.pth  — upload directly to the Space repo
  2. WandB artifact  global-model:latest  — auto-downloaded if key is set
"""

import os
from datetime import datetime, timezone
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import wandb


# ─── Architecture (must match training notebooks exactly) ─────────────────────

def fix_for_opacus(module):
    """Replace BatchNorm2d → GroupNorm recursively (no torch.load involved)."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            nc = child.num_features
            ng = min(32, nc)
            while nc % ng != 0:
                ng -= 1
            gn = nn.GroupNorm(ng, nc, eps=child.eps, affine=child.affine)
            if child.affine:
                gn.weight.data.copy_(child.weight.data)
                gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            fix_for_opacus(child)
    return module


def create_model():
    m = models.efficientnet_b0(weights=None)
    nf = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(nf, 2))
    return fix_for_opacus(m)


# ─── Constants & globals ──────────────────────────────────────────────────────

WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
WANDB_PROJECT = "fl-pneumonia-detection"
WANDB_ENTITY  = "chiwacych"          # update if your W&B username differs
MODEL_CACHE   = "./model_cache"
CLASSES       = ["NORMAL", "PNEUMONIA"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_model:      nn.Module | None = None
_model_meta: dict             = {}
_wandb_run                    = None
_history:    deque            = deque(maxlen=100)   # in-memory audit log

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ─── Model loading ────────────────────────────────────────────────────────────

def _load_ckpt(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _build_model(state_dict: dict) -> nn.Module:
    m = create_model()
    m.load_state_dict(state_dict)
    m.eval()
    return m


def startup() -> str:
    """Load model + init WandB at module start. Returns a status string."""
    global _model, _model_meta, _wandb_run
    lines: list[str] = []

    # 1 ── Load model ──────────────────────────────────────────────────────────
    local_path = os.path.join(MODEL_CACHE, "global_model.pth")
    ckpt: dict | None = None

    if os.path.exists(local_path):
        ckpt = _load_ckpt(local_path)
        lines.append(f"✅ Model loaded from cache  (FL round {ckpt.get('round', '?')})")
    elif WANDB_API_KEY:
        try:
            wandb.login(key=WANDB_API_KEY, relogin=True)
            api      = wandb.Api()
            artifact = api.artifact(
                f"{WANDB_ENTITY}/{WANDB_PROJECT}/global-model:latest", type="model"
            )
            art_dir = artifact.download(root=MODEL_CACHE)
            ckpt    = _load_ckpt(os.path.join(art_dir, "global_model.pth"))
            lines.append(f"✅ Downloaded from WandB  (FL round {ckpt.get('round', '?')})")
        except Exception as exc:
            return f"❌ Model load failed: {exc}"
    else:
        return (
            "⚠️  No model found.\n"
            "Either upload global_model.pth to the Space repo "
            "or add WANDB_API_KEY as a Space secret."
        )

    _model = _build_model(ckpt["model_state_dict"])

    hosp   = ckpt.get("hospital_metadata", [])
    eps    = max((h["epsilon"] for h in hosp if h.get("epsilon")), default=None)
    _model_meta = {
        "round":    ckpt.get("round", "?"),
        "test_acc": ckpt.get("test_accuracy"),
        "epsilon":  eps,
    }

    # 2 ── Init WandB inference run ────────────────────────────────────────────
    if WANDB_API_KEY:
        try:
            _wandb_run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=f"demo-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
                job_type="inference",
                config={
                    "interface":   "gradio-hf-space",
                    "model_round": _model_meta["round"],
                    "test_acc":    _model_meta["test_acc"],
                    "epsilon":     _model_meta["epsilon"],
                },
            )
            lines.append(f"✅ WandB logging active → {_wandb_run.get_url()}")
        except Exception as exc:
            lines.append(f"⚠️  WandB unavailable: {exc}")

    return "\n".join(lines)


# ─── Inference helpers ────────────────────────────────────────────────────────

def _infer(pil_img: Image.Image) -> np.ndarray:
    t = _transform(pil_img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(_model(t), dim=1)[0].numpy()


def _risk_level(label: str, prob: float) -> str:
    if label == "NORMAL":
        return "LOW"
    return "HIGH" if prob >= 0.80 else "MODERATE"


def _table_rows() -> list[list]:
    return [list(r.values()) for r in _history]


# ─── Main Gradio handler ──────────────────────────────────────────────────────

def analyze(
    image:          Image.Image | None,
    patient_id:     str,
    patient_age:    float | None,
    clinical_notes: str,
):
    if _model is None:
        return None, "❌ Model not loaded — check System Status above.", _table_rows()
    if image is None:
        return None, "⚠️ Please upload a chest X-ray image.", _table_rows()

    probs      = _infer(image)
    pred_idx   = int(np.argmax(probs))
    pred_label = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])
    risk       = _risk_level(pred_label, confidence)
    ts         = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pid        = patient_id.strip() or "—"

    # ── Clinical summary ──────────────────────────────────────────────────────
    icon  = "🔴" if pred_label == "PNEUMONIA" else "🟢"
    meta  = _model_meta
    acc_s = f"{meta['test_acc']:.1%}" if meta.get("test_acc") else "N/A"
    eps_s = f"{meta['epsilon']:.2f}"  if meta.get("epsilon")  else "N/A"

    summary = f"""
### {icon} {pred_label} &nbsp;—&nbsp; {confidence:.1%} confidence

| Field | Value |
|---|---|
| **Risk level** | {risk} |
| **Normal probability** | {probs[0]:.1%} |
| **Pneumonia probability** | {probs[1]:.1%} |
| **Patient ID** | {pid} |
| **Timestamp** | {ts} |
| **FL round** | {meta['round']} |
| **Global test accuracy** | {acc_s} |
| **Privacy budget (ε)** | {eps_s} |

{"**Clinical notes:** " + clinical_notes if clinical_notes.strip() else ""}

> ⚠️ *Research prototype only. Not for clinical use.*
"""

    # ── WandB logging ─────────────────────────────────────────────────────────
    if _wandb_run is not None:
        try:
            wandb.log({
                "diagnosis":      pred_label,
                "confidence":     confidence,
                "prob_normal":    float(probs[0]),
                "prob_pneumonia": float(probs[1]),
                "risk_level":     risk,
                "patient_id":     pid,
                "age":            int(patient_age) if patient_age else None,
                "clinical_notes": clinical_notes or "",
                "x_ray": wandb.Image(
                    image,
                    caption=f"{pred_label} ({confidence:.1%}) | risk={risk} | {ts}",
                ),
            })
        except Exception:
            pass   # never crash the UI over logging

    # ── Audit log ─────────────────────────────────────────────────────────────
    _history.appendleft({
        "Time (UTC)":  ts,
        "Patient ID":  pid,
        "Diagnosis":   pred_label,
        "Confidence":  f"{confidence:.1%}",
        "Risk":        risk,
    })

    return (
        {"NORMAL": float(probs[0]), "PNEUMONIA": float(probs[1])},
        summary,
        _table_rows(),
    )


# ─── Build UI ─────────────────────────────────────────────────────────────────

STARTUP_STATUS = startup()
meta  = _model_meta
acc_s = f"{meta['test_acc']:.1%}" if meta.get("test_acc") else "N/A"
eps_s = f"{meta['epsilon']:.2f}"  if meta.get("epsilon")  else "N/A"

with gr.Blocks(
    title="FL Pneumonia Detection",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
) as demo:

    gr.Markdown("""
# 🏥 FL Pneumonia Detection — Clinical Decision Support

> **Privacy-preserving AI** trained across 3 simulated hospitals using FedAvg + Differential Privacy (ε ≤ 10, δ = 10⁻⁵).
> No raw patient data is ever shared — only locally computed model updates leave each hospital.
""")

    gr.Textbox(
        value=STARTUP_STATUS,
        label="System Status",
        interactive=False,
        lines=max(2, STARTUP_STATUS.count("\n") + 1),
    )

    with gr.Row():

        # ── Input panel ───────────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Patient Information")
            patient_id  = gr.Textbox(
                label="Patient ID",
                placeholder="e.g. PT-2024-001  (optional — anonymized in logs)",
            )
            patient_age = gr.Number(
                label="Age", minimum=0, maximum=120,
                precision=0, value=None,
            )

            gr.Markdown("### 🩻 Chest X-Ray")
            xray_input = gr.Image(label="Upload Chest X-Ray", type="pil")

            notes_input = gr.Textbox(
                label="Clinical Notes  (optional)",
                placeholder=(
                    "e.g. 45-year-old presenting with 3-day fever, "
                    "productive cough, and reduced breath sounds on left side…"
                ),
                lines=3,
            )

            analyze_btn = gr.Button("🔬 Run Screening", variant="primary", size="lg")

        # ── Result panel ──────────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Screening Result")
            label_out   = gr.Label(label="Diagnosis Confidence", num_top_classes=2)
            summary_out = gr.Markdown()

    gr.Markdown("### 📁 Prediction Audit Log")
    history_out = gr.Dataframe(
        headers=["Time (UTC)", "Patient ID", "Diagnosis", "Confidence", "Risk"],
        datatype=["str"] * 5,
        interactive=False,
        wrap=True,
    )

    with gr.Accordion("ℹ️ System Information", open=False):
        gr.Markdown(f"""
| Parameter | Value |
|---|---|
| **Architecture** | EfficientNet-B0 with GroupNorm (Opacus-compatible) |
| **Training** | Federated Learning — FedAvg across 3 hospitals |
| **Privacy** | Opacus Differential Privacy (ε ≤ 10, δ = 1×10⁻⁵) |
| **FL rounds completed** | {meta.get('round', 'N/A')} |
| **Global test accuracy** | {acc_s} |
| **Privacy budget used (ε)** | {eps_s} |
| **Dataset** | Chest X-Ray Images (Pneumonia) — Kaggle |
| **WandB project** | [fl-pneumonia-detection](https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}) |

⚠️ *Research prototype. Not intended for clinical use.*
""")

    analyze_btn.click(
        fn=analyze,
        inputs=[xray_input, patient_id, patient_age, notes_input],
        outputs=[label_out, summary_out, history_out],
    )

if __name__ == "__main__":
    demo.launch()
