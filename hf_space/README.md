---
title: FL Pneumonia Detection
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
---

# FL Pneumonia Detection — Clinical Decision Support

Privacy-preserving chest X-ray screening powered by **Federated Learning** and **Differential Privacy**.

## How it works

- **3 simulated hospitals** each train locally on their own patient data — raw data never leaves the hospital
- **FedAvg aggregation** combines hospital model updates into a shared global model each round
- **Opacus Differential Privacy** (ε ≤ 10, δ = 10⁻⁵) guarantees individual patient records cannot be reverse-engineered from the model
- **EfficientNet-B0** backbone with GroupNorm layers for Opacus compatibility

## Setup

### Space secrets (Settings → Repository secrets)

| Secret | Purpose |
|---|---|
| `WANDB_API_KEY` | Download the trained model from WandB + log predictions |

### Alternative: upload the model directly

Place your `global_model.pth` (final WandB artifact) inside a `model_cache/` folder in the Space repo. The app will use it automatically without needing a WandB key.

## What gets logged to WandB

Every prediction is logged to the `fl-pneumonia-detection` project under a `demo-inference-*` run:

- Diagnosis label (NORMAL / PNEUMONIA)
- Confidence scores for both classes
- Risk level (LOW / MODERATE / HIGH)
- The uploaded X-ray image (with caption)
- Patient ID (if provided — purely for demo traceability)
- Timestamp

This creates a live dashboard you can show during presentations alongside the training metrics.

## Disclaimer

This is a **research prototype** built as a university final project. It is **not validated for clinical use** and must not be used to make medical decisions.
