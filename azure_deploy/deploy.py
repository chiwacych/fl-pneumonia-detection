"""
Azure ML Deployment Script
FL-Pneumonia-Detection

Run this ONCE after training on Kaggle:

    1. Download fl_dp_eps10_iid_best.pth from Kaggle output
    2. Place it in azure_deploy/
    3. Run:  python azure_deploy/deploy.py

Prerequisites:
    pip install azure-ai-ml azure-identity
    az login  (or set AZURE_CLIENT_ID / TENANT_ID / CLIENT_SECRET env vars)

You need:
    - Azure subscription ID
    - Azure ML workspace name + resource group
"""

import os
import time
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential

# ── Configuration — fill these in ────────────────────────────
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "YOUR_SUBSCRIPTION_ID")
RESOURCE_GROUP  = os.getenv("AZURE_RESOURCE_GROUP",  "YOUR_RESOURCE_GROUP")
WORKSPACE_NAME  = os.getenv("AZURE_WORKSPACE_NAME",  "YOUR_WORKSPACE_NAME")

ENDPOINT_NAME   = "fl-pneumonia-endpoint"
DEPLOYMENT_NAME = "fl-dp-v1"
MODEL_NAME      = "fl-dp-pneumonia"

SCRIPT_DIR      = Path(__file__).parent
CHECKPOINT_PATH = SCRIPT_DIR / "fl_dp_eps10_iid_best.pth"
# ─────────────────────────────────────────────────────────────


def main():
    print("🔗 Connecting to Azure ML workspace...")
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )
    print(f"✅ Connected: {WORKSPACE_NAME}")

    # ── 1. Register model ─────────────────────────────────────
    print("\n📦 Registering model...")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Download fl_dp_eps10_iid_best.pth from Kaggle output first."
        )

    model = ml_client.models.create_or_update(
        Model(
            name=MODEL_NAME,
            path=str(CHECKPOINT_PATH),
            description=(
                "EfficientNet-B0 trained with FedAvg FL + Opacus DP "
                "(ε=10, δ=1e-5) on chest X-ray pneumonia dataset. "
                "3-hospital simulation, IID partitioning."
            ),
            tags={
                "framework":  "pytorch",
                "task":       "binary-classification",
                "privacy":    "differential-privacy",
                "epsilon":    "10",
                "delta":      "1e-5",
            },
        )
    )
    print(f"✅ Model registered: {model.name} v{model.version}")

    # ── 2. Create environment ─────────────────────────────────
    print("\n🐍 Creating endpoint environment...")
    env = ml_client.environments.create_or_update(
        Environment(
            name="fl-pneumonia-env",
            conda_file=str(SCRIPT_DIR / "conda_env.yml"),
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
            description="PyTorch + Opacus environment for FL-Pneumonia inference",
        )
    )
    print(f"✅ Environment: {env.name}")

    # ── 3. Create endpoint ────────────────────────────────────
    print("\n🚀 Creating online endpoint (this takes ~2 min)...")
    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="Pneumonia detection from chest X-rays via FL+DP model",
        auth_mode="key",
        tags={"project": "fl-pneumonia-detection"},
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"✅ Endpoint created: {ENDPOINT_NAME}")

    # ── 4. Create deployment ──────────────────────────────────
    print("\n⚙️  Creating deployment (CPU, this takes ~5 min)...")
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=f"{MODEL_NAME}:{model.version}",
        environment=f"fl-pneumonia-env:{env.version}",
        code_configuration=CodeConfiguration(
            code=str(SCRIPT_DIR),
            scoring_script="score.py",
        ),
        instance_type="Standard_DS2_v2",   # CPU — within $100 student credit
        instance_count=1,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"✅ Deployment ready: {DEPLOYMENT_NAME}")

    # ── 5. Route all traffic to this deployment ───────────────
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    # ── 6. Print endpoint URI + key ───────────────────────────
    endpoint_obj = ml_client.online_endpoints.get(ENDPOINT_NAME)
    keys         = ml_client.online_endpoints.get_keys(ENDPOINT_NAME)

    scoring_uri = endpoint_obj.scoring_uri
    api_key     = keys.primary_key

    print("\n" + "="*60)
    print("DEPLOYMENT COMPLETE")
    print("="*60)
    print(f"Scoring URI : {scoring_uri}")
    print(f"API Key     : {api_key}")
    print("\nSave these — you need them in web_ui/index.html")
    print("="*60)

    # Write to a local file for convenience (gitignored via *.env)
    env_file = SCRIPT_DIR.parent / ".env"
    with open(env_file, "w") as f:
        f.write(f"AZURE_SCORING_URI={scoring_uri}\n")
        f.write(f"AZURE_API_KEY={api_key}\n")
    print(f"\n✅ Credentials saved to .env (gitignored)")


if __name__ == "__main__":
    main()
