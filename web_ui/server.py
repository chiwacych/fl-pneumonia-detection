"""
Local demo server for FL-Pneumonia-Detection web UI.

Reads AZURE_SCORING_URI and AZURE_API_KEY from the project .env file,
serves index.html, and proxies inference requests to Azure ML so the
API key is never exposed in the browser.

Usage:
    cd FL-Pneumonia-Detection
    python web_ui/server.py
    # then open http://localhost:5000
"""

import os
import json
import urllib.request
import urllib.error
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

# ── Load .env from project root ───────────────────────────────
def load_dotenv(path: Path):
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

load_dotenv(Path(__file__).parent.parent / ".env")

SCORING_URI = os.environ.get("AZURE_SCORING_URI", "")
API_KEY     = os.environ.get("AZURE_API_KEY", "")

if not SCORING_URI or not API_KEY:
    print("⚠️  AZURE_SCORING_URI or AZURE_API_KEY not set in .env")
    print("   Run azure_deploy/deploy.py first, then restart this server.")

# ── Flask app ─────────────────────────────────────────────────
app = Flask(__name__, static_folder=Path(__file__).parent)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Proxy endpoint: receives {image: base64} from the browser,
    forwards it to Azure ML, returns the result.
    The Azure API key never reaches the browser.
    """
    if not SCORING_URI or not API_KEY:
        return jsonify({"error": "Azure endpoint not configured. Check .env file."}), 503

    payload = request.get_json(force=True)
    body    = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        SCORING_URI,
        data=body,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        return jsonify(result)
    except urllib.error.HTTPError as e:
        return jsonify({"error": f"Azure error {e.code}: {e.reason}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Starting demo server at http://localhost:5000")
    print(f"   Scoring URI : {SCORING_URI or '(not set)'}")
    app.run(host="0.0.0.0", port=5000, debug=False)
