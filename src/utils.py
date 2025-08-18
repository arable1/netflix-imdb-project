from pathlib import Path
import joblib
import json

# Anchor to project root: src/ → parent is project root
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"

def load_joblib(name: str):
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)

def load_json(name: str):
    path = ARTIFACTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, "r") as f:
        return json.load(f)