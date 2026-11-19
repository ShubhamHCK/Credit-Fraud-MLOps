from pathlib import Path
import json
import joblib

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

def save_bundle(name, model, metrics):
    joblib.dump(model, ARTIFACTS / f"{name}.joblib")
    (ARTIFACTS / f"{name}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

def load_bundle(name):
    model_path = ARTIFACTS / f"{name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run python -m src.train_{name} first."
        )
    return joblib.load(model_path)

def load_metrics(name):
    path = ARTIFACTS / f"{name}_metrics.json"
    if not path.exists():
        return {"accuracy": None, "precision": None, "recall": None, "roc_auc": None}
    return json.loads(path.read_text(encoding="utf-8"))
