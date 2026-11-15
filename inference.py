"""Model loading and prediction. Confidence is returned ONLY if the saved model is flagged calibrated."""
import joblib
import numpy as np
import config as C


def load_model(path=C.MODELS_DIR / "pose_model.pkl"):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None
    except Exception as e:  # corrupt / incompatible sklearn version
        print(f"[inference] could not load model: {e}")
        return None


def predict(bundle, feats):
    """Returns (label 0/1, probability or None)."""
    x = np.asarray(feats, float).reshape(1, -1)
    if x.shape[1] != len(bundle["feature_names"]):
        raise ValueError("Feature length mismatch: retrain the model with the current code.")
    p = float(bundle["model"].predict_proba(x)[0][1])
    return int(p >= 0.5), (p if bundle.get("calibrated") else None)
