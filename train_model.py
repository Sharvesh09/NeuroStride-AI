"""Train + evaluate a Random Forest with subject/recording-wise validation (no window leakage).
  python train_model.py --mode daphnet     (accelerometer baseline; needs Daphnet files)
  python train_model.py --mode pose        (webcam model; needs recordings from collect_data.py)"""
import argparse, json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix,
                             ConfusionMatrixDisplay)
import config as C
import preprocess as P
import feature_extraction as F


def make_rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                  n_jobs=-1, random_state=C.RANDOM_STATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["daphnet", "pose"], required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=None, help="where to save model/graphs/report (default: project folders)")
    a = ap.parse_args()
    models_dir = C.MODELS_DIR if a.out_dir is None else __import__("pathlib").Path(a.out_dir)
    graphs_dir = C.GRAPHS_DIR if a.out_dir is None else models_dir
    reports_dir = C.REPORTS_DIR if a.out_dir is None else models_dir
    for d in (models_dir, graphs_dir, reports_dir): d.mkdir(parents=True, exist_ok=True)

    if a.mode == "daphnet":
        X, y, g = P.build_daphnet_windows(a.data_dir or C.DAPHNET_DIR); names = F.accel_feature_names()
    else:
        X, y, g = P.build_pose_windows(a.data_dir or C.POSE_DIR); names = F.pose_feature_names()
    groups = np.unique(g)
    print(f"Windows: {len(y)} | freeze-like: {int(y.sum())} | normal: {int((y == 0).sum())} | groups: {len(groups)}")
    if len(groups) < 3 or y.sum() < 5 or (y == 0).sum() < 5:
        raise SystemExit("Not enough data: need >=3 participants/recordings and >=5 windows of each class.")

    cv = LeaveOneGroupOut() if len(groups) <= 15 else GroupKFold(n_splits=5)
    y_true, y_pred = [], []
    for tr, te in cv.split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            print("  skipped a fold (training part has one class)"); continue
        m = make_rf().fit(X[tr], y[tr])
        y_true += list(y[te]); y_pred += list(m.predict(X[te]))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], zero_division=0)
    metrics = {"mode": a.mode, "n_windows_evaluated": int(len(y_true)), "n_groups": int(len(groups)),
               "validation": "leave-one-group-out" if len(groups) <= 15 else "5-fold GroupKFold",
               "accuracy": float(accuracy_score(y_true, y_pred)), "precision_fog": float(pr[0]),
               "recall_fog": float(rc[0]), "f1_fog": float(f1[0]),
               "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()}
    print(json.dumps(metrics, indent=2))

    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=[0, 1]),
                           display_labels=["Normal", "FoG-like"]).plot(cmap="GnBu")
    plt.title(f"Confusion matrix ({a.mode}, group-wise validation)")
    plt.savefig(graphs_dir / f"confusion_matrix_{a.mode}.png", dpi=150, bbox_inches="tight"); plt.close()

    final = make_rf().fit(X, y)
    imp = np.argsort(final.feature_importances_)[::-1][:12]
    plt.figure(figsize=(7, 4)); plt.barh([names[i] for i in imp][::-1], final.feature_importances_[imp][::-1], color="#00A6A6")
    plt.title("Top feature importances"); plt.tight_layout()
    plt.savefig(graphs_dir / f"feature_importance_{a.mode}.png", dpi=150); plt.close()

    # calibrated=False -> the dashboard will NOT display a probability as 'confidence'
    bundle = {"model": final, "feature_names": names, "mode": a.mode, "calibrated": False, "metrics": metrics}
    fname = "pose_model.pkl" if a.mode == "pose" else "daphnet_accel_model.pkl"
    joblib.dump(bundle, models_dir / fname)
    (reports_dir / f"metrics_{a.mode}.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model -> {models_dir / fname}")


if __name__ == "__main__":
    main()
