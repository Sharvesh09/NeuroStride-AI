"""Dataset loading + leakage-safe windowing.
Leakage control: every window carries the ID of the participant/recording it came from;
splits are made on those IDs (never on individual windows)."""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import config as C
import feature_extraction as F


def load_daphnet_file(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=C.DAPHNET_COLS, engine="python")
    return df


def daphnet_subject_id(path):
    m = re.match(r"S(\d+)R\d+", Path(path).stem)
    return int(m.group(1)) if m else -1


def build_daphnet_windows(data_dir=C.DAPHNET_DIR):
    files = sorted(Path(data_dir).rglob("S*R*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No Daphnet files (S01R01.txt ...) found under {data_dir}. See README for the download link.")
    win, step = int(C.DAPHNET_WINDOW_S * C.DAPHNET_FS), int(C.DAPHNET_STEP_S * C.DAPHNET_FS)
    Xs, ys, groups = [], [], []
    for f in files:
        df = load_daphnet_file(f)
        sig = df[C.DAPHNET_COLS[1:10]].to_numpy(float)
        lab = df["label"].to_numpy(int)
        idx = F.window_indices(len(df), win, step)
        keep, y = F.window_labels(lab, idx, positive=2, negative=1, min_pos_frac=C.DAPHNET_MIN_FREEZE_FRACTION)
        sid = daphnet_subject_id(f)
        for k, (s, e) in enumerate(idx):
            if keep[k]:
                Xs.append(F.accel_window_features(sig[s:e])); ys.append(y[k]); groups.append(sid)
    return np.vstack(Xs), np.array(ys), np.array(groups)


def build_pose_windows(data_dir=C.POSE_DIR):
    """Each CSV = one recording: columns t, <POSE_CHANNELS>, label (0 normal, 1 freeze-like).
    Group = recording file stem (use one file per subject/session; see README)."""
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No pose recordings in {data_dir}. Record some with collect_data.py first.")
    win, step = int(C.POSE_WINDOW_S * C.POSE_FS), int(C.POSE_STEP_S * C.POSE_FS)
    Xs, ys, groups = [], [], []
    for f in files:
        df = pd.read_csv(f)
        sig = F.resample_signals(df["t"].to_numpy(), df[C.POSE_CHANNELS].to_numpy(float))
        lab_raw = np.interp(np.arange(len(sig)) / C.POSE_FS + df["t"].iloc[0], df["t"], df["label"]).round().astype(int)
        idx = F.window_indices(len(sig), win, step)
        keep, y = F.window_labels(lab_raw, idx, positive=1, negative=0, min_pos_frac=C.POSE_MIN_FREEZE_FRACTION)
        for k, (s, e) in enumerate(idx):
            if keep[k]:
                Xs.append(F.pose_window_features(sig[s:e])); ys.append(y[k]); groups.append(f.stem.split("_")[0])
    return np.vstack(Xs), np.array(ys), np.array(groups)
