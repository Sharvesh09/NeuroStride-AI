"""Windowing + feature extraction for accelerometer (Daphnet) and webcam pose signals."""
import numpy as np
from scipy.signal import welch, find_peaks, butter, filtfilt
import config as C


# ---------- generic helpers ----------
def band_power(x, fs, lo, hi):
    x = np.asarray(x, float)
    x = x - np.mean(x)
    nper = min(len(x), int(fs * 2))
    if nper < 8:
        return 0.0
    f, p = welch(x, fs=fs, nperseg=nper)
    m = (f >= lo) & (f < hi)
    if m.sum() < 2:
        return 0.0
    ff, pp = f[m], p[m]
    return float(np.sum(np.diff(ff) * (pp[1:] + pp[:-1]) / 2.0))  # trapezoid rule (NumPy 1.x/2.x safe)


def freeze_index(x, fs):
    loco = band_power(x, fs, *C.LOCO_BAND)
    frz = band_power(x, fs, *C.FREEZE_BAND)
    return float(min(frz / (loco + 1e-6), 100.0))  # capped for numerical stability


def dominant_freq(x, fs):
    x = np.asarray(x, float) - np.mean(x)
    nper = min(len(x), int(fs * 2))
    if nper < 8:
        return 0.0
    f, p = welch(x, fs=fs, nperseg=nper)
    m = (f >= 0.3) & (f <= fs / 2)
    return float(f[m][np.argmax(p[m])]) if m.any() else 0.0


def lowpass(x, fs, cutoff=8.0, order=3):
    if len(x) < 15 or cutoff >= fs / 2:
        return np.asarray(x, float)
    b, a = butter(order, cutoff / (fs / 2), btype="low")
    return filtfilt(b, a, x)


def window_indices(n, win, step):
    return [(s, s + win) for s in range(0, n - win + 1, step)]


def window_labels(frame_labels, idx, positive, negative, min_pos_frac):
    """Return (window_mask, y). Windows are kept only if dominated by the two valid classes."""
    keep, y = [], []
    for (s, e) in idx:
        seg = frame_labels[s:e]
        valid = np.isin(seg, [positive, negative])
        if valid.mean() < 0.9:
            keep.append(False); y.append(-1); continue
        frac_pos = np.mean(seg == positive)
        keep.append(True)
        y.append(1 if frac_pos >= min_pos_frac else 0)
    return np.array(keep), np.array(y)


# ---------- Daphnet accelerometer features ----------
def accel_feature_names():
    names = []
    for s in ["ankle", "thigh", "trunk"]:
        names += [f"{s}_{k}" for k in
                  ["mean_mag", "std_mag", "energy", "loco_pow", "freeze_pow", "FI", "dom_freq", "std_x", "std_y", "std_z"]]
    return names


def accel_window_features(win, fs=C.DAPHNET_FS):
    """win: (n, 9) array -> ankle, thigh, trunk xyz."""
    feats = []
    for s in range(3):
        xyz = win[:, 3 * s:3 * s + 3].astype(float)
        mag = np.linalg.norm(xyz, axis=1)
        feats += [mag.mean(), mag.std(), float(np.mean((mag - mag.mean()) ** 2)),
                  band_power(mag, fs, *C.LOCO_BAND), band_power(mag, fs, *C.FREEZE_BAND),
                  freeze_index(mag, fs), dominant_freq(mag, fs),
                  xyz[:, 0].std(), xyz[:, 1].std(), xyz[:, 2].std()]
    return np.array(feats, float)


# ---------- Webcam pose features ----------
def resample_signals(t, X, fs=C.POSE_FS):
    """Resample irregular webcam timestamps (s) to a fixed rate. X: (n, k)."""
    t = np.asarray(t, float)
    if len(t) < 2 or t[-1] - t[0] <= 0:
        return np.empty((0, X.shape[1]))
    tn = np.arange(t[0], t[-1], 1.0 / fs)
    return np.column_stack([np.interp(tn, t, X[:, j]) for j in range(X.shape[1])])


def pose_feature_names():
    names = []
    for ch in C.POSE_CHANNELS:
        names += [f"{ch}_{k}" for k in ["std", "range", "loco_pow", "freeze_pow", "FI", "dom_freq"]]
    names += ["step_freq_hz", "step_interval_cv", "ankle_sep_std", "hip_speed_mean", "ankle_asym", "knee_ang_std_mean"]
    return names


def step_metrics(ankle_sep, fs=C.POSE_FS):
    """Step frequency and regularity from the inter-ankle vertical separation signal.
    Each step produces one |separation| peak, so peaks of |sep| ~ steps."""
    s = lowpass(np.abs(ankle_sep - np.mean(ankle_sep)), fs, 4.0)
    thr = max(0.15 * (s.max() - s.min()), 1e-4)
    peaks, _ = find_peaks(s, distance=max(1, int(0.25 * fs)), prominence=thr)
    dur = len(ankle_sep) / fs
    freq = len(peaks) / dur if dur > 0 else 0.0
    if len(peaks) >= 3:
        iv = np.diff(peaks) / fs
        cv = float(np.std(iv) / (np.mean(iv) + 1e-9))
    else:
        cv = 1.0  # too few steps -> treated as maximally irregular
    return float(freq), cv


def pose_window_features(win, fs=C.POSE_FS):
    """win: (n, len(POSE_CHANNELS)) normalised pose signals (see utils/pose_detection.py)."""
    win = np.asarray(win, float)
    feats = []
    for j in range(win.shape[1]):
        x = lowpass(win[:, j], fs, 8.0)
        feats += [x.std(), x.max() - x.min(),
                  band_power(x, fs, *C.LOCO_BAND), band_power(x, fs, *C.FREEZE_BAND),
                  freeze_index(x, fs), dominant_freq(x, fs)]
    la, ra = win[:, 0], win[:, 1]
    step_f, step_cv = step_metrics(la - ra, fs)
    hip_v = np.hypot(np.diff(win[:, 4]), np.diff(win[:, 5])) * fs
    feats += [step_f, step_cv, float(np.std(la - ra)), float(hip_v.mean()),
              float(abs(la.std() - ra.std())), float(np.mean([win[:, 2].std(), win[:, 3].std()]))]
    return np.array(feats, float)
