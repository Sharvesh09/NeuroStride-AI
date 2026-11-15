"""Central configuration for NeuroStride AI (research prototype, NOT a medical device)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DAPHNET_DIR = ROOT / "dataset" / "daphnet"
POSE_DIR = ROOT / "dataset" / "pose_recordings"
MODELS_DIR = ROOT / "models"
GRAPHS_DIR = ROOT / "outputs" / "graphs"
REPORTS_DIR = ROOT / "outputs" / "reports"

# ---- Daphnet (wearable accelerometer, 64 Hz) ----
DAPHNET_FS = 64
DAPHNET_WINDOW_S = 4.0          # window length (seconds)
DAPHNET_STEP_S = 2.0            # 50% overlap
DAPHNET_COLS = ["t_ms",
                "ankle_x", "ankle_y", "ankle_z",
                "thigh_x", "thigh_y", "thigh_z",
                "trunk_x", "trunk_y", "trunk_z", "label"]
# label: 0 = outside experiment, 1 = no freeze, 2 = freeze
DAPHNET_MIN_FREEZE_FRACTION = 0.5

# ---- Webcam pose pipeline ----
POSE_FS = 20                    # signals are resampled to this rate
POSE_WINDOW_S = 4.0
POSE_STEP_S = 1.0
POSE_CHANNELS = ["l_ankle_y", "r_ankle_y", "l_knee_ang", "r_knee_ang", "hip_x", "hip_y"]
POSE_CLASSES = {0: "Normal Walking", 1: "Possible FoG"}
POSE_MIN_FREEZE_FRACTION = 0.5

# ---- Alert logic (no alert from a single frame/window) ----
ALERT_CONSECUTIVE_WINDOWS = 3   # consecutive positive windows needed to raise an alert
ALERT_COOLDOWN_S = 10

# ---- Frequency bands (Bachlin et al. freeze-index convention) ----
LOCO_BAND = (0.5, 3.0)
FREEZE_BAND = (3.0, 8.0)

RANDOM_STATE = 42
