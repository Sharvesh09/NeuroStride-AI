"""Moving window buffer + alert logic (an alert needs several consecutive positive windows)."""
from collections import deque
from datetime import datetime
import numpy as np
import config as C
import feature_extraction as F


class SignalBuffer:
    def __init__(self, keep_s=12.0):
        self.t, self.x, self.keep = deque(), deque(), keep_s

    def add(self, t, vec):
        self.t.append(t); self.x.append(vec)
        while self.t and t - self.t[0] > self.keep:
            self.t.popleft(); self.x.popleft()

    def span(self):
        return (self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    def window(self, length_s=C.POSE_WINDOW_S):
        """Last `length_s` seconds, resampled to POSE_FS. None if not enough data."""
        if self.span() < length_s:
            return None
        t = np.array(self.t); X = np.array(self.x)
        m = t >= t[-1] - length_s
        sig = F.resample_signals(t[m], X[m])
        return sig if len(sig) >= int(0.9 * length_s * C.POSE_FS) else None


class AlertManager:
    def __init__(self):
        self.streak, self.last_alert_t, self.events, self.active = 0, -1e9, [], False

    def update(self, now_s, positive):
        """Returns True only when a NEW suspected event is recorded."""
        self.streak = self.streak + 1 if positive else 0
        self.active = self.streak >= C.ALERT_CONSECUTIVE_WINDOWS
        if self.active and now_s - self.last_alert_t > C.ALERT_COOLDOWN_S:
            self.last_alert_t = now_s
            self.events.append({"time": datetime.now().strftime("%H:%M:%S"), "session_s": round(now_s, 1),
                                "note": "Suspected freeze-like movement (not a clinical diagnosis)"})
            return True
        return False
