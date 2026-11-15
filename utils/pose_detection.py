"""MediaPipe Pose wrapper -> scale-normalised gait channels (see config.POSE_CHANNELS)."""
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
L = mp_pose.PoseLandmark
_NEEDED = [L.LEFT_HIP, L.RIGHT_HIP, L.LEFT_KNEE, L.RIGHT_KNEE, L.LEFT_ANKLE, L.RIGHT_ANKLE,
           L.LEFT_SHOULDER, L.RIGHT_SHOULDER]


def joint_angle(a, b, c):
    v1, v2 = a - b, c - b
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))


class PoseExtractor:
    def __init__(self, complexity=0, min_vis=0.5):
        self.pose = mp_pose.Pose(model_complexity=complexity, min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
        self.min_vis = min_vis

    def process(self, bgr, draw=True):
        """Returns (annotated_frame, channel_vector or None). None = body/legs not reliably visible."""
        h, w = bgr.shape[:2]
        res = self.pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return bgr, None
        if draw:
            mp_draw.draw_landmarks(bgr, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(166, 200, 0), thickness=2, circle_radius=3),
                                   mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2))
        lm = res.pose_landmarks.landmark
        if min(lm[i].visibility for i in _NEEDED) < self.min_vis:
            return bgr, None
        P = lambda i: np.array([lm[i].x * w, lm[i].y * h])  # pixel units keep correct aspect ratio
        sh = (P(L.LEFT_SHOULDER) + P(L.RIGHT_SHOULDER)) / 2
        hip = (P(L.LEFT_HIP) + P(L.RIGHT_HIP)) / 2
        torso = np.linalg.norm(sh - hip)
        if torso < 20:
            return bgr, None
        vec = np.array([
            (P(L.LEFT_ANKLE)[1] - hip[1]) / torso,
            (P(L.RIGHT_ANKLE)[1] - hip[1]) / torso,
            joint_angle(P(L.LEFT_HIP), P(L.LEFT_KNEE), P(L.LEFT_ANKLE)),
            joint_angle(P(L.RIGHT_HIP), P(L.RIGHT_KNEE), P(L.RIGHT_ANKLE)),
            hip[0] / torso, hip[1] / torso])
        return bgr, vec

    def close(self):
        self.pose.close()
