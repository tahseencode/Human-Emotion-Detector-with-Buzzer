"""
landmarks.py
Wraps MediaPipe Face Mesh and computes EAR (Eye Aspect Ratio) and
MAR (Mouth Aspect Ratio) from the 468 facial landmarks.
"""

import numpy as np
import mediapipe as mp

# MediaPipe landmark index groups (subset of the 468-point mesh)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Ordered as [left_corner, upper_left, upper_right, right_corner, lower_right, lower_left]
# to match the same 6-point ratio pattern used for the eyes.
MOUTH = [61, 39, 269, 291, 405, 181]

EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.6


class LandmarkDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def _euclidean(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _ratio(self, landmarks, indices, w, h):
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
        vertical1 = self._euclidean(pts[1], pts[5])
        vertical2 = self._euclidean(pts[2], pts[4])
        horizontal = self._euclidean(pts[0], pts[3])
        if horizontal == 0:
            return 0.0
        return (vertical1 + vertical2) / (2.0 * horizontal)

    def process(self, rgb_frame, w, h):
        """Returns dict with ear, mar, eyes_closed, yawning, landmarks (or None)."""
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark
        left_ear = self._ratio(lm, LEFT_EYE, w, h)
        right_ear = self._ratio(lm, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2.0
        mar = self._ratio(lm, MOUTH, w, h)

        return {
            "ear": ear,
            "mar": mar,
            "eyes_closed": ear < EAR_THRESHOLD,
            "yawning": mar > MAR_THRESHOLD,
            "raw_landmarks": lm,
        }