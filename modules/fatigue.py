"""
fatigue.py
Maintains a rolling window of EAR/yawn/emotion history to compute PERCLOS,
assembles the feature vector, and scores fatigue (0-100) using a trained
XGBoost regressor. Falls back to a rule-based score if no model is loaded,
so the pipeline still runs before you've trained on real session data.
"""

import time
from collections import deque

import numpy as np
import pandas as pd
import xgboost as xgb

FEATURE_COLUMNS = ["ear_avg", "perclos", "yawn_freq", "head_tilt", "emotion_valence"]


class FatigueScorer:
    def __init__(self, model_path=None, window_seconds=60):
        self.window_seconds = window_seconds
        self.history = deque()  # list of (timestamp, ear, eyes_closed)
        self.yawn_timestamps = deque()
        self.model = None

        if model_path:
            try:
                candidate = xgb.XGBRegressor()
                candidate.load_model(model_path)
                self.model = candidate
            except Exception as exc:
                print(f"[FatigueScorer] could not load model ({exc}); using fallback rule")
                self.model = None

    def update(self, ear, eyes_closed, yawning):
        now = time.time()
        self.history.append((now, ear, eyes_closed))
        if yawning:
            self.yawn_timestamps.append(now)

        # drop anything older than the rolling window
        cutoff = now - self.window_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()
        while self.yawn_timestamps and self.yawn_timestamps[0] < cutoff:
            self.yawn_timestamps.popleft()

    def _perclos(self):
        if not self.history:
            return 0.0
        closed_frames = sum(1 for _, _, closed in self.history if closed)
        return 100.0 * closed_frames / len(self.history)

    def _ear_avg(self):
        if not self.history:
            return 0.3
        return float(np.mean([e for _, e, _ in self.history]))

    def _yawn_freq(self):
        # yawns per minute, scaled to the actual window length
        return len(self.yawn_timestamps) * (60.0 / self.window_seconds)

    def score(self, head_tilt=0.0, emotion_valence=0.0):
        features = {
            "ear_avg": self._ear_avg(),
            "perclos": self._perclos(),
            "yawn_freq": self._yawn_freq(),
            "head_tilt": head_tilt,
            "emotion_valence": emotion_valence,
        }

        if self.model is not None:
            df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
            pred = float(self.model.predict(df)[0])
            return max(0.0, min(100.0, pred)), features

        # --- rule-based fallback (used until a trained model is supplied) ---
        score = (
            features["perclos"] * 0.6
            + features["yawn_freq"] * 5
            + max(0.0, -features["emotion_valence"]) * 20
            + max(0.0, features["head_tilt"]) * 0.3
        )
        return max(0.0, min(100.0, score)), features