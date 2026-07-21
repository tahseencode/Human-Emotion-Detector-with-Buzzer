"""
emotion.py
Wraps DeepFace for facial emotion classification (7 classes).
Converts the emotion distribution into a single "valence" score
(-1 = very negative, +1 = very positive) used as a fatigue feature.
"""

from deepface import DeepFace

VALENCE_WEIGHTS = {
    "happy": 1.0,
    "surprise": 0.3,
    "neutral": 0.0,
    "sad": -0.6,
    "fear": -0.7,
    "disgust": -0.5,
    "angry": -0.8,
}


class EmotionDetector:
    def analyze(self, bgr_frame):
        try:
            result = DeepFace.analyze(
                bgr_frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
            )
            data = result[0] if isinstance(result, list) else result
            scores = data["emotion"]
            dominant = data["dominant_emotion"]

            total = sum(scores.values()) or 1.0
            valence = sum(
                (scores.get(e, 0) / total) * w for e, w in VALENCE_WEIGHTS.items()
            )

            return {
                "dominant_emotion": dominant,
                "scores": scores,
                "valence": valence,
            }
        except Exception as exc:
            print(f"[EmotionDetector] analysis failed: {exc}")
            return None