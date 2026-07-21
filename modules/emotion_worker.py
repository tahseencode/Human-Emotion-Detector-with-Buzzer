"""
emotion_worker.py
Runs DeepFace emotion analysis on a background thread so it never
blocks the webcam capture / display loop.
"""

import threading
from modules.emotion import EmotionDetector


class EmotionWorker:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.dominant_emotion = "neutral"
        self.valence = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, frame):
        """Call this from the main loop to hand off the latest frame."""
        with self.lock:
            self.latest_frame = frame.copy()

    def _loop(self):
        detector = EmotionDetector()
        while self.running:
            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame
                    self.latest_frame = None
            if frame is not None:
                result = detector.analyze(frame)
                if result is not None:
                    with self.lock:
                        self.dominant_emotion = result["dominant_emotion"]
                        self.valence = result["valence"]

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)