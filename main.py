"""
main.py
Entry point. Captures webcam frames, runs the full pipeline every frame,
and updates the live dashboard.
Run: python main.py   |   Press 'q' to quit.
"""

import cv2

from modules.landmarks import LandmarkDetector
from modules.emotion_worker import EmotionWorker
from modules.fatigue import FatigueScorer
from modules.alert import AlertEngine
from modules.dashboard import Dashboard
from modules.alert_sound import AlertSound

EMOTION_EVERY_N_FRAMES = 10


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera index / permissions.")

    landmark_detector = LandmarkDetector()
    emotion_worker = EmotionWorker()
    fatigue_scorer = FatigueScorer(model_path="data/fatigue_model.json")
    alert_engine = AlertEngine()
    alert_sound = AlertSound()
    dashboard = Dashboard()

    frame_count = 0
    dominant_emotion = "neutral"
    emotion_valence = 0.0
    last_ear = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed, stopping.")
                break

            frame = cv2.resize(frame, (640, 480))
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            lm_result = landmark_detector.process(rgb, w, h)

            if lm_result is not None:
                last_ear = lm_result["ear"]
                fatigue_scorer.update(
                    ear=lm_result["ear"],
                    eyes_closed=lm_result["eyes_closed"],
                    yawning=lm_result["yawning"],
                )

            if frame_count % EMOTION_EVERY_N_FRAMES == 0:
                emotion_worker.submit(frame)

            dominant_emotion = emotion_worker.dominant_emotion
            emotion_valence = emotion_worker.valence

            fatigue_score, features = fatigue_scorer.score(
                head_tilt=0.0,
                emotion_valence=emotion_valence,
            )

            alert_triggered = alert_engine.evaluate(
                fatigue_score=fatigue_score, perclos=features["perclos"]
            )
            alert_sound.set_active(alert_triggered or alert_engine.active)

            dashboard.update(
                ear=last_ear,
                fatigue_score=fatigue_score,
                dominant_emotion=dominant_emotion,
            )

            status_text = (
                f"{dominant_emotion.upper()} | EAR:{last_ear:.2f} "
                f"| Fatigue:{fatigue_score:.0f}"
            )
            color = (0, 0, 255) if alert_triggered or alert_engine.active else (0, 255, 0)
            cv2.putText(
                frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

            cv2.imshow("Driver Safety AI - press q to quit", frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        alert_sound.stop()
        emotion_worker.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()