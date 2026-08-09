# Human Emotion Detector with Buzzer

A real-time driver-safety system that watches a webcam feed, reads facial landmarks and emotional state, scores fatigue on the fly, and sounds a buzzer the moment drowsiness crosses a safe threshold.

The idea is simple: tiredness shows up in the face long before a driver notices it themselves — drooping eyelids, longer blinks, yawns, a slack expression. This project turns those signals into a single live fatigue score and reacts before the driver dozes off.

## How it works

Every frame from the webcam goes through the same pipeline:

1. **Landmark detection** (`modules/landmarks.py`) — uses MediaPipe to find facial landmarks and compute the Eye Aspect Ratio (EAR), detect closed eyes, and detect yawning.
2. **Emotion recognition** (`modules/emotion_worker.py`) — runs DeepFace on every 10th frame (not every frame, to keep things fast) in a background worker, producing a dominant emotion and a valence score.
3. **Fatigue scoring** (`modules/fatigue.py`) — combines EAR, PERCLOS (percentage of eye closure over time), yawn frequency, head tilt, and emotional valence into a single 0–100 fatigue score, using a pre-trained XGBoost regression model (`data/fatigue_model.json`).
4. **Alerting** (`modules/alert.py` + `modules/alert_sound.py`) — once the fatigue score or PERCLOS crosses a threshold, the alert engine triggers a buzzer sound to wake the driver up.
5. **Dashboard** (`modules/dashboard.py`) — renders a live overlay on the video feed showing EAR, dominant emotion, and fatigue score, with the frame border turning red when an alert is active.

The whole loop runs in `main.py`, which opens the webcam, ties all the modules together, and shows the annotated feed until you press `q` to quit.

There's also a small **FastAPI** app (`api/`) that exposes parts of this pipeline as an API, built with `fastapi` and served via `uvicorn`.

## Fatigue model

`train_fatigue_model.py` trains the XGBoost regressor used at inference time. It currently generates synthetic labeled data (PERCLOS, average EAR, yawn frequency, head tilt, emotion valence → fatigue score) to bootstrap the model, and saves it to `data/fatigue_model.json`. This is meant to be swapped out for real logged session data over time.

To retrain the model:

```bash
python train_fatigue_model.py
```

This regenerates `data/fatigue_model.json`, which `main.py` loads at startup.

## Tech stack

| Component | Library |
|---|---|
| Face/eye landmark detection | MediaPipe |
| Emotion recognition | DeepFace |
| Fatigue scoring model | XGBoost |
| Video capture & overlay | OpenCV |
| API layer | FastAPI + Uvicorn |
| Data handling | NumPy, Pandas, scikit-learn |
| Visualization (model dev) | Matplotlib, Seaborn |

## Project structure

```
.
├── main.py                  # Entry point — webcam loop + live dashboard
├── train_fatigue_model.py   # Trains the XGBoost fatigue-scoring model
├── requirements.txt
├── installed_packages.txt
├── modules/
│   ├── landmarks.py          # MediaPipe-based EAR / eye-closure / yawn detection
│   ├── emotion_worker.py     # Threaded DeepFace emotion inference
│   ├── fatigue.py            # Fatigue scoring (PERCLOS + XGBoost model)
│   ├── alert.py               # Alert-triggering logic
│   ├── alert_sound.py         # Buzzer / alert sound playback
│   └── dashboard.py           # Live on-screen dashboard overlay
├── api/                      # FastAPI app exposing the pipeline
└── data/
    └── fatigue_model.json     # Trained fatigue model
```

## Getting started

**Requirements:** Python 3.10+ and a webcam.

```bash
git clone https://github.com/tahseencode/Human-Emotion-Detector-with-Buzzer.git
cd Human-Emotion-Detector-with-Buzzer
pip install -r requirements.txt
```

Run the live detector:

```bash
python main.py
```

Press `q` in the video window to quit.

If you want to retrain the fatigue model first:

```bash
python train_fatigue_model.py
python main.py
```

## What the dashboard shows

- **Dominant emotion** — current emotion classification from DeepFace
- **EAR** — live Eye Aspect Ratio (lower = eyes more closed)
- **Fatigue score** — a 0–100 score blending eye closure, yawning, head position, and emotional state
- **Alert state** — the video border/overlay turns red and the buzzer sounds when fatigue crosses the alert threshold

## Notes / future work

- The fatigue model is currently trained on **synthetic data** — accuracy will improve significantly once trained on real, logged driver sessions.
- Head tilt is passed as a placeholder (`0.0`) in `main.py` — a real head-pose estimation step would make the fatigue score more accurate.
- The `api/` layer is set up for FastAPI but designed to be extended (e.g. for streaming scores to a companion app or dashboard).

## License

No license file is currently included in this repository — all rights reserved by default. Add a `LICENSE` file if you'd like to open this up for reuse.
