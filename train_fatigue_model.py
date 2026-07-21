"""
train_fatigue_model.py
Generates synthetic labeled session data and trains an XGBoost regressor
to predict fatigue score (0-100). Replace with real logged data later.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

np.random.seed(42)
N = 2000

perclos = np.random.uniform(0, 80, N)
ear_avg = np.random.uniform(0.15, 0.35, N)
yawn_freq = np.random.uniform(0, 10, N)
head_tilt = np.random.uniform(-30, 30, N)
emotion_valence = np.random.uniform(-1, 1, N)

fatigue = (
    perclos * 0.7
    + yawn_freq * 4
    + np.abs(head_tilt) * 0.25
    + np.maximum(0, -emotion_valence) * 15
    + np.random.normal(0, 5, N)
)
fatigue = np.clip(fatigue, 0, 100)

df = pd.DataFrame(
    {
        "ear_avg": ear_avg,
        "perclos": perclos,
        "yawn_freq": yawn_freq,
        "head_tilt": head_tilt,
        "emotion_valence": emotion_valence,
        "fatigue_score": fatigue,
    }
)

X = df.drop(columns=["fatigue_score"])
y = df["fatigue_score"]

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X, y)

model.save_model("data/fatigue_model.json")
print("Saved model to data/fatigue_model.json")
print(f"Training R^2: {model.score(X, y):.3f}")