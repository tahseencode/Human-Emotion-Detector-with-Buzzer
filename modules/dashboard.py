"""
dashboard.py
Live multi-panel dashboard: EAR curve and fatigue score trend, updated
in place using matplotlib's interactive mode.
"""

from collections import deque

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

MAX_POINTS = 200


class Dashboard:
    def __init__(self):
        self.t = deque(maxlen=MAX_POINTS)
        self.ear_vals = deque(maxlen=MAX_POINTS)
        self.fatigue_vals = deque(maxlen=MAX_POINTS)
        self.step = 0

        plt.ion()
        self.fig, (self.ax_ear, self.ax_fatigue) = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.suptitle("Driver Drowsiness & Emotion Monitoring Dashboard")

    def update(self, ear, fatigue_score, dominant_emotion):
        self.step += 1
        self.t.append(self.step)
        self.ear_vals.append(ear)
        self.fatigue_vals.append(fatigue_score)

        self.ax_ear.clear()
        self.ax_ear.plot(self.t, self.ear_vals, color="cyan")
        self.ax_ear.axhline(0.22, color="red", linestyle="--", linewidth=0.8)
        self.ax_ear.set_title(f"Eye Aspect Ratio  |  Emotion: {dominant_emotion}")
        self.ax_ear.set_ylim(0, 0.5)

        self.ax_fatigue.clear()
        self.ax_fatigue.plot(self.t, self.fatigue_vals, color="orange")
        self.ax_fatigue.axhline(70, color="red", linestyle="--", linewidth=0.8)
        self.ax_fatigue.set_title("Fatigue Score (0-100)")
        self.ax_fatigue.set_ylim(0, 100)

        plt.tight_layout()
        plt.pause(0.001)