"""
alert_sound.py
Plays a looping beep on the laptop speaker while an alert is active,
without blocking the main video loop.
"""

import threading
import winsound


class AlertSound:
    def __init__(self, frequency=1000, duration_ms=400, gap_ms=200):
        self.frequency = frequency
        self.duration_ms = duration_ms
        self.gap_ms = gap_ms
        self.active = False
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def set_active(self, is_active: bool):
        with self.lock:
            self.active = is_active

    def _loop(self):
        while self.running:
            with self.lock:
                should_beep = self.active
            if should_beep:
                winsound.Beep(self.frequency, self.duration_ms)
            else:
                threading.Event().wait(0.1)  # idle check, avoids busy-loop

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)