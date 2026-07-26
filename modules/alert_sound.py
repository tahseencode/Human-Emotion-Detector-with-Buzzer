"""
alert_sound.py
Plays a beep on the laptop speaker while an alert is active,
without blocking the main video loop.
"""

import threading
import winsound


class AlertSound:
    def __init__(self, frequency=1000, duration_ms=400):
        self.frequency = frequency
        self.duration_ms = duration_ms
        self.active = False
        self.running = True
        self.lock = threading.Lock()
        self._wake = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def set_active(self, is_active: bool):
        with self.lock:
            changed = is_active != self.active
            self.active = is_active
        if changed:
            self._wake.set()  # wake the thread immediately on state change

    def _loop(self):
        while self.running:
            with self.lock:
                should_beep = self.active
            if should_beep:
                winsound.Beep(self.frequency, self.duration_ms)
            else:
                self._wake.wait(timeout=0.5)
                self._wake.clear()

    def stop(self):
        self.running = False
        self._wake.set()
        self.thread.join(timeout=1)