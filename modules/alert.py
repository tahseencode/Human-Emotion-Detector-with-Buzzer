"""
alert.py
Handles alert triggering logic. Prints to console / simulates the LCD
message for now. Swap `_send_to_device` for real serial/WiFi calls once
the ESP32 hardware is wired up.
"""

import time

FATIGUE_THRESHOLD = 70
PERCLOS_THRESHOLD = 30
COOLDOWN_SECONDS = 5


class AlertEngine:
    def __init__(self):
        self.last_alert_time = 0
        self.active = False

    def evaluate(self, fatigue_score, perclos):
        should_alert = fatigue_score > FATIGUE_THRESHOLD or perclos > PERCLOS_THRESHOLD
        now = time.time()

        if should_alert and (now - self.last_alert_time) > COOLDOWN_SECONDS:
            self.last_alert_time = now
            self.active = True
            self._send_to_device("DROWSY - PULL OVER")
            return True

        if not should_alert:
            self.active = False

        return False

    def _send_to_device(self, message):
        print(f"[ALERT] Buzzer ON | LCD: '{message}'")

        # --- ESP32 over serial (uncomment once hardware is wired up) ---
        # import serial
        # ser = serial.Serial('COM3', 115200, timeout=1)
        # ser.write(f"{message}\n".encode())

        # --- ESP32 over WiFi (alternative) ---
        # import requests
        # requests.post("http://<esp32-ip>/alert", json={"message": message})