"""
api/server.py
FastAPI service exposing the latest driver state.
Run with: uvicorn api.server:app --reload
"""

from fastapi import FastAPI

app = FastAPI(title="Driver Safety AI - Fleet API")

latest_status = {
    "emotion": "neutral",
    "ear": 0.3,
    "fatigue_score": 0.0,
    "eyes_closed": False,
    "alert_active": False,
}


@app.get("/status")
def get_status():
    return latest_status


@app.post("/internal/update")
def update_status(payload: dict):
    latest_status.update(payload)
    return {"ok": True}