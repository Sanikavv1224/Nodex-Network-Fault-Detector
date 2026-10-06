"""
monitor.py
----------------
JOB: The main brain of the project. Every few seconds it:
   1. Collects a reading         (collect_data.py)
   2. Smooths it                 (median of the last 5 readings)
   3. Predicts the verdict       (using the trained model)
   4. Logs it to a CSV           (for history + dashboard)
   5. Checks recent trend        (INNOVATION: early-warning, see trend.py)

 - SMOOTHING: the model sees the MEDIAN of the last 5 readings, not one raw
   reading. A phone hotspot's ping jumps between ~10 ms and ~130 ms for no
   real reason; a single spike no longer flips the verdict.
 - history.csv still stores the RAW readings (the truth), only the verdict is smoothed.
"""

import time
import csv
import os
from collections import deque
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from collect_data import collect_reading
from trend import early_warning, warning_text, FAULT_THRESHOLDS, WINDOW

LOG_FILE = "history.csv"
MODEL_FILE = "model.joblib"
CHECK_INTERVAL_SECONDS = 5   # shorter for demo; real use could be 60 s
SMOOTH_WINDOW = 5            # verdict uses the median of this many readings
FEATURES = ["wifi_signal", "router_ping", "external_ping"]


def log_reading(reading, verdict, confidence):
    """Append one row (timestamp, RAW readings, verdict, confidence) to history.csv."""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "wifi_signal", "router_ping",
                             "external_ping", "verdict", "confidence"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            reading["wifi_signal"], reading["router_ping"], reading["external_ping"],
            verdict, round(confidence, 1),
        ])


def smoothed(readings):
    """Median of each metric over the recent readings (kills one-off spikes)."""
    return {k: float(np.median([r[k] for r in readings])) for k in FEATURES}


def predict_verdict(model, reading):
    """Feed a (smoothed) reading into the trained model -> verdict + confidence %."""
    X = pd.DataFrame([reading])[FEATURES]
    verdict = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0]) * 100
    return verdict, confidence


def run():
    model = joblib.load(MODEL_FILE)
    recent = deque(maxlen=WINDOW)   # in-memory history for smoothing + trend
    print("NodeX monitor running... (Ctrl+C to stop)\n")

    while True:
        reading = collect_reading()
        recent.append(reading)

        stable = smoothed(list(recent)[-SMOOTH_WINDOW:])
        verdict, confidence = predict_verdict(model, stable)
        log_reading(reading, verdict, confidence)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"Wifi={reading['wifi_signal']}%  Router={reading['router_ping']:.0f}ms  "
              f"Ext={reading['external_ping']:.0f}ms  ->  {verdict} ({confidence:.0f}% confidence)")

        for metric, threshold in FAULT_THRESHOLDS.items():
            result = early_warning([r[metric] for r in recent], threshold, CHECK_INTERVAL_SECONDS)
            if result:
                print(f"   WARNING: {warning_text(metric, result)}")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    run()