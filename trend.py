"""
trend.py
----------------
Shared early-warning logic, used by BOTH monitor.py and dashboard.py
so the terminal and the web page always agree (and use the same wording).

A warning only fires when the rise is real:
    1. Spikes are damped first (rolling median of 3).
    2. The trend must be STEEP enough (MIN_SLOPE ms per check).
    3. The trend must be CONSISTENT (R^2 >= MIN_R2 = the line really fits).
    4. The total rise over the window must be meaningful (MIN_TOTAL_RISE).
    5. The projected fault must be within MAX_SECONDS_AHEAD (2 min).
    6. Windows containing timeouts (999) are skipped - a timeout is a lost
       packet, not a slow reply, so it can't be used for a latency trend.
"""

import numpy as np

TIMEOUT_MS = 999

# Latency above these values is treated as a "fault" (kept in sync with train_model.py).
FAULT_THRESHOLDS = {"router_ping": 100, "external_ping": 170}

WINDOW = 8               # readings looked at (8 x 5 s = 40 seconds of history)
MIN_SLOPE = 3.0          # ms of increase per check
MIN_R2 = 0.6             # how well a straight line fits (0 to 1)
MIN_TOTAL_RISE = 20      # ms of increase across the whole window
MAX_SECONDS_AHEAD = 120  # only warn if the fault is projected within 2 minutes

FRIENDLY_NAMES = {"router_ping": "Router ping", "external_ping": "External ping"}


def early_warning(values, threshold, interval_seconds=5, window=WINDOW):
    """
    values: recent readings of ONE metric, oldest -> newest (NaN or 999 = timeout).
    Returns None (no warning) or a dict with the estimated seconds left.
    """
    v = np.asarray(values, dtype=float)[-window:]
    if len(v) < window or np.isnan(v).any() or (v >= TIMEOUT_MS).any():
        return None

    # 1) damp one-off spikes: rolling median of 3
    smooth = np.array([np.median(v[max(0, i - 2): i + 1]) for i in range(len(v))])

    # 2) fit a straight line through the smoothed points
    x = np.arange(len(smooth))
    slope, intercept = np.polyfit(x, smooth, 1)

    # 3) how well does the line fit?  (R^2)
    fit = slope * x + intercept
    ss_res = float(((smooth - fit) ** 2).sum())
    ss_tot = float(((smooth - smooth.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    current = float(smooth[-1])
    total_rise = slope * (len(smooth) - 1)

    if (slope < MIN_SLOPE or r2 < MIN_R2 or total_rise < MIN_TOTAL_RISE
            or current >= threshold):
        return None

    seconds_left = (threshold - current) / slope * interval_seconds
    if seconds_left > MAX_SECONDS_AHEAD:   # too far away to be a useful warning
        return None
    return {"seconds_left": seconds_left, "slope": float(slope), "r2": r2,
            "current": current}


def format_eta(seconds):
    """Friendly time text instead of '~0.4 min'."""
    if seconds < 10:
        return "very soon"
    if seconds < 60:
        return f"~{int(round(seconds / 5) * 5)} seconds"
    return f"~{seconds / 60:.1f} min"


def warning_text(metric, result):
    name = FRIENDLY_NAMES.get(metric, metric)
    return f"{name} is rising steadily - may reach fault level in {format_eta(result['seconds_left'])}"