"""
dashboard.py
----------------
JOB: Show the results in a simple web page (Streamlit).

Run with:  streamlit run dashboard.py
(monitor.py must be running separately in another terminal so history.csv keeps growing)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np

from trend import early_warning, warning_text, FAULT_THRESHOLDS, WINDOW

LOG_FILE = "history.csv"
TIMEOUT_MS = 999  # sentinel value monitor.py writes when a ping fails/times out
GAP_SECONDS = 30  # a pause longer than this = monitor was off; break the chart line there
MIN_EPISODE_READINGS = 3  # a fault must last >= 3 readings (15 s) to count as an episode

st.set_page_config(page_title="NodeX - Network Fault Detector", layout="wide", page_icon="📡")

# ---------- Light custom styling ----------
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background-color: #1a1c24;
        border: 1px solid #2a2d3a;
        border-radius: 10px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; opacity: 0.75; }
    .verdict-banner {
        padding: 20px 24px;
        border-radius: 12px;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .verdict-normal { background-color: rgba(46, 160, 67, 0.15); color: #4ade80; border: 1px solid rgba(46,160,67,0.4); }
    .verdict-fault  { background-color: rgba(220, 38, 38, 0.15); color: #f87171; border: 1px solid rgba(220,38,38,0.4); }
    .verdict-warning { background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.4);
                        padding: 14px 20px; border-radius: 10px; font-size: 1rem; font-weight: 600; margin-bottom: 10px; }
    .subtle { opacity: 0.6; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

title_col, refresh_col = st.columns([5, 1])
with title_col:
    st.title("📡 NodeX: Network Fault Detector")
    st.caption("Cross-layer diagnosis with predictive early-warning")
with refresh_col:
    st.write("")
    st.write("")
    if st.button("🔄 Refresh"):
        st.rerun()

if not os.path.isfile(LOG_FILE):
    st.warning("No data yet. Run monitor.py first to start collecting readings.")
    st.stop()

df = pd.read_csv(LOG_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ---- Treat timeout sentinel as missing, not a real reading ----
df["router_ping_timeout"] = df["router_ping"] >= TIMEOUT_MS
df["external_ping_timeout"] = df["external_ping"] >= TIMEOUT_MS
df["router_ping_clean"] = df["router_ping"].where(~df["router_ping_timeout"])
df["external_ping_clean"] = df["external_ping"].where(~df["external_ping_timeout"])


# ---- Early-Warning check (same logic as monitor.py, from trend.py) ----
def current_warnings(df):
    """Look at the last WINDOW readings. Skip if they span a pause (monitor was restarted)."""
    tail = df.tail(WINDOW)
    if len(tail) < WINDOW:
        return []
    span = (tail["timestamp"].iloc[-1] - tail["timestamp"].iloc[0]).total_seconds()
    if span > WINDOW * 5 * 3:      # readings aren't continuous -> trend would be meaningless
        return []
    found = []
    for metric in FAULT_THRESHOLDS:
        result = early_warning(tail[metric + "_clean"].values, FAULT_THRESHOLDS[metric])
        if result:
            found.append(warning_text(metric, result))
    return found


for w in current_warnings(df):
    st.markdown(f'<div class="verdict-warning">⏳ Early Warning: {w}</div>', unsafe_allow_html=True)

# ---- Current Status ----
latest = df.iloc[-1]
readable_time = latest["timestamp"].strftime("%I:%M:%S %p").lstrip("0")

if latest["verdict"] == "Normal":
    st.markdown(
        f'<div class="verdict-banner verdict-normal">✅ Everything looks Normal '
        f'({latest["confidence"]}% confidence)</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="verdict-banner verdict-fault">⚠️ {latest["verdict"]} is likely the cause '
        f'— {latest["confidence"]}% confidence</div>',
        unsafe_allow_html=True,
    )
st.markdown(f'<p class="subtle">Last checked: {readable_time}</p>', unsafe_allow_html=True)

st.write("")
col1, col2, col3 = st.columns(3)
col1.metric("📶 Wifi Signal", f"{latest['wifi_signal']}%")

router_display = "Timeout ⚠️" if latest["router_ping_timeout"] else f"{latest['router_ping']:.0f} ms"
external_display = "Timeout ⚠️" if latest["external_ping_timeout"] else f"{latest['external_ping']:.0f} ms"
col2.metric("🌐 Router Ping", router_display)
col3.metric("🛰️ External Ping", external_display)

st.write("")
st.divider()

# ---- History Graphs: 3 separate properly-scaled charts ----
st.subheader("History Over Time")
st.caption("Each metric has its own scale. Timeouts and pauses in monitoring show as gaps, not fake lines.")

window_choice = st.radio("Show", ["Last 15 min", "Last hour", "Last 24 hours", "All"],
                         index=1, horizontal=True)
window_minutes = {"Last 15 min": 15, "Last hour": 60, "Last 24 hours": 1440}.get(window_choice)
if window_minutes:
    view = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(minutes=window_minutes)]
else:
    view = df


def with_gaps(d, column):
    """Insert a blank point wherever the monitor was paused, so the line breaks
    instead of drawing a fake slope across hours/days with no data."""
    d = d[["timestamp", column]].copy()
    pause = d["timestamp"].diff().dt.total_seconds() > GAP_SECONDS
    breaks = d[pause].copy()
    breaks["timestamp"] = breaks["timestamp"] - pd.Timedelta(seconds=1)
    breaks[column] = np.nan
    return pd.concat([d, breaks]).sort_values("timestamp")


def line_chart(df, column, title, color, y_suffix=""):
    df = with_gaps(df, column)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df[column],
        mode="lines", name=title,
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=color.replace(")", ", 0.08)").replace("rgb", "rgba"),
        connectgaps=False,  # timeouts (NaN) show as a real gap, not a bridged line
    ))
    fig.update_layout(
        title=title,
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", ticksuffix=y_suffix),
        font=dict(color="#e5e7eb"),
    )
    return fig

c1, c2, c3 = st.columns(3)
with c1:
    st.plotly_chart(line_chart(view, "wifi_signal", "Wifi Signal", "rgb(59,130,246)", "%"),
                     width='stretch')
with c2:
    st.plotly_chart(line_chart(view, "router_ping_clean", "Router Ping", "rgb(251,191,36)", "ms"),
                     width='stretch')
with c3:
    st.plotly_chart(line_chart(view, "external_ping_clean", "External Ping", "rgb(248,113,113)", "ms"),
                     width='stretch')

timeout_count = int(view["router_ping_timeout"].sum() + view["external_ping_timeout"].sum())
if timeout_count:
    st.caption(f"⚠️ {timeout_count} timed-out ping(s) in this window — shown as gaps above, excluded from the charts.")

st.divider()

# ---- Pattern Analysis: fault EPISODES by hour (not raw polls) ----
st.subheader("When Do Faults Happen Most?")
st.caption("Counts continuous fault episodes (e.g. a 3-minute router outage = 1 fault), not every poll. Blips shorter than 15 s are ignored.")

df["hour"] = df["timestamp"].dt.hour

# An "episode" is a run of consecutive rows with the same non-Normal verdict.
paused = df["timestamp"].diff().dt.total_seconds() > GAP_SECONDS
df["_new_episode"] = df["verdict"].ne(df["verdict"].shift()) | paused
df["_episode_id"] = df["_new_episode"].cumsum()

episodes = (
    df[df["verdict"] != "Normal"]
    .groupby("_episode_id")
    .agg(verdict=("verdict", "first"), start_hour=("hour", "first"), length=("verdict", "size"))
    .reset_index(drop=True)
)
episodes = episodes[episodes["length"] >= MIN_EPISODE_READINGS]  # ignore 1-2 reading blips

if not episodes.empty:
    counts_by_hour = episodes.groupby(["start_hour", "verdict"]).size().unstack(fill_value=0)

    fig = go.Figure()
    palette = {"WiFi": "rgb(59,130,246)", "Router": "rgb(251,191,36)", "ISP": "rgb(248,113,113)"}
    for verdict_type in counts_by_hour.columns:
        fig.add_trace(go.Bar(
            x=counts_by_hour.index, y=counts_by_hour[verdict_type],
            name=verdict_type, marker_color=palette.get(verdict_type, "gray"),
        ))
    fig.update_layout(
        barmode="stack", height=320,
        xaxis_title="Hour of day", yaxis_title="Number of fault episodes",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, width='stretch')
else:
    st.info("No faults logged yet - keep the monitor running to build up history.")

st.divider()
st.caption(f"Total readings logged: {len(df)} · Data source: {LOG_FILE}")