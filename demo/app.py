# app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
from mock_data import MockICS
from collections import deque

st.set_page_config(layout="wide", page_title="ICS IDS Demo")

# --- Session state init
if "stream" not in st.session_state:
    st.session_state.stream = deque(maxlen=300)  # keep last 300 samples
if "ics" not in st.session_state:
    st.session_state.ics = MockICS(n_sensors=4)
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "running" not in st.session_state:
    st.session_state.running = True

# --- Layout
st.title("AI for Automated Intrusion Detection (Demo)")
col1, col2 = st.columns([3,1])

with col2:
    st.header("Controls")
    if st.button("Start Stream"):
        st.session_state.running = True
    if st.button("Stop Stream"):
        st.session_state.running = False

    st.markdown("### Simulate Attack")
    if st.button("Sensor Spoof (High)"):
        st.session_state.ics.start_attack("sensor_spoof")
        st.session_state.alerts.insert(0, f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Attack started: SENSOR SPOOF")
    if st.button("Unauthorized Write"):
        st.session_state.ics.start_attack("unauth_write")
        st.session_state.alerts.insert(0, f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Attack started: UNAUTH WRITE")
    if st.button("Stop Attack"):
        st.session_state.ics.stop_attack()
        st.session_state.alerts.insert(0, f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Attack stopped")

    st.markdown("---")
    st.write("Detection rule (demo):")
    st.write("- If sensor_1 deviates > 15 from rolling mean → ALERT")
    st.write("- If sensor_4 toggles suspiciously → ALERT")

with col1:
    st.header("Live Sensor Streams")
    sensor_chart = st.empty()
    st.markdown("### Event Log")
    log_box = st.empty()

# --- helper detection (very simple rule-based for demo)
def detect_alerts(df):
    alerts = []
    if len(df) < 10:
        return alerts
    # rolling mean for sensor_1
    s1 = df["sensor_1"].astype(float)
    mean = s1.rolling(30, min_periods=5).mean().iloc[-1]
    if not np.isnan(mean) and abs(s1.iloc[-1] - mean) > 15:
        alerts.append(("sensor_spoof_detected", f"sensor_1 large deviation (val={s1.iloc[-1]:.1f}, mean={mean:.1f})"))
    # toggle detection on last sensor
    s4 = df["sensor_4"].astype(float)
    if s4.tail(8).nunique() > 1 and s4.tail(8).std() > 0.2:
        # quick heuristic: frequent toggling
        alerts.append(("unauth_write_detected", f"sensor_4 toggling (std={s4.tail(8).std():.2f})"))
    return alerts

# --- streaming loop (uses small sleep to simulate realtime)
for _ in range(30):  # produce initial backlog quickly
    row = st.session_state.ics.step()
    st.session_state.stream.append(row)

# Main update: run in a while-like small loop controlled by "running"
if st.session_state.running:
    # run one update per loop iteration
    row = st.session_state.ics.step()
    st.session_state.stream.append(row)

# convert to DataFrame for plotting and detection
df = pd.DataFrame(list(st.session_state.stream))
df_display = df.copy()
df_display["time_str"] = df_display["time"].dt.strftime("%H:%M:%S")

# Plots
if not df.empty:
    import plotly.express as px
    fig = px.line(df, x="time", y=[c for c in df.columns if c.startswith("sensor_")], markers=False)
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
    sensor_chart.plotly_chart(fig, use_container_width=True)

# Detection
new_alerts = detect_alerts(df)
for kind, text in new_alerts:
    msg = f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {kind.upper()}: {text}"
    # add to alerts if not duplicate recent
    if not st.session_state.alerts or msg != st.session_state.alerts[0]:
        st.session_state.alerts.insert(0, msg)

# show alerts (top)
if st.session_state.alerts:
    st.markdown("### Alerts (most recent)")
    for a in st.session_state.alerts[:10]:
        st.warning(a)

# show event log
with log_box:
    st.markdown("### Full Log (latest first)")
    for a in st.session_state.alerts[:50]:
        st.write(a)

# small sleep to make UI smoother when rerun
time.sleep(0.1)
st.markdown("---")
st.caption("Demo app: rule-based detection to mimic AI alerts. For full project, replace detect_alerts() with model inference.")
