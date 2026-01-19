import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Smart Health Monitor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Smart Health Monitoring System")
st.caption("A simple real-time health indicator dashboard")

# ---------------- Inputs ----------------
c1, c2 = st.columns(2)

with c1:
    heart_rate = st.slider("Heart Rate (bpm)", 50, 160, 80)
    activity = st.slider("Activity Level (1–10)", 1, 10, 5)

with c2:
    sleep = st.slider("Sleep Hours (last 24h)", 0.0, 12.0, 6.0, 0.5)
    temperature = st.slider("Body Temperature (°C)", 35.0, 40.0, 36.7, 0.1)

# ---------------- Simple Rule-Based Logic ----------------
risk_score = 0

if heart_rate > 100:
    risk_score += 1
if sleep < 5:
    risk_score += 1
if activity <= 3:
    risk_score += 1
if temperature > 37.8:
    risk_score += 1

# ---------------- Result ----------------
st.subheader("Health Status")

if risk_score >= 2:
    st.error("⚠️ Potential Health Risk Detected")
else:
    st.success("✅ Health Parameters Look Normal")

st.write(f"**Risk Score:** {risk_score} / 4")

# ---------------- Recommendations ----------------
st.subheader("Recommendations")

recommendations = []

if sleep < 6:
    recommendations.append("Improve sleep duration (7–8 hours recommended).")
if heart_rate > 100:
    recommendations.append("Practice deep breathing and stay hydrated.")
if activity <= 3:
    recommendations.append("Add light walking or stretching today.")
if temperature > 37.8:
    recommendations.append("Rest and avoid intense activity.")

if not recommendations:
    recommendations.append("Maintain current routine. Good job.")

for r in recommendations:
    st.write("•", r)

# ---------------- Trend Chart (Session Only) ----------------
st.subheader("📊 Session Trends")

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=["Time", "Heart Rate", "Sleep", "Activity", "Temperature"]
    )

st.session_state.history.loc[len(st.session_state.history)] = [
    datetime.now(),
    heart_rate,
    sleep,
    activity,
    temperature
]

fig = px.line(
    st.session_state.history,
    x="Time",
    y=["Heart Rate", "Sleep", "Activity", "Temperature"],
    markers=True
)

st.plotly_chart(fig, use_container_width=True)

st.caption("Note: Data resets on refresh.")
