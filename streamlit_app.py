# streamlit_app.py
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px

# Optional SHAP (explainability)
shap = None

# Load Model + Config
@st.cache_resource
def load_assets():
    model = joblib.load("models/anomaly_model.pkl")
    config = json.loads(Path("models/config.json").read_text())
    data = pd.read_csv("data/health_data.csv")
    return model, config, data

MODEL, CONFIG, DF = load_assets()
FEATURES = CONFIG["feature_order"]
THRESHOLD = float(CONFIG["threshold"])
RULES = CONFIG.get("rule_safety_net", {"hr_high":110, "sleep_low":5.0, "temp_high":37.8})

# AI Advice Generator
try:
    from nlp_advice.ai_advice import generate_dynamic_advice
except:
    generate_dynamic_advice = None

# Rules
def rule_safety_net(hr, slp, act, temp):
    if slp < 2: return True
    if hr > 100 and slp < 4: return True
    if temp > RULES["temp_high"]: return True
    return False

# Model Prediction
def predict_proba(hr, slp, act, temp):
    X = np.array([[hr, slp, act, temp]], dtype=float)
    return float(MODEL.predict_proba(X)[:,1][0])

# Causes
def derive_causes(hr, slp, act, temp):
    causes = []
    if slp < 4: causes.append("Severely low sleep")
    elif slp < 6: causes.append("Mild sleep deficit")
    if hr > 100: causes.append("Elevated heart rate")
    if act <= 3: causes.append("Low activity level")
    if temp > 37.8: causes.append("Elevated body temperature")
    return causes or ["Overall healthy pattern detected"]

# Recommendations
def recommendations(hr, slp, act, temp, pred):
    recs = []
    if slp < 4: recs.append("Prioritize sleep today (target 7–8 hours). Keep room dark and cool.")
    elif slp < 6: recs.append("Try to increase sleep duration by 1–2 hours tonight.")
    if hr > 100: recs.append("Hydrate and try deep breathing (4-7-8) to reduce heart strain.")
    if act <= 3: recs.append("Add light walking or stretching today.")
    if temp > 37.8: recs.append("Rest, hydrate, avoid intense activity today.")
    if pred == 1 and not recs:
        recs.append("Your body may be under stress — rest and hydrate today.")
    if not recs:
        recs.append("Keep up balanced hydration, sleep, and consistent activity.")
    return recs

# ---------------- UI ----------------
st.set_page_config(page_title="Smart Health Monitor", page_icon="🩺", layout="centered")
st.title("🩺 Smart Health Monitoring — Live & Intelligent")

# Input sliders
c1, c2 = st.columns(2)
with c1:
    hr = st.slider("Heart Rate (bpm)", 50, 160, 90)
    act = st.slider("Activity Level (1–10)", 1, 10, 4)
with c2:
    slp = st.slider("Sleep Hours (last 24h)", 0.0, 12.0, 5.0, 0.5)
    temp = st.slider("Body Temperature (°C)", 35.0, 40.5, 36.8, 0.1)

# Live Prediction
proba = predict_proba(hr, slp, act, temp)
pred = int(proba >= THRESHOLD)
if rule_safety_net(hr, slp, act, temp): pred = 1

st.subheader("Prediction Result")
st.write(f"Probability (Anomaly): **{proba:.3f}**  |  Threshold: **{THRESHOLD:.3f}**")
st.success("✅ Normal") if pred == 0 else st.error("⚠️ Anomaly Detected")

# Causes
st.subheader("Likely Causes")
for c in derive_causes(hr, slp, act, temp):
    st.write("•", c)

# SHAP
st.subheader("Top Influential Factors")
st.caption("SHAP not installed. Run: `pip install shap==0.44.1`")

# Recommendations
st.subheader("What You Should Do")
for r in recommendations(hr, slp, act, temp, pred):
    st.write("•", r)

# AI Dynamic Advice
st.subheader("AI Personalized Guidance")
if generate_dynamic_advice:
    advice, fallback = generate_dynamic_advice(hr, slp, act, temp, pred, temperature=0.9, top_p=0.92, jitter=2)
    st.write(advice)
else:
    st.caption("AI advice not enabled. (Install transformers + torch)")

# ---------------- Trend Chart ----------------
st.subheader("📊 Trend of Your Health Metrics (Session)")

# Initialize session state for storing historical readings
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["Timestamp", "Heart Rate", "Sleep Hours", "Activity Level", "Temperature"])

# Append current reading with timestamp
st.session_state.history.loc[len(st.session_state.history)] = {
    "Timestamp": datetime.now(),
    "Heart Rate": hr,
    "Sleep Hours": slp,
    "Activity Level": act,
    "Temperature": temp
}

# Sort by timestamp
st.session_state.history.sort_values("Timestamp", inplace=True)

# Plot using Plotly
fig = px.line(
    st.session_state.history,
    x="Timestamp",
    y=["Heart Rate", "Sleep Hours", "Activity Level", "Temperature"],
    labels={"value": "Metric Value", "variable": "Metric"},
    title="Session Health Trends",
    markers=True
)
fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Value",
    legend_title="Metrics",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
st.caption("Note: Trends are based on current session data only.")
