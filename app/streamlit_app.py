import os
import streamlit as st
import numpy as np
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌍",
    layout="centered"
)

# --------------------------------------------------
# DARK THEME STYLING
# --------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #e6e6e6;
}
.stApp {
    background-color: #0e1117;
}
.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    color: #4dd0e1;
}
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #b0bec5;
    margin-bottom: 30px;
}
.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "aqi_xgboost_model.pkl")

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# AQI CATEGORY LOGIC
# --------------------------------------------------
def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

def health_advisory(category):
    advice = {
        "Good": "Air quality is satisfactory. Ideal for outdoor activities.",
        "Satisfactory": "Minor discomfort for sensitive individuals.",
        "Moderate": "Sensitive groups should limit prolonged outdoor exertion.",
        "Poor": "People with respiratory issues should avoid outdoor exposure.",
        "Very Poor": "Avoid outdoor activities. Wear masks if necessary.",
        "Severe": "Health alert! Everyone should stay indoors."
    }
    return advice[category]

def category_color(category):
    colors = {
        "Good": "#2ecc71",
        "Satisfactory": "#3498db",
        "Moderate": "#f1c40f",
        "Poor": "#e67e22",
        "Very Poor": "#e74c3c",
        "Severe": "#c0392b"
    }
    return colors[category]

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<div class='title'>🌍 AQI Prediction System</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>Air Quality Index Prediction using XGBoost</div>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# INPUT CARD
# --------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 1000.0, 50.0)
pm10 = st.number_input("PM10 (µg/m³)", 0.0, 1000.0, 80.0)
no2 = st.number_input("NO₂ (µg/m³)", 0.0, 500.0, 40.0)
so2 = st.number_input("SO₂ (µg/m³)", 0.0, 500.0, 15.0)
co = st.number_input("CO (mg/m³)", 0.0, 50.0, 1.2)
o3 = st.number_input("O₃ (µg/m³)", 0.0, 500.0, 30.0)

month = st.selectbox("Month", list(range(1, 13)))
day = st.slider("Day", 1, 31, 15)

# NEW YEAR INPUT
year = st.number_input("Year", 2010, 2035, 2024)

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("🚀 Predict AQI"):

    # Feature Engineering (same as training)
    pm_ratio = pm25 / (pm10 + 1)
    pm_interaction = pm25 * pm10

    no2_so2 = no2 * so2
    co_o3 = co * o3
    no2_o3_ratio = no2 / (o3 + 1)

    pm25_rolling = pm25

    # MODEL INPUT (15 FEATURES)
    data = np.array([[
        pm25, pm10, no2, so2, co, o3,
        month, day, year,
        pm_ratio, pm_interaction,
        no2_so2, co_o3, no2_o3_ratio,
        pm25_rolling
    ]])

    # Predict log(AQI)
    prediction_log = model.predict(data)[0]

    # Convert back
    prediction = np.expm1(prediction_log)

    category = aqi_category(prediction)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.success(f"Predicted AQI: {round(prediction,2)}")

    st.markdown(
        f"<h3 style='color:{category_color(category)}'>AQI Category: {category}</h3>",
        unsafe_allow_html=True
    )

    st.info(health_advisory(category))

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")

st.markdown("""
<div style="text-align:center; color:#90a4ae;">
<b>AQI Prediction System</b><br>
Dataset: India AQI (2015–2020)<br>
Machine Learning • XGBoost • Streamlit
</div>
""", unsafe_allow_html=True)