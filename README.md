# 🌍 AQI Prediction System

A  web-based application to predict **Air Quality Index (AQI)** using pollutant concentration data.  
The application is built using **Streamlit** and provides a clean, user-friendly interface for AQI prediction.


🔗 **Live App**:  
https://aqi-prediction-system-9jkyneu4su63e7tlxtx7vz.streamlit.app/
---

## 📌 Features

- Predicts **AQI value** based on pollutant inputs
- Displays **AQI category** (Good, Moderate, Poor, etc.)
- Provides **health advisory messages** based on AQI level
- Interactive and **dark-themed UI**
- Deployed using **Streamlit Community Cloud**

---

## 🧪 Input Parameters

The model takes the following pollutant concentrations:

- **PM2.5** (µg/m³)
- **PM10** (µg/m³)
- **NO₂** (µg/m³)
- **SO₂** (µg/m³)
- **CO** (mg/m³)
- **O₃** (µg/m³)
- **Month** (1–12)

---

## 📊 AQI Categories

| AQI Range | Category |
|---------|----------|
| 0–50 | Good |
| 51–100 | Satisfactory |
| 101–200 | Moderate |
| 201–300 | Poor |
| 301–400 | Very Poor |
| 401+ | Severe |

---
## 🛠 Tech Stack

- **Python**
- **Streamlit**
- **Scikit-learn**
- **XGBoost**
- **NumPy & Pandas**
- **Joblib**
- **Git & GitHub**

---

## 🚀 Deployment

The application is deployed on **Streamlit Cloud** and allows users to
predict AQI values interactively through a web interface.

## 📁 Project Structure
AQI-Prediction-System/
│
├── app/
│   ├── streamlit_app.py      
│   ├── app.py               
│   └── utils.py             
│
├── models/
│   ├── aqi_xgboost_model.pkl
│   └── scaler.pkl            
├── data/
│   ├── raw/                  
│   └── processed/          
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_xgboost_model.ipynb
│
├── requirements.txt        
├── README.md                
└── .gitignore  
