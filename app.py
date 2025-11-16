import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# Load Model and Scaler
# ===============================
model = joblib.load("optimized_heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# Streamlit App UI
# ===============================
st.set_page_config(page_title="❤️ Heart Disease Predictor", layout="centered")

# Custom CSS for UI styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #83a4d4, #b6fbff);
        font-family: 'Segoe UI';
    }
    .main-title {
        text-align: center;
        font-size: 36px !important;
        color: #FF4B4B;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        font-size: 18px !important;
        color: #444;
        margin-bottom: 40px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">❤️ Heart Disease Risk Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Enter your health details below to predict the likelihood of heart disease.</p>', unsafe_allow_html=True)

# ===============================
# User Input Section
# ===============================
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 40)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
    
with col2:
    restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
    exang = st.selectbox("Exercise-Induced Angina (1 = Yes, 0 = No)", [0, 1])
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2])

# Combine user input into array
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Scale the input
scaled_input = scaler.transform(input_data)

# ===============================
# Prediction
# ===============================
if st.button("🔍 Predict"):
    prediction = model.predict(scaled_input)
    prob = model.predict_proba(scaled_input)[0][1] * 100

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Heart Disease\nPrediction Confidence: {prob:.2f}%")
    else:
        st.success(f"✅ You are at Low Risk of Heart Disease\nPrediction Confidence: {prob:.2f}%")

# ===============================
# Footer
# ===============================
st.markdown("""
---
Developed with ❤️ using Python, Streamlit, and Random Forest Classifier.
""")
