import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI-Powered Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="centered"
)

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# -----------------------------
# Header
# -----------------------------
st.title("🫀 AI-Powered Heart Disease Risk Predictor")

st.markdown("""
Welcome! This application uses a trained **Machine Learning (KNN)** model to estimate
the likelihood of heart disease based on patient clinical information.

Please enter the required details below and click **Predict Risk**.
""")

st.divider()

# -----------------------------
# User Input
# -----------------------------
age = st.slider("Age", 18, 100, 40)

sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=200,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    min_value=100,
    max_value=600,
    value=200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.slider(
    "Maximum Heart Rate",
    60,
    220,
    150
)

exercise_angina = st.selectbox(
    "Exercise-Induced Angina",
    ["Y", "N"]
)

oldpeak = st.slider(
    "Oldpeak (ST Depression)",
    0.0,
    6.0,
    1.0
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Risk", use_container_width=True):

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    st.divider()

    if prediction == 1:
        st.error("⚠️ **Prediction Result: High Risk of Heart Disease**")
        st.warning(
            "The model predicts a higher likelihood of heart disease. "
            "Please consult a qualified healthcare professional for proper medical evaluation."
        )

    else:
        st.success("✅ **Prediction Result: Low Risk of Heart Disease**")
        st.info(
            "The model predicts a lower likelihood of heart disease based on the provided information."
        )

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "⚠️ Disclaimer: This application is intended for educational and demonstration "
    "purposes only. It should not be used as a substitute for professional medical "
    "advice, diagnosis, or treatment."
)

st.markdown(
    "**Developed by:** Abhishek Tiwari | **Technology:** Python, Streamlit, Scikit-learn, Machine Learning"
)
