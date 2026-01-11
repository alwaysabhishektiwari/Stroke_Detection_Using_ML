import streamlit as st
import pandas as pd
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="HeartCare AI",
    page_icon="❤️",
    layout="wide"
)

# ================= DARK UI CSS (HIGH CONTRAST) =================
st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
html, body, [class*="css"] {
    color: #f1f1f1 !important;
    font-size: 15px;
}

/* ---------- BACKGROUND ---------- */
.stApp {
    background: radial-gradient(circle at top, #111827, #020617);
}

/* ---------- HEADINGS ---------- */
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 700;
}

/* ---------- LABELS ---------- */
label {
    color: #e5e7eb !important;
    font-weight: 500;
}

/* ---------- INPUT FIELDS ---------- */
.stNumberInput input,
.stSelectbox div,
.stTextInput input {
    background-color: #020617 !important;
    color: #ffffff !important;
    border: 1px solid #334155;
}

/* ---------- SLIDERS ---------- */
.stSlider span {
    color: #f87171 !important;
    font-weight: 600;
}

/* ---------- RADIO ---------- */
.stRadio label {
    color: #ffffff !important;
}

/* ---------- CARDS ---------- */
.card {
    background: linear-gradient(145deg, #020617, #020617);
    padding: 24px;
    border-radius: 14px;
    border: 1px solid #1e293b;
    box-shadow: 0 10px 25px rgba(0,0,0,0.6);
    margin-bottom: 20px;
}

/* ---------- BUTTON ---------- */
.stButton>button {
    width: 100%;
    height: 3.5em;
    background: linear-gradient(90deg,#dc2626,#ef4444);
    color: white !important;
    font-size: 16px;
    font-weight: 700;
    border-radius: 14px;
}

/* ---------- RESULT CARDS ---------- */
.result-high {
    background: #7f1d1d;
    border-left: 6px solid #ef4444;
}
.result-low {
    background: #064e3b;
    border-left: 6px solid #22c55e;
}

/* ---------- DIVIDER ---------- */
hr {
    border-color: #1e293b;
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_assets():
    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    expected_columns = joblib.load("heart_columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_assets()

# ================= HEADER =================
st.markdown("## ❤️ HeartCare AI – Heart Disease Prediction System")
st.markdown("AI-based clinical decision support for early cardiovascular risk detection")
st.divider()

# ================= INPUT SECTION =================
left, right = st.columns(2)

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👤 Patient Profile")

    age = st.slider("Age (Years)", 18, 100, 40)
    sex = st.radio("Sex", ["M", "F"], horizontal=True)
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    exercise_angina = st.radio("Exercise-Induced Angina", ["Y", "N"], horizontal=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🩺 Clinical Measurements")

    c1, c2 = st.columns(2)
    with c1:
        resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    with c2:
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ================= PREDICTION =================
if st.button("🔍 Run Health Assessment"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.markdown("""
        <div class="card result-high">
            <h3>⚠️ High Risk Detected</h3>
            <p>The model indicates a high probability of heart disease.
            Immediate clinical consultation is recommended.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card result-low">
            <h3>✅ Low Risk Detected</h3>
            <p>The model indicates a low probability of heart disease.
            Maintain a healthy lifestyle and routine checkups.</p>
        </div>
        """, unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<center><small>© 2026 HeartCare AI | Machine Learning Healthcare System</small></center>",
    unsafe_allow_html=True
)
