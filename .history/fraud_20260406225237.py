import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
import os

warnings.filterwarnings('ignore')
import plotly.express as px

ds = pd.read_csv('nibss_fraud_dataset.csv')
st.markdown("<h1 style = 'color: #DD5746; text-align: center; font-size:60px; font-family: Monospace'>FRAUD DETECTION PREDICTION SYSTEM / MODEL </h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin:30px; color:#FFC470; text-align:center; font-family:Serif'>Build by Adeseye Ademola </h4>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html = True)

st.image('pngwing.com fraud.png')
st.divider()

st.markdown("<h2 style = 'color: #F7C566; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)
st.markdown('Fraud detection is a critical component of Data-Driven Decision Making Systems (DDDM) in digital banking platforms such as Moniepoint, where transaction integrity and customer trust directly impact financial stability and platform credibility. In modern fintech environments, a high volume of digital transactions is processed in real time, creating opportunities for both legitimate activity and sophisticated fraudulent behavior. Fraudulent transactions are typically rare but high-impact events, making them difficult to detect using traditional rule-based systems alone')

st.divider()

st.dataframe(ds.head(50), use_container_width=True)

st.sidebar.image('user icon.png', caption = 'Welcome User')

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE_PATH = os.path.join(BASE_DIR, "fraud_detection_model.joblib")

# Load model, scaler, and encoders
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_FILE_PATH)
        scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
        encoders = joblib.load(os.path.join(BASE_DIR, "encoders.joblib"))
        # id_encoders if you have separate file for transaction_id & customer_id
        id_encoders = joblib.load(os.path.join(BASE_DIR, "id_encoders.joblib")) if os.path.exists(os.path.join(BASE_DIR, "id_encoders.joblib")) else {}
        return model, scaler, encoders, id_encoders
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.stop()

model, scaler, encoders, id_encoders = load_artifacts()

# Fixed FEATURE_NAMES - removed 'is_fraud'
FEATURE_NAMES = [
'transaction_id', 'customer_id', 'timestamp', 'amount', 'channel','merchant_category', 'bank', 'location', 'age_group', 'hour','day_of_week', 'month', 'is_weekend', 'is_peak_hour', 'tx_count_24h',
'amount_sum_24h', 'amount_mean_7d', 'amount_std_7d', 'tx_count_total',
'amount_mean_total', 'amount_std_total', 'channel_diversity',
'location_diversity', 'amount_vs_mean_ratio', 'online_channel_ratio',
'hour_sin', 'hour_cos', 'day_sin',
'day_cos', 'month_sin', 'month_cos', 'amount_log', 'amount_rounded',
'velocity_score', 'merchant_risk_score', 'composite_risk'
]

# ---------------------------
# INPUT SECTION
# ---------------------------
st.header("Enter Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input("Amount", value=1000.0)
    hour = st.slider("Hour", 0, 23, 12)
    day_of_week = st.slider("Day of Week", 0, 6, 2)

with col2:
    month = st.slider("Month", 1, 12, 6)
    is_weekend = st.selectbox("Is Weekend", [0, 1])
    is_peak_hour = st.selectbox("Is Peak Hour", [0, 1])

with col3:
    tx_count_24h = st.number_input("Tx Count 24h", value=5.0)
    amount_sum_24h = st.number_input("Amount Sum 24h", value=5000.0)
    amount_mean_7d = st.number_input("Amount Mean 7d", value=800.0)

# More features
amount_std_7d = st.number_input("Amount Std 7d", value=100.0)
tx_count_total = st.number_input("Total Tx Count", value=50.0)
amount_mean_total = st.number_input("Mean Total", value=900.0)
amount_std_total = st.number_input("Std Total", value=150.0)

channel_diversity = st.number_input("Channel Diversity", value=2)
location_diversity = st.number_input("Location Diversity", value=2)

amount_vs_mean_ratio = st.number_input("Amount vs Mean Ratio", value=1.2)
online_channel_ratio = st.number_input("Online Channel Ratio", value=0.5)

velocity_score = st.number_input("Velocity Score", value=0.3)
merchant_risk_score = st.number_input("Merchant Risk Score", value=0.4)
composite_risk = st.number_input("Composite Risk", value=0.5)

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("Predict Fraud"):

    # Create a DataFrame with all expected features (use defaults for missing ones)
    input_dict = {col: 0 for col in FEATURE_NAMES}  # default values
    
    # Fill the features you collected
    input_dict.update({
        'amount': amount,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': is_weekend,
        'is_peak_hour': is_peak_hour,
        'tx_count_24h': tx_count_24h,
        'amount_sum_24h': amount_sum_24h,
        'amount_mean_7d': amount_mean_7d,
        'amount_std_7d': amount_std_7d,
        'tx_count_total': tx_count_total,
        'amount_mean_total': amount_mean_total,
        'amount_std_total': amount_std_total,
        'channel_diversity': channel_diversity,
        'location_diversity': location_diversity,
        'amount_vs_mean_ratio': amount_vs_mean_ratio,
        'online_channel_ratio': online_channel_ratio,
        'velocity_score': velocity_score,
        'merchant_risk_score': merchant_risk_score,
        'composite_risk': composite_risk,
        # Add more mappings here if you collect them (channel, merchant_category, etc.)
    })

    input_df = pd.DataFrame([input_dict])

    # Apply the same preprocessing as during training
    # 1. Encode categorical columns using saved encoders
    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col].astype(str))

    # 2. Encode transaction_id & customer_id if you collected them
    for col, le in id_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col].astype(str))

    # 3. Scale numeric features
    input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Result")

    if prediction[0] == 1:
        st.error("⚠️ Fraud Detected")
    else:
        st.success("✅ Legitimate Transaction")

    st.info(f"Fraud Probability: {probability*100:.2f}%")

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("Model Info")
st.sidebar.write("Model: Logistic Regression (Balanced)")
st.sidebar.write("Focus: Fraud Recall over Accuracy")