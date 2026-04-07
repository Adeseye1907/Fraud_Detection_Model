import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# -------------------------------
# LOAD DATA (SAFE)
# -------------------------------
DATA_FILE = "nibss_fraud_dataset.csv"

if os.path.exists(DATA_FILE):
    ds = pd.read_csv(DATA_FILE)
else:
    st.error("❌ Dataset file not found. Please ensure **nibss_fraud_dataset.csv** is in the app folder.")
    st.stop()

# -------------------------------
# HEADER
# -------------------------------
st.markdown(
    "<h1 style='color:#DD5746; text-align:center;'>🔍 FRAUD DETECTION SYSTEM</h1>",
    unsafe_allow_html=True
)
st.markdown("<h4 style='text-align:center; color:gray;'>Built by Adeseye</h4>", unsafe_allow_html=True)

# Optional header image
if os.path.exists("fraud_detection.png"):
    st.image("fraud_detection.png", use_container_width=True)

st.divider()

# -------------------------------
# BACKGROUND
# -------------------------------
with st.expander("📖 Background of Study", expanded=False):
    st.markdown("""
    Fraud detection is a critical component of Data-Driven Decision Making in digital banking.  
    This system analyzes transaction behavior and risk indicators to predict fraudulent activities 
    using machine learning models.
    """)

# -------------------------------
# DATASET PREVIEW
# -------------------------------
st.subheader("Dataset Preview")
st.dataframe(ds.head(), use_container_width=True)

st.divider()

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    if os.path.exists("user icon.png"):
        st.image("user icon.png", caption="Welcome User", use_column_width=True)
    
    st.header("Model Performance")
    st.metric("ROC-AUC Score", "0.55")
    st.caption("Note: This is a demo model. Performance can be improved with better features/training.")

# -------------------------------
# LOAD MODEL (Cached)
# -------------------------------
MODEL_FILE = "fraud_detection_model.joblib"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        st.error("❌ Model file not found. Please ensure **fraud_detection_model.joblib** is in the app folder.")
        st.stop()
    return joblib.load(MODEL_FILE)

model = load_model()

# -------------------------------
# FEATURE NAMES (Must match training exactly)
# -------------------------------
FEATURE_NAMES = [
    'amount', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
    'tx_count_24h', 'amount_sum_24h', 'amount_mean_7d', 'amount_std_7d',
    'tx_count_total', 'amount_mean_total', 'amount_std_total',
    'channel_diversity', 'location_diversity', 'amount_vs_mean_ratio',
    'online_channel_ratio', 'velocity_score', 'merchant_risk_score', 'composite_risk'
]

# -------------------------------
# MAIN INPUT SECTION
# -------------------------------
st.title("Fraud Prediction")

st.markdown("### Enter Transaction Details")

# Basic Transaction Info
col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input("Transaction Amount (₦)", min_value=0.0, value=1000.0, step=100.0)
    hour = st.slider("Hour of Transaction", 0, 23, 12)
    day_of_week = st.slider("Day of Week (0 = Monday)", 0, 6, 2)

with col2:
    month = st.slider("Month", 1, 12, 6)
    is_weekend = st.selectbox("Is Weekend?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_peak_hour = st.selectbox("Is Peak Hour?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col3:
    tx_count_24h = st.number_input("Transactions in Last 24h", min_value=0, value=5)
    amount_sum_24h = st.number_input("Total Amount in 24h (₦)", min_value=0.0, value=5000.0)
    amount_mean_7d = st.number_input("Average Amount (7 days)", min_value=0.0, value=800.0)

# Advanced Features
with st.expander("Advanced Risk Features", expanded=False):
    col4, col5 = st.columns(2)
    
    with col4:
        amount_std_7d = st.number_input("Std Dev Amount (7 days)", value=100.0)
        tx_count_total = st.number_input("Total Transactions (Lifetime)", value=50)
        amount_mean_total = st.number_input("Mean Amount (Lifetime)", value=900.0)
        amount_std_total = st.number_input("Std Dev Amount (Lifetime)", value=150.0)
        
    with col5:
        channel_diversity = st.number_input("Channel Diversity", min_value=1, value=2)
        location_diversity = st.number_input("Location Diversity", min_value=1, value=2)
        amount_vs_mean_ratio = st.number_input("Amount vs Mean Ratio", value=1.2)
        online_channel_ratio = st.number_input("Online Channel Ratio", value=0.5)
        
        velocity_score = st.number_input("Velocity Score", value=0.3)
        merchant_risk_score = st.number_input("Merchant Risk Score", value=0.4)
        composite_risk = st.number_input("Composite Risk Score", value=0.5)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🚨 Predict Fraud", type="primary", use_container_width=True):
    
    input_data = np.array([
        amount, hour, day_of_week, month, is_weekend, is_peak_hour,
        tx_count_24h, amount_sum_24h, amount_mean_7d, amount_std_7d,
        tx_count_total, amount_mean_total, amount_std_total,
        channel_diversity, location_diversity,
        amount_vs_mean_ratio, online_channel_ratio,
        velocity_score, merchant_risk_score, composite_risk
    ]).reshape(1, -1)

    input_df = pd.DataFrame(input_data, columns=FEATURE_NAMES)

    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ **FRAUDULENT TRANSACTION DETECTED**")
            st.markdown(f"**Fraud Probability:** `{probability * 100:.2f}%`")
        else:
            st.success("✅ **Legitimate Transaction**")
            st.markdown(f"**Fraud Probability:** `{probability * 100:.2f}%`")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Footer
st.divider()
st.caption("NIBSS Fraud Detection System • Demo Version")