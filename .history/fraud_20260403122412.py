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
# PATH SETUP
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "fraud_detection_model.joblib")
DATA_FILE = os.path.join(BASE_DIR, "nibss_fraud_dataset.csv")

# -------------------------------
# LOAD MODEL (with good error message)
# -------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        st.error(f"""
        ❌ **Model file not found!**

        Expected location:  
        `{MODEL_FILE}`

        Please place the file `fraud_detection_model.joblib` in the same folder as your `fraud.py` script.
        """)
        st.stop()
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# -------------------------------
# LOAD DATA (Safe)
# -------------------------------
if not os.path.exists(DATA_FILE):
    st.error(f"""
    ❌ Dataset not found!  
    Expected: `{DATA_FILE}`
    """)
    st.stop()

ds = pd.read_csv(DATA_FILE)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<h1 style='color:#DD5746; text-align:center;'>🔍 FRAUD DETECTION SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Built by Adeseye Ademola</h4>", unsafe_allow_html=True)

if os.path.exists(os.path.join(BASE_DIR, "fraud detection.png")):
    st.image(os.path.join(BASE_DIR, "fraud detection.png"), use_container_width=True)
else:
    st.warning("Header image 'fraud detection.png' not found.")

st.divider()

# -------------------------------
# BACKGROUND
# -------------------------------
st.markdown("## Background of Study")
st.markdown("""
Fraud detection is a critical component of Data-Driven Decision Making Systems (DDDM) in digital banking platforms such as Moniepoint. 
With the rapid increase in digital transactions, detecting fraudulent activities in real-time has become essential.

This system uses machine learning to analyze transaction patterns such as amount, transaction frequency, behavioral trends, 
and risk scores to predict whether a transaction is fraudulent or legitimate.
""")

st.divider()

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
    if os.path.exists(os.path.join(BASE_DIR, "user icon.png")):
        st.image(os.path.join(BASE_DIR, "user icon.png"), caption="Welcome User")
    
    st.header("Model Performance")
    st.metric("ROC-AUC Score", "0.55")

# -------------------------------
# LOAD MODEL
# -------------------------------
model = load_model()

# -------------------------------
# FEATURE NAMES
# -------------------------------
FEATURE_NAMES = [
    'amount', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
    'tx_count_24h', 'amount_sum_24h', 'amount_mean_7d', 'amount_std_7d',
    'tx_count_total', 'amount_mean_total', 'amount_std_total',
    'channel_diversity', 'location_diversity', 'amount_vs_mean_ratio',
    'online_channel_ratio', 'velocity_score', 'merchant_risk_score', 'composite_risk'
]

# -------------------------------
# MAIN APP
# -------------------------------
def main():
    st.title("Fraud Prediction App")
    st.markdown("Enter transaction details to predict fraud")

    col1, col2, col3 = st.columns(3)

    with col1:
        amount = st.number_input("Transaction Amount (₦)", value=1000.0, min_value=0.0)
        hour = st.slider("Hour", 0, 23, 12)
        day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 2)

    with col2:
        month = st.slider("Month", 1, 12, 6)
        is_weekend = st.selectbox("Is Weekend", [0, 1], format_func=lambda x: "Yes" if x else "No")
        is_peak_hour = st.selectbox("Is Peak Hour", [0, 1], format_func=lambda x: "Yes" if x else "No")

    with col3:
        tx_count_24h = st.number_input("Transactions in 24h", value=5, min_value=0)
        amount_sum_24h = st.number_input("Total Amount in 24h (₦)", value=5000.0, min_value=0.0)
        amount_mean_7d = st.number_input("Avg Amount (7 days)", value=800.0, min_value=0.0)

    # Additional features
    st.subheader("Advanced Features")
    col4, col5 = st.columns(2)
    with col4:
        amount_std_7d = st.number_input("Std Amount (7d)", value=100.0)
        tx_count_total = st.number_input("Total Transactions", value=50.0)
        amount_mean_total = st.number_input("Mean Amount (Total)", value=900.0)
        amount_std_total = st.number_input("Std Amount (Total)", value=150.0)

    with col5:
        channel_diversity = st.number_input("Channel Diversity", value=2, min_value=1)
        location_diversity = st.number_input("Location Diversity", value=2, min_value=1)
        amount_vs_mean_ratio = st.number_input("Amount vs Mean Ratio", value=1.2)
        online_channel_ratio = st.number_input("Online Channel Ratio", value=0.5)
        velocity_score = st.number_input("Velocity Score", value=0.3)
        merchant_risk_score = st.number_input("Merchant Risk Score", value=0.4)
        composite_risk = st.number_input("Composite Risk Score", value=0.5)

    # -------------------------------
    # PREDICTION
    # -------------------------------
    if st.button("🚨 Predict Fraud", type="primary"):
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
            else:
                st.success("✅ **Legitimate Transaction**")

            st.info(f"Fraud Probability: **{probability*100:.2f}%**")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# Run the app
if __name__ == "__main__":
    main()