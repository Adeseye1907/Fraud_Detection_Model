import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings

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


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration ---
# LOAD MODEL + SCALER (FIXED PATH)
# -------------------------------
import os
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE_PATH = os.path.join(BASE_DIR, "fraud_detection_model.joblib")



@st.cache_resource
def load_objects():
    if not os.path.exists(MODEL_FILE):
        st.error(f"Model not found: {MODEL_FILE}")
        st.stop()

    if not os.path.exists(SCALER_FILE):
        st.error(f"Scaler not found: {SCALER_FILE}")
        st.stop()

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    return model, scaler

model, scaler = load_objects()

FEATURE_NAMES = [
'transaction_id', 'customer_id', 'timestamp', 'amount', 'channel','merchant_category', 'bank', 'location', 'age_group', 'hour','day_of_week', 'month', 'is_weekend', 'is_peak_hour', 'tx_count_24h',
'amount_sum_24h', 'amount_mean_7d', 'amount_std_7d', 'tx_count_total',
'amount_mean_total', 'amount_std_total', 'channel_diversity',
'location_diversity', 'amount_vs_mean_ratio', 'online_channel_ratio',
'is_fraud', 'fraud_technique', 'hour_sin', 'hour_cos', 'day_sin',
'day_cos', 'month_sin', 'month_cos', 'amount_log', 'amount_rounded',
'velocity_score', 'merchant_risk_score', 'composite_risk' # Must use the name listed during your fit time
]



# --- Load Model with Caching for Speed ---
@st.cache_resource
def load_model():
    """Loads the trained model from the file."""
    try:
        model = joblib.load(MODEL_FILE_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{MODEL_FILE_PATH}'. Please check the file name.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

model = load_model()

# --- Streamlit UI and Prediction Logic ---

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

    input_data = np.array([[
        amount, hour, day_of_week, month, is_weekend, is_peak_hour,
        tx_count_24h, amount_sum_24h, amount_mean_7d, amount_std_7d,
        tx_count_total, amount_mean_total, amount_std_total,
        channel_diversity, location_diversity,
        amount_vs_mean_ratio, online_channel_ratio,
        velocity_score, merchant_risk_score, composite_risk
    ]])

    # SCALE INPUT

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Result")

    if prediction == 1:
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



