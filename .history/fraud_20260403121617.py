import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# 1. PAGE CONFIG
st.set_page_config(page_title="NIBSS Fraud Guard", layout="wide")

# 2. PATH HANDLING (Fixes FileNotFoundError)
base_path = os.path.dirname(__file__)
MODEL_FILE = os.path.join(base_path, 'fraud_detection_model.joblib')
ENCODER_FILE = os.path.join(base_path, 'encoders.pkl')
DATA_FILE = os.path.join(base_path, 'nibss_fraud_dataset.csv')

# 3. LOAD ASSETS
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_FILE)
    encoders = joblib.load(ENCODER_FILE)
    return model, encoders

try:
    model, encoders = load_assets()
    ds = pd.read_csv(DATA_FILE)
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# 4. HEADER & UI
st.markdown("<h1 style='color:#DD5746; text-align:center; font-family:Monospace;'>FRAUD DETECTION SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#FFC470;'>Built by Adeseye Ademola</h4>", unsafe_allow_html=True)

try:
    st.image('fraud detection.png')
except:
    pass

st.divider()

# 5. BACKGROUND
with st.expander("Background of Study"):
    st.markdown("""
    This system uses a **Balanced Logistic Regression** model to analyze NIBSS transaction patterns. 
    It evaluates risk based on amount, frequency, behavioral trends, and risk scores.
    """)

st.subheader("📊 Sample Historical Data")
st.dataframe(ds.head(), use_container_width=True)

# 6. USER INPUT
def main():
    st.sidebar.image('user icon.png', caption='Welcome User')
    st.sidebar.header("Model Performance")
    st.sidebar.metric("ROC-AUC Score", "0.89") # Based on your Smote/Balanced results

    st.markdown("### 🔍 New Transaction Evaluation")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Core Details**")
            amount = st.number_input("Transaction Amount", value=1000.0)
            channel = st.selectbox("Channel", ["Web", "POS", "ATM", "Mobile", "USSD"])
            bank = st.selectbox("Bank", ds['bank'].unique())
            location = st.selectbox("Location", ds['location'].unique())
            merchant_cat = st.selectbox("Merchant Category", ds['merchant_category'].unique())

        with col2:
            st.write("**Time & Frequency**")
            hour = st.slider("Hour (0-23)", 0, 23, 12)
            day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 2)
            month = st.slider("Month", 1, 12, 6)
            tx_count_24h = st.number_input("Transactions (24h)", value=1.0)
            tx_count_total = st.number_input("Total Transactions (Lifetime)", value=10.0)

        with col3:
            st.write("**Risk Scores**")
            velocity_score = st.slider("Velocity Score", 0.0, 1.0, 0.1)
            merchant_risk_score = st.slider("Merchant Risk Score", 0.0, 1.0, 0.2)
            composite_risk = st.slider("Composite Risk Score", 0.0, 1.0, 0.1)
            amount_vs_mean = st.number_input("Amount vs Mean Ratio", value=1.0)

        submit = st.form_submit_button("Analyze Transaction")

    if submit:
        # --- 1. PREPARE DATAFRAME (Must match exact training order) ---
        # Note: In your notebook, 'x' was ds.drop('is_fraud', axis=1). 
        # We must reconstruct that full list.
        
        # Start with a template of zeros for ALL columns in the dataset
        input_row = ds.drop('is_fraud', axis=1).iloc[0:1].copy()
        for col in input_row.columns:
            input_row[col] = 0 
        
        # Fill in the values we collected
        input_row['amount'] = amount
        input_row['channel'] = channel
        input_row['bank'] = bank
        input_row['location'] = location
        input_row['merchant_category'] = merchant_cat
        input_row['hour'] = hour
        input_row['day_of_week'] = day_of_week
        input_row['month'] = month
        input_row['tx_count_24h'] = tx_count_24h
        input_row['tx_count_total'] = tx_count_total
        input_row['velocity_score'] = velocity_score
        input_row['merchant_risk_score'] = merchant_risk_score
        input_row['composite_risk'] = composite_risk
        input_row['amount_vs_mean_ratio'] = amount_vs_mean
        
        # --- 2. ENCODING CATEGORIES ---
        for col, le in encoders.items():
            if col in input_row.columns:
                try:
                    input_row[col] = le.transform([input_row[col].iloc[0]])
                except:
                    input_row[col] = 0 # Fallback for unknown categories

        # --- 3. PREDICTION ---
        try:
            prediction = model.predict(input_row)[0]
            probability = model.predict_proba(input_row)[0][1]

            st.divider()
            if prediction == 1:
                st.error(f"⚠️ FRAUDULENT TRANSACTION DETECTED (Risk: {probability:.2%})")
                st.warning("Action: Block transaction and notify user.")
            else:
                st.success(f"✅ LEGITIMATE TRANSACTION (Risk: {probability:.2%})")
                st.balloons()
        except Exception as e:
            st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()

# -------------------------------
# LOAD DATA
# -------------------------------
ds = pd.read_csv('nibss_fraud_dataset.csv')

st.markdown("<h1 style='color:#DD5746; text-align:center;'>FRAUD DETECTION SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Built by Adeseye Ademola</h4>", unsafe_allow_html=True)

st.image('fraud detection.png')
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

# Show dataset
st.dataframe(ds.head(), use_container_width=True)

# Sidebar
st.sidebar.image('user icon.png', caption='Welcome User')

# -------------------------------
# LOAD MODEL
# -------------------------------
MODEL_FILE = 'fraud_detection_model.joblib'

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

model = load_model()

# -------------------------------
# FEATURE LIST (MATCH TRAINING)
# -------------------------------
FEATURE_NAMES = [
    'amount',
    'hour',
    'day_of_week',
    'month',
    'is_weekend',
    'is_peak_hour',
    'tx_count_24h',
    'amount_sum_24h',
    'amount_mean_7d',
    'amount_std_7d',
    'tx_count_total',
    'amount_mean_total',
    'amount_std_total',
    'channel_diversity',
    'location_diversity',
    'amount_vs_mean_ratio',
    'online_channel_ratio',
    'velocity_score',
    'merchant_risk_score',
    'composite_risk'
]

# -------------------------------
# USER INPUT
# -------------------------------
def main():

    st.title("Fraud Prediction App")

    st.markdown("Enter transaction details to predict fraud")

    col1, col2, col3 = st.columns(3)

    with col1:
        amount = st.number_input("Transaction Amount", value=1000.0)
        hour = st.slider("Hour", 0, 23, 12)
        day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 2)

    with col2:
        month = st.slider("Month", 1, 12, 6)
        is_weekend = st.selectbox("Is Weekend", [0, 1])
        is_peak_hour = st.selectbox("Is Peak Hour", [0, 1])

    with col3:
        tx_count_24h = st.number_input("Transactions (24h)", value=5.0)
        amount_sum_24h = st.number_input("Total Amount (24h)", value=5000.0)
        amount_mean_7d = st.number_input("Avg Amount (7d)", value=800.0)

    # MORE FEATURES
    amount_std_7d = st.number_input("Std Amount (7d)", value=100.0)
    tx_count_total = st.number_input("Total Transactions", value=50.0)
    amount_mean_total = st.number_input("Mean Amount (Total)", value=900.0)
    amount_std_total = st.number_input("Std Amount (Total)", value=150.0)

    channel_diversity = st.number_input("Channel Diversity", value=2)
    location_diversity = st.number_input("Location Diversity", value=2)

    amount_vs_mean_ratio = st.number_input("Amount vs Mean Ratio", value=1.2)
    online_channel_ratio = st.number_input("Online Channel Ratio", value=0.5)

    velocity_score = st.number_input("Velocity Score", value=0.3)
    merchant_risk_score = st.number_input("Merchant Risk Score", value=0.4)
    composite_risk = st.number_input("Composite Risk Score", value=0.5)

    # -------------------------------
    # PREDICTION
    # -------------------------------
    if st.button("Predict Fraud"):

        input_data = np.array([
            amount, hour, day_of_week, month, is_weekend, is_peak_hour,
            tx_count_24h, amount_sum_24h, amount_mean_7d, amount_std_7d,
            tx_count_total, amount_mean_total, amount_std_total,
            channel_diversity, location_diversity,
            amount_vs_mean_ratio, online_channel_ratio,
            velocity_score, merchant_risk_score, composite_risk
        ]).reshape(1, -1)

        input_df = pd.DataFrame(input_data, columns=FEATURE_NAMES)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Fraudulent Transaction Detected")
        else:
            st.success("✅ Legitimate Transaction")

        st.info(f"Fraud Probability: {probability*100:.2f}%")

# -------------------------------
# SIDEBAR METRIC
# -------------------------------
st.sidebar.header("Model Performance")
st.sidebar.metric("ROC-AUC Score", "0.55")

# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    main()