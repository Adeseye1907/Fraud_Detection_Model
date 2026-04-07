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