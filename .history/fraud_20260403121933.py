import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')
FileNotFoundError: [Errno 2] No such file or directory: 'fraud_detection_model.joblib'

File "C:\Users\user\Desktop\Machine Learning\Project\Fruad detection\fraud.py", line 57, in <module>
    model = load_model()
File "C:\Users\user\Desktop\Machine Learning\streamlit3_13\Lib\site-packages\streamlit\runtime\caching\cache_utils.py", line 227, in __call__
    return self._get_or_create_cached_value(args, kwargs, spinner_message)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\user\Desktop\Machine Learning\streamlit3_13\Lib\site-packages\streamlit\runtime\caching\cache_utils.py", line 269, in _get_or_create_cached_value
    return self._handle_cache_miss(cache, value_key, func_args, func_kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\user\Desktop\Machine Learning\streamlit3_13\Lib\site-packages\streamlit\runtime\caching\cache_utils.py", line 328, in _handle_cache_miss
    computed_value = self._info.func(*func_args, **func_kwargs)
File "C:\Users\user\Desktop\Machine Learning\Project\Fruad detection\fraud.py", line 55, in load_model
    return joblib.load(MODEL_FILE)
           ~~~~~~~~~~~^^^^^^^^^^^^
File "C:\Users\user\Desktop\Machine Learning\streamlit3_13\Lib\site-packages\joblib\numpy_pickle.py", line 735, in load
    with open(filename, "rb") as f:
         ~~~~^^^^^^^^^^^^^^^^

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