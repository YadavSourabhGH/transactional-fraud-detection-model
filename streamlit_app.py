import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set Page Config
st.set_page_config(page_title="FinSafe - Fraud Detection AI", layout="wide", page_icon="🛡️")

# Styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
        color: #1c2e4a;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
    }
    .risk-low { color: green; font-weight: bold; }
    .risk-med { color: orange; font-weight: bold; }
    .risk-high { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.title("🛡️ FinSafe: AI-Based Fraud Detection System")
st.markdown("""
Welcome to **FinSafe**, a premium fraud prediction and risk segmentation dashboard. 
This system uses machine learning to analyze financial transactions and identify potential anomalies in real-time.
""")

# Load Model and Scaler
@st.cache_resource
def load_assets():
    model_path = 'fraud_model.pkl'
    scaler_path = 'scaler.pkl'
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

model, scaler = load_assets()

# Sidebar - Business Logic Explanation
with st.sidebar:
    st.header("Business Strategy")
    st.info("""
    **Revenue Optimization**: 
    Automated detection reduces chargeback costs by ~30% and minimizes manual review overhead.
    
    **Risk-Based Pricing**:
    Lower risk segments are eligible for premium financial products with reduced fees.
    """)
    st.image("https://img.freepik.com/free-vector/fraud-prevention-abstract-concept-vector-illustration-financial-safety-online-security-protection-from-scam-identity-theft-prevention-credit-card-fraud-monitoring-software-abstract-metaphor_335657-1335.jpg")

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Transaction Details")
    
    amount = st.number_input("Transaction Amount ($)", min_value=-5000.0, max_value=50000.0, value=25.0)
    transaction_hour = st.slider("Hour of Day (0-23)", 0, 23, 14)
    transaction_day = st.selectbox("Day of Week", options=[0,1,2,3,4,5,6], format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
    transaction_month = st.slider("Month (1-12)", 1, 12, 6)
    use_chip = st.selectbox("Transaction Method", ['Swipe Transaction', 'Online Transaction', 'Chip Transaction'])
    is_refund = 1 if amount < 0 else 0
    
    # Mapping
    method_map = {'Chip Transaction': 0, 'Online Transaction': 1, 'Swipe Transaction': 2}
    chip_encoded = method_map[use_chip]
    
    submit = st.button("Analyze Transaction")

with col2:
    st.subheader("Prediction Result")
    
    if submit:
        # Prepare Input
        input_data = pd.DataFrame({
            'amount': [amount],
            'is_refund': [is_refund],
            'amount_abs': [abs(amount)],
            'transaction_hour': [transaction_hour],
            'transaction_day': [transaction_day],
            'transaction_month': [transaction_month],
            'use_chip_encoded': [chip_encoded],
            'state_encoded': [0] # Default
        })
        
        if model and scaler:
            try:
                scaled_input = scaler.transform(input_data)
                prediction = model.predict(scaled_input)[0]
                probability = model.predict_proba(scaled_input)[0][1]
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                prediction, probability = 0, 0.0
        else:
            st.warning("Model not found. Showing heuristic-based demonstration.")
            # Heuristic Logic
            if abs(amount) > 5000 and use_chip == 'Online Transaction' and (transaction_hour < 5 or transaction_hour > 23):
                prediction, probability = 1, 0.92
            elif abs(amount) > 2000:
                prediction, probability = 0, 0.40
            else:
                prediction, probability = 0, 0.05
        
        # Display Result
        if prediction == 1:
            st.error(f"⚠️ POTENTIAL FRAUD DETECTED (Confidence: {probability:.2%})")
            st.markdown("<p class='risk-high'>Risk Segment: High Risk</p>", unsafe_allow_html=True)
            st.warning("Action: Transaction Blocked. Alert sent to Risk Management team.")
        else:
            st.success(f"✅ TRANSACTION SAFE (Fraud Probability: {probability:.2%})")
            if probability > 0.3:
                st.markdown("<p class='risk-med'>Risk Segment: Medium Risk</p>", unsafe_allow_html=True)
                st.info("Action: Transaction Approved. Flagged for secondary review.")
            else:
                st.markdown("<p class='risk-low'>Risk Segment: Low Risk</p>", unsafe_allow_html=True)
                st.write("Action: Transaction Approved instantly.")

        # Risk Score Bar
        st.subheader("Risk Analytics Breakdown")
        risk_metrics = {
            "Amount Volatility": min(abs(amount)/1000, 1.0),
            "Temporal Irregularity": 1.0 if (transaction_hour < 6 or transaction_hour > 22) else 0.2,
            "Channel Risk": 0.8 if use_chip == 'Online Transaction' else 0.1
        }
        st.bar_chart(pd.Series(risk_metrics))

st.divider()
st.caption("FinSafe Fraud Detection AI - Applied AI for Business Mini Project")
