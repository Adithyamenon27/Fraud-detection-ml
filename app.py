import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(
    page_title="Transaction Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# =======================
# THE "SECRET" TO CLEARING INPUTS
# =======================
# We use a version counter in the session state.
# Incrementing this 'form_id' changes the key of every widget, forcing a reset.
if 'form_id' not in st.session_state:
    st.session_state.form_id = 0


def clear_all_inputs():
    st.session_state.form_id += 1


# =======================
# LOAD MODEL ASSETS
# =======================
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open("model1.pkl", "rb"))
        scaler = pickle.load(open("scaler1.pkl", "rb"))
        features = pickle.load(open("features1.pkl", "rb"))
        return model, scaler, features
    except FileNotFoundError:
        return None, None, None


model, scaler, features = load_assets()

# =======================
# UI DESIGN (CSS)
# =======================
st.markdown("""
<style>
    .stNumberInput input { font-weight: bold; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3.5em; }
    div[data-testid="stMetricValue"] { font-size: 28px; }
</style>
""", unsafe_allow_html=True)

# =======================
# HEADER
# =======================
st.image("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&q=80&w=1200&h=300")
st.title("🛡️ FraudGuard:Fraud Transaction Detection")
st.info("Fill in the transaction details.")

# =======================
# INPUT SECTION
# =======================
# Note: Every 'key' is tied to st.session_state.form_id
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Senders Info")
        amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01, key=f"amt_{st.session_state.form_id}")
        oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, step=0.01,
                                        key=f"so_{st.session_state.form_id}")
        newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, step=0.01,
                                         key=f"sn_{st.session_state.form_id}")

    with col2:
        st.subheader("🏦 Receiver's Info")
        oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, step=0.01,
                                         key=f"ro_{st.session_state.form_id}")
        newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, step=0.01,
                                         key=f"rn_{st.session_state.form_id}")
        transaction_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"],
                                        key=f"type_{st.session_state.form_id}")

# =======================
# ACTION BUTTONS
# =======================
st.write(" ")  # Spacer
btn_col1, btn_col2 = st.columns([2, 1])

with btn_col1:
    predict_btn = st.button("🚀 Run Fraud Analysis", type="primary")

with btn_col2:
    # This button triggers the clear function
    st.button("🧹 Clear All Inputs", on_click=clear_all_inputs)

# =======================
# PREDICTION LOGIC
# =======================
if predict_btn:
    if model is None:
        st.error("Model files not found. Please upload model.pkl, scaler.pkl, and features.pkl.")
    else:
        with st.spinner("Analyzing patterns..."):
            # Feature Engineering
            errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
            errorBalanceDest = oldbalanceDest + amount - newbalanceDest
            type_CASH_OUT = 1 if transaction_type == "CASH_OUT" else 0

            input_data = {
                "amount": amount, "oldbalanceOrg": oldbalanceOrg, "newbalanceOrig": newbalanceOrig,
                "oldbalanceDest": oldbalanceDest, "newbalanceDest": newbalanceDest,
                "errorBalanceOrig": errorBalanceOrig, "errorBalanceDest": errorBalanceDest,
                "type_CASH_OUT": type_CASH_OUT
            }

            input_df = pd.DataFrame([input_data]).reindex(columns=features, fill_value=0)
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            st.divider()

            if prediction == 1:
                st.error(f"### 🚨 Fraud Detected! Risk Score: {probability:.2%}")
                st.write(
                    "This transaction shows high-risk characteristics common in money laundering or unauthorized transfers.")
            else:
                st.success(f"### ✅ Transaction Legitimate. Risk Score: {probability:.2%}")
                st.balloons()

st.divider()
st.caption("AI-Powered Fraud Prevention System | Real-time Analysis enabled")
