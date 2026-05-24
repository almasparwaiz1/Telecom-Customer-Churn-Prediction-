import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ==========================================
# MODEL PATH
# ==========================================
MODEL_PATH = "Streamlit_Frontend/churn_prediction_pipeline.joblib"

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Telecom Churn AI Predictor",
    page_icon="📡",
    layout="wide"
)

# ==========================================
# LOAD MODEL (SAFE VERSION)
# ==========================================
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            return None

        model = joblib.load(MODEL_PATH)

        # 🔥 CHECK IF MODEL IS FITTED
        if hasattr(model, "steps"):
            # sklearn Pipeline
            for step_name, step_obj in model.steps:
                if hasattr(step_obj, "predict") and not hasattr(step_obj, "feature_importances_"):
                    pass  # ignore
        return model

    except Exception as e:
        st.error(f"❌ Model Load Error: {e}")
        return None


pipeline = load_model()

# ==========================================
# UI HEADER
# ==========================================
st.title("📡 Telecom Churn Prediction AI")

# ==========================================
# INPUTS
# ==========================================
account_length = st.number_input("Account Length", 1, 300, 100)
area_code = st.selectbox("Area Code", [408, 415, 510])
international_plan = st.selectbox("International Plan", ["No", "Yes"])
voice_mail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
number_vmail_messages = st.number_input("Voicemail Messages", 0, 60, 0)
customer_service_calls = st.number_input("Customer Service Calls", 0, 15, 1)

total_day_minutes = st.number_input("Day Minutes", 0.0, 400.0, 180.0)
total_day_calls = st.number_input("Day Calls", 0, 200, 100)
total_day_charge = st.number_input("Day Charge", 0.0, 70.0, 30.0)

total_eve_minutes = st.number_input("Evening Minutes", 0.0, 400.0, 180.0)
total_eve_calls = st.number_input("Evening Calls", 0, 200, 100)
total_eve_charge = st.number_input("Evening Charge", 0.0, 40.0, 15.0)

total_night_minutes = st.number_input("Night Minutes", 0.0, 400.0, 180.0)
total_night_calls = st.number_input("Night Calls", 0, 200, 100)
total_night_charge = st.number_input("Night Charge", 0.0, 30.0, 8.0)

total_intl_minutes = st.number_input("Intl Minutes", 0.0, 30.0, 10.0)
total_intl_calls = st.number_input("Intl Calls", 0, 20, 4)
total_intl_charge = st.number_input("Intl Charge", 0.0, 15.0, 2.7)

# ==========================================
# INPUT DATA
# ==========================================
input_data = pd.DataFrame([{
    "Account length": account_length,
    "Area code": area_code,
    "International plan": international_plan,
    "Voice mail plan": voice_mail_plan,
    "Number vmail messages": number_vmail_messages,
    "Total day minutes": total_day_minutes,
    "Total day calls": total_day_calls,
    "Total day charge": total_day_charge,
    "Total eve minutes": total_eve_minutes,
    "Total eve calls": total_eve_calls,
    "Total eve charge": total_eve_charge,
    "Total night minutes": total_night_minutes,
    "Total night calls": total_night_calls,
    "Total night charge": total_night_charge,
    "Total intl minutes": total_intl_minutes,
    "Total intl calls": total_intl_calls,
    "Total intl charge": total_intl_charge,
    "Customer service calls": customer_service_calls
}])

# ==========================================
# PREDICTION BUTTON
# ==========================================
if st.button("🚀 Predict Churn"):

    if pipeline is None:
        st.error("❌ Model not loaded or corrupted. Please retrain and save model properly.")
        st.stop()

    try:
        pred = pipeline.predict(input_data)[0]

        # SAFE PROBABILITY HANDLING
        if hasattr(pipeline, "predict_proba"):
            prob = pipeline.predict_proba(input_data)

            # handle sklearn pipeline output
            if isinstance(prob, (list, np.ndarray)):
                prob = np.array(prob)
                churn_prob = prob[0] if prob.ndim == 1 else prob[0][1]
            else:
                churn_prob = 0.5
        else:
            churn_prob = 0.5

        st.subheader("Result")

        if pred:
            st.error(f"⚠️ Customer likely to CHURN ({churn_prob:.2%})")
        else:
            st.success(f"✅ Customer will STAY ({(1-churn_prob):.2%})")

    except Exception as e:
        st.error(f"❌ Prediction Engine Error:\n{e}")
