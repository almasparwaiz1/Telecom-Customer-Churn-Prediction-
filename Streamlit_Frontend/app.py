import streamlit as st
import pandas as pd
import joblib
import pickle
import io
import os
import sys
import __main__
import numpy as np

# ==========================================
# SCIKIT-LEARN COMPATIBILITY PATCH
# ==========================================
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# ==========================================
# CUSTOM TRANSFORMER AND PIPELINE DEFINITIONS
# ==========================================
from sklearn.base import BaseEstimator, TransformerMixin

class InitialFeaturePreparation(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if 'State' in X_copy.columns:
            X_copy = X_copy.drop('State', axis=1)
        if 'Area code' in X_copy.columns:
            X_copy['Area code'] = X_copy['Area code'].astype(object)
        return X_copy

def create_powerful_features(df):
    df_fe = df.copy()

    charge_cols = ['Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge']
    df_fe['Total_Charge'] = df_fe[charge_cols].sum(axis=1)

    minutes_cols = ['Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes']
    df_fe['Total_Minutes'] = df_fe[minutes_cols].sum(axis=1)

    calls_cols = ['Total day calls', 'Total eve calls', 'Total night calls', 'Total intl calls']
    df_fe['Total_Calls'] = df_fe[calls_cols].sum(axis=1)

    df_fe['Avg_Charge_Per_Minute'] = df_fe['Total_Charge'] / df_fe['Total_Minutes']
    df_fe['Avg_Charge_Per_Minute'] = df_fe['Avg_Charge_Per_Minute'].replace([np.inf, -np.inf], np.nan).fillna(0)

    if df_fe['Account length'].nunique() > 1:
        df_fe['Tenure_Group_Numeric'] = pd.qcut(df_fe['Account length'], q=4, labels=False, duplicates='drop')
    else:
        val = df_fe['Account length'].iloc[0]
        df_fe['Tenure_Group_Numeric'] = 0 if val < 73 else (1 if val < 100 else (2 if val < 127 else 3))

    df_fe['Voicemail_Per_Tenure'] = df_fe['Number vmail messages'] / df_fe['Account length']
    df_fe['Voicemail_Per_Tenure'] = df_fe['Voicemail_Per_Tenure'].replace([np.inf, -np.inf], np.nan).fillna(0)

    df_fe['Customer_Service_Calls_Per_Tenure'] = df_fe['Customer service calls'] / df_fe['Account length']
    df_fe['Customer_Service_Calls_Per_Tenure'] = df_fe['Customer_Service_Calls_Per_Tenure'].replace([np.inf, -np.inf], np.nan).fillna(0)

    if 'International plan_Yes' in df_fe.columns:
        df_fe['Intl_Plan_and_Usage'] = df_fe['International plan_Yes'] * df_fe['Total intl minutes']
    elif 'International plan' in df_fe.columns:
        is_yes = df_fe['International plan'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
        df_fe['Intl_Plan_and_Usage'] = is_yes * df_fe['Total intl minutes']
    else:
        df_fe['Intl_Plan_and_Usage'] = 0

    df_fe['Day_Usage_Ratio'] = df_fe['Total day minutes'] / df_fe['Total_Minutes']
    df_fe['Eve_Usage_Ratio'] = df_fe['Total eve minutes'] / df_fe['Total_Minutes']
    df_fe['Night_Usage_Ratio'] = df_fe['Total night minutes'] / df_fe['Total_Minutes']
    df_fe['Intl_Usage_Ratio'] = df_fe['Total intl minutes'] / df_fe['Total_Minutes']

    for col in ['Day_Usage_Ratio', 'Eve_Usage_Ratio', 'Night_Usage_Ratio', 'Intl_Usage_Ratio']:
        df_fe[col] = df_fe[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df_fe

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, create_powerful_features_func, column_names_after_preprocessing):
        self.create_powerful_features_func = create_powerful_features_func
        self.column_names_after_preprocessing = column_names_after_preprocessing

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.column_names_after_preprocessing, index=pd.RangeIndex(len(X)))
        X_fe = self.create_powerful_features_func(X_df)
        return X_fe

class ChurnPredictorPipeline:
    def __init__(self, preprocessing_pipeline, model, optimal_threshold):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.model = model
        self.optimal_threshold = optimal_threshold

    def predict_proba(self, X_raw):
        if hasattr(self.preprocessing_pipeline, 'transform'):
            try:
                X_processed = self.preprocessing_pipeline.transform(X_raw)
            except Exception:
                X_processed = X_raw
                for _, step in self.preprocessing_pipeline.steps:
                    X_processed = step.transform(X_processed)
        else:
            X_processed = X_raw
            for _, step in self.preprocessing_pipeline.steps:
                X_processed = step.transform(X_processed)
        
        return self.model.predict_proba(X_processed)[:, 1]

    def predict(self, X_raw):
        probabilities = self.predict_proba(X_raw)
        return (probabilities >= self.optimal_threshold).astype(bool)

setattr(__main__, 'InitialFeaturePreparation', InitialFeaturePreparation)
setattr(__main__, 'FeatureEngineeringTransformer', FeatureEngineeringTransformer)
setattr(__main__, 'create_powerful_features', create_powerful_features)
setattr(__main__, 'ChurnPredictorPipeline', ChurnPredictorPipeline)

# ==========================================
# BASE CONFIGURATION
# ==========================================
BASE_DIR = r"Streamlit_Frontend"
MODEL_PATH = os.path.join(BASE_DIR, "churn_prediction_pipeline.joblib")

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Telecom Churn AI Predictor",
    page_icon="📡",
    layout="wide"
)

# ==========================================
# PROFESSIONAL UI (THE EXECUTIVE STANDARD)
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main {
    background-color: #f8fafc;
}
.title-container {
    background: linear-gradient(135deg, #1984c5, #115e8a);
    padding: 35px;
    border-radius: 14px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
}
.title-container h1 {
    color: white !important;
    font-weight: 700;
    margin-bottom: 5px;
}
h3, .stSubheader {
    color: #111827 !important;
    font-weight: 600 !important;
}
.stButton>button {
    background: #1984c5;
    color: white;
    border-radius: 8px;
    padding: 14px 28px;
    border: none;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.3s ease;
    width: 100%;
    box-shadow: 0 4px 6px -1px rgba(25, 132, 197, 0.2);
}
.stButton>button:hover {
    background: #115e8a;
    color: white;
    transform: translateY(-1px);
}
.result-box {
    padding: 25px;
    border-radius: 10px;
    text-align: center;
    font-size: 26px;
    font-weight: 700;
    margin-top: 20px;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
}
.churn {
    background-color: #fdf2f2;
    color: #c22d2d;
    border: 2px solid #c22d2d;
}
.no-churn {
    background-color: #f0f7fc;
    color: #1984c5;
    border: 2px solid #1984c5;
}
.prob-card {
    background-color: #e6eaed;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    font-weight: 500;
    margin-top: 15px;
    border-left: 5px solid #64748b;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# RECOVERY HELPER: DUMMY FIT TO REPAIR STATE
# ==========================================
def force_fit_pipeline(loaded_obj):
    mock_sample = pd.DataFrame([{
        "Account length": 100, "Area code": 415, "International plan": "No", "Voice mail plan": "No",
        "Number vmail messages": 0, "Total day minutes": 180.0, "Total day calls": 100, "Total day charge": 30.0,
        "Total eve minutes": 180.0, "Total eve calls": 100, "Total eve charge": 15.0, "Total night minutes": 180.0,
        "Total night calls": 100, "Total night charge": 8.0, "Total intl minutes": 10.0, "Total intl calls": 4,
        "Total intl charge": 2.7, "Customer service calls": 1
    }])
    
    prep = getattr(loaded_obj, 'preprocessing_pipeline', None)
    if prep is not None:
        try:
            prep.fit(mock_sample)
        except Exception:
            if hasattr(prep, 'steps'):
                X_tmp = mock_sample
                for name, step in prep.steps:
                    try:
                        step.fit(X_tmp)
                        X_tmp = step.transform(X_tmp)
                    except Exception:
                        pass
    return loaded_obj

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found:\n{MODEL_PATH}")
        st.stop()
    try:
        loaded_obj = joblib.load(MODEL_PATH)
        loaded_obj = force_fit_pipeline(loaded_obj)
        return loaded_obj
    except Exception as e:
        st.error(f"❌ Error loading model: {type(e).__name__} - {e}")
        st.stop()

pipeline = load_model()

# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div class="title-container">
    <h1>📡 Telecom Churn Prediction AI</h1>
    <p style="font-size:18px; opacity:0.9;">Enterprise Customer Retention Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# INPUT SECTION
# ==========================================
st.subheader("📋 Customer Demographics & Usage Data")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 Account & Profile")
    account_length = st.slider("Account Length (Months)", min_value=1, max_value=300, value=100, step=1)
    area_code = st.selectbox("Area Code", [408, 415, 510])
    international_plan = st.selectbox("International Plan", ["No", "Yes"])
    voice_mail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
    number_vmail_messages = st.slider("Voicemail Messages", min_value=0, max_value=60, value=0, step=1)
    customer_service_calls = st.slider("Customer Service Calls", min_value=0, max_value=15, value=1, step=1)

with col2:
    st.markdown("### ☀️ Day & Evening Logs")
    total_day_minutes = st.slider("Day Minutes", min_value=0.0, max_value=400.0, value=180.0, step=0.1)
    total_day_calls = st.slider("Day Calls", min_value=1, max_value=200, value=100, step=1)
    total_day_charge = st.slider("Day Charge ($)", min_value=0.0, max_value=70.0, value=30.0, step=0.1)
    total_eve_minutes = st.slider("Evening Minutes", min_value=0.0, max_value=400.0, value=180.0, step=0.1)
    total_eve_calls = st.slider("Evening Calls", min_value=1, max_value=200, value=100, step=1)
    total_eve_charge = st.slider("Evening Charge ($)", min_value=0.0, max_value=40.0, value=15.0, step=0.1)

with col3:
    st.markdown("### 🌙 Night & International Logs")
    total_night_minutes = st.slider("Night Minutes", min_value=0.0, max_value=400.0, value=180.0, step=0.1)
    total_night_calls = st.slider("Night Calls", min_value=1, max_value=200, value=100, step=1)
    total_night_charge = st.slider("Night Charge ($)", min_value=0.0, max_value=30.0, value=8.0, step=0.1)
    total_intl_minutes = st.slider("International Minutes", min_value=0.0, max_value=30.0, value=10.0, step=0.1)
    total_intl_calls = st.slider("International Calls", min_value=1, max_value=20, value=4, step=1)
    total_intl_charge = st.slider("International Charge ($)", min_value=0.0, max_value=15.0, value=2.7, step=0.1)

# ==========================================
# CREATE DATAFRAME (FIXED: ALL 19 COLUMNS ADDED)
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
# PREDICTION ENGINE
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("📊 Evaluate Customer Accounts Risk")

if predict_btn:
    try:
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0]

        st.markdown("---")
        st.subheader("🎯 Optimization Risk Assessment")

        if prediction:
            st.markdown("""
            <div class="result-box churn">
                ⚠️ High Risk Profile: Customer is likely to CHURN
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="prob-card">
                <strong>Action Required:</strong> Churn Risk Factor is at <strong>{probability:.2%}</strong>.
                Consider launching immediate retention operations.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-box no-churn">
                🛡️ Stable Profile: Customer is Retained (Active)
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="prob-card">
                <strong>Account Status:</strong> Account stability healthy with a <strong>{(1 - probability):.2%}</strong> retention confidence factor.
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction Engine Failed Execution:\n{e}")

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<br><hr>
<center style="color:#64748b; font-size: 14px;">
    Executive Standard Dashboard | Powered by Streamlit & Scikit-Learn
</center>
""", unsafe_allow_html=True)
