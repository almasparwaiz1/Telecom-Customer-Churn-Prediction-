import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Enterprise Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Professional UI Injecting Color Combos
st.markdown("""
    <style>
        .reportview-container { background: #fdfdfd; }
        .metric-card {
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        /* Color Palette 1: Teal to Coral/Orange (Risk Status) */
        .status-low-risk { background-color: #E6F4F1; border-left: 5px solid #008080; color: #004D4D; }
        .status-high-risk { background-color: #FFF2E6; border-left: 5px solid #FF7F50; color: #993300; }
        
        /* Color Palette 2: Dark Blue to Crimson (Executive Focus) */
        .exec-baseline { background-color: #EAEFF5; border-left: 5px solid #002D62; color: #001A3A; }
        .exec-critical { background-color: #FCE8E6; border-left: 5px solid #DC143C; color: #7A0A1D; }
        
        /* Color Palette 3: Light Gray to Dark Purple (Distribution Profiles) */
        .dist-neutral { background-color: #F5F5F5; border-left: 5px solid #808080; color: #333333; }
        .dist-complex { background-color: #F3E6F7; border-left: 5px solid #4B0082; color: #2E004F; }
    </style>
""", unsafe_allow_html=True)

# --- File Paths Configuration ---
BASE_DIR = r"F:\AI and Data Science Projects\Telecom churn prediction app\Streamlit_Frontend"

MODEL_PATH = os.path.join(BASE_DIR, 'stacking_classifier_model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.joblib')
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, 'feature_columns.joblib')
OPTIMAL_THRESHOLD = 0.8539040781841282

# --- Helper Functions (Loading Assets with Caching) ---
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Critical Error: Predictive Engine asset missing at configuration path: {path}")
        st.stop()
    return joblib.load(path)

@st.cache_resource
def load_scaler(path):
    if not os.path.exists(path):
        st.error(f"Critical Error: Standardization Engine asset missing at configuration path: {path}")
        st.stop()
    return joblib.load(path)

@st.cache_resource
def load_feature_columns(path):
    if not os.path.exists(path):
        st.error(f"Critical Error: Feature Metadata schema asset missing at configuration path: {path}")
        st.stop()
    return joblib.load(path)

# --- Feature Engineering Function ---
def create_powerful_features(df):
    df_fe = df.copy()

    # 1. Total Charges (Avg monthly spend proxy)
    charge_cols = ['Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge']
    df_fe['Total_Charge'] = df_fe[charge_cols].sum(axis=1)

    # 2. Total Minutes & Total Calls
    minutes_cols = ['Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes']
    df_fe['Total_Minutes'] = df_fe[minutes_cols].sum(axis=1)

    calls_cols = ['Total day calls', 'Total eve calls', 'Total night calls', 'Total intl calls']
    df_fe['Total_Calls'] = df_fe[calls_cols].sum(axis=1)

    # 3. Average Charge per Minute (Overall)
    df_fe['Avg_Charge_Per_Minute'] = df_fe['Total_Charge'] / df_fe['Total_Minutes']
    df_fe['Avg_Charge_Per_Minute'] = df_fe['Avg_Charge_Per_Minute'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 4. Tenure Groups
    if df_fe['Account length'].nunique() > 1:
        df_fe['Tenure_Group_Numeric'] = pd.qcut(df_fe['Account length'], q=4, labels=False, duplicates='drop')
    else:
        df_fe['Tenure_Group_Numeric'] = 0

    # 5. Voicemail Usage Intensity
    df_fe['Voicemail_Per_Tenure'] = df_fe['Number vmail messages'] / df_fe['Account length']
    df_fe['Voicemail_Per_Tenure'] = df_fe['Voicemail_Per_Tenure'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 6. Customer Service Call Intensity
    df_fe['Customer_Service_Calls_Per_Tenure'] = df_fe['Customer service calls'] / df_fe['Account length']
    df_fe['Customer_Service_Calls_Per_Tenure'] = df_fe['Customer_Service_Calls_Per_Tenure'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 7. Interaction: International Plan and International Usage
    if 'International plan_Yes' in df_fe.columns:
        df_fe['Intl_Plan_and_Usage'] = df_fe['International plan_Yes'] * df_fe['Total intl minutes']
    else:
        df_fe['Intl_Plan_and_Usage'] = 0

    # 8. Usage Ratios based on total minutes
    df_fe['Day_Usage_Ratio'] = df_fe['Total day minutes'] / df_fe['Total_Minutes']
    df_fe['Eve_Usage_Ratio'] = df_fe['Total eve minutes'] / df_fe['Total_Minutes']
    df_fe['Night_Usage_Ratio'] = df_fe['Total night minutes'] / df_fe['Total_Minutes']
    df_fe['Intl_Usage_Ratio'] = df_fe['Total intl minutes'] / df_fe['Total_Minutes']

    for col in ['Day_Usage_Ratio', 'Eve_Usage_Ratio', 'Night_Usage_Ratio', 'Intl_Usage_Ratio']:
        df_fe[col] = df_fe[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df_fe

# --- Preprocessing and Prediction Function ---
def preprocess_and_predict(input_df_raw, model, scaler, all_feature_columns):
    df = input_df_raw.copy()

    original_numerical_features = [
        'Account length', 'Number vmail messages', 'Total day minutes', 'Total day calls',
        'Total day charge', 'Total eve minutes', 'Total eve calls', 'Total eve charge',
        'Total night minutes', 'Total night calls', 'Total night charge',
        'Total intl minutes', 'Total intl calls', 'Total intl charge', 'Customer service calls'
    ]
    original_categorical_features = ['International plan', 'Voice mail plan', 'Area code']

    df['Area code'] = df['Area code'].astype(object)
    df_encoded = pd.get_dummies(df, columns=original_categorical_features, drop_first=True)

    expected_ohe_cols = ['International plan_Yes', 'Voice mail plan_Yes', 'Area code_415', 'Area code_510']
    for col in expected_ohe_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    if 'State' in df_encoded.columns:
        df_encoded = df_encoded.drop('State', axis=1)

    numerical_cols_for_scaling = [col for col in original_numerical_features if col in df_encoded.columns]
    df_for_scaling = df_encoded[numerical_cols_for_scaling].copy()

    df_scaled_numerical = pd.DataFrame(scaler.transform(df_for_scaling), columns=numerical_cols_for_scaling, index=df_encoded.index)

    for col in numerical_cols_for_scaling:
        df_encoded[col] = df_scaled_numerical[col]

    df_fe = create_powerful_features(df_encoded.copy())

    missing_cols = set(all_feature_columns) - set(df_fe.columns)
    for c in missing_cols:
        df_fe[c] = 0

    final_input_df = df_fe[all_feature_columns]

    churn_probability = model.predict_proba(final_input_df)[:, 1][0]
    churn_prediction = (churn_probability >= OPTIMAL_THRESHOLD).astype(bool)

    return churn_prediction, churn_probability

# --- Load Operational Model Assets ---
model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)
all_feature_columns = load_feature_columns(FEATURE_COLUMNS_PATH)

# --- Executive User Interface ---
st.title("💼 Enterprise Customer Churn Intelligence")
st.markdown("""
    This advanced analytical tool leverages an ensemble stacking architecture to evaluate account risk profiles.
    Adjust tactical customer account parameters via the slider matrix in the control panel below to compute precise churn risk probabilities.
""")

# Sidebar Structure using Interactive Sliders for Premium Feel
with st.sidebar:
    st.header("🎛️ Account Matrix Control Panel")
    
    with st.expander("📊 Account Metadata Profiles", expanded=True):
        account_length = st.slider("Account Vintage Length (Days)", min_value=1, max_value=250, value=100, step=1)
        area_code = st.selectbox("Regional Area Code", options=[408, 415, 510], index=1)
        international_plan = st.radio("Active International Provisioning Plan", options=["No", "Yes"], index=0)
        voice_mail_plan = st.radio("Active Voicemail Provisioning Plan", options=["No", "Yes"], index=0)
        number_vmail_messages = st.slider("Stored Voicemail Message Volume", min_value=0, max_value=50, value=0, step=1)
        customer_service_calls = st.slider("Customer Support Communications Escalations", min_value=0, max_value=9, value=1, step=1)

    with st.expander("📈 Temporal Usage & Charges Matrix", expanded=False):
        st.markdown("### Daytime Call Metrology")
        total_day_minutes = st.slider("Accumulated Daytime Duration (Mins)", min_value=0.0, max_value=350.0, value=180.0, step=0.1)
        total_day_calls = st.slider("Total Daytime Outbound Calls", min_value=0, max_value=170, value=100, step=1)
        total_day_charge = st.slider("Total Accrued Daytime Invoice ($)", min_value=0.0, max_value=60.0, value=30.0, step=0.01)

        st.markdown("### Evening Call Metrology")
        total_eve_minutes = st.slider("Accumulated Evening Duration (Mins)", min_value=0.0, max_value=360.0, value=200.0, step=0.1)
        total_eve_calls = st.slider("Total Evening Outbound Calls", min_value=0, max_value=170, value=100, step=1)
        total_eve_charge = st.slider("Total Accrued Evening Invoice ($)", min_value=0.0, max_value=35.0, value=17.0, step=0.01)

        st.markdown("### Nocturnal Call Metrology")
        total_night_minutes = st.slider("Accumulated Night Duration (Mins)", min_value=0.0, max_value=400.0, value=200.0, step=0.1)
        total_night_calls = st.slider("Total Night Outbound Calls", min_value=0, max_value=170, value=100, step=1)
        total_night_charge = st.slider("Total Accrued Night Invoice ($)", min_value=0.0, max_value=18.0, value=9.0, step=0.01)

        st.markdown("### International Call Metrology")
        total_intl_minutes = st.slider("Accumulated Cross-Border Duration (Mins)", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
        total_intl_calls = st.slider("Total Cross-Border Outbound Calls", min_value=0, max_value=20, value=4, step=1)
        total_intl_charge = st.slider("Total Accrued Cross-Border Invoice ($)", min_value=0.0, max_value=6.0, value=2.70, step=0.01)

    st.markdown("---")
    predict_button = st.button("Execute Risk Evaluation Model", use_container_width=True)

# --- Executive Insight & Main Content Execution Area ---
if predict_button:
    input_data = {
        'Account length': [account_length], 'Area code': [area_code], 'International plan': [international_plan],
        'Voice mail plan': [voice_mail_plan], 'Number vmail messages': [number_vmail_messages],
        'Total day minutes': [total_day_minutes], 'Total day calls': [total_day_calls], 'Total day charge': [total_day_charge],
        'Total eve minutes': [total_eve_minutes], 'Total eve calls': [total_eve_calls], 'Total eve charge': [total_eve_charge],
        'Total night minutes': [total_night_minutes], 'Total night calls': [total_night_calls], 'Total night charge': [total_night_charge],
        'Total intl minutes': [total_intl_minutes], 'Total intl calls': [total_intl_calls], 'Total intl charge': [total_intl_charge],
        'Customer service calls': [customer_service_calls]
    }
    input_df_raw = pd.DataFrame(input_data)

    try:
        churn_pred, churn_prob = preprocess_and_predict(input_df_raw, model, scaler, all_feature_columns)

        st.subheader("📋 Executive Strategic Risk Assessment")
        
        # Defining Dynamic HTML Classes based on requested palettes
        status_class = "status-high-risk" if churn_pred else "status-low-risk"
        exec_class = "exec-critical" if churn_prob > 0.50 else "exec-baseline"
        dist_class = "dist-complex" if churn_prob > OPTIMAL_THRESHOLD else "dist-neutral"
        
        # Layout metrics with clean CSS Injector grids
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Teal to Coral/Orange System
            status_text = "CRITICAL CHURN RISK FLAG" if churn_pred else "SECURE ACCOUNT HOLDER"
            st.markdown(f"""
                <div class="metric-card {status_class}">
                    <h3>Account Security State</h3>
                    <h2 style='margin: 0;'>{status_text}</h2>
                    <small>Classification relative to system baseline optimal parameters</small>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            # Navy to Crimson Executive Metric
            st.markdown(f"""
                <div class="metric-card {exec_class}">
                    <h3>Executive Churn Weight</h3>
                    <h2 style='margin: 0;'>{churn_prob:.2%} Probability</h2>
                    <small>Raw output computed by underlying stacking ensemble</small>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            # Light Gray to Dark Purple distribution representation
            st.markdown(f"""
                <div class="metric-card {dist_class}">
                    <h3>Mathematical Threshold Distribution</h3>
                    <h2 style='margin: 0;'>{OPTIMAL_THRESHOLD:.4f}</h2>
                    <small>F1 Optimized operational cutoff point</small>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        with st.expander("🔍 Inspect Raw Feature Payload Vector"):
            st.dataframe(input_df_raw.style.format(precision=2))

        with st.expander("⚙️ Core Methodology Architecture"):
            st.markdown("""
                **Operational Pipeline Execution:**
                * **Mathematical Data Engineering:** Converts raw temporal distributions into multi-tiered operational metrics (such as Interaction Vectors, Service Call Intensities, and Charge Ratios per Minute).
                * **Model Pipeline Topology:** Feeds downstream standardized inputs into an advanced stacking ensemble tier (combining high-performance **LightGBM** and **XGBoost** frameworks) designed for complex data pattern extraction.
            """)

    except Exception as e:
        st.error(f"Operational pipeline execution failed: {e}")
        st.warning("Ensure that model asset matrices line up exactly with target data directory endpoints.")

# --- Footer System ---
st.sidebar.markdown("---")
st.sidebar.caption("🔒 Enterprise Intelligence Node | Encrypted Predictive Pipeline")
st.markdown("---")
st.caption("Strategic Advisory Disclaimer: This information is derived from predictive statistical inference models. Strategic operational workflows must evaluate localized market trends alongside model parameters.")