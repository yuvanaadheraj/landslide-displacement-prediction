# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Landslide Prediction",
    page_icon="🏔️",
    layout="centered"
)

# --- Load Model and Scalers ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("baseline_model_xgboost.pkl")
        scaler_X = joblib.load("scaler_X_xgboost.pkl")
        scaler_y = joblib.load("scaler_y_xgboost.pkl")
        return model, scaler_X, scaler_y
    except FileNotFoundError:
        return None, None, None

model, scaler_X, scaler_y = load_model()

if model is None:
    st.error("Model files not found! Make sure you downloaded the three .pkl files from Colab.")
    st.stop()


# --- Get the list of feature names ---
@st.cache_data
def get_feature_names():
    try:
        data = pd.read_csv("datalandslide.csv")
        data['dates'] = pd.to_datetime(data['dates'], errors='coerce')
        data = data.dropna(subset=['dates'])
        
        data['year'] = data['dates'].dt.year
        data['month'] = data['dates'].dt.month
        data['day'] = data['dates'].dt.day
        data['day_of_week'] = data['dates'].dt.dayofweek
        data['day_of_year'] = data['dates'].dt.dayofyear
        
        data['month_sin'] = np.sin(2*np.pi*data['month']/12)
        data['month_cos'] = np.cos(2*np.pi*data['month']/12)
        data['day_sin'] = np.sin(2*np.pi*data['day']/31)
        data['day_cos'] = np.cos(2*np.pi*data['day']/31)
        data = data.drop(columns=['dates', 'stationid'])
        
        target_cols = ['dispx', 'dispy', 'dispz']
        numeric_cols = data.select_dtypes(include='number').columns.tolist()
        feature_names = [c for c in numeric_cols if c not in target_cols]
        return feature_names
    except FileNotFoundError:
        return None

FEATURE_NAMES = get_feature_names()

if FEATURE_NAMES is None:
    st.error("datalandslide.csv not found.")
    st.stop()


# --- Main Application ---
st.title("🏔️ AI Landslide Displacement Prediction")
st.write("Enter the sensor and date information to predict the landslide displacement.")

with st.form("prediction_form"):
    st.header("Date Information")
    input_date = st.date_input("Select the date for prediction")

    st.header("Sensor Readings (Key Inputs)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temperature = st.number_input("Temperature (°C)", value=25.0)
        humidity = st.number_input("Humidity (%)", value=80.0)
        pressure = st.number_input("Pressure (hPa)", value=1036.0)
    
    with col2:
        rain = st.number_input("Rain (mm)", value=0.0)
        lightavg = st.number_input("Light Avg", value=150.0)
        lightmax = st.number_input("Light Max", value=300.0)

    with col3:
        sumax = st.number_input("Sum-AX (accel.)", value=0.06)
        sumay = st.number_input("Sum-AY (accel.)", value=0.11)
        sumaz = st.number_input("Sum-AZ (accel.)", value=-1.0)

    submitted = st.form_submit_button("Predict Displacement")

# --- Prediction Logic ---
if submitted:
    # 1. Create the full feature array, starting with defaults
    feature_dict = {feature: 0.0 for feature in FEATURE_NAMES}
    
    # 2. Process date inputs
    date = pd.to_datetime(input_date)
    feature_dict['year'] = date.year
    feature_dict['month'] = date.month
    feature_dict['day'] = date.day
    feature_dict['day_of_week'] = date.dayofweek
    feature_dict['day_of_year'] = date.dayofyear
    feature_dict['month_sin'] = np.sin(2 * np.pi * date.month / 12)
    feature_dict['month_cos'] = np.cos(2 * np.pi * date.month / 12)
    feature_dict['day_sin'] = np.sin(2 * np.pi * date.day / 31)
    feature_dict['day_cos'] = np.cos(2 * np.pi * date.day / 31)
    
    # 3. Add user inputs
    feature_dict['temperature'] = temperature
    feature_dict['humidity'] = humidity
    feature_dict['pressure'] = pressure
    feature_dict['rain'] = rain
    feature_dict['lightavg'] = lightavg
    feature_dict['lightmax'] = lightmax
    feature_dict['sumax'] = sumax
    feature_dict['sumay'] = sumay
    feature_dict['sumaz'] = sumaz
    
    # --- WARNING LINE REMOVED ---
    
    # 4. Create the final feature vector in the correct order
    try:
        final_features = [feature_dict[name] for name in FEATURE_NAMES]
        final_features_array = np.array([final_features]).astype('float32')
        
        # 5. Scale the features
        scaled_features = scaler_X.transform(final_features_array)
        
        # 6. Predict
        pred_scaled = model.predict(scaled_features)
        
        # 7. Inverse scale the prediction
        pred_actual = scaler_y.inverse_transform(pred_scaled)
        
        # 8. Display results
        st.subheader("✅ Prediction Results")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Displacement X (mm)", f"{pred_actual[0][0]:.4f}")
        res_col2.metric("Displacement Y (mm)", f"{pred_actual[0][1]:.4f}")
        res_col3.metric("Displacement Z (mm)", f"{pred_actual[0][2]:.4f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")