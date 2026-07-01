# src/baselines/train_xgboost.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from src.config import Config
from src.utils import evaluate_model
import joblib
import numpy as np

def get_tabular_data(csv_path, split_ratio):
    """
    A separate data loader for sklearn models.
    This re-uses the logic from your data_loader.py
    """
    data = pd.read_csv(csv_path)
    data['dates'] = pd.to_datetime(data['dates'], errors='coerce') 
    data = data.dropna(subset=['dates'])

    # --- Create date parts ---
    data['year'] = data['dates'].dt.year
    data['month'] = data['dates'].dt.month
    data['day'] = data['dates'].dt.day
    data['day_of_week'] = data['dates'].dt.dayofweek
    data['day_of_year'] = data['dates'].dt.dayofyear

    # Cyclic features
    data['month_sin'] = np.sin(2*np.pi*data['month']/12)
    data['month_cos'] = np.cos(2*np.pi*data['month']/12)
    data['day_sin'] = np.sin(2*np.pi*data['day']/31)
    data['day_cos'] = np.cos(2*np.pi*data['day']/31)
    
    data = data.drop(columns=['dates', 'stationid'])

    target_cols = ['dispx', 'dispy', 'dispz']
    numeric_cols = data.select_dtypes(include='number').columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in target_cols]

    X = data[feature_cols].values.astype('float32')
    y = data[target_cols].values.astype('float32')
    
    # --- Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1 - split_ratio), random_state=42
    )

    # --- Scale Data ---
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

def train_xgboost():
    print("--- Starting Baseline Training: XGBoost ---")
    
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = get_tabular_data(
        Config.DATA_PATH, Config.TRAIN_SPLIT
    )

    # Create the XGBoost model
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    
    model = MultiOutputRegressor(base_model)
    
    print("Training XGBoost...")
    model.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform
    y_actual = scaler_y.inverse_transform(y_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Save scalers and model
    joblib.dump(scaler_X, "scaler_X_xgboost.pkl")
    # --- THIS IS THE FIX ---
    joblib.dump(scaler_y, "scaler_y_xgboost.pkl") # Was 'scaler_Y'
    # -----------------------
    joblib.dump(model, "baseline_model_xgboost.pkl")

    rmse, mae, r2 = evaluate_model(y_actual, y_pred)
    print("--- XGBoost Model Evaluation ---")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print("--------------------------------\n")