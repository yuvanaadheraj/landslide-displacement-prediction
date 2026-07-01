# train_fis.py
# A standalone script to build a classic Fuzzy Inference System (FIS)

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- Imports for FIS ---
import skfuzzy as fuzz
from skfuzzy import control as ctrl
# -----------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- We include all helper functions here to make this file standalone ---

def evaluate_model(y_actual, y_pred):
    """Calculates and returns RMSE, MAE, and R² metrics."""
    # Remove NaNs that can occur if no fuzzy rule is fired
    clean_data = list(zip(y_actual.flatten(), y_pred.flatten()))
    clean_data = [(a, p) for a, p in clean_data if not np.isnan(p) and not np.isnan(a)]
    if not clean_data:
        print("Warning: All predictions were NaN. Returning worst-case score.")
        return np.inf, np.inf, 0.0
        
    y_actual_clean, y_pred_clean = zip(*clean_data)
    y_actual_clean = np.array(y_actual_clean)
    y_pred_clean = np.array(y_pred_clean)

    rmse = np.sqrt(mean_squared_error(y_actual_clean, y_pred_clean))
    mae = mean_absolute_error(y_actual_clean, y_pred_clean)
    r2 = r2_score(y_actual_clean, y_pred_clean)
    return rmse, mae, r2

def get_tabular_data(csv_path, split_ratio):
    """Loads and processes the data from the CSV file."""
    data = pd.read_csv(csv_path)
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
    
    if 'stationid' in data.columns:
        data = data.drop(columns=['stationid'])
    data = data.drop(columns=['dates'])
    
    target_cols = ['dispx', 'dispy', 'dispz']
    numeric_cols = data.select_dtypes(include='number').columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in target_cols]

    X = data[feature_cols].values.astype('float32')
    y = data[target_cols].values.astype('float32')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1 - split_ratio), random_state=42
    )
    
    # --- IMPORTANT ---
    # We return the ORIGINAL, UNSCALED data for FIS
    # Fuzzy logic works best with real-world values (e.g., rain in mm)
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)
    
    return X_train, X_test, y_train, y_test, scaler_y, feature_cols

def find_top_features(csv_path, feature_names):
    """Trains a temporary XGB model to find feature importances."""
    print("\n--- Finding Top Features (Training temporary XGBoost) ---")
    
    # We need scaled data for XGBoost to work well
    data = pd.read_csv(csv_path)
    # ... (Re-doing data loading just for scaling) ...
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
    if 'stationid' in data.columns: data = data.drop(columns=['stationid'])
    data = data.drop(columns=['dates'])
    target_cols = ['dispx', 'dispy', 'dispz']
    numeric_cols = data.select_dtypes(include='number').columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in target_cols]
    X = data[feature_cols].values.astype('float32')
    y = data[target_cols].values.astype('float32')
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler_X_xgb = StandardScaler().fit(X_train)
    scaler_y_xgb = StandardScaler().fit(y_train)
    X_train_scaled = scaler_X_xgb.transform(X_train)
    y_train_scaled = scaler_y_xgb.transform(y_train)
    # --- End of scaling ---
    
    base_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train_scaled, y_train_scaled)
    
    all_importances = [est.feature_importances_ for est in model.estimators_]
    avg_importances = np.mean(all_importances, axis=0)
    
    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': avg_importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)
    
    # We use TOP 2 FEATURES. FIS with more inputs gets very complex.
    top_2_features = feature_df.head(2)['Feature'].tolist()
    top_2_indices = [feature_names.index(f) for f in top_2_features]
    
    print("Top 2 features found:", top_2_features)
    print("NOTE: We use only 2 features as FIS rules grow exponentially.")
    return top_2_features, top_2_indices

def train_fis_system():
    """Main function to build and run the FIS."""
    
    # 1. Load Data
    print("--- Loading and Processing Data ---")
    # We get the ORIGINAL, UNSCALED data for FIS
    X_train, X_test, y_train, y_test, scaler_y, feature_names = get_tabular_data(
        "datalandslide.csv", 0.8
    )
    
    # 2. Find Top Features
    TOP_FEATURES_LIST, TOP_FEATURE_INDICES = find_top_features("datalandslide.csv", feature_names)
    
    # 3. Filter data to only include top features
    X_train = X_train[:, TOP_FEATURE_INDICES]
    X_test = X_test[:, TOP_FEATURE_INDICES]
    
    # --- Start of FIS Definition ---
    print(f"\n--- Building Fuzzy System with {TOP_FEATURES_LIST[0]} and {TOP_FEATURES_LIST[1]} ---")
    
    # Find the min/max range of our data for defining the "universe"
    x0_min, x0_max = np.min(X_train[:, 0]), np.max(X_train[:, 0])
    x1_min, x1_max = np.min(X_train[:, 1]), np.max(X_train[:, 1])
    y_min, y_max = np.min(y_train), np.max(y_train)

    # 1. Fuzzification: Define Antecedents (Inputs)
    # Create 3 fuzzy sets (low, medium, high) for each input
    input1 = ctrl.Antecedent(np.arange(x0_min, x0_max, (x0_max - x0_min) / 100), TOP_FEATURES_LIST[0])
    input2 = ctrl.Antecedent(np.arange(x1_min, x1_max, (x1_max - x1_min) / 100), TOP_FEATURES_LIST[1])

    input1['low'] = fuzz.trimf(input1.universe, [x0_min, x0_min, (x0_min+x0_max)/2])
    input1['medium'] = fuzz.trimf(input1.universe, [x0_min, (x0_min+x0_max)/2, x0_max])
    input1['high'] = fuzz.trimf(input1.universe, [(x0_min+x0_max)/2, x0_max, x0_max])
    
    input2['low'] = fuzz.trimf(input2.universe, [x1_min, x1_min, (x1_min+x1_max)/2])
    input2['medium'] = fuzz.trimf(input2.universe, [x1_min, (x1_min+x1_max)/2, x1_max])
    input2['high'] = fuzz.trimf(input2.universe, [(x1_min+x1_max)/2, x1_max, x1_max])
    
    # Define Consequent (Output)
    # We will predict a combined displacement (average of x, y, z)
    # This is a simplification, as FIS is complex with multi-output
    y_train_avg = np.mean(y_train, axis=1)
    y_test_avg = np.mean(y_test, axis=1)
    y_avg_min, y_avg_max = np.min(y_train_avg), np.max(y_train_avg)
    
    output = ctrl.Consequent(np.arange(y_avg_min, y_avg_max, (y_avg_max - y_avg_min) / 100), 'displacement')
    output['low'] = fuzz.trimf(output.universe, [y_avg_min, y_avg_min, (y_avg_min+y_avg_max)/2])
    output['medium'] = fuzz.trimf(output.universe, [y_avg_min, (y_avg_min+y_avg_max)/2, y_avg_max])
    output['high'] = fuzz.trimf(output.universe, [(y_avg_min+y_avg_max)/2, y_avg_max, y_avg_max])

    # 2. Inference Engine: Define Rules (We define 9 simple rules)
    rule1 = ctrl.Rule(input1['low'] & input2['low'], output['low'])
    rule2 = ctrl.Rule(input1['low'] & input2['medium'], output['low'])
    rule3 = ctrl.Rule(input1['low'] & input2['high'], output['medium'])
    rule4 = ctrl.Rule(input1['medium'] & input2['low'], output['low'])
    rule5 = ctrl.Rule(input1['medium'] & input2['medium'], output['medium'])
    rule6 = ctrl.Rule(input1['medium'] & input2['high'], output['high'])
    rule7 = ctrl.Rule(input1['high'] & input2['low'], output['medium'])
    rule8 = ctrl.Rule(input1['high'] & input2['medium'], output['high'])
    rule9 = ctrl.Rule(input1['high'] & input2['high'], output['high'])

    fis_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    fis = ctrl.ControlSystemSimulation(fis_ctrl)
    
    # 3. Defuzzification: Run the simulation
    print("\n--- Running Fuzzy Simulation on Test Data ---")
    y_pred_list = []
    for i in range(len(X_test)):
        fis.input[TOP_FEATURES_LIST[0]] = X_test[i, 0]
        fis.input[TOP_FEATURES_LIST[1]] = X_test[i, 1]
        try:
            fis.compute()
            y_pred_list.append(fis.output['displacement'])
        except:
            y_pred_list.append(np.nan) # Handle cases where no rule is fired
            
    y_pred_fis = np.array(y_pred_list)

    # 6. Evaluate the combined ANFIS model
    print("\n--- Fuzzy System Evaluation (Combined) ---")
    
    # We are evaluating the *unscaled* average displacement
    rmse, mae, r2 = evaluate_model(y_test_avg, y_pred_fis)
    print(f"Overall RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # 7. Generate and Save Graphs
    print("\n--- Generating Graphs ---")
    
    # Graph 1: Membership Functions (The "Fuzzy Rules" we defined)
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))
    
    ax0.plot(input1.universe, fuzz.trimf(input1.universe, [x0_min, x0_min, (x0_min+x0_max)/2]), 'b', label='Low')
    ax0.plot(input1.universe, fuzz.trimf(input1.universe, [x0_min, (x0_min+x0_max)/2, x0_max]), 'g', label='Medium')
    ax0.plot(input1.universe, fuzz.trimf(input1.universe, [(x0_min+x0_max)/2, x0_max, x0_max]), 'r', label='High')
    ax0.set_title(f'Membership Function for {TOP_FEATURES_LIST[0]}')
    ax0.legend()

    ax1.plot(input2.universe, fuzz.trimf(input2.universe, [x1_min, x1_min, (x1_min+x1_max)/2]), 'b', label='Low')
    ax1.plot(input2.universe, fuzz.trimf(input2.universe, [x1_min, (x1_min+x1_max)/2, x1_max]), 'g', label='Medium')
    ax1.plot(input2.universe, fuzz.trimf(input2.universe, [(x1_min+x1_max)/2, x1_max, x1_max]), 'r', label='High')
    ax1.set_title(f'Membership Function for {TOP_FEATURES_LIST[1]}')
    ax1.legend()
    
    ax2.plot(output.universe, fuzz.trimf(output.universe, [y_avg_min, y_avg_min, (y_avg_min+y_avg_max)/2]), 'b', label='Low')
    ax2.plot(output.universe, fuzz.trimf(output.universe, [y_avg_min, (y_avg_min+y_avg_max)/2, y_avg_max]), 'g', label='Medium')
    ax2.plot(output.universe, fuzz.trimf(output.universe, [(y_avg_min+y_avg_max)/2, y_avg_max, y_avg_max]), 'r', label='High')
    ax2.set_title(f'Output Membership Function for Displacement')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("fis_membership_functions.png")
    print("Saved 'fis_membership_functions.png'")

# --- This makes the file runnable ---
if __name__ == "__main__":
    train_fis_system()