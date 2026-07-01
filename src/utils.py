# src/utils.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_actual, y_pred):
    """
    Calculates and returns RMSE, MAE, and R² metrics.
    
    Args:
        y_actual (np.array): The true target values.
        y_pred (np.array): The predicted target values.
        
    Returns:
        tuple: (rmse, mae, r2)
    """
    
    # --- RMSE (Root Mean Squared Error) ---
    # We calculate it for all targets combined
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    
    # --- MAE (Mean Absolute Error) ---
    # We calculate it for all targets combined
    mae = mean_absolute_error(y_actual, y_pred)
    
    # --- R² (R-squared) ---
    # We calculate an overall R² score
    r2 = r2_score(y_actual, y_pred)
    
    return rmse, mae, r2