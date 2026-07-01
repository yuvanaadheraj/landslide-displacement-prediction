# src/baselines/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import get_dataloaders
from src.config import Config
from src.utils import evaluate_model
from src.baselines.models import MLPModel, GRUOnlyModel
import joblib
import numpy as np

def train_pytorch_baseline(model_name):
    print(f"--- Starting Baseline Training: {model_name} ---")
    
    train_loader, test_loader, dataset = get_dataloaders(
        Config.DATA_PATH, Config.BATCH_SIZE, Config.TRAIN_SPLIT
    )

    in_features = dataset.X.shape[1]
    out_features = dataset.y.shape[1] # This will be 3

    if model_name == 'mlp':
        model = MLPModel(in_features, Config.HIDDEN_DIM, out_features)
    elif model_name == 'gru':
        model = GRUOnlyModel(in_features, Config.HIDDEN_DIM, out_features)
    else:
        raise ValueError("Unknown model name. Use 'mlp' or 'gru'.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    print(f"Training {model_name} for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

    # --- Evaluation ---
    model.eval()
    all_preds = []
    all_actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            all_preds.append(preds.numpy())
            all_actuals.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds)
    all_actuals = np.concatenate(all_actuals)

    # Inverse transform to original scale
    y_actual = dataset.scaler_y.inverse_transform(all_actuals)
    y_pred = dataset.scaler_y.inverse_transform(all_preds)
    
    # Save scalers and model
    joblib.dump(dataset.scaler_X, f"scaler_X_{model_name}.pkl")
    joblib.dump(dataset.scaler_y, f"scaler_y_{model_name}.pkl")
    torch.save(model.state_dict(), f"baseline_model_{model_name}.pth")

    rmse, mae, r2 = evaluate_model(y_actual, y_pred)
    print(f"--- {model_name} Model Evaluation ---")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print(f"-----------------------------------\n")