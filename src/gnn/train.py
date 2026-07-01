# src/gnn/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.gnn.utils import build_spatial_graph
from src.gnn.data_loader import SpatioTemporalDataset
from src.gnn.model import GNN_GRU_SpatioTemporal
from src.config import Config
from src.utils import evaluate_model
import joblib
import numpy as np

def train_gnn():
    print("--- Starting GNN+GRU (Spatio-Temporal) Training ---")
    
    # 1. Build the static station graph
    node_map, num_nodes, edge_index = build_spatial_graph(Config.DATA_PATH)
    
    # 2. Create Train and Test Datasets
    train_dataset = SpatioTemporalDataset(
        csv_path=Config.DATA_PATH,
        node_map=node_map,
        seq_len=Config.SEQ_LEN,
        is_train=True
    )
    
    test_dataset = SpatioTemporalDataset(
        csv_path=Config.DATA_PATH,
        node_map=node_map,
        seq_len=Config.SEQ_LEN,
        is_train=False
    )
    
    # 3. Fit scalers on *training* data and apply to *both*
    train_dataset.scale_data()
    test_dataset.set_scalers(train_dataset.scaler_X, train_dataset.scaler_y)
    test_dataset.scale_data()

    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 5. Initialize Model
    in_features = train_dataset.X_data.shape[-1]
    out_features = Config.OUTPUT_DIM
    
    model = GNN_GRU_SpatioTemporal(
        in_features=in_features,
        gnn_hidden=Config.HIDDEN_DIM,
        gru_hidden=Config.HIDDEN_DIM,
        out_features=out_features,
        num_nodes=num_nodes
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    
    # Move edge_index to device (if GPU is available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    edge_index = edge_index.to(device)
    
    print(f"Training GNN on {device} for {Config.EPOCHS} epochs...")
    
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        for X_seq_batch, y_batch in train_loader:
            X_seq_batch, y_batch = X_seq_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_seq_batch, edge_index)
            
            # y_batch shape: [batch, num_nodes, out_features]
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

    # --- Evaluation ---
    model.eval()
    all_preds = []
    all_actuals = []
    with torch.no_grad():
        for X_seq_batch, y_batch in test_loader:
            X_seq_batch, y_batch = X_seq_batch.to(device), y_batch.to(device)
            preds = model(X_seq_batch, edge_index)
            all_preds.append(preds.cpu().numpy())
            all_actuals.append(y_batch.cpu().numpy())

    # Reshape: from [num_batches, batch, num_nodes, features] to [total_samples, num_nodes, features]
    all_preds = np.concatenate(all_preds, axis=0)
    all_actuals = np.concatenate(all_actuals, axis=0)

    # Reshape for scaler: [total_samples * num_nodes, features]
    all_preds_flat = all_preds.reshape(-1, out_features)
    all_actuals_flat = all_actuals.reshape(-1, out_features)

    # Inverse transform to original scale
    y_actual = test_dataset.scaler_y.inverse_transform(all_actuals_flat)
    y_pred = test_dataset.scaler_y.inverse_transform(all_preds_flat)
    
    # Save scalers and model
    joblib.dump(test_dataset.scaler_X, "scaler_X_gnn.pkl")
    joblib.dump(test_dataset.scaler_y, "scaler_y_gnn.pkl")
    torch.save(model.state_dict(), "gnn_spatiotemporal_model.pth")

    rmse, mae, r2 = evaluate_model(y_actual, y_pred)
    print("--- GNN+GRU (Spatio-Temporal) Model Evaluation ---")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print("--------------------------------------------------\n")