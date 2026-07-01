# src/gnn/data_loader.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from src.config import Config

class SpatioTemporalDataset(Dataset):
    def __init__(self, csv_path, node_map, seq_len, is_train=True):
        self.seq_len = seq_len
        self.node_map = node_map
        self.num_nodes = len(node_map)
        
        # Load and preprocess data
        df = self.load_data(csv_path)
        
        self.target_cols = ['dispx', 'dispy', 'dispz']
        self.feature_cols = [c for c in df.columns if c not in ['dates', 'stationid'] + self.target_cols]
        
        # --- Pivot data to [time, nodes, features] ---
        pivoted_df = df.pivot(index='dates', columns='stationid')
        
        # Reorder columns to group features
        pivoted_df = pivoted_df.reorder_levels([1, 0], axis=1)
        
        # Sort nodes to match node_map
        pivoted_df = pivoted_df[sorted(self.node_map.keys())]
        
        # --- FIX 1: Silencing fillna warning ---
        # Handle missing data: fill forward, then backward
        pivoted_df = pivoted_df.ffill().bfill() # Replaced fillna(method=...)
        
        # --- FIX 2: Silencing stack warning ---
        # Now, separate X (features) and y (targets)
        X_df = pivoted_df.stack(level=0, future_stack=True)[self.feature_cols].unstack(level=1)
        y_df = pivoted_df.stack(level=0, future_stack=True)[self.target_cols].unstack(level=1)
        
        # Convert to numpy arrays
        self.X_data = X_df.values.reshape(len(X_df), self.num_nodes, len(self.feature_cols)).astype('float32')
        self.y_data = y_df.values.reshape(len(y_df), self.num_nodes, len(self.target_cols)).astype('float32')

        # --- Train/Test Split (by time) ---
        split_idx = int(len(self.X_data) * Config.TRAIN_SPLIT)
        
        if is_train:
            self.X_data = self.X_data[:split_idx]
            self.y_data = self.y_data[:split_idx]
        else:
            self.X_data = self.X_data[split_idx:]
            self.y_data = self.y_data[split_idx:]

        # --- Scaling ---
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_train_flat = self.X_data.reshape(-1, len(self.feature_cols))
        y_train_flat = self.y_data.reshape(-1, len(self.target_cols))
        
        if is_train:
            self.scaler_X.fit(X_train_flat)
            self.scaler_y.fit(y_train_flat)

    def set_scalers(self, scaler_X, scaler_y):
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def scale_data(self):
        X_flat = self.X_data.reshape(-1, len(self.feature_cols))
        y_flat = self.y_data.reshape(-1, len(self.target_cols))
        
        X_scaled = self.scaler_X.transform(X_flat)
        y_scaled = self.scaler_y.transform(y_flat)
        
        self.X_data = X_scaled.reshape(self.X_data.shape)
        self.y_data = y_scaled.reshape(self.y_data.shape)
        
        self.sequences = self.create_sequences()

    def load_data(self, csv_path):
        data = pd.read_csv(csv_path)
        data['dates'] = pd.to_datetime(data['dates'], errors='coerce')
        data = data.dropna(subset=['dates', 'stationid'])
        
        data['year'] = data['dates'].dt.year
        data['month'] = data['dates'].dt.month
        data['day'] = data['dates'].dt.day
        data['day_of_week'] = data['dates'].dt.dayofweek
        data['day_of_year'] = data['dates'].dt.dayofyear
        
        data['month_sin'] = np.sin(2*np.pi*data['month']/12)
        data['month_cos'] = np.cos(2*np.pi*data['month']/12)
        data['day_sin'] = np.sin(2*np.pi*data['day']/31)
        data['day_cos'] = np.cos(2*np.pi*data['day']/31)
        
        data = data.drop(columns=['month', 'day', 'year', 'day_of_week', 'day_of_year'])
        return data

    def create_sequences(self):
        X_seq, y_seq = [], []
        for i in range(len(self.X_data) - self.seq_len):
            X_seq.append(self.X_data[i : i + self.seq_len])
            y_seq.append(self.y_data[i + self.seq_len])
            
        return list(zip(X_seq, y_seq))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X, y = self.sequences[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)