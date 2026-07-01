# src/data_loader.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

class LandslideDataset(Dataset):
    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)

        # --- Robust date parsing ---
        data['dates'] = pd.to_datetime(data['dates'], infer_datetime_format=True, errors='coerce')

        # Drop rows with invalid dates
        data = data.dropna(subset=['dates'])

        # Extract temporal features
        # Note: Your original file had 'year', 'month', etc. here.
        # My baseline 'train_xgboost.py' re-does this, which is a bit redundant
        # but this is the file your 'train.py' (for MLP/GRU) will use.
        data['year'] = data['dates'].dt.year
        data['month'] = data['dates'].dt.month
        data['day'] = data['dates'].dt.day
        data['day_of_week'] = data['dates'].dt.dayofweek
        data['day_of_year'] = data['dates'].dt.dayofyear

        # Cyclic encoding
        data['month_sin'] = np.sin(2*np.pi*data['month']/12)
        data['month_cos'] = np.cos(2*np.pi*data['month']/12)
        data['day_sin'] = np.sin(2*np.pi*data['day']/31)
        data['day_cos'] = np.cos(2*np.pi*data['day']/31)

        # Drop original date and non-numeric columns
        # We keep the originals (month, day) for the XGBoost loader
        # but this loader drops everything it doesn't need.
        data = data.drop(columns=['dates', 'stationid'])

        # --- Multi-output targets ---
        self.target_cols = ['dispx', 'dispy', 'dispz']
        
        # Find all numeric columns
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        
        # Features are all numeric columns *except* the targets
        feature_cols = [c for c in numeric_cols if c not in self.target_cols]

        self.X = data[feature_cols].values.astype('float32')
        self.y = data[self.target_cols].values.astype('float32')

        # Scale features and targets
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X = self.scaler_X.fit_transform(self.X)
        self.y = self.scaler_y.fit_transform(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


def get_dataloaders(csv_path, batch_size=32, split=0.8):
    dataset = LandslideDataset(csv_path)
    train_size = int(split * len(dataset))
    test_size = len(dataset) - train_size
    
    # We add a generator for reproducible splits
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # We return the whole 'dataset' object so we can access the scalers
    return train_loader, test_loader, dataset