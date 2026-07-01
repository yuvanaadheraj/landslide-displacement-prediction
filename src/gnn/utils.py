# src/gnn/utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix
import torch

def build_spatial_graph(csv_path):
    """
    Builds a static graph of stations based on feature similarity.
    Nodes = stations
    Edges = K-Nearest-Neighbors based on average sensor readings.
    """
    data = pd.read_csv(csv_path)
    
    # Drop rows with no station ID
    data = data.dropna(subset=['stationid'])
    
    # Get list of unique stations (our nodes)
    # We sort them to ensure a consistent mapping
    stations = sorted(data['stationid'].unique())
    node_map = {station: i for i, station in enumerate(stations)}
    num_nodes = len(stations)

    # Calculate average features for each station
    target_cols = ['dispx', 'dispy', 'dispz']
    drop_cols = ['dates', 'stationid'] + target_cols
    
    # Get mean of all numeric features *except* targets
    feature_df = data.drop(columns=drop_cols)
    # Get all numeric columns that are left
    feature_cols = feature_df.select_dtypes(include='number').columns
    
    # Add stationid back for grouping
    feature_df['stationid'] = data['stationid']
    
    # Calculate mean features per station
    avg_features = feature_df.groupby('stationid')[feature_cols].mean()
    
    # Make sure we have the same order as our node_map
    avg_features = avg_features.reindex(stations)

    # Scale the features before KNN
    scaler = StandardScaler()
    avg_features_scaled = scaler.fit_transform(avg_features)

    # Build graph using KNN (K=4 neighbors)
    # This connects each station to its 4 most similar stations
    adj_matrix = kneighbors_graph(
        avg_features_scaled, 
        n_neighbors=4, 
        mode='connectivity', 
        include_self=False
    )
    
    # Convert to PyTorch Geometric edge_index format
    edge_index, edge_attr = from_scipy_sparse_matrix(adj_matrix)
    
    print(f"Built graph with {num_nodes} nodes (stations) and {edge_index.shape[1]} edges.")
    
    # node_map: {'t1': 0, 't10': 1, ...}
    # edge_index: [2, num_edges] tensor
    return node_map, num_nodes, edge_index.long()