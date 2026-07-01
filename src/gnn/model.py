# src/gnn/model.py
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GNN_GRU_SpatioTemporal(nn.Module):
    def __init__(self, in_features, gnn_hidden, gru_hidden, out_features, num_nodes):
        super(GNN_GRU_SpatioTemporal, self).__init__()
        
        self.num_nodes = num_nodes
        self.gru_hidden = gru_hidden
        
        # GNN Layer
        self.gnn = GCNConv(in_features, gnn_hidden)
        
        # GRU Layer
        # It takes the output of the GNN (gnn_hidden) as input
        self.gru = nn.GRU(gnn_hidden, gru_hidden, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(gru_hidden, out_features)

    def forward(self, x_seq, edge_index):
        # x_seq shape: [batch, seq_len, num_nodes, num_features]
        
        batch_size = x_seq.shape[0]
        seq_len = x_seq.shape[1]
        
        gnn_out_seq = []
        
        # Process each time step in the sequence
        for t in range(seq_len):
            x_t = x_seq[:, t, :, :]  # Shape: [batch, num_nodes, num_features]
            
            # GNNs in PyG are tricky with batches. 
            # We reshape to process all nodes in the batch at once.
            x_t_reshaped = x_t.reshape(-1, x_t.shape[-1]) # Shape: [batch*num_nodes, num_features]
            
            # We need to adjust edge_index for the batch
            # This creates a "batched" edge_index
            batch_edge_index = edge_index.repeat(1, batch_size) + \
                               torch.arange(batch_size, device=x_seq.device).repeat_interleave(edge_index.shape[1]) * self.num_nodes
            
            gnn_out = self.gnn(x_t_reshaped, batch_edge_index)
            gnn_out = torch.relu(gnn_out)
            
            # Reshape back to [batch, num_nodes, gnn_hidden]
            gnn_out = gnn_out.reshape(batch_size, self.num_nodes, -1)
            gnn_out_seq.append(gnn_out)

        # Stack GNN outputs along the time dimension
        x_gnn_out = torch.stack(gnn_out_seq, dim=1) # Shape: [batch, seq_len, num_nodes, gnn_hidden]
        
        # Now, pass to GRU. GRU must process each node's time series.
        # Reshape to [batch*num_nodes, seq_len, gnn_hidden]
        x_gru_in = x_gnn_out.permute(0, 2, 1, 3) # [batch, num_nodes, seq_len, gnn_hidden]
        x_gru_in = x_gru_in.reshape(-1, seq_len, x_gru_in.shape[-1])
        
        # Run GRU
        _, h_n = self.gru(x_gru_in) # h_n shape: [1, batch*num_nodes, gru_hidden]
        
        # Get the last hidden state
        gru_out_last = h_n.squeeze(0) # Shape: [batch*num_nodes, gru_hidden]
        
        # Final prediction
        out = self.fc(gru_out_last) # Shape: [batch*num_nodes, out_features]
        
        # Reshape to [batch, num_nodes, out_features]
        out = out.reshape(batch_size, self.num_nodes, -1)
        
        return out