# src/baselines/models.py
import torch
import torch.nn as nn

class MLPModel(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) baseline."""
    def __init__(self, in_features, hidden_dim, output_dim):
        super(MLPModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class GRUOnlyModel(nn.Module):
    """
    This is your original GNN_GRU_Model, but with the GNN
    layer removed. This tests if the GRU provides any benefit.
    The input `x` is unsqueezed to `[batch, 1, features]`
    to mimic a sequence of length 1 for the GRU.
    """
    def __init__(self, in_features, hidden_dim, output_dim):
        super(GRUOnlyModel, self).__init__()
        self.hidden_dim = hidden_dim
        # GRU layer
        self.gru = nn.GRU(in_features, hidden_dim, batch_first=True, num_layers=2)
        # Final fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape is [batch, features]
        # GRU expects [batch, seq_len, features]
        x = x.unsqueeze(1)  # Shape becomes [batch, 1, features]

        # Pass through GRU
        gru_out, _ = self.gru(x)  # gru_out shape: [batch, 1, hidden_dim]

        # Get the last time step's output
        last_step_out = gru_out[:, -1, :]  # Shape: [batch, hidden_dim]
        
        # Pass through the final linear layer
        out = self.fc(last_step_out)  # Shape: [batch, output_dim]
        return out