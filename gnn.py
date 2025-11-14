# gnn.py
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class GNNEncoder(nn.Module):
    def __init__(self, in_dim=7, hidden=64, out_dim=32):
        super().__init__()
        self.conv1 = pyg_nn.SAGEConv(in_dim, hidden)
        self.conv2 = pyg_nn.SAGEConv(hidden, out_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return x