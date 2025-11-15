# train.py — FIXED: Better architecture with attention over candidates
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from gnn import GNNEncoder

class GNNFeatureExtractor(BaseFeaturesExtractor):
    """Extract graph features using GNN with attention pooling over candidates"""
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.gnn = GNNEncoder(in_dim=7, hidden=64, out_dim=32)
        
        # Attention mechanism for pooling candidates
        self.attention = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )

    def forward(self, obs):
        # Handle batched input from vec_env
        batch_size = obs["node_feat"].shape[0]
        
        # Process each item in batch
        features_list = []
        for i in range(batch_size):
            x = obs["node_feat"][i]  # (N, 7)
            edge_index = obs["edge_index"][i].long()  # (2, E)
            cand = obs["candidates"][i].long()  # (200, 2)

            # Get node embeddings
            node_emb = self.gnn(x, edge_index)  # (N, 32)
            
            # Get candidate pair embeddings
            cand_emb = (node_emb[cand[:, 0]] + node_emb[cand[:, 1]]) / 2.0  # (200, 32)
            
            # Attention pooling over candidates
            attn_weights = self.attention(cand_emb)  # (200, 1)
            attn_weights = torch.softmax(attn_weights, dim=0)  # (200, 1)
            pooled = (cand_emb * attn_weights).sum(dim=0)  # (32,)
            
            features_list.append(pooled)
        
        # Stack batch and project
        batch_features = torch.stack(features_list)  # (batch, 32)
        output = self.projection(batch_features)  # (batch, features_dim)
        
        return output
