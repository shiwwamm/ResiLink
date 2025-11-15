# train.py — FIXED: Proper feature extractor + custom policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
from gymnasium import spaces
from gnn import GNNEncoder

class GNNFeatureExtractor(BaseFeaturesExtractor):
    """Extract graph features using GNN - returns pooled graph representation"""
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.gnn = GNNEncoder(in_dim=7, hidden=64, out_dim=32)
        self.mlp = nn.Sequential(
            nn.Linear(32 * 200, 256),  # Flatten all candidate embeddings
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )

    def forward(self, obs):
        # Handle batched input from vec_env
        batch_size = obs["node_feat"].shape[0]
        
        # Process each item in batch
        features_list = []
        for i in range(batch_size):
            x = obs["node_feat"][i].clone().detach()  # (N, 7)
            edge_index = obs["edge_index"][i].clone().detach().long()  # (2, E)
            cand = obs["candidates"][i].clone().detach().long()  # (200, 2)

            node_emb = self.gnn(x, edge_index)  # (N, 32)
            cand_emb = (node_emb[cand[:, 0]] + node_emb[cand[:, 1]]) / 2.0  # (200, 32)
            
            # Flatten candidate embeddings
            flat_features = cand_emb.flatten()  # (200*32,)
            features_list.append(flat_features)
        
        # Stack batch
        batch_features = torch.stack(features_list)  # (batch, 200*32)
        output = self.mlp(batch_features)  # (batch, features_dim)
        
        return output
