# train.py
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from gnn import GNNEncoder

class GNNPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=32):
        super().__init__(observation_space, features_dim)
        self.gnn = GNNEncoder()
        self.mlp = nn.Sequential(nn.Linear(64, features_dim), nn.ReLU())

    def forward(self, obs):
        x = torch.tensor(obs["node_feat"], dtype=torch.float)
        edge_index = torch.tensor(obs["edge_index"], dtype=torch.long)
        cand = torch.tensor(obs["candidates"], dtype=torch.long)
        node_emb = self.gnn(x, edge_index)
        cand_emb = node_emb[cand].reshape(cand.shape[0], -1)
        return self.mlp(cand_emb)