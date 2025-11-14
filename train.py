# train.py — FINAL FIXED (Handles (1,200) input)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from gnn import GNNEncoder

class GNNPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.gnn = GNNEncoder(in_dim=7, hidden=64, out_dim=32)
        self.mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim)
        )
        self.logit_head = nn.Linear(features_dim, 1)

    def forward(self, obs):
        # Handle batched input from vec_env (1, N, F) → (N, F)
        x = obs["node_feat"].squeeze(0).clone().detach()  # (N, 7)
        edge_index = obs["edge_index"].squeeze(0).clone().detach().long()  # (2, E)
        cand = obs["candidates"].squeeze(0).clone().detach().long()  # (200, 2)

        node_emb = self.gnn(x, edge_index)  # (N, 32)
        cand_emb = (node_emb[cand[:, 0]] + node_emb[cand[:, 1]]) / 2.0  # (200, 32)

        hidden = self.mlp(cand_emb)  # (200, 128)
        logits = self.logit_head(hidden).squeeze(-1)  # (200,)

        return logits  # PPO samples from this
