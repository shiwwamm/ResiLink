#!/usr/bin/env python3
"""
ResiLink v2: GIN + PPO for Network Topology Optimization
========================================================
- GIN: Expressive embeddings [Xu et al., ICLR 2019]
- PPO: Stable policy [Schulman et al., 2017]
- Reward: Discrete ternary (GraphRARE-inspired) [Shu et al., 2023]
"""

import argparse
import json
import logging
import os
from pathlib import Path
import time
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
from scipy.sparse.linalg import eigsh  # For algebraic connectivity
from collections import deque

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class GraphPPOEnv(gym.Env):
    """PPO Environment for Graph Topology Optimization."""
    def __init__(self, graphml_path: str, max_steps: int = 10, delta: float = 0.01, reward_type: str = "graphrare"):
        super().__init__()
        self.graphml_path = graphml_path
        self.max_steps = max_steps
        self.delta = delta
        self.reward_type = reward_type
        self.G = None
        self.G_prev = None
        self.step_count = 0
        self.candidates = []

        # Load initial graph
        self._load_graph()

        # Spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Discrete(200)  # Max candidates

    def _load_graph(self):
        self.G = nx.read_graphml(self.graphml_path)
        for n in self.G.nodes():
            self.G.nodes[n]['id'] = str(n)  # Ensure string IDs

    def reset(self, seed=None, options=None):
        # Gymnasium compatibility: accept seed and options parameters
        if seed is not None:
            np.random.seed(seed)
        
        self._load_graph()
        self.step_count = 0
        self.G_prev = self.G.copy()
        self.candidates = self._get_candidates()
        
        # Gymnasium expects (observation, info) tuple
        return self._get_state(), {}

    def _get_candidates(self) -> List[Tuple[str, str]]:
        """Feasible link candidates (degree < 8, no self-loops)."""
        nodes = list(self.G.nodes())
        cands = [(u, v) for i, u in enumerate(nodes) for v in nodes[i+1:] if not self.G.has_edge(u, v)
                 and self.G.degree(u) < 8 and self.G.degree(v) < 8]
        return cands[:200]  # Limit for discrete space

    def _get_state(self) -> np.ndarray:
        """12-dim state: topology + capacity features."""
        if not nx.is_connected(self.G):
            return np.zeros(12)
        
        props = [
            self.G.number_of_nodes() / 100.0,
            self.G.number_of_edges() / 100.0,
            nx.density(self.G),
            float(nx.is_connected(self.G)),
            np.mean([d for _, d in self.G.degree()]),
            nx.diameter(self.G) / 10.0,
        ]
        caps = [self._parse_capacity(e.get('capacity', '0 Gbps')) for _, _, e in self.G.edges(data=True)]
        props += [np.mean(caps)/1e9, np.std(caps)/1e9, np.max(caps)/1e9, len(caps)/100.0, np.median(caps)/1e9]
        return np.clip(np.array(props, dtype=np.float32), -1, 1)

    def _parse_capacity(self, cap_str: str) -> float:
        """Parse '10 Gbps' to bps."""
        try:
            cap_str = cap_str.lower().replace('<', '').replace(' ', '')
            if 'gbps' in cap_str:
                return float(cap_str.replace('gbps', '')) * 1e9
            elif 'mbps' in cap_str:
                return float(cap_str.replace('mbps', '')) * 1e6
            return 1e9  # Default 1 Gbps
        except:
            return 1e9

    def step(self, action: int):
        if action >= len(self.candidates) or self.step_count >= self.max_steps:
            # Gymnasium expects 5 values: (obs, reward, terminated, truncated, info)
            return self._get_state(), 0.0, True, False, {}

        u, v = self.candidates[action]
        self.G_prev = self.G.copy()
        self.G.add_edge(u, v, capacity='10 Gbps')  # Default new link

        reward = self._compute_reward()
        self.step_count += 1
        self.candidates = self._get_candidates()  # Update
        terminated = self.step_count >= self.max_steps
        truncated = False  # Not truncated by time limit
        
        # Gymnasium expects 5 values: (obs, reward, terminated, truncated, info)
        return self._get_state(), reward, terminated, truncated, {}

    def _compute_reward(self) -> float:
        """GraphRARE Discrete Ternary Reward [Shu et al., 2023]."""
        if not nx.is_connected(self.G):
            return -1.0

        U_current = self._graph_utility(self.G)
        U_prev = self._graph_utility(self.G_prev)

        if self.reward_type == "graphrare":
            if U_current > U_prev + self.delta:
                return 1.0
            elif abs(U_current - U_prev) <= self.delta:
                return 0.0
            else:
                return -1.0
        else:  # Legacy linear
            return U_current - U_prev

    def _graph_utility(self, G: nx.Graph) -> float:
        """Network utility: Resilience + Efficiency [Newman, 2010]."""
        if not nx.is_connected(G):
            return 0.0
        lambda2 = nx.algebraic_connectivity(G)  # Spectral resilience
        dens = nx.density(G)
        bis = nx.stoer_wagner(G)[0] if G.number_of_edges() > 0 else 0
        return 0.4 * lambda2 + 0.3 * dens + 0.3 * (bis / (G.number_of_nodes() ** 2))

class DeepGIN(nn.Module):
    """GIN for Expressive Embeddings [Xu et al., ICLR 2019]."""
    def __init__(self, in_dim: int = 12, hidden: int = 128, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

        for _ in range(depth):
            nn = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINConv(nn, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden))

        self.link_pred = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        return x

    def link_score(self, emb: torch.Tensor, i: int, j: int):
        return self.link_pred(torch.cat([emb[i], emb[j]], dim=-1)).squeeze()

def run_optimization(graphml_path: str, max_steps: int = 10, delta: float = 0.01, reward_type: str = "graphrare", num_episodes: int = 1000):
    """End-to-End Training Loop."""
    env = DummyVecEnv([lambda: GraphPPOEnv(graphml_path, max_steps, delta, reward_type)])
    model = PPO("MlpPolicy", env, verbose=1, seed=SEED, learning_rate=1e-4)
    model.learn(total_timesteps=num_episodes * max_steps)

    # Extract final suggestions
    obs = env.reset()
    history = []
    for step in range(max_steps):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        # Decode action to link
        env_instance = env.envs[0]
        candidates = env_instance._get_candidates()
        if action < len(candidates):
            u, v = candidates[action]
            history.append({
                "step": step + 1,
                "added_link": {"src": u, "dst": v},
                "reward": reward[0],
                "utility": env_instance._graph_utility(env_instance.G)
            })

    # Save
    Path("resilink_v2_results").mkdir(exist_ok=True)
    with open("resilink_v2_results/history.json", "w") as f:
        json.dump(history, f, indent=2)
    model.save("resilink_v2_results/ppo_model")
    log.info(f"v2 Optimization complete. History saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResiLink v2: GIN + PPO")
    parser.add_argument("graphml", help="GraphML file")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--reward-type", choices=["graphrare", "linear"], default="graphrare")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    run_optimization(args.graphml, args.max_steps, args.delta, args.reward_type, args.episodes)