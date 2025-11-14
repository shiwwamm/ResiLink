# env.py
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np
from pathlib import Path
import torch

class GraphPPOEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, graphml_path: str, max_steps: int = 20, plateau_steps: int = 5, plateau_threshold: float = 0.01):
        super().__init__()
        self.graphml_path = Path(graphml_path)
        self.max_steps = max_steps
        self.plateau_steps = plateau_steps
        self.plateau_threshold = plateau_threshold

        self.G = None
        self.G_prev = None
        self.step_count = 0
        self.candidates = []
        self.original_degrees = {}
        self.recent_U = []
        self.best_U = -float('inf')

        self._load_graph()
        self.observation_space = spaces.Dict({
            "node_feat": spaces.Box(low=-np.inf, high=np.inf, shape=(self.G.number_of_nodes(), 7), dtype=np.float32),
            "edge_index": spaces.Box(low=0, high=self.G.number_of_nodes()-1, shape=(2, self.G.number_of_edges()*2), dtype=np.int64),
            "candidates": spaces.Box(low=0, high=self.G.number_of_nodes()-1, shape=(200, 2), dtype=np.int64)
        })
        self.action_space = spaces.Discrete(200)

    def _load_graph(self):
        self.G = nx.read_graphml(self.graphml_path)
        for n in self.G.nodes():
            self.G.nodes[n]['id'] = str(n)

    def reset(self, seed=None, options=None):
        if seed: np.random.seed(seed)
        self._load_graph()
        self.step_count = 0
        self.G_prev = self.G.copy()
        self.original_degrees = {n: self.G.degree(n) for n in self.G.nodes()}
        self.candidates = self._get_candidates()
        self.recent_U = []
        self.best_U = self._compute_U()
        return self._get_obs(), {}

    def _get_candidates(self):
        nodes = list(self.G.nodes())
        current_added = {n: 0 for n in nodes}
        for u, v in self.G.edges():
            if not self.G_prev.has_edge(u, v):
                current_added[u] += 1
                current_added[v] += 1
        cands = [(i, j) for i in range(len(nodes)) for j in range(i+1, len(nodes))
                 if not self.G.has_edge(nodes[i], nodes[j])
                 and self.original_degrees[nodes[i]] < 8
                 and self.original_degrees[nodes[j]] < 8
                 and current_added[nodes[i]] < 1
                 and current_added[nodes[j]] < 1]
        return cands[:200]

    def _get_obs(self):
        node_list = list(self.G.nodes())
        node_feat = self._node_features()
        edge_index = torch.tensor([
            [node_list.index(u), node_list.index(v)]
            for u, v in self.G.edges()
        ], dtype=torch.long).t()
        
        cand_idx = torch.tensor([
            [node_list.index(u), node_list.index(v)]
            for u, v in self.candidates[:200]
        ], dtype=torch.long)
        
        if cand_idx.shape[0] < 200:
            pad = torch.zeros((200 - cand_idx.shape[0], 2), dtype=torch.long)
            cand_idx = torch.cat([cand_idx, pad], dim=0)
        
        return {
            "node_feat": node_feat,
            "edge_index": edge_index,
            "candidates": cand_idx
        }

    def _node_features(self):
        betweenness = nx.betweenness_centrality(self.G)
        clustering = nx.clustering(self.G)
        ecc = nx.eccentricity(self.G) if nx.is_connected(self.G) else {n: 1 for n in self.G.nodes()}
        
        feats = []
        for n in self.G.nodes():
            neighbors = list(self.G.neighbors(n))
            local_conn = 0
            if neighbors:
                target = neighbors[0]
                try:
                    local_conn = nx.node_connectivity(self.G, n, target)
                except:
                    local_conn = 0
            else:
                local_conn = 0

            feats.append([
                self.G.degree(n),
                betweenness.get(n, 0),
                clustering.get(n, 0),
                1.0 if 'core' in str(n).lower() else 0.0,
                1.1 if self.G.degree(n) >= 4 else 0.0,
                local_conn,
                1.0 / (ecc.get(n, 1) + 1)
            ])
        return np.array(feats, dtype=np.float32)

    def _compute_U(self):
        if not nx.is_connected(self.G): return -100.0
        min_cut = nx.stoer_wagner(self.G)[0]
        λ2 = nx.algebraic_connectivity(self.G)
        imbalance = np.var([self.G.degree(n) for n in self.G.nodes()])
        return 0.6 * min_cut + 0.3 * λ2 - 0.1 * imbalance + (1.0 if min_cut >= 2 else 0.0)

    def _check_plateau(self):
        if len(self.recent_U) < self.plateau_steps + 1: return False
        return self.recent_U[-1] - self.recent_U[-(self.plateau_steps + 1)] < self.plateau_threshold

    def step(self, action):
        if action >= len(self.candidates):
            return self._get_obs(), 0.0, True, False, {"plateau": False, "links": self.step_count}

        u_idx, v_idx = self.candidates[action]
        node_list = list(self.G.nodes())
        u, v = node_list[u_idx], node_list[v_idx]
        self.G_prev = self.G.copy()
        self.G.add_edge(u, v, capacity='10 Gbps')

        current_U = self._compute_U()
        reward = max(0, current_U - self.best_U)
        self.best_U = max(self.best_U, current_U)
        self.recent_U.append(current_U)
        if len(self.recent_U) > self.plateau_steps + 1: self.recent_U.pop(0)

        self.step_count += 1
        self.candidates = self._get_candidates()
        plateau = self._check_plateau()
        done = self.step_count >= self.max_steps or plateau

        info = {"plateau": plateau, "min_cut": nx.stoer_wagner(self.G)[0], "links": self.step_count}
        return self._get_obs(), reward, done, False, info