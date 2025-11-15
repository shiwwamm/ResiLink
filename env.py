# env.py — FINAL FIXED: PADDED EDGE_INDEX
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np
from pathlib import Path
import torch

class GraphPPOEnv(gym.Env):
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
        n_nodes = self.G.number_of_nodes()
        max_edges = n_nodes * (n_nodes - 1) // 2  # Complete graph

        # FIXED: PADDED SHAPES
        self.observation_space = spaces.Dict({
            "node_feat": spaces.Box(low=-np.inf, high=np.inf, shape=(n_nodes, 7), dtype=np.float32),
            "edge_index": spaces.Box(low=0, high=n_nodes-1, shape=(2, max_edges), dtype=np.int64),
            "candidates": spaces.Box(low=0, high=n_nodes-1, shape=(200, 2), dtype=np.int64)
        })
        self.action_space = spaces.Discrete(200)

    def _load_graph(self):
        self.G = nx.read_graphml(self.graphml_path)
        for n in list(self.G.nodes()):
            self.G.nodes[n]['id'] = str(n)

    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        self._load_graph()
        self.step_count = 0
        self.G_prev = self.G.copy()
        self.original_degrees = {n: self.G.degree(n) for n in self.G.nodes()}
        self.candidates = self._get_candidates()
        self.recent_U = []
        self.best_U = self._compute_U()
        return self._get_obs(), {}

    def _get_candidates(self):
        node_list = list(self.G.nodes())
        current_added = {n: 0 for n in node_list}
        for u, v in self.G.edges():
            if not self.G_prev.has_edge(u, v):
                current_added[u] += 1
                current_added[v] += 1
        
        # Compute betweenness to identify bottleneck nodes
        try:
            edge_betweenness = nx.edge_betweenness_centrality(self.G)
            node_betweenness = nx.betweenness_centrality(self.G)
        except:
            edge_betweenness = {}
            node_betweenness = {n: 0 for n in node_list}
        
        cands = []
        for i in range(len(node_list)):
            for j in range(i+1, len(node_list)):
                u, v = node_list[i], node_list[j]
                if (not self.G.has_edge(u, v)
                    and self.original_degrees[u] < 8
                    and self.original_degrees[v] < 8
                    and current_added[u] < 1
                    and current_added[v] < 1):
                    # Prioritize links between high-betweenness nodes (bottlenecks)
                    # and links that span long distances (bridge communities)
                    try:
                        shortest_path = nx.shortest_path_length(self.G, u, v)
                    except:
                        shortest_path = 999  # Not connected
                    
                    priority = (node_betweenness.get(u, 0) + node_betweenness.get(v, 0)) * shortest_path
                    cands.append((u, v, priority))
        
        # Sort by priority (higher is better) and take top 200
        cands.sort(key=lambda x: x[2], reverse=True)
        return [(u, v) for u, v, _ in cands[:200]]

    def _get_obs(self):
        node_list = list(self.G.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        node_feat = self._node_features()

        # PAD edge_index to max_edges
        edge_list = [(node_to_idx[u], node_to_idx[v]) for u, v in self.G.edges()]
        edge_tensor = torch.zeros((2, self.observation_space["edge_index"].shape[1]), dtype=torch.long)
        if edge_list:
            edge_tensor[:, :len(edge_list)] = torch.tensor(edge_list, dtype=torch.long).t()

        cand_idx = torch.tensor([
            [node_to_idx[u], node_to_idx[v]] for u, v in self.candidates
        ], dtype=torch.long)
        if cand_idx.shape[0] < 200:
            pad = torch.zeros((200 - cand_idx.shape[0], 2), dtype=torch.long)
            cand_idx = torch.cat([cand_idx, pad], dim=0)

        return {
            "node_feat": node_feat,
            "edge_index": edge_tensor,
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
                try:
                    local_conn = nx.node_connectivity(self.G, n, neighbors[0])
                except: local_conn = 0
            feats.append([
                float(self.G.degree(n)),
                betweenness.get(n, 0.0),
                clustering.get(n, 0.0),
                1.0 if 'core' in str(n).lower() else 0.0,
                1.0 if self.G.degree(n) >= 4 else 0.0,
                float(local_conn),
                1.0 / (ecc.get(n, 1) + 1)
            ])
        return np.array(feats, dtype=np.float32)

    def _compute_U(self):
        if not nx.is_connected(self.G): return -100.0
        min_cut = nx.stoer_wagner(self.G)[0]
        λ2 = nx.algebraic_connectivity(self.G)
        imbalance = np.var([self.G.degree(n) for n in self.G.nodes()])
        
        # Higher weight on min-cut, bonus for reaching min-cut >= 2
        min_cut_bonus = 5.0 if min_cut >= 2 else 0.0
        return 2.0 * min_cut + 0.5 * λ2 - 0.1 * imbalance + min_cut_bonus

    def _check_plateau(self):
        if len(self.recent_U) < self.plateau_steps + 1: return False
        return self.recent_U[-1] - self.recent_U[-(self.plateau_steps + 1)] < self.plateau_threshold

    def step(self, action):
        if action >= len(self.candidates):
            return self._get_obs(), 0.0, True, False, {"plateau": False, "links": self.step_count}
        u, v = self.candidates[action]
        self.G_prev = self.G.copy()
        self.G.add_edge(u, v, capacity='10 Gbps')
        
        # Compute metrics before and after
        prev_U = self.best_U
        current_U = self._compute_U()
        
        # Better reward shaping: reward any improvement + exploration bonus
        utility_gain = current_U - prev_U
        
        # Small reward for adding links that improve connectivity even slightly
        if utility_gain > 0:
            reward = utility_gain
        else:
            # Small exploration bonus to encourage trying different placements
            reward = 0.01
        
        self.best_U = max(self.best_U, current_U)
        self.recent_U.append(current_U)
        if len(self.recent_U) > self.plateau_steps + 1: self.recent_U.pop(0)
        self.step_count += 1
        self.candidates = self._get_candidates()
        plateau = self._check_plateau()
        done = self.step_count >= self.max_steps or plateau
        info = {"plateau": plateau, "min_cut": nx.stoer_wagner(self.G)[0], "links": self.step_count}
        return self._get_obs(), reward, done, False, info
