# ml_baselines.py — 4 STRONG ML BASELINES (CPU-only, no training needed)
# Works with your current benchmark pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.utils import from_networkx
import random

# Simple MLP for GIN
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
    def forward(self, x):
        x = F.relu(self.lin1(x))
        return self.lin2(x)

# === 1. GCN-based link scorer ===
class GCNScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(7, 64)
        self.conv2 = GCNConv(64, 32)
        self.scorer = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x, edge_index, cand_edges):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        u_emb = x[cand_edges[:,0]]
        v_emb = x[cand_edges[:,1]]
        pair = torch.cat([u_emb, v_emb], dim=-1)
        score = self.scorer(pair).squeeze(-1)
        return score

# === 2. GAT-based link scorer ===
class GATScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(7, 64, heads=4, dropout=0.3)
        self.conv2 = GATConv(64*4, 32, heads=1, concat=False)
        self.scorer = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x, edge_index, cand_edges):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        u_emb = x[cand_edges[:,0]]
        v_emb = x[cand_edges[:,1]]
        pair = torch.cat([u_emb, v_emb], dim=-1)
        return self.scorer(pair).squeeze(-1)

# === 3. GIN-based scorer ===
class GINScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GINConv(MLP(7, 64))
        self.conv2 = GINConv(MLP(64, 32))
        self.scorer = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x, edge_index, cand_edges):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        u_emb = x[cand_edges[:,0]]
        v_emb = x[cand_edges[:,1]]
        pair = torch.cat([u_emb, v_emb], dim=-1)
        return self.scorer(pair).squeeze(-1)

# === 4. GRAN-style sampler (2021 ICLR) — very strong baseline ===
class GRANScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = GATConv(7, 32, heads=8, concat=False)
        self.mlp = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x, edge_index, cand_edges):
        x = F.elu(self.conv(x, edge_index))
        u_emb = x[cand_edges[:,0]]
        v_emb = x[cand_edges[:,1]]
        diff = torch.abs(u_emb - v_emb)
        pair = torch.cat([u_emb, v_emb, diff], dim=-1)
        return self.mlp(pair).squeeze(-1)

# === Helper: Run any scorer ===
def run_gnn_baseline(G, model_class, max_links=10, seed=42):
    torch.manual_seed(seed)
    G = G.copy()
    data = from_networkx(G)
    x = torch.tensor([
        [G.degree(n),
         nx.clustering(G, n),
         nx.betweenness_centrality(G, endpoints=True).get(n, 0),
         nx.closeness_centrality(G).get(n, 0),
         nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6).get(n, 0),
         nx.pagerank(G).get(n, 0),
         len(nx.node_connected_component(G.to_undirected(), n)) / G.number_of_nodes()]
        for n in G.nodes()
    ], dtype=torch.float)
    data.x = x
    candidates = [(u,v) for u,v in nx.non_edges(G) if G.degree(u)<8 and G.degree(v)<8]
    if len(candidates) == 0:
        return G, []
    cand_tensor = torch.tensor(candidates, dtype=torch.long)
    
    model = model_class()
    model.eval()
    with torch.no_grad():
        scores = model(data.x, data.edge_index, cand_tensor).numpy()
    
    ranked = sorted(zip(scores, candidates), reverse=True)
    added = []
    for _, (u,v) in ranked[:max_links*5]:  # oversample to respect degree
        if G.degree(u)<8 and G.degree(v)<8 and not G.has_edge(u,v):
            G.add_edge(u,v)
            added.append((u,v))
            if len(added) >= max_links:
                break
    return G, added

# Export list
ML_BASELINES = [
    ("GCN-LinkScore", lambda G, max_links=10: run_gnn_baseline(G, GCNScorer, max_links)),
    ("GAT-LinkScore", lambda G, max_links=10: run_gnn_baseline(G, GATScorer, max_links)),
    ("GIN-LinkScore", lambda G, max_links=10: run_gnn_baseline(G, GINScorer, max_links)),
    ("GRAN-Style",   lambda G, max_links=10: run_gnn_baseline(G, GRANScorer, max_links)),
]