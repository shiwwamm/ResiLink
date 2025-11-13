#!/usr/bin/env python3
"""
ResiLink – Topology-Only Optimizer (Enhanced)
--------------------------------------------

Now with:
- Capacity-aware scoring
- Node type constraints
- Max degree limits
- Bisection bandwidth
- Visualization
- Mininet export
- Interactive mode
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
from torch_geometric.nn import GATConv
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from networkx.drawing.nx_agraph import graphviz_layout
import random

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# GNN
# --------------------------------------------------------------------------- #
class SimpleGAT(nn.Module):
    def __init__(self, in_dim=10, hidden=64, heads=4, dropout=0.1):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=dropout)
        self.pred = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

    def link_score(self, emb, u, v):
        return self.pred(torch.cat([emb[u], emb[v]], dim=0)).squeeze()

# --------------------------------------------------------------------------- #
# RL Agent
# --------------------------------------------------------------------------- #
class TinyRL:
    def __init__(self, state_dim=8, lr=1e-3):
        self.q = nn.Sequential(
            nn.Linear(state_dim + 1, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

    @staticmethod
    def graph_state(G: nx.Graph, use_capacity: bool = False) -> torch.Tensor:
        props = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "connected": float(nx.is_connected(G)),
            "avg_degree": np.mean([d for _, d in G.degree()]),
            "diameter": nx.diameter(G) if nx.is_connected(G) else 0,
            "total_capacity": sum(TinyRL._parse_capacity(e.get("capacity", "0")) for _, _, e in G.edges(data=True)) if use_capacity else 0,
            "avg_capacity": np.mean([TinyRL._parse_capacity(e.get("capacity", "0")) for _, _, e in G.edges(data=True)]) if use_capacity and G.edges else 0,
        }
        vec = np.array([props[k] for k in props.keys()])
        vec[0] /= 1000
        vec[1] /= 1000
        vec[6] /= 1e6 if use_capacity else 1
        vec[7] /= 1e6 if use_capacity else 1
        return torch.tensor(vec, dtype=torch.float32)

    @staticmethod
    def _parse_capacity(cap: str) -> float:
        try:
            cap = cap.strip().lower()
            if "gbps" in cap: return float(cap.replace("gbps", "").replace("<", "").strip()) * 1e9
            if "mbps" in cap: return float(cap.replace("mbps", "").replace("<", "").strip()) * 1e6
            return 0
        except: return 0

    def select(self, state, candidates):
        if np.random.rand() < 0.05:
            return np.random.randint(len(candidates))
        scores = [self.q(torch.cat([state, torch.tensor([i])])).item() for i in range(len(candidates))]
        return int(np.argmax(scores))

    def train(self, s, a, r, s_next):
        self.opt.zero_grad()
        pred = self.q(torch.cat([s, torch.tensor([a])]))
        target = r + 0.95 * self.q(torch.cat([s_next, torch.tensor([0.])])).detach()
        loss = F.mse_loss(pred, target)
        loss.backward()
        self.opt.step()

# --------------------------------------------------------------------------- #
# Main Optimizer
# --------------------------------------------------------------------------- #
class TopologyOnlyOptimizer:
    def __init__(
        self,
        graphml_path: str,
        max_cycles: int = 10,
        use_capacity: bool = False,
        node_types_file: Optional[str] = None,
        max_degree: Optional[int] = None,
        show_bisection: bool = False,
        visualize: bool = False,
        export_mininet: bool = False,
        interactive: bool = False,
    ):
        from topology_only_analyzer import TopologyOnlyAnalyzer
        self.analyzer = TopologyOnlyAnalyzer()
        if not self.analyzer.load_graphml(graphml_path):
            raise FileNotFoundError(graphml_path)

        self.use_capacity = use_capacity
        self.max_degree = max_degree
        self.show_bisection = show_bisection
        self.visualize = visualize
        self.export_mininet = export_mininet
        self.interactive = interactive

        self.gnn = SimpleGAT()
        self.rl = TinyRL()
        self.max_cycles = max_cycles
        self.history: List[Dict] = []

        # Load node types
        self.node_types = {}
        if node_types_file:
            self._load_node_types(node_types_file)

        if visualize:
            self._init_plot()

    def _load_node_types(self, path: str):
        import json
        with open(path) as f:
            data = json.load(f)
            for nid, typ in data.items():
                if nid in self.analyzer.G.nodes:
                    self.node_types[nid] = typ

    def _init_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def _node_features(self) -> torch.Tensor:
        G = self.analyzer.G
        deg = np.array([G.degree(n) for n in G.nodes()])
        bet = nx.betweenness_centrality(G, weight=None, normalized=True)
        bet_vec = np.array([bet.get(n, 0) for n in G.nodes()])

        deg = deg / (deg.max() + 1e-6)
        bet_vec = bet_vec / (bet_vec.max() + 1e-6)

        # Add type one-hot
        types = list(set(self.node_types.values())) if self.node_types else ["default"]
        type_idx = {t: i for i, t in enumerate(types)}
        type_vec = np.zeros((len(G), len(types)))
        for i, n in enumerate(G.nodes()):
            t = self.node_types.get(n, "default")
            if t in type_idx:
                type_vec[i, type_idx[t]] = 1.0

        feats = np.concatenate([deg[:, None], bet_vec[:, None], type_vec], axis=1)
        pad = np.zeros((len(G), 10 - feats.shape[1]))
        return torch.tensor(np.concatenate([feats, pad], axis=1), dtype=torch.float32)

    def _edge_index(self) -> torch.Tensor:
        edges = [(i, j) for i, (u, v) in enumerate(self.analyzer.G.edges()) for i, j in [(list(self.analyzer.G.nodes()).index(u), list(self.analyzer.G.nodes()).index(v)), (list(self.analyzer.G.nodes()).index(v), list(self.analyzer.G.nodes()).index(u))]]
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _respect_constraints(self, u: str, v: str) -> bool:
        if self.max_degree is not None:
            if self.analyzer.G.degree(u) >= self.max_degree or self.analyzer.G.degree(v) >= self.max_degree:
                return False
        if self.node_types:
            tu, tv = self.node_types.get(u, "default"), self.node_types.get(v, "default")
            if tu == "leaf" and tv == "leaf":
                return False
        return True

    def _calculate_bisection(self, G: nx.Graph) -> float:
        if not nx.is_connected(G): return 0.0
        try:
            cut_value, _ = nx.stoer_wagner(G)
            return cut_value
        except:
            return 0.0

    def run_one_cycle(self) -> Dict:
        G = self.analyzer.G
        candidates = [(u, v) for u, v in self.analyzer.candidate_pairs() if self._respect_constraints(u, v)]
        if not candidates:
            log.info("No valid candidate links.")
            return {}

        x = self._node_features()
        edge_index = self._edge_index()
        emb = self.gnn(x, edge_index)

        gnn_scores = []
        for u, v in candidates:
            i = list(G.nodes()).index(u)
            j = list(G.nodes()).index(v)
            score = self.gnn.link_score(emb, i, j).item()
            gnn_scores.append(score)

        state = TinyRL.graph_state(G, self.use_capacity)
        rl_idx = self.rl.select(state, candidates)
        rl_score = 1.0 if rl_idx == np.argmax(gnn_scores) else 0.1

        ensemble = np.array(gnn_scores) * 0.7 + rl_score * 0.3
        best = int(np.argmax(ensemble))
        u, v = candidates[best]

        if self.interactive:
            print(f"\nProposed: {u} ↔ {v} (score: {ensemble[best]:.3f})")
            if input("Add? [y/N] ").lower() != 'y':
                return {}

        G.add_edge(u, v, capacity="suggested")
        new_state = TinyRL.graph_state(G, self.use_capacity)
        reward = self._reward(G)

        self.rl.train(state, best, reward, new_state)

        result = {
            "cycle": len(self.history) + 1,
            "added_link": {"src": u, "dst": v, "gnn_score": float(gnn_scores[best])},
            "graph_metrics": {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "density": nx.density(G),
                "connected": nx.is_connected(G),
                "diameter": nx.diameter(G) if nx.is_connected(G) else None,
                "bisection_bandwidth": self._calculate_bisection(G) if self.show_bisection else None,
            },
            "reward": reward,
        }
        self.history.append(result)
        log.info(f"Cycle {result['cycle']}: {u} ↔ {v}")

        if self.visualize:
            self._update_plot(G)

        return result

    def _reward(self, G: nx.Graph) -> float:
        conn = 1.0 if nx.is_connected(G) else 0.0
        dens = nx.density(G)
        diam = nx.diameter(G) if nx.is_connected(G) else 999
        bis = self._calculate_bisection(G) / 1e9 if self.show_bisection else 0
        return conn * 0.4 + dens * 0.3 - (diam / 100.0) * 0.2 + bis * 0.1

    def _update_plot(self, G: nx.Graph):
        self.ax.clear()
        pos = graphviz_layout(G, prog="neato")
        colors = ['lightblue' if n in self.node_types and self.node_types[n] == 'spine' else 'lightgreen' for n in G.nodes()]
        nx.draw(G, pos, ax=self.ax, node_color=colors, with_labels=True, node_size=500, font_size=8)
        self.ax.set_title(f"Cycle {len(self.history)}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.5)

    def export_to_mininet(self, path: str):
        out = ["#!/usr/bin/python", 'from mininet.topo import Topo', 'class MyTopo(Topo):', '    def __init__(self):', '        Topo.__init__(self)']
        hosts = []
        for i, n in enumerate(self.analyzer.G.nodes()):
            h = f"h{i}"
            hosts.append(h)
            out.append(f"        {h} = self.addHost('{n}')")
        for u, v in self.analyzer.G.edges():
            i = list(self.analyzer.G.nodes()).index(u)
            j = list(self.analyzer.G.nodes()).index(v)
            out.append(f"        self.addLink({hosts[i]}, {hosts[j]})")
        out.append("topos = { 'mytopo': ( lambda: MyTopo() ) }")
        Path(path).write_text("\n".join(out))
        log.info(f"Mininet topology exported to {path}")

    def run(self):
        for c in range(self.max_cycles):
            res = self.run_one_cycle()
            if not res:
                break
            if res["graph_metrics"]["density"] > 0.6 and res["graph_metrics"]["connected"]:
                log.info("Target topology quality reached.")
                break
            time.sleep(0.1)

        out_dir = Path("resilink_topology_results")
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        torch.save(self.gnn.state_dict(), out_dir / "gnn_final.pth")

        if self.export_mininet:
            self.export_to_mininet(out_dir / "mininet_topo.py")

        if self.visualize:
            plt.ioff()
            plt.show()

        log.info(f"Optimization complete. Results in {out_dir}")

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="ResiLink – topology-only optimizer")
    parser.add_argument("graphml", help="GraphML file")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--use-capacity", action="store_true", help="Use edge capacity in scoring")
    parser.add_argument("--node-types-file", type=str, help="JSON: {node: type}")
    parser.add_argument("--max-degree", type=int, help="Max links per node")
    parser.add_argument("--show-bisection", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--export-mininet", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    opt = TopologyOnlyOptimizer(
        args.graphml,
        max_cycles=args.cycles,
        use_capacity=args.use_capacity,
        node_types_file=args.node_types_file,
        max_degree=args.max_degree,
        show_bisection=args.show_bisection,
        visualize=args.visualize,
        export_mininet=args.export_mininet,
        interactive=args.interactive,
    )
    opt.run()

if __name__ == "__main__":
    main()