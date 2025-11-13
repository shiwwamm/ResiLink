#!/usr/bin/env python3
"""
ResiLink v2 – LIGHT BENCHMARK (FINAL)
========================================
* 5 added links
* 1,600 PPO timesteps (~10 min)
* Hybrid reward: Spectral + Flow + Degree Penalty
* Fixed: degree constraint, Gymnasium API, correct metric
* Saves: optimized graph, links, metrics
"""
import argparse
import json
import logging
import os
from pathlib import Path
import numpy as np
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

# ----------------------------------------------------------------------
# GraphPPOEnv – Hybrid Reward + Degree Constraint
# ----------------------------------------------------------------------
class GraphPPOEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, graphml_path: str, max_steps: int = 5):
        super().__init__()
        self.graphml_path = Path(graphml_path)
        self.max_steps = max_steps
        self.G = None
        self.G_prev = None
        self.step_count = 0
        self.candidates = []
        self.original_degrees = {}

        self._load_graph()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(12,), dtype=np.float32)
        self.action_space = spaces.Discrete(200)

    def _load_graph(self):
        self.G = nx.read_graphml(self.graphml_path)
        for n in self.G.nodes():
            self.G.nodes[n]['id'] = str(n)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._load_graph()
        self.step_count = 0
        self.G_prev = self.G.copy()
        self.original_degrees = {n: self.G.degree(n) for n in self.G.nodes()}
        self.candidates = self._get_candidates()
        return self._get_state(), {}

    def _get_candidates(self):
        """Only allow nodes with ORIGINAL degree < 8"""
        nodes = list(self.G.nodes())
        cands = [(u, v) for i, u in enumerate(nodes)
                 for v in nodes[i+1:]
                 if not self.G.has_edge(u, v)
                 and self.original_degrees[u] < 8
                 and self.original_degrees[v] < 8]
        return cands[:200]

    def _get_state(self) -> np.ndarray:
        if not nx.is_connected(self.G):
            return np.zeros(12, dtype=np.float32)

        degrees = [d for _, d in self.G.degree()]
        caps = [self._parse_capacity(e.get('capacity', '10 Gbps'))
                for _, _, e in self.G.edges(data=True)]

        props = [
            self.G.number_of_nodes() / 100.0,
            self.G.number_of_edges() / 100.0,
            nx.density(self.G),
            float(nx.is_connected(self.G)),
            np.mean(degrees),
            np.std(degrees),
            nx.diameter(self.G) / 10.0,
            np.mean(caps) / 1e9,
            np.std(caps) / 1e9,
            np.max(caps) / 1e9,
            len(caps) / 100.0,
            np.median(caps) / 1e9,
        ]
        return np.clip(np.array(props, dtype=np.float32), -1, 1)

    def _parse_capacity(self, cap_str: str) -> float:
        try:
            s = str(cap_str).lower().replace('<', '').replace(' ', '')
            if 'gbps' in s: return float(s.replace('gbps', '')) * 1e9
            if 'mbps' in s: return float(s.replace('mbps', '')) * 1e6
            return 1e9
        except:
            return 1e9

    def _maxflow_robustness(self, G):
        nodes = list(G.nodes())
        if len(nodes) < 2: return 0.0
        s, t = np.random.choice(nodes, 2, replace=False)
        try:
            return nx.maximum_flow_value(G, s, t, capacity='capacity')
        except:
            return 0.0

    def _compute_reward(self) -> float:
        if not nx.is_connected(self.G):
            return -1.0

        # Spectral + Flow + Degree Penalty
        λ2 = nx.algebraic_connectivity(self.G)
        flow = self._maxflow_robustness(self.G)
        deg_penalty = sum(max(0, self.G.degree(n) - 8)**2 for n in self.G.nodes())

        U = 0.4 * λ2 + 0.4 * (flow / 10.0) - 0.2 * deg_penalty

        # Previous
        λ2_prev = nx.algebraic_connectivity(self.G_prev)
        flow_prev = self._maxflow_robustness(self.G_prev)
        deg_penalty_prev = sum(max(0, self.G_prev.degree(n) - 8)**2 for n in self.G_prev.nodes())
        U_prev = 0.4 * λ2_prev + 0.4 * (flow_prev / 10.0) - 0.2 * deg_penalty_prev

        return np.clip(U - U_prev, -1.0, 1.0)

    def step(self, action: int):
        if action >= len(self.candidates) or self.step_count >= self.max_steps:
            return self._get_state(), 0.0, True, False, {}

        u, v = self.candidates[action]
        self.G_prev = self.G.copy()
        self.G.add_edge(u, v, capacity='10 Gbps')
        reward = self._compute_reward()
        self.step_count += 1
        self.candidates = self._get_candidates()
        terminated = self.step_count >= self.max_steps
        return self._get_state(), reward, terminated, False, {}


# ----------------------------------------------------------------------
# Paper-specific metric functions
# ----------------------------------------------------------------------
def throughput_proxy(G):
    """NeuroPlan-style: max flow between random s-t pair (10 Gbps edges)"""
    if not nx.is_connected(G):
        return 0.0
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return 0.0
    s, t = np.random.choice(nodes, 2, replace=False)
    try:
        flow = nx.maximum_flow_value(G, s, t, capacity='capacity')
        return float(flow)
    except:
        return 0.0


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ResiLink v2 – light benchmark")
    parser.add_argument("graphml", help="Path to fixed GraphML")
    parser.add_argument("--steps", type=int, default=5, help="Links to add")
    parser.add_argument("--paper", choices=["neuroplan"], default="neuroplan",
                        help="Use NeuroPlan metric (throughput)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Train PPO (FAST)
    # ------------------------------------------------------------------
    env_fn = lambda: GraphPPOEnv(args.graphml, max_steps=args.steps)
    env = DummyVecEnv([env_fn])

    model = PPO(
        "MlpPolicy", env,
        n_steps=64,
        batch_size=64,
        n_epochs=4,
        learning_rate=5e-4,
        verbose=1,
        seed=42,
        device="cpu"
    )
    print("Training for 1,600 timesteps (~10 min)...")
    model.learn(total_timesteps=1_600)

    # ------------------------------------------------------------------
    # 2. Extract deterministic policy
    # ------------------------------------------------------------------
    print("Training done. Extracting links...")
    obs = env.reset()[0]
    G_final = nx.read_graphml(args.graphml)
    added_links = []

    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step([action])
        terminated = dones[0]
        obs = obs[0]

        env_inst = env.envs[0]
        cand = env_inst._get_candidates()
        if action < len(cand):
            u, v = cand[action]
            G_final.add_edge(u, v, capacity="10 Gbps")
            added_links.append((u, v))
            print(f"  Added link: {u} — {v}")

        if terminated:
            break

    # ------------------------------------------------------------------
    # 3. Compute metrics
    # ------------------------------------------------------------------
    core = {
        "density": nx.density(G_final),
        "diameter": nx.diameter(G_final) if nx.is_connected(G_final) else -1,
        "bisection": nx.stoer_wagner(G_final)[0],
        "alg_conn": nx.algebraic_connectivity(G_final)
    }

    metric_val = throughput_proxy(G_final)
    unit = "Gbps"

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    out_dir = Path("optimized_graphs")
    out_dir.mkdir(exist_ok=True)
    opt_path = out_dir / f"opt_{Path(args.graphml).name}"
    nx.write_graphml(G_final, opt_path)

    links_path = out_dir / f"links_{Path(args.graphml).stem}.txt"
    with open(links_path, "w") as f:
        for u, v in added_links:
            f.write(f"{u} — {v}\n")

    # ------------------------------------------------------------------
    # 5. Print result
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"Topology : {Path(args.graphml).name}")
    print(f"Paper    : NEUROPLAN")
    print(f"Metric   : {metric_val:.3f} {unit}")
    print("-"*60)
    print(f"Core metrics → dens={core['density']:.4f}, diam={core['diameter']}, "
          f"bis={core['bisection']}, λ₂={core['alg_conn']:.4f}")
    print(f"Saved graph → {opt_path}")
    print(f"Saved links → {links_path}")
    print("="*60)


if __name__ == "__main__":
    main()
