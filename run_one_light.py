#!/usr/bin/env python3
"""
ResiLink v2 – LIGHT benchmark (one GraphML at a time)
=====================================================
* 5 added links
* 5 000 PPO timesteps (≈ 2 min on a VM)
* Prints the paper-specific metric + core ResiLink metrics
"""
import argparse, json, logging, os
from pathlib import Path
import numpy as np
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

# ----------------------------------------------------------------------
# Import the *fixed* environment (copy-paste from hybrid_resilink_v2.py)
# ----------------------------------------------------------------------
class GraphPPOEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, graphml_path: str, max_steps: int = 5,
                 delta: float = 0.01, reward_type: str = "graphrare"):
        super().__init__()
        self.graphml_path = Path(graphml_path)
        self.max_steps = max_steps
        self.delta = delta
        self.reward_type = reward_type
        self.G = None
        self.G_prev = None
        self.step_count = 0
        self.candidates = []

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
        self.candidates = self._get_candidates()
        return self._get_state(), {}

    def _get_candidates(self):
        nodes = list(self.G.nodes())
        cands = [(u, v) for i, u in enumerate(nodes)
                 for v in nodes[i+1:]
                 if not self.G.has_edge(u, v)
                 and self.G.degree(u) < 8 and self.G.degree(v) < 8]
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

    def _compute_reward(self) -> float:
        if not nx.is_connected(self.G): return -1.0
        U_cur = self._graph_utility(self.G)
        U_pre = self._graph_utility(self.G_prev)
        if self.reward_type == "graphrare":
            if U_cur > U_pre + self.delta: return 1.0
            if abs(U_cur - U_pre) <= self.delta: return 0.0
            return -1.0
        return U_cur - U_pre

    def _graph_utility(self, G: nx.Graph) -> float:
        if not nx.is_connected(G): return 0.0
        lam2 = nx.algebraic_connectivity(G)
        dens = nx.density(G)
        bis = nx.stoer_wagner(G)[0] if G.number_of_edges() > 0 else 0
        return 0.4 * lam2 + 0.3 * dens + 0.3 * (bis / (G.number_of_nodes() ** 2))


# ----------------------------------------------------------------------
# Paper-specific metric functions
# ----------------------------------------------------------------------
def citation_accuracy_proxy(G):   # density × 100
    return nx.density(G) * 100.0

def throughput_proxy(G):
    return float(nx.stoer_wagner(G)[0]) if nx.is_connected(G) else 0.0

def utilization_score(G):
    return 1.0 / nx.diameter(G) if nx.is_connected(G) else 0.0

def coverage_proxy(G):
    return nx.density(G)

def delay_reduction_proxy(G, orig_path):
    orig = nx.read_graphml(orig_path)
    if nx.is_connected(G) and nx.is_connected(orig):
        red = (nx.diameter(orig) - nx.diameter(G)) / nx.diameter(orig) * 100.0
        return max(red, 0.0)
    return 0.0


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ResiLink v2 – light benchmark")
    parser.add_argument("graphml", help="Path to fixed GraphML")
    parser.add_argument("--steps", type=int, default=5, help="Links to add")
    parser.add_argument("--paper", choices=["graphrare","neuroplan","drlgs","wsn","marl_wmn"],
                        required=True, help="Which paper metric to report")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Train PPO (tiny settings)
    # ------------------------------------------------------------------
    env = DummyVecEnv([lambda: GraphPPOEnv(args.graphml, max_steps=args.steps)])
    model = PPO(
        "MlpPolicy", env,
        n_steps=32,           # tiny rollout
        batch_size=64,
        learning_rate=3e-4,
        verbose=1,
        seed=42,
        device="cpu"
    )
    model.learn(total_timesteps=5_000)   # ~2 min on VM

    # ------------------------------------------------------------------
    # 2. Extract deterministic policy (no noise)
    # ------------------------------------------------------------------
    obs, _ = env.reset()
    G_final = nx.read_graphml(args.graphml)
    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, _ = env.step(action)
        env_inst = env.envs[0]
        cand = env_inst._get_candidates()
        if action < len(cand):
            u, v = cand[action]
            G_final.add_edge(u, v, capacity="10 Gbps")
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

    if args.paper == "graphrare":
        metric_val = citation_accuracy_proxy(G_final)
        unit = "% accuracy"
    elif args.paper == "neuroplan":
        metric_val = throughput_proxy(G_final)
        unit = "Gbps"
    elif args.paper == "drlgs":
        metric_val = utilization_score(G_final)
        unit = "utilization"
    elif args.paper == "wsn":
        metric_val = coverage_proxy(G_final)
        unit = "coverage"
    elif args.paper == "marl_wmn":
        metric_val = delay_reduction_proxy(G_final, args.graphml)
        unit = "% delay reduction"

    # ------------------------------------------------------------------
    # 4. Print result
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"Topology : {Path(args.graphml).name}")
    print(f"Paper    : {args.paper.upper()}")
    print(f"Metric   : {metric_val:.3f} {unit}")
    print("-"*60)
    print(f"Core metrics → dens={core['density']:.4f}, diam={core['diameter']}, "
          f"bis={core['bisection']}, λ₂={core['alg_conn']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()