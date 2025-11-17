# compare_jsac.py — FINAL (100% safe)
import os, time, csv
from pathlib import Path
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import GraphPPOEnv
from train import GNNFeatureExtractor
import numpy as np

TOPO = "AttMpls"
PATH = f"real_world_topologies/{TOPO}.graphml"
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

def run_jsac_reward(G_path):
    env_fn = lambda: GraphPPOEnv(G_path, max_steps=40, plateau_steps=5, plateau_threshold=0.1)
    env = DummyVecEnv([env_fn])
    policy_kwargs = dict(
        features_extractor_class=GNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64])
    )
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=0, device="cpu", seed=42)
    model.learn(total_timesteps=5000)

    obs = env.reset()
    G_final = nx.read_graphml(G_path)
    added = []
    for _ in range(40):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        candidates = env.envs[0].candidates
        if action >= len(candidates): break
        u, v = candidates[action]
        G_final.add_edge(u, v, capacity="10 Gbps")
        added.append((u, v))
        obs, _, done, _ = env.step([action])
        if done[0]: break
    return G_final, added

def compute_jsac_reward(G):
    if not nx.is_connected(G): return -100.0
    min_cut = nx.stoer_wagner(G)[0]
    λ2 = nx.algebraic_connectivity(G)
    var_deg = np.var([G.degree(n) for n in G.nodes()])
    bonus = 1.0 if λ2 >= 0.5 else 0.0
    return 0.5 * min_cut + 0.4 * λ2 - 0.1 * var_deg + bonus

# Run
print("RESILINK v4 vs. IEEE JSAC 2023")
print("="*80)
G_orig = nx.read_graphml(PATH)
orig_cut = nx.stoer_wagner(G_orig)[0]
orig_U = compute_jsac_reward(G_orig)
print(f"Original: min-cut={orig_cut}, U={orig_U:.2f}")

start = time.time()
G_final, added = run_jsac_reward(PATH)
final_cut = nx.stoer_wagner(G_final)[0]
final_U = compute_jsac_reward(G_final)
gain = (final_cut - orig_cut) / orig_cut * 100 if orig_cut > 0 else 0

print(f"JSAC Reward → {len(added)} links → min-cut={final_cut} (+{gain:.1f}%) in {time.time()-start:.1f}s")
print(f"JSAC U: {orig_U:.2f} → {final_U:.2f}")

# Run ResiLink v4
from benchmark import run_rl
G_final_v4, added_v4 = run_rl(PATH)
final_cut_v4 = nx.stoer_wagner(G_final_v4)[0]
gain_v4 = (final_cut_v4 - orig_cut) / orig_cut * 100

print(f"ResiLink v4 → {len(added_v4)} links → min-cut={final_cut_v4} (+{gain_v4:.1f}%)")

# Save
results = [
    {"Method": "JSAC 2023", "MinCut": final_cut, "Gain%": gain, "Links": len(added)},
    {"Method": "ResiLink v4", "MinCut": final_cut_v4, "Gain%": gain_v4, "Links": len(added_v4)}
]
with open(OUT_DIR / "compare_jsac.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"→ results/compare_jsac.csv")
