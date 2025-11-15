# benchmark.py — FULL BENCHMARK
import os, time, csv
from pathlib import Path
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import GraphPPOEnv
from train import GNNFeatureExtractor
from baselines import greedy_topology, random_topology, degree_topology

TOPOLOGIES = ["Aarnet", "Abilene", "Nsfnet", "Geant2012"]
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

results = []

def run_rl(G_path, reward_name, max_links=10):
    env_fn = lambda: GraphPPOEnv(G_path, max_steps=30, plateau_steps=5, plateau_threshold=0.05)
    env = DummyVecEnv([env_fn])
    policy_kwargs = dict(
        features_extractor_class=GNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64])
    )
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=0, device="cpu", seed=42)
    model.learn(total_timesteps=20000)

    obs = env.reset()
    G_final = nx.read_graphml(G_path)
    added = []
    for _ in range(30):
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

for topo in TOPOLOGIES:
    path = f"real_world_topologies/{topo}.graphml"
    if not os.path.exists(path): continue

    G_orig = nx.read_graphml(path)
    orig_cut = nx.stoer_wagner(G_orig)[0]

    # Baselines
    for name, func in [("Greedy", greedy_topology), ("Random", random_topology), ("Degree", degree_topology)]:
        start = time.time()
        G_final, added = func(G_orig.copy(), max_links=10)
        final_cut = nx.stoer_wagner(G_final)[0]
        gain = (final_cut - orig_cut) / orig_cut * 100 if orig_cut > 0 else 0
        results.append({
            "Topology": topo,
            "Method": name,
            "Final_TPut": f"{final_cut*10:.1f}",
            "Gain_%": f"{gain:.1f}",
            "Links": len(added),
            "Time_s": f"{time.time()-start:.1f}"
        })

    # RL Methods
    for name in ["Paper_Reward", "Ours_v1", "Ours_v2"]:
        start = time.time()
        G_final, added = run_rl(path, name)
        final_cut = nx.stoer_wagner(G_final)[0]
        gain = (final_cut - orig_cut) / orig_cut * 100 if orig_cut > 0 else 0
        results.append({
            "Topology": topo,
            "Method": name,
            "Final_TPut": f"{final_cut*10:.1f}",
            "Gain_%": f"{gain:.1f}",
            "Links": len(added),
            "Time_s": f"{time.time()-start:.1f}"
        })

# Save
csv_path = OUT_DIR / "benchmark.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"→ {csv_path}")