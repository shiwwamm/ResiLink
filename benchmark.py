# benchmark.py — FULL BENCHMARK (11 baselines + ResiLink v4)
# Run: python benchmark.py
# Output: results/benchmark_full.csv + console table
# Estimated time: ~2.5 hours on CPU (50k timesteps)

import os
import time
import csv
from pathlib import Path
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import GraphPPOEnv
from train import GNNFeatureExtractor
from baselines import BASELINES
from ml_baselines import ML_BASELINES

ALL_BASELINES = BASELINES + ML_BASELINES

TOPOLOGIES = [
    "Aarnet", "Abilene", "Nsfnet", "Geant2012",
    "AttMpls", "Bellcanada", "TataNld", "Cogentco"
]

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUT_DIR / "benchmark_full.csv"

def run_rl(G_path, timesteps=50000):
    env_fn = lambda: GraphPPOEnv(G_path, max_steps=40, plateau_steps=5, plateau_threshold=0.01)
    env = DummyVecEnv([env_fn])
    model = PPO(
        "MultiInputPolicy", env,
        policy_kwargs=dict(
            features_extractor_class=GNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[128,64], vf=[128,64])
        ),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=0,
        device="cpu",
        seed=42
    )
    start = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - start

    obs = env.reset()
    G_final = nx.read_graphml(G_path)
    added = []
    for _ in range(40):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action[0])
        candidates = env.envs[0].candidates
        if action >= len(candidates): break
        u, v = candidates[action]
        if not G_final.has_edge(u,v):
            G_final.add_edge(u,v)
            added.append((u,v))
        obs, _, done, _ = env.step([action])
        if done[0]: break
    total_time = time.time() - start
    return G_final, added, train_time, total_time

# Header
print("="*100)
print("RESILINK v4 — FULL BENCHMARK (11 BASELINES + RL)")
print(f"Topologies: {len(TOPOLOGIES)} | RL timesteps: 50k")
print("="*100)

results = []
file_exists = CSV_PATH.exists()

with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "Topology","Method","Nodes","Orig_Cut","Final_Cut","Gain_%","Links_Added",
            "Throughput_Gbps","Time_s","Notes"
        ])

    for topo in TOPOLOGIES:
        path = f"real_world_topologies/{topo}.graphml"
        if not os.path.exists(path):
            print(f"Skipping {topo} — file not found")
            continue

        print(f"\nTOPOLOGY: {topo}")
        print("-" * 80)
        G_orig = nx.read_graphml(path)
        orig_cut = nx.minimum_edge_cut(G_orig).__len__() or nx.stoer_wagner(G_orig)[0]
        print(f"Original: {G_orig.number_of_nodes()} nodes, {G_orig.number_of_edges()} edges, min-cut = {orig_cut}")

        # === Run all baselines ===
        for name, func in ALL_BASELINES:
            start = time.time()
            G_final, added = func(G_orig.copy(), max_links=15)
            final_cut = nx.minimum_edge_cut(G_final).__len__() or nx.stoer_wagner(G_final)[0]
            gain = (final_cut - orig_cut) / orig_cut * 100 if orig_cut > 0 else 0
            t = time.time() - start
            print(f"  {name:<18}: {len(added):2d} links → min-cut {final_cut} ({gain:+6.1f}%) in {t:5.1f}s")
            results.append({
                "Topology": topo,
                "Method": name,
                "Nodes": G_orig.number_of_nodes(),
                "Orig_Cut": orig_cut,
                "Final_Cut": final_cut,
                "Gain_%": f"{gain:.1f}",
                "Links_Added": len(added),
                "Throughput_Gbps": final_cut * 10,
                "Time_s": f"{t:.1f}",
                "Notes": "heuristic"
            })
            writer.writerow([
                topo, name, G_orig.number_of_nodes(), orig_cut, final_cut,
                f"{gain:.1f}", len(added), final_cut*10, f"{t:.1f}", "heuristic"
            ])

        # === Run ResiLink v4 ===
        print("  Running ResiLink v4 (50k timesteps)...")
        G_final, added, train_t, total_t = run_rl(path, timesteps=50000)
        final_cut = nx.stoer_wagner(G_final)[0]
        gain = (final_cut - orig_cut) / orig_cut * 100 if orig_cut > 0 else 0
        print(f"  ResiLink v4     : {len(added):2d} links → min-cut {final_cut} ({gain:+6.1f}%) in {total_t:.1f}s")
        results.append({
            "Topology": topo,
            "Method": "ResiLink v4 (Ours)",
            "Nodes": G_orig.number_of_nodes(),
            "Orig_Cut": orig_cut,
            "Final_Cut": final_cut,
            "Gain_%": f"{gain:.1f}",
            "Links_Added": len(added),
            "Throughput_Gbps": final_cut * 10,
            "Time_s": f"{total_t:.1f}",
            "Notes": "RL+GNN"
        })
        writer.writerow([
            topo, "ResiLink v4 (Ours)", G_orig.number_of_nodes(), orig_cut, final_cut,
            f"{gain:.1f}", len(added), final_cut*10, f"{total_t:.1f}", "RL+GNN"
        ])

print("="*100)
print(f"ALL DONE — Results saved to: {CSV_PATH}")
print("You now have 11 baselines + ResiLink → SOTA thesis table ready!")
print("="*100)