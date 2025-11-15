#!/usr/bin/env python3
# benchmark_v2.py — Better benchmark with pre-training
import os
import time
import csv
from pathlib import Path
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from env import GraphPPOEnv
from train import GNNFeatureExtractor
from baselines import greedy_topology, random_topology, degree_topology, betweenness_topology

TRAIN_TOPOLOGIES = ["Aarnet", "Abilene", "Nsfnet", "Geant2012"]
TEST_TOPOLOGIES = ["Aarnet", "Abilene", "Nsfnet", "Geant2012"]
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

results = []

print("=" * 80)
print("RESILINK BENCHMARK V2 - Multi-Topology Training")
print("=" * 80)

def train_rl_model(topologies, total_timesteps=100000):
    """Train a single model on multiple topologies"""
    print(f"\nTraining RL model on {len(topologies)} topologies...")
    print(f"Total timesteps: {total_timesteps}")
    
    # Create multiple environments (one per topology)
    def make_env(topo_path):
        def _init():
            return GraphPPOEnv(topo_path, max_steps=20, plateau_steps=5, plateau_threshold=0.01)
        return _init
    
    env_fns = [make_env(f"real_world_topologies/{t}.graphml") for t in topologies]
    env = DummyVecEnv(env_fns)
    
    policy_kwargs = dict(
        features_extractor_class=GNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )
    
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="cpu",
        seed=42,
    )
    
    start = time.time()
    model.learn(total_timesteps=total_timesteps)
    train_time = time.time() - start
    
    print(f"✓ Training completed in {train_time:.1f}s")
    return model, train_time

def evaluate_rl(model, topo_path, max_links=20):
    """Evaluate trained model on a topology"""
    env = DummyVecEnv([lambda: GraphPPOEnv(topo_path, max_steps=max_links)])
    
    obs = env.reset()
    G_final = nx.read_graphml(topo_path)
    added = []
    
    for _ in range(max_links):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        candidates = env.envs[0].candidates
        if action >= len(candidates):
            break
        u, v = candidates[action]
        G_final.add_edge(u, v, capacity="10 Gbps")
        added.append((u, v))
        obs, _, done, _ = env.step([action])
        if done[0]:
            break
    
    return G_final, added

# Train RL model on all training topologies
print(f"\n{'='*80}")
print("PHASE 1: Training RL Model")
print(f"{'='*80}")
rl_model, train_time = train_rl_model(TRAIN_TOPOLOGIES, total_timesteps=100000)

# Evaluate all methods
print(f"\n{'='*80}")
print("PHASE 2: Evaluation")
print(f"{'='*80}")

for topo in TEST_TOPOLOGIES:
    path = f"real_world_topologies/{topo}.graphml"
    if not os.path.exists(path):
        print(f"Skipping {topo} - file not found")
        continue

    print(f"\n{'='*80}")
    print(f"Topology: {topo}")
    print(f"{'='*80}")

    G_orig = nx.read_graphml(path)
    orig_cut = nx.stoer_wagner(G_orig)[0]
    print(f"Original: {G_orig.number_of_nodes()} nodes, {G_orig.number_of_edges()} edges, min-cut={orig_cut}")

    # Baselines
    baseline_methods = [
        ("Random", random_topology),
        ("Degree", degree_topology),
        ("Betweenness", betweenness_topology),
        ("Greedy", greedy_topology),
    ]

    for name, func in baseline_methods:
        print(f"  {name:12s}...", end=" ", flush=True)
        start = time.time()
        try:
            G_final, added = func(G_orig.copy(), max_links=10)
            final_cut = nx.stoer_wagner(G_final)[0]
            gain = (final_cut - orig_cut) / orig_cut * 100 if orig_cut > 0 else 0
            elapsed = time.time() - start
            print(f"{len(added):2d} links → {final_cut*10:5.1f} Gbps (+{gain:5.1f}%) in {elapsed:5.1f}s")
            results.append({
                "Topology": topo,
                "Method": name,
                "Orig_TPut": f"{orig_cut*10:.1f}",
                "Final_TPut": f"{final_cut*10:.1f}",
                "Gain_%": f"{gain:.1f}",
                "Links": len(added),
                "Time_s": f"{elapsed:.1f}",
            })
        except Exception as e:
            print(f"FAILED: {e}")

    # RL Method (using pre-trained model)
    print(f"  {'RL-GNN':12s}...", end=" ", flush=True)
    start = time.time()
    try:
        G_final, added = evaluate_rl(rl_model, path, max_links=20)
        final_cut = nx.stoer_wagner(G_final)[0]
        gain = (final_cut - orig_cut) / orig_cut * 100 if orig_cut > 0 else 0
        elapsed = time.time() - start
        print(f"{len(added):2d} links → {final_cut*10:5.1f} Gbps (+{gain:5.1f}%) in {elapsed:5.1f}s")
        results.append({
            "Topology": topo,
            "Method": "RL-GNN",
            "Orig_TPut": f"{orig_cut*10:.1f}",
            "Final_TPut": f"{final_cut*10:.1f}",
            "Gain_%": f"{gain:.1f}",
            "Links": len(added),
            "Time_s": f"{elapsed:.1f}",
        })
    except Exception as e:
        print(f"FAILED: {e}")

# Save results
csv_path = OUT_DIR / "benchmark_v2.csv"
if results:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Print summary table
    print(f"\n{'='*100}")
    print("BENCHMARK RESULTS (V2)")
    print(f"{'='*100}")
    print(f"Training time: {train_time:.1f}s for {len(TRAIN_TOPOLOGIES)} topologies")
    print(f"\n| {'Topology':<12} | {'Method':<12} | {'Orig':<6} | {'Final':<6} | {'Gain%':<6} | {'Links':<5} | {'Time':<6} |")
    print(f"|{'-'*14}|{'-'*14}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*7}|{'-'*8}|")
    for r in results:
        print(
            f"| {r['Topology']:<12} | {r['Method']:<12} | {r['Orig_TPut']:<6} | {r['Final_TPut']:<6} | {r['Gain_%']:<6} | {r['Links']:<5} | {r['Time_s']:<6} |"
        )
    print(f"{'='*100}")
    
    # Calculate average gains per method
    print(f"\nAverage Gains by Method:")
    methods = {}
    for r in results:
        method = r['Method']
        gain = float(r['Gain_%'])
        if method not in methods:
            methods[method] = []
        methods[method].append(gain)
    
    for method, gains in sorted(methods.items()):
        avg_gain = sum(gains) / len(gains)
        print(f"  {method:12s}: {avg_gain:6.1f}%")
    
    print(f"\n→ Results saved to: {csv_path}")
else:
    print("\nNo results to save.")
