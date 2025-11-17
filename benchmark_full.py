#!/usr/bin/env python3
# benchmark_full.py — Complete Benchmark: Baselines + Papers + ResiLink on 8 Topologies
# Combines benchmark.py and compare_papers.py for comprehensive evaluation
# Runtime: ~3-4 hours on CPU

import os
import time
import csv
from pathlib import Path
import networkx as nx
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import GraphPPOEnv
from train import GNNFeatureExtractor
from baselines import greedy_topology, random_topology, degree_topology, betweenness_topology

# Configuration
TOPOLOGIES = [
    "Aarnet", "Abilene", "Nsfnet", "Geant2012",
    "AttMpls", "Bellcanada", "TataNld", "Cogentco"
]

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUT_DIR / "benchmark_full.csv"

BASELINE_TIMESTEPS = 0  # Baselines don't need training
PAPER_TIMESTEPS = 5000  # Fast comparison (2-3 min each)
RESILINK_TIMESTEPS = 50000  # Full training (15 min)

print("=" * 100)
print("RESILINK FULL BENCHMARK")
print(f"Topologies: {len(TOPOLOGIES)}")
print(f"Methods: 4 Baselines + 4 Papers + ResiLink = 9 total")
print(f"Paper methods: {PAPER_TIMESTEPS} timesteps")
print(f"ResiLink: {RESILINK_TIMESTEPS} timesteps")
print("=" * 100)

# ============================================================================
# UTILITY FUNCTIONS (from compare_papers.py)
# ============================================================================

def compute_jsac_utility(G):
    """IEEE JSAC 2023 (Li et al.)"""
    if not nx.is_connected(G):
        return -100.0
    min_cut = nx.stoer_wagner(G)[0]
    λ2 = nx.algebraic_connectivity(G)
    var_deg = np.var([G.degree(n) for n in G.nodes()])
    bonus = 1.0 if λ2 >= 0.5 else 0.0
    return 0.5 * min_cut + 0.4 * λ2 - 0.1 * var_deg + bonus

def compute_neuroplan_utility(G):
    """SIGCOMM 2021 (Zhu et al., NeuroPlan)"""
    if not nx.is_connected(G):
        return -100.0
    min_cut = nx.stoer_wagner(G)[0]
    deg_penalty = np.var([G.degree(n) for n in G.nodes()])
    return 0.8 * min_cut - 0.2 * deg_penalty

def compute_graphrare_utility(G):
    """arXiv 2023 (Shu et al., GraphRARE)"""
    if not nx.is_connected(G):
        return -100.0
    λ2 = nx.algebraic_connectivity(G)
    var_deg = np.var([G.degree(n) for n in G.nodes()])
    return λ2 - 0.2 * var_deg

def compute_wsn_utility(G):
    """GLOBECOM 2019 (Meng et al., Deep RL for WSN)"""
    if not nx.is_connected(G):
        return -100.0
    coverage = nx.density(G)
    energy = np.mean([G.degree(n) for n in G.nodes()])
    return 0.7 * coverage - 0.3 * energy

def compute_resilink_utility(G):
    """ResiLink v4 utility function"""
    if not nx.is_connected(G):
        return -100.0
    min_cut = nx.stoer_wagner(G)[0]
    λ2 = nx.algebraic_connectivity(G)
    var_deg = np.var([G.degree(n) for n in G.nodes()])
    min_cut_bonus = 5.0 if min_cut >= 2 else 0.0
    return 2.0 * min_cut + 0.5 * λ2 - 0.1 * var_deg + min_cut_bonus

# ============================================================================
# TRAINING FUNCTION (from compare_papers.py)
# ============================================================================

def train_rl_method(G_path, method_name, utility_func, timesteps, max_steps=20):
    """Train RL model with specific utility function"""
    print(f"    Training {method_name} ({timesteps} timesteps)...", end=" ", flush=True)
    
    # Override utility for this run
    class MethodEnv(GraphPPOEnv):
        def _compute_U(self):
            return utility_func(self.G)
    
    env_fn = lambda: MethodEnv(G_path, max_steps=max_steps, plateau_steps=5, plateau_threshold=0.01)
    env = DummyVecEnv([env_fn])
    
    policy_kwargs = dict(
        features_extractor_class=GNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64])
    )
    
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
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
    
    # Evaluate
    obs = env.reset()
    G_final = nx.read_graphml(G_path)
    added = []
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        candidates = env.envs[0].candidates
        if action >= len(candidates):
            break
        u, v = candidates[action]
        if not G_final.has_edge(u, v):
            G_final.add_edge(u, v, capacity="10 Gbps")
            added.append((u, v))
        obs, _, done, _ = env.step([action])
        if done[0]:
            break
    
    final_cut = nx.stoer_wagner(G_final)[0]
    total_time = time.time() - start
    
    print(f"{len(added)} links → min-cut {final_cut} in {total_time:.1f}s")
    
    return G_final, len(added), final_cut, total_time

# ============================================================================
# MAIN BENCHMARK LOOP
# ============================================================================

results = []

# CSV header
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Topology", "Method", "Type", "Nodes", "Edges",
        "Orig_MinCut", "Final_MinCut", "Gain_%", "Links_Added",
        "Throughput_Gbps", "Time_s"
    ])

for topo in TOPOLOGIES:
    path = f"real_world_topologies/{topo}.graphml"
    if not os.path.exists(path):
        print(f"\n⚠️  Skipping {topo} — file not found")
        continue

    print(f"\n{'='*100}")
    print(f"TOPOLOGY: {topo}")
    print(f"{'='*100}")
    
    # Load original graph
    G_orig = nx.read_graphml(path)
    orig_cut = nx.stoer_wagner(G_orig)[0]
    
    print(f"Original: {G_orig.number_of_nodes()} nodes, {G_orig.number_of_edges()} edges, min-cut={orig_cut}")
    print()
    
    # ========================================================================
    # 1. BASELINES (from benchmark.py)
    # ========================================================================
    print("  [1/3] Running Baselines...")
    
    baseline_methods = [
        ("Random", random_topology),
        ("Degree", degree_topology),
        ("Betweenness", betweenness_topology),
        ("Greedy", greedy_topology),
    ]
    
    for name, func in baseline_methods:
        print(f"    {name:15s}...", end=" ", flush=True)
        start = time.time()
        try:
            G_final, added = func(G_orig.copy(), max_links=15)
            final_cut = nx.stoer_wagner(G_final)[0]
            gain = (final_cut - orig_cut) / orig_cut * 100 if orig_cut > 0 else 0
            elapsed = time.time() - start
            
            print(f"{len(added):2d} links → min-cut {final_cut} (+{gain:5.1f}%) in {elapsed:5.1f}s")
            
            # Save result
            result = [
                topo, name, "Baseline",
                G_orig.number_of_nodes(), G_orig.number_of_edges(),
                orig_cut, final_cut, f"{gain:.1f}", len(added),
                final_cut * 10, f"{elapsed:.1f}"
            ]
            results.append(result)
            
            with open(CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow(result)
                
        except Exception as e:
            print(f"FAILED: {e}")
    
    # ========================================================================
    # 2. PAPER METHODS (from compare_papers.py)
    # ========================================================================
    print(f"\n  [2/3] Running Paper Methods ({PAPER_TIMESTEPS} timesteps each)...")
    
    paper_methods = [
        ("JSAC 2023", compute_jsac_utility),
        ("NeuroPlan 2021", compute_neuroplan_utility),
        ("GraphRARE 2023", compute_graphrare_utility),
        ("WSN 2019", compute_wsn_utility),
    ]
    
    for name, utility_func in paper_methods:
        try:
            G_final, links, final_cut, elapsed = train_rl_method(
                path, name, utility_func, PAPER_TIMESTEPS, max_steps=20
            )
            gain = (final_cut - orig_cut) / orig_cut * 100 if orig_cut > 0 else 0
            
            # Save result
            result = [
                topo, name, "Paper",
                G_orig.number_of_nodes(), G_orig.number_of_edges(),
                orig_cut, final_cut, f"{gain:.1f}", links,
                final_cut * 10, f"{elapsed:.1f}"
            ]
            results.append(result)
            
            with open(CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow(result)
                
        except Exception as e:
            print(f"    {name:15s}... FAILED: {e}")
    
    # ========================================================================
    # 3. RESILINK v4 (Ours)
    # ========================================================================
    print(f"\n  [3/3] Running ResiLink v4 ({RESILINK_TIMESTEPS} timesteps)...")
    
    try:
        G_final, links, final_cut, elapsed = train_rl_method(
            path, "ResiLink v4", compute_resilink_utility, RESILINK_TIMESTEPS, max_steps=20
        )
        gain = (final_cut - orig_cut) / orig_cut * 100 if orig_cut > 0 else 0
        
        # Save result
        result = [
            topo, "ResiLink v4 (Ours)", "RL+GNN",
            G_orig.number_of_nodes(), G_orig.number_of_edges(),
            orig_cut, final_cut, f"{gain:.1f}", links,
            final_cut * 10, f"{elapsed:.1f}"
        ]
        results.append(result)
        
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow(result)
            
    except Exception as e:
        print(f"    ResiLink v4... FAILED: {e}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 100)
print("BENCHMARK COMPLETE!")
print("=" * 100)

# Calculate average gains per method
method_gains = {}
for result in results:
    method = result[1]
    gain = float(result[7])
    if method not in method_gains:
        method_gains[method] = []
    method_gains[method].append(gain)

print("\nAverage Gains by Method:")
print("-" * 50)
for method in sorted(method_gains.keys()):
    gains = method_gains[method]
    avg_gain = sum(gains) / len(gains)
    print(f"  {method:25s}: {avg_gain:6.1f}% (n={len(gains)})")

print("\n" + "=" * 100)
print(f"Results saved to: {CSV_PATH}")
print(f"Total methods tested: {len(method_gains)}")
print(f"Total topologies: {len(TOPOLOGIES)}")
print(f"Total experiments: {len(results)}")
print("=" * 100)

# Create summary table
print("\nTop 3 Methods by Average Gain:")
sorted_methods = sorted(method_gains.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)
for i, (method, gains) in enumerate(sorted_methods[:3], 1):
    avg_gain = sum(gains) / len(gains)
    print(f"  {i}. {method:25s}: {avg_gain:6.1f}%")

print("\n✅ Benchmark complete! Check results/benchmark_full.csv for details.")
