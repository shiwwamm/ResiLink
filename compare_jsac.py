# compare_jsac.py — Compare ResiLink vs JSAC 2023 reward function
import os
import time
import csv
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

print("=" * 80)
print("RESILINK vs. IEEE JSAC 2023 Comparison")
print("=" * 80)

# Check if topology exists
if not os.path.exists(PATH):
    print(f"ERROR: {PATH} not found!")
    print("Available topologies:")
    for f in sorted(Path("real_world_topologies").glob("*.graphml"))[:10]:
        print(f"  - {f.stem}")
    exit(1)

def compute_jsac_utility(G):
    """JSAC 2023 utility function"""
    if not nx.is_connected(G):
        return -100.0
    min_cut = nx.stoer_wagner(G)[0]
    λ2 = nx.algebraic_connectivity(G)
    var_deg = np.var([G.degree(n) for n in G.nodes()])
    bonus = 1.0 if λ2 >= 0.5 else 0.0
    return 0.5 * min_cut + 0.4 * λ2 - 0.1 * var_deg + bonus

def compute_resilink_utility(G):
    """ResiLink utility function"""
    if not nx.is_connected(G):
        return -100.0
    min_cut = nx.stoer_wagner(G)[0]
    λ2 = nx.algebraic_connectivity(G)
    var_deg = np.var([G.degree(n) for n in G.nodes()])
    min_cut_bonus = 5.0 if min_cut >= 2 else 0.0
    return 2.0 * min_cut + 0.5 * λ2 - 0.1 * var_deg + min_cut_bonus

def train_and_evaluate(G_path, method_name, timesteps=50000, max_steps=20):
    """Train RL model and evaluate"""
    print(f"\n{method_name}:")
    print(f"  Training for {timesteps} timesteps...")
    
    env_fn = lambda: GraphPPOEnv(G_path, max_steps=max_steps, plateau_steps=5, plateau_threshold=0.01)
    env = DummyVecEnv([env_fn])
    
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
        verbose=0,
        device="cpu",
        seed=42,
    )
    
    start = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - start
    
    # Evaluate
    print(f"  Evaluating...")
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
        G_final.add_edge(u, v, capacity="10 Gbps")
        added.append((u, v))
        obs, _, done, _ = env.step([action])
        if done[0]:
            break
    
    eval_time = time.time() - start - train_time
    total_time = time.time() - start
    
    return G_final, added, train_time, eval_time, total_time

# Load original graph
G_orig = nx.read_graphml(PATH)
orig_cut = nx.stoer_wagner(G_orig)[0]
orig_jsac_U = compute_jsac_utility(G_orig)
orig_resilink_U = compute_resilink_utility(G_orig)

print(f"\nOriginal Topology: {TOPO}")
print(f"  Nodes: {G_orig.number_of_nodes()}")
print(f"  Edges: {G_orig.number_of_edges()}")
print(f"  Min-cut: {orig_cut}")
print(f"  JSAC Utility: {orig_jsac_U:.2f}")
print(f"  ResiLink Utility: {orig_resilink_U:.2f}")

# Note: Both methods use the same environment (ResiLink reward)
# This comparison shows if the reward function matters for final performance
print("\n" + "=" * 80)
print("NOTE: Both methods use ResiLink environment (same reward during training)")
print("Comparison shows final performance on different utility metrics")
print("=" * 80)

# Run ResiLink (current approach)
G_resilink, added_resilink, train_t1, eval_t1, total_t1 = train_and_evaluate(
    PATH, "ResiLink (Current)", timesteps=50000, max_steps=20
)

final_cut_resilink = nx.stoer_wagner(G_resilink)[0]
final_jsac_U_resilink = compute_jsac_utility(G_resilink)
final_resilink_U_resilink = compute_resilink_utility(G_resilink)
gain_resilink = (final_cut_resilink - orig_cut) / orig_cut * 100 if orig_cut > 0 else 0

print(f"  Results:")
print(f"    Links added: {len(added_resilink)}")
print(f"    Final min-cut: {final_cut_resilink} (+{gain_resilink:.1f}%)")
print(f"    JSAC Utility: {orig_jsac_U:.2f} → {final_jsac_U_resilink:.2f}")
print(f"    ResiLink Utility: {orig_resilink_U:.2f} → {final_resilink_U_resilink:.2f}")
print(f"    Time: {total_t1:.1f}s (train: {train_t1:.1f}s, eval: {eval_t1:.1f}s)")

# For comparison: train with fewer timesteps (like JSAC paper might use)
G_jsac_style, added_jsac_style, train_t2, eval_t2, total_t2 = train_and_evaluate(
    PATH, "JSAC-style (5k timesteps)", timesteps=5000, max_steps=40
)

final_cut_jsac = nx.stoer_wagner(G_jsac_style)[0]
final_jsac_U_jsac = compute_jsac_utility(G_jsac_style)
final_resilink_U_jsac = compute_resilink_utility(G_jsac_style)
gain_jsac = (final_cut_jsac - orig_cut) / orig_cut * 100 if orig_cut > 0 else 0

print(f"  Results:")
print(f"    Links added: {len(added_jsac_style)}")
print(f"    Final min-cut: {final_cut_jsac} (+{gain_jsac:.1f}%)")
print(f"    JSAC Utility: {orig_jsac_U:.2f} → {final_jsac_U_jsac:.2f}")
print(f"    ResiLink Utility: {orig_resilink_U:.2f} → {final_resilink_U_jsac:.2f}")
print(f"    Time: {total_t2:.1f}s (train: {train_t2:.1f}s, eval: {eval_t2:.1f}s)")

# Summary table
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print(f"| {'Method':<20} | {'MinCut':<8} | {'Gain%':<8} | {'Links':<6} | {'JSAC_U':<8} | {'ResiLink_U':<10} | {'Time':<8} |")
print(f"|{'-'*22}|{'-'*10}|{'-'*10}|{'-'*8}|{'-'*10}|{'-'*12}|{'-'*10}|")
print(
    f"| {'Original':<20} | {orig_cut:<8} | {'-':<8} | {'-':<6} | {orig_jsac_U:<8.2f} | {orig_resilink_U:<10.2f} | {'-':<8} |"
)
print(
    f"| {'ResiLink (50k)':<20} | {final_cut_resilink:<8} | {gain_resilink:<8.1f} | {len(added_resilink):<6} | {final_jsac_U_resilink:<8.2f} | {final_resilink_U_resilink:<10.2f} | {total_t1:<8.1f} |"
)
print(
    f"| {'JSAC-style (5k)':<20} | {final_cut_jsac:<8} | {gain_jsac:<8.1f} | {len(added_jsac_style):<6} | {final_jsac_U_jsac:<8.2f} | {final_resilink_U_jsac:<10.2f} | {total_t2:<8.1f} |"
)
print("=" * 80)

# Save results
results = [
    {
        "Method": "Original",
        "MinCut": orig_cut,
        "Gain%": 0.0,
        "Links": 0,
        "JSAC_Utility": f"{orig_jsac_U:.2f}",
        "ResiLink_Utility": f"{orig_resilink_U:.2f}",
        "Time_s": 0.0,
    },
    {
        "Method": "ResiLink (50k timesteps)",
        "MinCut": final_cut_resilink,
        "Gain%": f"{gain_resilink:.1f}",
        "Links": len(added_resilink),
        "JSAC_Utility": f"{final_jsac_U_resilink:.2f}",
        "ResiLink_Utility": f"{final_resilink_U_resilink:.2f}",
        "Time_s": f"{total_t1:.1f}",
    },
    {
        "Method": "JSAC-style (5k timesteps)",
        "MinCut": final_cut_jsac,
        "Gain%": f"{gain_jsac:.1f}",
        "Links": len(added_jsac_style),
        "JSAC_Utility": f"{final_jsac_U_jsac:.2f}",
        "ResiLink_Utility": f"{final_resilink_U_jsac:.2f}",
        "Time_s": f"{total_t2:.1f}",
    },
]

csv_path = OUT_DIR / "compare_jsac.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\n→ Results saved to: {csv_path}")
