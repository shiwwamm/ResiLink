# benchmark.py — RESILINK v4 (FULL UPGRADE)
import os, time, csv
from pathlib import Path
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import GraphPPOEnv
from train import GNNFeatureExtractor
from baselines import greedy_topology, random_topology, degree_topology, betweenness_topology

TOPOLOGIES = ["Aarnet", "Abilene", "Nsfnet", "Geant2012"]
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

results = []

def run_rl(G_path):
    env_fn = lambda: GraphPPOEnv(G_path, max_steps=40, plateau_steps=5, plateau_threshold=0.1)
    env = DummyVecEnv([env_fn])
    policy_kwargs = dict(
        features_extractor_class=GNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64])
    )
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=0, device="cpu", seed=42)
    model.learn(total_timesteps=100000)

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

print("="*80)
print("RESILINK v4 BENCHMARK - FULL RUN")
print("="*80)

for topo in TOPOLOGIES:
    path = f"real_world_topologies/{topo}.graphml"
    if not os.path.exists(path): 
        print(f"Skipping {topo} – file not found")
        continue

    print(f"\nTopology: {topo}")
    print("="*80)
    G_orig = nx.read_graphml(path)
    orig_cut = nx.stoer_wagner(G_orig)[0]
    print(f"Original: {G_orig.number_of_nodes()} nodes, {G_orig.number_of_edges()} edges, min-cut={orig_cut}")

    # Baselines
    for name, func in [
        ("Random", random_topology),
        ("Degree", degree_topology),
        ("Betweenness", betweenness_topology),
        ("Greedy", greedy_topology)
    ]:
        start = time.time()
        G_final, added = func(G_orig.copy(), max_links=10)
        final_cut = nx.stoer_wagner(G_final)[0]
        gain = (final_cut - orig_cut) / orig_cut * 100 if orig_cut > 0 else 0
        print(f"  Running {name}... {len(added)} links → {final_cut*10:.1f} Gbps (+{gain:.1f}%) in {time.time()-start:.1f}s")
        results.append({
            "Topology": topo,
            "Method": name,
            "Final_TPut": f"{final_cut*10:.1f}",
            "Gain_%": f"{gain:.1f}",
            "Links": len(added),
            "Time_s": f"{time.time()-start:.1f}"
        })

    # RL-GNN (ResiLink v4)
    start = time.time()
    G_final, added = run_rl(path)
    final_cut = nx.stoer_wagner(G_final)[0]
    gain = (final_cut - orig_cut) / orig_cut * 100 if orig_cut > 0 else 0
    print(f"  Running RL-GNN... {len(added)} links → {final_cut*10:.1f} Gbps (+{gain:.1f}%) in {time.time()-start:.1f}s")
    results.append({
        "Topology": topo,
        "Method": "RL-GNN",
        "Final_TPut": f"{final_cut*10:.1f}",
        "Gain_%": f"{gain:.1f}",
        "Links": len(added),
        "Time_s": f"{time.time()-start:.1f}"
    })

# Save CSV
csv_path = OUT_DIR / "benchmark_v4.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("="*100)
print("BENCHMARK RESULTS (v4)")
print("="*100)
print("| Topology | Method       | Orig | Final | Gain% | Links | Time |")
print("|----------|--------------|------|-------|-------|-------|------|")
for r in results:
    orig = float(r['Final_TPut']) / (1 + float(r['Gain_%'])/100) if float(r['Gain_%']) != 0 else float(r['Final_TPut'])
    print(f"| {r['Topology']:8} | {r['Method']:12} | {orig:4.1f} | {r['Final_TPut']:5} | {r['Gain_%']:5} | {r['Links']:5} | {r['Time_s']:4} |")
print("="*100)
print(f"→ Results saved to: {csv_path}")