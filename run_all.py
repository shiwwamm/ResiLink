# run_all.py
import os, time, csv
from pathlib import Path
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import GraphPPOEnv
from train import GNNPolicy

# topologies = ["Abilene", "Aarnet", "Geant", "Nsfnet", "TataNld", "Chinanet", "BtNorthAmerica", "BellCanada"]
topologies = ["Aarnet"]
results = []
out_dir = Path("optimized_graphs")
out_dir.mkdir(exist_ok=True)

print("RESILINK v3 – BATCH RUN (Adaptive Stopping)")
print("="*80)

for topo in topologies:
    path = f"real_world_topologies/{topo}.graphml"
    if not os.path.exists(path): continue

    print(f"\n→ {topo}...")
    start = time.time()
    env_fn = lambda: GraphPPOEnv(path, max_steps=20, plateau_steps=5, plateau_threshold=0.01)
    env = DummyVecEnv([env_fn])
    policy_kwargs = dict(features_extractor_class=GNNPolicy, features_extractor_kwargs=dict(features_dim=32))
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=0, device="cpu", seed=42)
    model.learn(total_timesteps=5000)

    # Extract links
    env_inst = env.envs[0]
    obs = env_inst.reset()[0]
    G_final = nx.read_graphml(path)
    added = []
    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        if action >= len(env_inst.candidates): break
        u_idx, v_idx = env_inst.candidates[action]
        u, v = list(G_final.nodes())[u_idx], list(G_final.nodes())[v_idx]
        G_final.add_edge(u, v, capacity="10 Gbps")
        added.append((u, v))
        obs, _, done, _, info = env.step([action])
        if done[0]: break

    orig_cut = nx.stoer_wagner(nx.read_graphml(path))[0]
    final_cut = nx.stoer_wagner(G_final)[0]
    gain = (final_cut - orig_cut) / orig_cut * 100 if orig_cut > 0 else 0

    nx.write_graphml(G_final, out_dir / f"opt_{topo}.graphml")
    with open(out_dir / f"links_{topo}.txt", "w") as f:
        for u, v in added: f.write(f"{u} — {v}\n")

    results.append({
        "Topology": topo,
        "Nodes": G_final.number_of_nodes(),
        "Links_Added": len(added),
        "Orig_TPut": f"{orig_cut*10:.1f}",
        "Final_TPut": f"{final_cut*10:.1f}",
        "Gain_%": f"{gain:.1f}",
        "Time_s": f"{time.time()-start:.1f}"
    })
    print(f"  {len(added)} links → {final_cut*10:.1f} Gbps (+{gain:.1f}%) in {time.time()-start:.1f}s")

# Save CSV + Table
csv_path = out_dir / "results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n" + "="*100)
print("RESILINK v3 – FINAL RESULTS")
print("| Topology | Nodes | Links | Orig | Final | Gain | Time |")
print("|----------|-------|-------|------|-------|------|------|")
for r in results:
    print(f"| {r['Topology']:8} | {r['Nodes']:5} | {r['Links_Added']:5} | {r['Orig_TPut']:4} | {r['Final_TPut']:5} | {r['Gain_%']:4} | {r['Time_s']:4} |")
print("="*100)
print(f"→ {csv_path}")