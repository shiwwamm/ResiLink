# run_all.py — FINAL 100% WORKING
import os
import time
import csv
from pathlib import Path
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import GraphPPOEnv
from train import GNNFeatureExtractor

TOPOLOGIES = ["Aarnet"]

OUT_DIR = Path("optimized_graphs")
OUT_DIR.mkdir(exist_ok=True)

results = []

print("RESILINK v3 – BATCH RUN (8 Topologies)")
print("=" * 80)

for topo in TOPOLOGIES:
    path = f"real_world_topologies/{topo}.graphml"
    if not os.path.exists(path):
        print(f"Skipping {topo} – file not found")
        continue

    print(f"\n→ {topo}...", end="")
    start = time.time()

    try:
        env_fn = lambda: GraphPPOEnv(path, max_steps=20, plateau_steps=5, plateau_threshold=0.01)
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
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            device="cpu",
            seed=42
        )
        model.learn(total_timesteps=20000)

        # INFERENCE: FIXED
        obs = env.reset()
        G_final = nx.read_graphml(path)
        added = []

        for _ in range(20):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)  # ← FIXED: NumPy array → int
            candidates = env.envs[0].candidates
            if action >= len(candidates):
                break
            u, v = candidates[action]
            G_final.add_edge(u, v, capacity="10 Gbps")
            added.append((u, v))
            obs, _, done, _ = env.step([action])
            if done[0]:
                break

        orig_G = nx.read_graphml(path)
        orig_cut = nx.stoer_wagner(orig_G)[0]
        final_cut = nx.stoer_wagner(G_final)[0]
        gain = (final_cut - orig_cut) / orig_cut * 100 if orig_cut > 0 else 0

        opt_path = OUT_DIR / f"opt_{topo}.graphml"
        links_path = OUT_DIR / f"links_{topo}.txt"
        nx.write_graphml(G_final, opt_path)
        with open(links_path, "w") as f:
            for u, v in added:
                f.write(f"{u} — {v}\n")

        results.append({
            "Topology": topo,
            "Nodes": G_final.number_of_nodes(),
            "Links_Added": len(added),
            "Orig_TPut": f"{orig_cut * 10:.1f}",
            "Final_TPut": f"{final_cut * 10:.1f}",
            "Gain_%": f"{gain:.1f}",
            "Time_s": f"{time.time() - start:.1f}"
        })

        print(f" {len(added)} links → {final_cut * 10:.1f} Gbps (+{gain:.1f}%) in {time.time() - start:.1f}s")

    except Exception as e:
        print(f" FAILED: {e}")

# Save & Print
csv_path = OUT_DIR / "results.csv"
if results:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

print("\n" + "=" * 100)
print("RESILINK v3 – FINAL RESULTS")
print("| Topology     | Nodes | Links | Orig | Final | Gain  | Time |")
print("|--------------|-------|-------|------|-------|-------|------|")
for r in results:
    print(f"| {r['Topology']:12} | {r['Nodes']:5} | {r['Links_Added']:5} | {r['Orig_TPut']:4} | {r['Final_TPut']:5} | {r['Gain_%']:5} | {r['Time_s']:4} |")
print("=" * 100)
print(f"→ {csv_path}")
print(f"→ Optimized graphs: {OUT_DIR}/")
