#!/usr/bin/env python3
"""Dry run test for compare_jsac.py - checks all dependencies"""
import sys
import os
from pathlib import Path

print("=" * 80)
print("DRY RUN: Testing compare_jsac.py dependencies")
print("=" * 80)

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    import networkx as nx
    print("  ✓ networkx")
except ImportError as e:
    print(f"  ✗ networkx: {e}")
    sys.exit(1)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    print("  ✓ stable_baselines3")
except ImportError as e:
    print(f"  ✗ stable_baselines3: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("  ✓ numpy")
except ImportError as e:
    print(f"  ✗ numpy: {e}")
    sys.exit(1)

try:
    from env import GraphPPOEnv
    print("  ✓ env.GraphPPOEnv")
except ImportError as e:
    print(f"  ✗ env.GraphPPOEnv: {e}")
    sys.exit(1)

try:
    from train import GNNFeatureExtractor
    print("  ✓ train.GNNFeatureExtractor")
except ImportError as e:
    print(f"  ✗ train.GNNFeatureExtractor: {e}")
    sys.exit(1)

# Test 2: Check directories
print("\n2. Testing directories...")
if Path("real_world_topologies").exists():
    print("  ✓ real_world_topologies/ exists")
    topo_count = len(list(Path("real_world_topologies").glob("*.graphml")))
    print(f"    Found {topo_count} topologies")
else:
    print("  ✗ real_world_topologies/ not found")
    print("    Run: python3 fetch_zoo.py")
    sys.exit(1)

if Path("results").exists():
    print("  ✓ results/ exists")
else:
    print("  ⚠ results/ not found (will be created)")
    Path("results").mkdir(exist_ok=True)
    print("  ✓ results/ created")

# Test 3: Check specific topology
print("\n3. Testing AttMpls topology...")
topo_path = Path("real_world_topologies/AttMpls.graphml")
if topo_path.exists():
    print(f"  ✓ {topo_path} exists")
    try:
        G = nx.read_graphml(topo_path)
        print(f"    Nodes: {G.number_of_nodes()}")
        print(f"    Edges: {G.number_of_edges()}")
        print(f"    Connected: {nx.is_connected(G)}")
        if nx.is_connected(G):
            min_cut = nx.stoer_wagner(G)[0]
            print(f"    Min-cut: {min_cut}")
    except Exception as e:
        print(f"  ✗ Error reading topology: {e}")
        sys.exit(1)
else:
    print(f"  ✗ {topo_path} not found")
    print("\n  Available topologies:")
    for f in sorted(Path("real_world_topologies").glob("*.graphml"))[:10]:
        print(f"    - {f.stem}")
    print("\n  Suggestion: Change TOPO variable in compare_jsac.py to one of the above")
    sys.exit(1)

# Test 4: Test environment creation
print("\n4. Testing environment creation...")
try:
    env = GraphPPOEnv(str(topo_path), max_steps=5, plateau_steps=5, plateau_threshold=0.01)
    print("  ✓ GraphPPOEnv created")
    obs, info = env.reset()
    print(f"    Observation keys: {list(obs.keys())}")
    print(f"    Candidates: {len(env.candidates)}")
except Exception as e:
    print(f"  ✗ Error creating environment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test utility functions
print("\n5. Testing utility functions...")
try:
    G = nx.read_graphml(topo_path)
    
    # JSAC utility
    if nx.is_connected(G):
        min_cut = nx.stoer_wagner(G)[0]
        λ2 = nx.algebraic_connectivity(G)
        var_deg = np.var([G.degree(n) for n in G.nodes()])
        bonus = 1.0 if λ2 >= 0.5 else 0.0
        jsac_U = 0.5 * min_cut + 0.4 * λ2 - 0.1 * var_deg + bonus
        print(f"  ✓ JSAC utility: {jsac_U:.2f}")
    
    # ResiLink utility
    if nx.is_connected(G):
        min_cut = nx.stoer_wagner(G)[0]
        λ2 = nx.algebraic_connectivity(G)
        var_deg = np.var([G.degree(n) for n in G.nodes()])
        min_cut_bonus = 5.0 if min_cut >= 2 else 0.0
        resilink_U = 2.0 * min_cut + 0.5 * λ2 - 0.1 * var_deg + min_cut_bonus
        print(f"  ✓ ResiLink utility: {resilink_U:.2f}")
except Exception as e:
    print(f"  ✗ Error computing utilities: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test PPO model creation (quick)
print("\n6. Testing PPO model creation...")
try:
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    env_fn = lambda: GraphPPOEnv(str(topo_path), max_steps=5)
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
        verbose=0,
        device="cpu",
        seed=42,
    )
    print("  ✓ PPO model created")
    
    # Quick test (1 step)
    print("  Testing 1 training step...")
    model.learn(total_timesteps=10)
    print("  ✓ Training step successful")
    
except Exception as e:
    print(f"  ✗ Error creating PPO model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Check CSV writing
print("\n7. Testing CSV output...")
try:
    import csv
    test_results = [
        {"Method": "Test", "MinCut": 1, "Gain%": "0.0", "Links": 0}
    ]
    csv_path = Path("results/test_output.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=test_results[0].keys())
        writer.writeheader()
        writer.writerows(test_results)
    print(f"  ✓ CSV written to {csv_path}")
    csv_path.unlink()  # Clean up
    print("  ✓ Test file cleaned up")
except Exception as e:
    print(f"  ✗ Error writing CSV: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\ncompare_jsac.py should run successfully.")
print("\nTo run the actual comparison:")
print("  python3 compare_jsac.py")
print("\nNote: This will take ~30 minutes (trains 2 models)")
print("=" * 80)
