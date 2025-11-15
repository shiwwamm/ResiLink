#!/usr/bin/env python3
"""Debug script to understand environment behavior"""
import networkx as nx
from env import GraphPPOEnv

# Test on a few topologies
topologies = ["Abilene", "Aarnet", "Geant2012"]

for topo in topologies:
    path = f"real_world_topologies/{topo}.graphml"
    print(f"\n{'='*60}")
    print(f"Testing: {topo}")
    print('='*60)
    
    # Load original graph
    G_orig = nx.read_graphml(path)
    print(f"Original graph: {G_orig.number_of_nodes()} nodes, {G_orig.number_of_edges()} edges")
    
    if nx.is_connected(G_orig):
        orig_cut = nx.stoer_wagner(G_orig)[0]
        orig_lambda2 = nx.algebraic_connectivity(G_orig)
        print(f"Original min-cut: {orig_cut}")
        print(f"Original λ₂: {orig_lambda2:.4f}")
    else:
        print("WARNING: Graph is not connected!")
        continue
    
    # Create environment
    env = GraphPPOEnv(path, max_steps=20)
    obs, info = env.reset()
    
    print(f"\nInitial state:")
    print(f"  Candidates: {len(env.candidates)}")
    print(f"  Initial utility: {env.best_U:.4f}")
    
    # Take actions using prioritized candidates
    print(f"\nTaking 10 prioritized actions:")
    total_reward = 0
    for step in range(10):
        if len(env.candidates) == 0:
            print(f"  Step {step}: No candidates available!")
            break
            
        action = 0  # Take first (highest priority) candidate
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        current_cut = info.get('min_cut', 0)
        print(f"  Step {step}: reward={reward:.4f}, min_cut={current_cut}, utility={env.best_U:.4f}, candidates={len(env.candidates)}")
        
        if done:
            print(f"  Episode ended (plateau={info.get('plateau', False)})")
            break
    
    print(f"\nFinal utility: {env.best_U:.4f}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final min-cut: {info.get('min_cut', 0)}")
