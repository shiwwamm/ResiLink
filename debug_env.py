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
    
    # Take random actions
    print(f"\nTaking 5 random actions:")
    for step in range(5):
        if len(env.candidates) == 0:
            print(f"  Step {step}: No candidates available!")
            break
            
        action = 0  # Take first candidate
        obs, reward, done, truncated, info = env.step(action)
        
        current_cut = info.get('min_cut', 0)
        print(f"  Step {step}: Added link, reward={reward:.4f}, min_cut={current_cut}, candidates={len(env.candidates)}")
        
        if done:
            print(f"  Episode ended (plateau={info.get('plateau', False)})")
            break
    
    print(f"\nFinal utility: {env.best_U:.4f}")
    print(f"Improvement: {env.best_U - env._compute_U():.4f}")
