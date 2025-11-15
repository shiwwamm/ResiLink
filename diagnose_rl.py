#!/usr/bin/env python3
"""Diagnose why RL is underperforming"""
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import GraphPPOEnv
from train import GNNFeatureExtractor

def test_topology(topo_name, timesteps=50000):
    path = f"real_world_topologies/{topo_name}.graphml"
    G_orig = nx.read_graphml(path)
    orig_cut = nx.stoer_wagner(G_orig)[0]
    
    print(f"\n{'='*60}")
    print(f"Testing: {topo_name}")
    print(f"{'='*60}")
    print(f"Original: {G_orig.number_of_nodes()} nodes, {G_orig.number_of_edges()} edges")
    print(f"Original min-cut: {orig_cut}")
    
    # Create environment
    env_fn = lambda: GraphPPOEnv(path, max_steps=20, plateau_steps=5, plateau_threshold=0.01)
    env = DummyVecEnv([env_fn])
    
    # Train model
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
    
    print(f"\nTraining for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)
    
    # Evaluate
    print(f"\nEvaluating...")
    obs = env.reset()
    G_final = nx.read_graphml(path)
    added = []
    
    for step in range(20):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        candidates = env.envs[0].candidates
        
        if action >= len(candidates):
            print(f"  Step {step}: No valid action (action={action}, candidates={len(candidates)})")
            break
        
        u, v = candidates[action]
        G_final.add_edge(u, v, capacity="10 Gbps")
        added.append((u, v))
        
        obs, reward, done, info = env.step([action])
        current_cut = nx.stoer_wagner(G_final)[0]
        
        print(f"  Step {step}: Added {u}-{v}, min_cut={current_cut}, reward={reward[0]:.4f}, done={done[0]}")
        
        if done[0]:
            print(f"  Episode ended")
            break
    
    final_cut = nx.stoer_wagner(G_final)[0]
    gain = (final_cut - orig_cut) / orig_cut * 100 if orig_cut > 0 else 0
    
    print(f"\nResults:")
    print(f"  Links added: {len(added)}")
    print(f"  Final min-cut: {final_cut}")
    print(f"  Throughput: {orig_cut*10:.1f} → {final_cut*10:.1f} Gbps")
    print(f"  Gain: {gain:.1f}%")

# Test on a few topologies
for topo in ["Aarnet", "Nsfnet"]:
    test_topology(topo, timesteps=50000)
