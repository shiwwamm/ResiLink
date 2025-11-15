#!/usr/bin/env python3
"""Quick test to verify shape compatibility"""
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import GraphPPOEnv
from train import GNNFeatureExtractor

print("Testing shape compatibility...")

# Create env
env = DummyVecEnv([lambda: GraphPPOEnv("real_world_topologies/Aarnet.graphml", max_steps=5)])

# Create model
policy_kwargs = dict(
    features_extractor_class=GNNFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[dict(pi=[128, 64], vf=[128, 64])]
)

model = PPO(
    "MultiInputPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    device="cpu",
    seed=42
)

print("\n✓ Model created successfully!")

# Test forward pass
obs = env.reset()
print(f"\nObservation shapes:")
print(f"  node_feat: {obs['node_feat'].shape}")
print(f"  edge_index: {obs['edge_index'].shape}")
print(f"  candidates: {obs['candidates'].shape}")

# Test prediction
action, _ = model.predict(obs, deterministic=True)
print(f"\n✓ Prediction successful! Action: {action}")

# Test one training step
print("\nTesting training step...")
model.learn(total_timesteps=10)
print("✓ Training step successful!")

print("\n✅ All shape tests passed!")
