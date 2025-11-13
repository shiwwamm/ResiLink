#!/bin/bash
# Install missing dependencies for ResiLink benchmark

echo "Installing missing dependencies..."
pip3 install shimmy>=2.0.0
pip3 install gymnasium>=0.28.0
pip3 install stable-baselines3>=2.0.0

echo "✓ Dependencies installed!"
echo "You can now run: python3 benchmark_compare.py"
