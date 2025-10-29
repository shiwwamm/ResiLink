"""
Enhanced ResiLink: Hybrid Network Resilience Optimization
=========================================================

Hybrid GNN+RL network resilience optimization with complete academic justification.

This package provides hybrid optimization combining:
- Graph Neural Networks (Veličković et al. 2018) for pattern learning
- Reinforcement Learning (Mnih et al. 2015) for adaptive optimization  
- Ensemble methods (Breiman 2001) for principled combination
- Real-time SDN integration via Ryu controller

Usage:
    # Use the main implementation script
    python3 hybrid_resilink_implementation.py --training-mode
    
    # Or import for custom usage
    from enhanced_resilink import HybridResiLinkImplementation
    
    implementation = HybridResiLinkImplementation()
    result = implementation.run_optimization_cycle()

Authors: Research Team
License: MIT
Academic Foundation: Complete citations for all methods and parameters
"""

# Check for PyTorch availability
try:
    import torch
    import torch_geometric
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

__version__ = "2.0.0"
__all__ = []

# Note: Main implementation is in hybrid_resilink_implementation.py
# This package structure is maintained for compatibility