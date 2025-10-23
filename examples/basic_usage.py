#!/usr/bin/env python3
"""
Enhanced ResiLink: Basic Usage Example
=====================================

This example demonstrates the basic usage of Enhanced ResiLink
for network resilience optimization.
"""

from enhanced_resilink import TheoreticallyPerfectOptimizer
import networkx as nx

def main():
    """Basic usage demonstration."""
    print("Enhanced ResiLink: Basic Usage Example")
    print("=" * 40)
    
    # Create a test network
    G = nx.karate_club_graph()
    print(f"Test network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Initialize optimizer
    optimizer = TheoreticallyPerfectOptimizer()
    print("Optimizer initialized with theoretical validation")
    
    # Run optimization
    print("\nRunning optimization...")
    G_optimized, suggested_links = optimizer.optimize_with_complete_justification(G)
    
    # Display results
    print(f"\nResults:")
    print(f"  Links added: {len(suggested_links)}")
    print(f"  Original edges: {G.number_of_edges()}")
    print(f"  Optimized edges: {G_optimized.number_of_edges()}")
    
    # Show link details
    print("\nSuggested links:")
    for link in suggested_links:
        print(f"  {link['src_node']}-{link['dst_node']}: Score {link['score_breakdown']['total_score']:.4f}")

if __name__ == "__main__":
    main()