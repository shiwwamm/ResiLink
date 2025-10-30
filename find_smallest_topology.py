#!/usr/bin/env python3
"""
Find Smallest Topology
======================

Analyzes all available topologies to find the smallest one for testing.
"""

import os
import sys
from pathlib import Path
import networkx as nx

def analyze_topology_sizes():
    """Analyze all topology files and find the smallest ones."""
    topology_dir = Path("real_world_topologies")
    
    if not topology_dir.exists():
        print("❌ real_world_topologies directory not found")
        return
    
    # Get all GraphML files (prefer GraphML over GML for rich metadata)
    graphml_files = list(topology_dir.glob("*.graphml"))
    
    print(f"🔍 Analyzing {len(graphml_files)} topology files...")
    print()
    
    topology_sizes = []
    
    for file_path in graphml_files:
        try:
            # Load the graph
            G = nx.read_graphml(str(file_path))
            
            nodes = G.number_of_nodes()
            edges = G.number_of_edges()
            
            topology_sizes.append({
                'name': file_path.stem,
                'file': file_path.name,
                'nodes': nodes,
                'edges': edges,
                'density': nx.density(G),
                'connected': nx.is_connected(G)
            })
            
        except Exception as e:
            print(f"⚠️  Failed to load {file_path.name}: {e}")
    
    # Sort by number of nodes (smallest first)
    topology_sizes.sort(key=lambda x: x['nodes'])
    
    print("📊 TOPOLOGY SIZES (sorted by nodes):")
    print("=" * 80)
    print(f"{'Name':<25} {'Nodes':<6} {'Edges':<6} {'Density':<8} {'Connected':<10} {'File'}")
    print("-" * 80)
    
    # Show smallest 20 topologies
    for i, topo in enumerate(topology_sizes[:20]):
        connected_str = "✅ Yes" if topo['connected'] else "❌ No"
        print(f"{topo['name']:<25} {topo['nodes']:<6} {topo['edges']:<6} {topo['density']:<8.3f} {connected_str:<10} {topo['file']}")
    
    if len(topology_sizes) > 20:
        print(f"... and {len(topology_sizes) - 20} more topologies")
    
    print()
    print("🎯 SMALLEST TOPOLOGIES FOR TESTING:")
    print("=" * 50)
    
    # Find smallest connected topologies
    smallest_connected = [t for t in topology_sizes if t['connected']][:5]
    
    for i, topo in enumerate(smallest_connected, 1):
        print(f"{i}. {topo['name']} ({topo['nodes']} nodes, {topo['edges']} edges)")
        print(f"   File: {topo['file']}")
        print(f"   Density: {topo['density']:.3f}")
        print()
    
    # Recommend the smallest connected topology
    if smallest_connected:
        recommended = smallest_connected[0]
        print("🏆 RECOMMENDED FOR TESTING:")
        print(f"   Network: {recommended['name']}")
        print(f"   File: {recommended['file']}")
        print(f"   Size: {recommended['nodes']} nodes, {recommended['edges']} edges")
        print(f"   Connected: {'Yes' if recommended['connected'] else 'No'}")
        print(f"   Density: {recommended['density']:.3f}")
        print()
        print("🚀 Usage:")
        print(f"   sudo python3 sdn/mininet_builder.py real_world_topologies/{recommended['file']}")
        print(f"   sudo python3 test_complete_system.py --topology real_world_topologies/{recommended['file']}")
        
        return recommended['file']
    
    return None

if __name__ == "__main__":
    analyze_topology_sizes()