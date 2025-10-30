#!/usr/bin/env python3
"""
Quick Test for Enhanced ResiLink
===============================

Simple test to verify basic functionality without complex system validation.
"""

import os
import sys
import time
import requests
from pathlib import Path

def test_controller_api():
    """Test if the controller API is accessible."""
    print("🧪 Testing Controller API...")
    
    endpoints = {
        'switches': 'http://localhost:8080/v1.0/topology/switches',
        'links': 'http://localhost:8080/v1.0/topology/links',
        'hosts': 'http://localhost:8080/v1.0/topology/hosts'
    }
    
    for name, url in endpoints.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ {name}: {len(data)} items")
            else:
                print(f"  ❌ {name}: HTTP {response.status_code}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")

def test_topology_parsing():
    """Test topology parsing."""
    print("\n🧪 Testing Topology Parsing...")
    
    try:
        from core.enhanced_topology_parser import EnhancedTopologyParser
        
        parser = EnhancedTopologyParser()
        network = parser.parse_graphml("real_world_topologies/Arpanet196912.graphml")
        
        print(f"  ✅ Parsed: {network.metadata.name}")
        print(f"     Nodes: {len(network.nodes)}")
        print(f"     Edges: {len(network.edges)}")
        
        # Check if it's actually connected
        import networkx as nx
        G = nx.Graph()
        for edge_key in network.edges.keys():
            src, dst = edge_key
            G.add_edge(src, dst)
        
        print(f"     Connected: {nx.is_connected(G)}")
        
    except Exception as e:
        print(f"  ❌ Topology parsing failed: {e}")

def test_hybrid_implementation():
    """Test hybrid implementation initialization."""
    print("\n🧪 Testing Hybrid Implementation...")
    
    try:
        from hybrid_resilink_implementation import HybridResiLinkImplementation
        
        # Test with simulation mode
        impl = HybridResiLinkImplementation(simulation_mode=True)
        print("  ✅ Hybrid implementation initialized")
        
        if impl.use_enhanced_metrics:
            print("  ✅ Enhanced metrics available")
        else:
            print("  ⚠️  Enhanced metrics not available (using fallback)")
            
    except Exception as e:
        print(f"  ❌ Hybrid implementation failed: {e}")

def main():
    """Run quick tests."""
    print("🚀 Enhanced ResiLink Quick Test")
    print("=" * 40)
    
    print("\n💡 Make sure the controller is running:")
    print("   ryu-manager sdn/basic_controller.py ryu.app.rest_topology ryu.app.ofctl_rest --observe-links --wsapi-host 0.0.0.0 --wsapi-port 8080")
    print()
    
    test_controller_api()
    test_topology_parsing()
    test_hybrid_implementation()
    
    print("\n" + "=" * 40)
    print("🎯 Quick Test Complete")
    print("\nIf controller API tests pass, you can run:")
    print("  python3 hybrid_resilink_implementation.py --simulation-mode --max-cycles 3")

if __name__ == "__main__":
    main()