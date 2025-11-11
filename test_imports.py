#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import sys

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        print("✓ Testing geographic_network_analyzer...")
        from geographic_network_analyzer import GeographicNetworkAnalyzer, GeographicConstraints
        print("  ✓ GeographicNetworkAnalyzer imported")
        print("  ✓ GeographicConstraints imported")
        
        print("\n✓ Testing core.enhanced_topology_parser...")
        from core.enhanced_topology_parser import EnhancedTopologyParser, NodeMetadata, EdgeMetadata
        print("  ✓ EnhancedTopologyParser imported")
        print("  ✓ NodeMetadata imported")
        print("  ✓ EdgeMetadata imported")
        
        print("\n✓ Testing standard libraries...")
        import networkx as nx
        import numpy as np
        import torch
        print("  ✓ networkx imported")
        print("  ✓ numpy imported")
        print("  ✓ torch imported")
        
        print("\n✓ Testing torch_geometric...")
        from torch_geometric.nn import GATConv
        from torch_geometric.data import Data
        print("  ✓ GATConv imported")
        print("  ✓ Data imported")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        print(f"\nMissing dependency. Install with:")
        print(f"  pip install {str(e).split()[-1]}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of key classes."""
    print("\n" + "="*60)
    print("Testing basic functionality...")
    print("="*60)
    
    try:
        # Test GeographicConstraints
        print("\n1. Testing GeographicConstraints...")
        from geographic_network_analyzer import GeographicConstraints
        constraints = GeographicConstraints(
            max_distance_km=2000.0,
            cost_per_km=1000.0
        )
        penalty = constraints.get_distance_penalty(500.0)
        print(f"   Distance penalty for 500km: {penalty}")
        assert 0.0 <= penalty <= 1.0, "Penalty should be between 0 and 1"
        print("   ✓ GeographicConstraints working")
        
        # Test GeographicNetworkAnalyzer
        print("\n2. Testing GeographicNetworkAnalyzer...")
        from geographic_network_analyzer import GeographicNetworkAnalyzer
        analyzer = GeographicNetworkAnalyzer(constraints)
        print("   ✓ GeographicNetworkAnalyzer instantiated")
        
        # Test EnhancedTopologyParser
        print("\n3. Testing EnhancedTopologyParser...")
        from core.enhanced_topology_parser import EnhancedTopologyParser
        parser = EnhancedTopologyParser()
        print("   ✓ EnhancedTopologyParser instantiated")
        
        print("\n✅ All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("ResiLink Module Import Test")
    print("="*60)
    
    imports_ok = test_imports()
    
    if imports_ok:
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n" + "="*60)
            print("✅ ALL TESTS PASSED")
            print("="*60)
            print("\nYou can now run:")
            print("  python3 hybrid_resilink_implementation.py --help")
            return 0
        else:
            print("\n" + "="*60)
            print("⚠️  IMPORTS OK BUT FUNCTIONALITY ISSUES")
            print("="*60)
            return 1
    else:
        print("\n" + "="*60)
        print("❌ IMPORT TESTS FAILED")
        print("="*60)
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
