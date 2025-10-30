#!/usr/bin/env python3
"""
Test Geographic Integration with Hybrid ResiLink
===============================================

Tests the geographic constraint features integrated into the hybrid implementation.
"""

import subprocess
import time
import os
import sys
from pathlib import Path

def test_geographic_analyzer():
    """Test the standalone geographic analyzer."""
    print("🌍 Testing Geographic Network Analyzer")
    print("-" * 40)
    
    try:
        result = subprocess.run(['python3', 'geographic_network_analyzer.py'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Geographic analyzer working")
            print("📊 Sample output:")
            print(result.stdout[-500:])  # Last 500 characters
        else:
            print("❌ Geographic analyzer failed")
            print(f"Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Failed to test geographic analyzer: {e}")
    
    print()

def test_hybrid_with_geography():
    """Test hybrid implementation with geographic constraints."""
    print("🤖 Testing Hybrid Implementation with Geographic Constraints")
    print("-" * 60)
    
    # Check if GraphML file exists
    graphml_file = "real_world_topologies/Bellcanada.graphml"
    if not Path(graphml_file).exists():
        print(f"❌ GraphML file not found: {graphml_file}")
        return
    
    print(f"📁 Using GraphML file: {graphml_file}")
    
    # Test with geographic constraints
    cmd = [
        'python3', 'hybrid_resilink_implementation.py',
        '--single-cycle',
        '--training-mode',
        '--graphml-file', graphml_file,
        '--simulation-mode'  # Use simulation mode for testing
    ]
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Hybrid implementation with geography working")
            
            # Look for geographic information in output
            output = result.stdout
            if "Geographic Analysis:" in output:
                print("✅ Geographic analysis found in output")
                # Extract geographic section
                lines = output.split('\n')
                in_geo_section = False
                for line in lines:
                    if "Geographic Analysis:" in line:
                        in_geo_section = True
                    if in_geo_section:
                        print(f"   {line}")
                        if line.strip() == "" and in_geo_section:
                            break
            else:
                print("⚠️  No geographic analysis found in output")
            
            if "Distance:" in output:
                print("✅ Distance calculations working")
            if "Link Type:" in output:
                print("✅ Link type classification working")
            if "Cost Estimate:" in output:
                print("✅ Cost estimation working")
                
        else:
            print("❌ Hybrid implementation failed")
            print(f"Error: {result.stderr}")
            print(f"Output: {result.stdout[-1000:]}")  # Last 1000 chars
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out (60 seconds)")
    except Exception as e:
        print(f"❌ Failed to test hybrid implementation: {e}")
    
    print()

def test_distance_constraints():
    """Test specific distance constraint scenarios."""
    print("📏 Testing Distance Constraint Scenarios")
    print("-" * 40)
    
    from geographic_network_analyzer import GeographicNetworkAnalyzer, GeographicConstraints
    
    # Create analyzer with strict constraints
    strict_constraints = GeographicConstraints(
        max_distance_km=500.0,  # Very strict
        international_penalty=0.1,  # Heavy penalty
        submarine_cable_threshold=200.0
    )
    
    analyzer = GeographicNetworkAnalyzer(strict_constraints)
    
    # Load Bell Canada data
    graphml_file = Path("real_world_topologies/Bellcanada.graphml")
    if graphml_file.exists():
        if analyzer.load_graphml_geography(graphml_file):
            print(f"✅ Loaded {len(analyzer.nodes)} nodes")
            
            # Test some specific scenarios
            node_ids = list(analyzer.nodes.keys())
            
            if len(node_ids) >= 4:
                # Test short distance
                analysis1 = analyzer.analyze_link_feasibility(node_ids[0], node_ids[1])
                print(f"📊 Short link test:")
                print(f"   Distance: {analysis1['distance_km']:.0f} km")
                print(f"   Feasible: {'✅' if analysis1['feasible'] else '❌'}")
                print(f"   Type: {analysis1['link_type']}")
                
                # Test long distance (if available)
                if len(node_ids) >= 10:
                    analysis2 = analyzer.analyze_link_feasibility(node_ids[0], node_ids[-1])
                    print(f"📊 Long link test:")
                    print(f"   Distance: {analysis2['distance_km']:.0f} km")
                    print(f"   Feasible: {'✅' if analysis2['feasible'] else '❌'}")
                    print(f"   Type: {analysis2['link_type']}")
                    if not analysis2['feasible']:
                        print(f"   Reason: {analysis2['reason']}")
        else:
            print("❌ Failed to load geographic data")
    else:
        print(f"❌ GraphML file not found: {graphml_file}")
    
    print()

def main():
    """Main test function."""
    print("🧪 Geographic Integration Testing Suite")
    print("=" * 50)
    
    # Test 1: Geographic analyzer
    test_geographic_analyzer()
    
    # Test 2: Distance constraints
    test_distance_constraints()
    
    # Test 3: Hybrid implementation with geography
    test_hybrid_with_geography()
    
    print("🎯 Testing Summary:")
    print("=" * 20)
    print("✅ If all tests pass, geographic constraints are working")
    print("📍 Geographic data is loaded from GraphML files")
    print("📏 Distance calculations use great-circle formula")
    print("🌍 International links are penalized appropriately")
    print("💰 Cost estimates include geographic factors")
    print()
    print("🚀 Next steps:")
    print("1. Run with real controller and Mininet topology")
    print("2. Use --graphml-file parameter for geographic constraints")
    print("3. Observe distance-based feasibility decisions")

if __name__ == "__main__":
    main()