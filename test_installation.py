#!/usr/bin/env python3
"""
Enhanced ResiLink: Installation Test Script
==========================================

This script tests that Enhanced ResiLink is properly installed and working.
Run this after installation to verify everything is set up correctly.
"""

import sys
import traceback

def test_basic_imports():
    """Test basic package imports."""
    print("🔍 Testing basic imports...")
    
    try:
        import networkx as nx
        print("   ✅ NetworkX imported successfully")
    except ImportError as e:
        print(f"   ❌ NetworkX import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("   ✅ NumPy imported successfully")
    except ImportError as e:
        print(f"   ❌ NumPy import failed: {e}")
        return False
    
    try:
        from enhanced_resilink import TheoreticallyPerfectOptimizer
        print("   ✅ Enhanced ResiLink core imported successfully")
    except ImportError as e:
        print(f"   ❌ Enhanced ResiLink import failed: {e}")
        print("   💡 Make sure you ran: pip install -e .")
        return False
    
    return True

def test_classical_approach():
    """Test the classical optimization approach."""
    print("\n📚 Testing classical approach...")
    
    try:
        import networkx as nx
        from enhanced_resilink import TheoreticallyPerfectOptimizer
        
        # Create test network
        G = nx.karate_club_graph()
        print(f"   📊 Test network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Initialize optimizer
        optimizer = TheoreticallyPerfectOptimizer()
        print("   ✅ Optimizer initialized with theoretical validation")
        
        # Run optimization
        G_opt, links = optimizer.optimize_with_complete_justification(G)
        print(f"   ✅ Optimization completed: {len(links)} links suggested")
        
        # Verify results
        if len(links) > 0:
            print(f"   ✅ Results look good: {G.number_of_edges()} → {G_opt.number_of_edges()} edges")
            
            # Show first link justification
            link = links[0]
            print(f"   📋 Example link: {link['src_node']}-{link['dst_node']} (score: {link['score_breakdown']['total_score']:.4f})")
        else:
            print("   ⚠️  No links suggested (this might be normal for some networks)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Classical approach test failed: {e}")
        traceback.print_exc()
        return False

def test_hybrid_approach():
    """Test the hybrid ML approach (if available)."""
    print("\n🤖 Testing hybrid ML approach...")
    
    try:
        from enhanced_resilink import HYBRID_AVAILABLE
        
        if not HYBRID_AVAILABLE:
            print("   ⚠️  Hybrid approach not available (PyTorch not installed)")
            print("   💡 Install with: pip install torch torch-geometric")
            return True  # Not an error, just not installed
        
        from enhanced_resilink import HybridResiLinkOptimizer, HybridConfiguration
        print("   ✅ Hybrid components imported successfully")
        
        # Test configuration
        config = HybridConfiguration()
        print(f"   ✅ Configuration created: {config.classical_weight + config.gnn_weight + config.rl_weight:.3f} (should be 1.000)")
        
        # Test optimizer creation
        hybrid_optimizer = HybridResiLinkOptimizer(config)
        print("   ✅ Hybrid optimizer initialized")
        
        print("   🎯 Hybrid approach ready for use!")
        return True
        
    except Exception as e:
        print(f"   ❌ Hybrid approach test failed: {e}")
        traceback.print_exc()
        return False

def test_validation_framework():
    """Test the validation framework."""
    print("\n🧪 Testing validation framework...")
    
    try:
        from enhanced_resilink.validation import run_quick_validation
        
        results = run_quick_validation()
        print("   ✅ Quick validation completed")
        
        # Check results
        all_passed = all(results.values())
        if all_passed:
            print("   ✅ All validation tests passed!")
        else:
            print("   ⚠️  Some validation tests failed:")
            for test, result in results.items():
                status = "✅" if result else "❌"
                print(f"      {status} {test}")
        
        return all_passed
        
    except Exception as e:
        print(f"   ❌ Validation test failed: {e}")
        traceback.print_exc()
        return False

def test_metrics():
    """Test the academic metrics."""
    print("\n📊 Testing academic metrics...")
    
    try:
        import networkx as nx
        from enhanced_resilink.metrics import ExactAcademicMetrics
        
        # Create test network
        G = nx.path_graph(5)  # Simple connected graph
        metrics = ExactAcademicMetrics()
        
        # Test Fiedler connectivity
        fiedler = metrics.fiedler_algebraic_connectivity(G)
        print(f"   ✅ Fiedler connectivity: {fiedler:.4f}")
        
        # Test efficiency
        efficiency = metrics.latora_marchiori_efficiency(G)
        print(f"   ✅ Global efficiency: {efficiency:.4f}")
        
        # Test all metrics
        all_metrics = metrics.compute_all_metrics(G)
        print(f"   ✅ All metrics computed: {len(all_metrics)} metrics")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Metrics test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all installation tests."""
    print("🚀 Enhanced ResiLink Installation Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Classical Approach", test_classical_approach),
        ("Hybrid Approach", test_hybrid_approach),
        ("Validation Framework", test_validation_framework),
        ("Academic Metrics", test_metrics)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Enhanced ResiLink is ready to use!")
        print("\n🎯 Next steps:")
        print("   1. Run: python examples/basic_usage.py")
        print("   2. Review: docs/thesis_defense/defense_summary.md")
        print("   3. Practice: python examples/hybrid_comparison.py (if hybrid available)")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the error messages above.")
        print("\n💡 Common solutions:")
        print("   - Make sure you ran: pip install -e .")
        print("   - Check dependencies: pip install -r requirements.txt")
        print("   - Try fresh virtual environment if issues persist")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)