#!/usr/bin/env python3
"""
Simple Direct Test
=================

Direct test of the hybrid implementation.
"""

print("🚀 Testing Enhanced ResiLink...")

try:
    from hybrid_resilink_implementation import HybridResiLinkImplementation
    
    print("✅ Import successful")
    
    # Initialize with simulation mode
    impl = HybridResiLinkImplementation(simulation_mode=True, reward_threshold=0.85)
    print("✅ Initialization successful")
    
    print("🎯 Available methods:")
    methods = [method for method in dir(impl) if not method.startswith('_') and callable(getattr(impl, method))]
    for method in methods[:10]:  # Show first 10 methods
        print(f"   • {method}")
    
    print("\n🚀 Testing single optimization cycle...")
    
    # Test single cycle
    result = impl.run_optimization_cycle(training_mode=True)
    print(f"✅ Single cycle completed: {result}")
    
    print("\n🎉 Basic functionality test PASSED!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()