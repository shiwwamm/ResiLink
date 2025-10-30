#!/usr/bin/env python3
"""
Test Simulation Mode
===================

Test the hybrid implementation in simulation mode (no network required).
"""

def test_simulation_mode():
    """Test simulation mode."""
    print("🧪 Testing Simulation Mode...")
    
    try:
        from hybrid_resilink_implementation import HybridResiLinkImplementation
        
        # Test with simulation mode
        impl = HybridResiLinkImplementation(simulation_mode=True)
        print("✅ Hybrid implementation initialized")
        
        if impl.use_enhanced_metrics:
            print("✅ Enhanced metrics available")
        else:
            print("⚠️  Enhanced metrics not available (using fallback)")
        
        print("\n🚀 Running 2 optimization cycles...")
        
        # Run a few cycles
        successful_cycles = impl.run_continuous_optimization(
            max_cycles=2,
            cycle_interval=2,
            training_mode=True
        )
        
        print(f"✅ Completed {successful_cycles} optimization cycles")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test."""
    print("🚀 Enhanced ResiLink Simulation Test")
    print("=" * 40)
    
    success = test_simulation_mode()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 Simulation test PASSED!")
        print("\nYou can now run full optimization:")
        print("  python3 hybrid_resilink_implementation.py --simulation-mode --max-cycles 5")
    else:
        print("❌ Simulation test FAILED!")

if __name__ == "__main__":
    main()