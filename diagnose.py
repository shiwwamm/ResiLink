#!/usr/bin/env python3
"""
Diagnostic script for ResiLink implementation
Checks all dependencies and module availability
"""

import sys
import importlib.util

def check_module(module_name, package_name=None):
    """Check if a module is available."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            print(f"  ✓ {module_name}")
            return True
        else:
            print(f"  ✗ {module_name} - NOT FOUND")
            if package_name:
                print(f"    Install with: pip install {package_name}")
            return False
    except Exception as e:
        print(f"  ✗ {module_name} - ERROR: {e}")
        return False


def check_file(file_path):
    """Check if a file exists."""
    from pathlib import Path
    if Path(file_path).exists():
        print(f"  ✓ {file_path}")
        return True
    else:
        print(f"  ✗ {file_path} - NOT FOUND")
        return False


def main():
    """Run diagnostics."""
    print("="*70)
    print("ResiLink Implementation Diagnostics")
    print("="*70)
    
    all_ok = True
    
    # Check Python version
    print("\n1. Python Version:")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠️  Warning: Python 3.8+ recommended")
        all_ok = False
    else:
        print("  ✓ Version OK")
    
    # Check core dependencies
    print("\n2. Core Dependencies:")
    deps = [
        ("networkx", "networkx"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("torch", "torch"),
        ("torch_geometric", "torch-geometric"),
        ("sklearn", "scikit-learn"),
        ("requests", "requests"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("pandas", "pandas"),
    ]
    
    for module, package in deps:
        if not check_module(module, package):
            all_ok = False
    
    # Check optional dependencies
    print("\n3. Optional Dependencies:")
    optional_deps = [
        ("ryu", "ryu"),
        ("psutil", "psutil"),
    ]
    
    for module, package in optional_deps:
        check_module(module, package)
    
    # Check local modules
    print("\n4. Local Modules:")
    local_modules = [
        "geographic_network_analyzer.py",
        "core/enhanced_topology_parser.py",
        "hybrid_resilink_implementation.py",
    ]
    
    for module in local_modules:
        if not check_file(module):
            all_ok = False
    
    # Check data directory
    print("\n5. Data Directory:")
    if not check_file("real_world_topologies"):
        print("  ⚠️  Warning: Topology data directory not found")
    else:
        from pathlib import Path
        graphml_files = list(Path("real_world_topologies").glob("*.graphml"))
        gml_files = list(Path("real_world_topologies").glob("*.gml"))
        print(f"  Found {len(graphml_files)} GraphML files")
        print(f"  Found {len(gml_files)} GML files")
    
    # Try importing local modules
    print("\n6. Local Module Imports:")
    try:
        print("  Testing geographic_network_analyzer...")
        from geographic_network_analyzer import GeographicNetworkAnalyzer, GeographicConstraints
        print("    ✓ GeographicNetworkAnalyzer")
        print("    ✓ GeographicConstraints")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        all_ok = False
    
    try:
        print("  Testing core.enhanced_topology_parser...")
        from core.enhanced_topology_parser import EnhancedTopologyParser
        print("    ✓ EnhancedTopologyParser")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        all_ok = False
    
    # Summary
    print("\n" + "="*70)
    if all_ok:
        print("✅ ALL CHECKS PASSED")
        print("\nYou can now run:")
        print("  python3 hybrid_resilink_implementation.py --help")
        print("\nOr test with simulation mode:")
        print("  python3 hybrid_resilink_implementation.py --simulation-mode --max-cycles 3")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before running the implementation.")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
    print("="*70)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
