#!/usr/bin/env python3
"""
Test script to verify hybrid_resilink_implementation fixes
"""

import requests
import time

def test_controller_endpoints():
    """Test that all required endpoints are accessible."""
    base_url = "http://localhost:8080"
    
    endpoints = [
        "/v1.0/topology/switches",
        "/v1.0/topology/links", 
        "/v1.0/topology/hosts",
        "/stats/flow/1",
        "/stats/port/1",
        "/stats/portdesc/1"
    ]
    
    print("🔍 Testing Controller Endpoints:")
    print("-" * 40)
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            status = "✅" if response.status_code == 200 else f"❌ ({response.status_code})"
            print(f"{status} {endpoint}")
        except Exception as e:
            print(f"❌ {endpoint} - Error: {str(e)[:50]}")
    
    print()

def test_network_discovery():
    """Test network discovery functionality."""
    print("🌐 Testing Network Discovery:")
    print("-" * 40)
    
    try:
        # Test switches
        switches_resp = requests.get("http://localhost:8080/v1.0/topology/switches", timeout=5)
        switches = switches_resp.json() if switches_resp.status_code == 200 else []
        print(f"📊 Switches discovered: {len(switches)}")
        
        # Test links
        links_resp = requests.get("http://localhost:8080/v1.0/topology/links", timeout=5)
        links = links_resp.json() if links_resp.status_code == 200 else []
        print(f"🔗 Links discovered: {len(links)}")
        
        # Test hosts
        hosts_resp = requests.get("http://localhost:8080/v1.0/topology/hosts", timeout=5)
        hosts = hosts_resp.json() if hosts_resp.status_code == 200 else []
        print(f"🖥️  Hosts discovered: {len(hosts)}")
        
        if switches:
            print(f"📋 Sample switch: {switches[0]}")
        if links:
            print(f"📋 Sample link: {links[0]}")
            
    except Exception as e:
        print(f"❌ Network discovery failed: {e}")
    
    print()

def main():
    print("🧪 Testing Hybrid ResiLink Implementation Fixes")
    print("=" * 50)
    
    # Test controller connectivity
    test_controller_endpoints()
    
    # Test network discovery
    test_network_discovery()
    
    print("💡 If all endpoints show ✅, you can run:")
    print("   python3 hybrid_resilink_implementation.py --max-cycles 3 --training-mode")

if __name__ == "__main__":
    main()