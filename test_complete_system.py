#!/usr/bin/env python3
"""
Complete System Test for Enhanced ResiLink
==========================================

Comprehensive test that demonstrates the complete workflow:
1. Parse rich GraphML topology data
2. Extract geographic and link characteristics
3. Build Mininet network with realistic parameters
4. Test SDN controller functionality
5. Validate API endpoints for optimization

Usage:
    # Terminal 1: Start controller
    ryu-manager sdn/basic_controller.py ryu.app.rest_topology ryu.app.ofctl_rest --observe-links --wsapi-host 0.0.0.0 --wsapi-port 8080
    
    # Terminal 2: Run system test
    sudo python3 test_complete_system.py
"""

import os
import sys
import time
import json
import requests
import logging
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from core.enhanced_topology_parser import EnhancedTopologyParser
from sdn.mininet_builder import MininetBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompleteSystemTest:
    """Complete system test for Enhanced ResiLink."""
    
    def __init__(self):
        self.parser = EnhancedTopologyParser()
        self.builder = None
        self.enhanced_network = None
        self.test_results = {}
        
        logger.info("Complete System Test initialized")
    
    def run_complete_test(self, topology_file: str = "real_world_topologies/Geant2012.graphml"):
        """Run the complete system test."""
        print("🧪 Enhanced ResiLink Complete System Test")
        print("=" * 50)
        
        try:
            # Phase 1: Parse rich topology data
            print("\n📊 Phase 1: Parsing Rich Topology Data")
            self.test_topology_parsing(topology_file)
            
            # Phase 2: Build enhanced Mininet network
            print("\n🏗️  Phase 2: Building Enhanced Mininet Network")
            self.test_network_building()
            
            # Phase 3: Test SDN controller functionality
            print("\n🎮 Phase 3: Testing SDN Controller")
            self.test_controller_functionality()
            
            # Phase 4: Validate API endpoints
            print("\n📡 Phase 4: Validating API Endpoints")
            self.test_api_endpoints()
            
            # Phase 5: Test optimization readiness
            print("\n🚀 Phase 5: Testing Optimization Readiness")
            self.test_optimization_readiness()
            
            # Summary
            print("\n📋 Test Summary")
            self.print_test_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"System test failed: {e}")
            print(f"\n❌ System test failed: {e}")
            return False
        
        finally:
            if self.builder:
                self.builder.stop_network()
    
    def test_topology_parsing(self, topology_file: str):
        """Test rich topology parsing."""
        print(f"  📁 Loading topology: {topology_file}")
        
        if not os.path.exists(topology_file):
            raise FileNotFoundError(f"Topology file not found: {topology_file}")
        
        # Parse topology
        self.enhanced_network = self.parser.parse_graphml(topology_file)
        
        # Validate parsing results
        assert len(self.enhanced_network.nodes) > 0, "No nodes parsed"
        assert len(self.enhanced_network.edges) > 0, "No edges parsed"
        
        # Display rich metadata
        metadata = self.enhanced_network.metadata
        print(f"  ✅ Network: {metadata.name}")
        print(f"     Type: {metadata.network_type}")
        print(f"     Location: {metadata.geo_location}")
        print(f"     Date: {metadata.date_obtained}")
        print(f"     Nodes: {len(self.enhanced_network.nodes)}")
        print(f"     Edges: {len(self.enhanced_network.edges)}")
        
        # Geographic analysis
        geo_analysis = self.enhanced_network.geographic_analysis
        if 'num_countries' in geo_analysis:
            print(f"     Countries: {geo_analysis['num_countries']}")
        
        if 'link_distances' in geo_analysis:
            distances = geo_analysis['link_distances']
            print(f"     Avg link distance: {distances['mean']:.1f} km")
            print(f"     Total network span: {distances['total']:.0f} km")
        
        # Academic metrics
        metrics = self.enhanced_network.academic_metrics
        print(f"     Density: {metrics.get('density', 0):.3f}")
        print(f"     Connected: {metrics.get('is_connected', False)}")
        
        if 'average_shortest_path' in metrics:
            print(f"     Avg path length: {metrics['average_shortest_path']:.2f}")
        
        self.test_results['topology_parsing'] = {
            'status': 'PASS',
            'nodes': len(self.enhanced_network.nodes),
            'edges': len(self.enhanced_network.edges),
            'countries': geo_analysis.get('num_countries', 0),
            'connected': metrics.get('is_connected', False)
        }
    
    def test_network_building(self):
        """Test Mininet network building."""
        print("  🏗️  Building Mininet network from rich data...")
        
        # Create builder
        self.builder = MininetBuilder()
        
        # Build network
        net = self.builder.build_from_enhanced_network(self.enhanced_network)
        
        # Validate network structure
        assert len(net.switches) > 0, "No switches created"
        assert len(net.hosts) > 0, "No hosts created"
        assert len(net.links) > 0, "No links created"
        
        print(f"  ✅ Network built successfully")
        print(f"     Switches: {len(net.switches)}")
        print(f"     Hosts: {len(net.hosts)}")
        print(f"     Links: {len(net.links)}")
        
        # Start network
        print("  🚀 Starting network...")
        connectivity_result = self.builder.start_network()
        
        print(f"  📊 Connectivity test: {connectivity_result}% packet loss")
        
        if connectivity_result == 0:
            print("  ✅ Perfect connectivity!")
            connectivity_status = 'PERFECT'
        elif connectivity_result < 20:
            print("  ⚠️  Good connectivity")
            connectivity_status = 'GOOD'
        else:
            print("  ❌ Poor connectivity")
            connectivity_status = 'POOR'
        
        self.test_results['network_building'] = {
            'status': 'PASS',
            'switches': len(net.switches),
            'hosts': len(net.hosts),
            'links': len(net.links),
            'connectivity': connectivity_status,
            'packet_loss': connectivity_result
        }
    
    def test_controller_functionality(self):
        """Test SDN controller functionality."""
        print("  🎮 Testing controller connection...")
        
        # Wait for controller to be ready
        time.sleep(5)
        
        # Test basic connectivity to controller
        try:
            response = requests.get('http://localhost:8080/v1.0/topology/switches', timeout=5)
            if response.status_code == 200:
                print("  ✅ Controller API accessible")
                controller_status = 'ACCESSIBLE'
            else:
                print(f"  ❌ Controller API error: {response.status_code}")
                controller_status = 'ERROR'
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Controller not accessible: {e}")
            controller_status = 'INACCESSIBLE'
            raise
        
        # Test packet forwarding by doing additional pings
        print("  📡 Testing packet forwarding...")
        if self.builder and self.builder.net:
            hosts = self.builder.net.hosts
            if len(hosts) >= 2:
                h1, h2 = hosts[0], hosts[1]
                result = h1.cmd(f'ping -c 3 {h2.IP()}')
                
                if 'time=' in result:
                    print("  ✅ Packet forwarding working")
                    forwarding_status = 'WORKING'
                else:
                    print("  ❌ Packet forwarding failed")
                    forwarding_status = 'FAILED'
            else:
                forwarding_status = 'INSUFFICIENT_HOSTS'
        else:
            forwarding_status = 'NO_NETWORK'
        
        self.test_results['controller_functionality'] = {
            'status': 'PASS' if controller_status == 'ACCESSIBLE' and forwarding_status == 'WORKING' else 'FAIL',
            'api_status': controller_status,
            'forwarding_status': forwarding_status
        }
    
    def test_api_endpoints(self):
        """Test all API endpoints required for optimization."""
        print("  📡 Testing API endpoints...")
        
        endpoints = {
            'switches': '/v1.0/topology/switches',
            'links': '/v1.0/topology/links',
            'hosts': '/v1.0/topology/hosts',
            'all_topology': '/v1.0/topology/all',
            'controller_stats': '/v1.0/stats/controller'
        }
        
        endpoint_results = {}
        
        for name, endpoint in endpoints.items():
            try:
                response = requests.get(f'http://localhost:8080{endpoint}', timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if isinstance(data, list):
                        count = len(data)
                    elif isinstance(data, dict):
                        count = len(data) if 'error' not in data else 0
                    else:
                        count = 1
                    
                    print(f"    ✅ {name}: {count} items")
                    endpoint_results[name] = {'status': 'PASS', 'count': count}
                else:
                    print(f"    ❌ {name}: HTTP {response.status_code}")
                    endpoint_results[name] = {'status': 'FAIL', 'error': response.status_code}
                    
            except Exception as e:
                print(f"    ❌ {name}: {e}")
                endpoint_results[name] = {'status': 'ERROR', 'error': str(e)}
        
        # Validate data structure for optimization
        try:
            response = requests.get('http://localhost:8080/v1.0/topology/all', timeout=5)
            if response.status_code == 200:
                topology_data = response.json()
                
                # Check required fields
                required_fields = ['switches', 'links', 'hosts']
                missing_fields = [field for field in required_fields if field not in topology_data]
                
                if not missing_fields:
                    print("  ✅ Topology data structure valid for optimization")
                    data_structure_status = 'VALID'
                else:
                    print(f"  ❌ Missing required fields: {missing_fields}")
                    data_structure_status = 'INVALID'
            else:
                data_structure_status = 'ERROR'
        except Exception as e:
            print(f"  ❌ Data structure validation failed: {e}")
            data_structure_status = 'ERROR'
        
        self.test_results['api_endpoints'] = {
            'status': 'PASS' if all(r['status'] == 'PASS' for r in endpoint_results.values()) else 'FAIL',
            'endpoints': endpoint_results,
            'data_structure': data_structure_status
        }
    
    def test_optimization_readiness(self):
        """Test readiness for optimization algorithms."""
        print("  🚀 Testing optimization readiness...")
        
        # Check if we have sufficient network complexity
        nodes = len(self.enhanced_network.nodes)
        edges = len(self.enhanced_network.edges)
        
        # Calculate potential new links (complete graph - existing edges)
        max_possible_edges = nodes * (nodes - 1) // 2
        potential_new_links = max_possible_edges - edges
        
        print(f"    📊 Current edges: {edges}")
        print(f"    📊 Potential new links: {potential_new_links}")
        
        if potential_new_links > 0:
            print("  ✅ Network has optimization potential")
            optimization_potential = 'HIGH'
        else:
            print("  ⚠️  Network is complete (no optimization possible)")
            optimization_potential = 'NONE'
        
        # Check geographic constraints
        geo_analysis = self.enhanced_network.geographic_analysis
        if 'link_distances' in geo_analysis:
            avg_distance = geo_analysis['link_distances']['mean']
            print(f"    🌍 Average link distance: {avg_distance:.1f} km")
            
            if avg_distance < 1000:
                geographic_feasibility = 'HIGH'
            elif avg_distance < 5000:
                geographic_feasibility = 'MEDIUM'
            else:
                geographic_feasibility = 'LOW'
        else:
            geographic_feasibility = 'UNKNOWN'
        
        print(f"    🌍 Geographic feasibility: {geographic_feasibility}")
        
        # Academic metrics readiness
        metrics = self.enhanced_network.academic_metrics
        has_centrality = 'degree_centrality' in metrics
        has_efficiency = 'global_efficiency' in metrics
        is_connected = metrics.get('is_connected', False)
        
        academic_readiness = 'READY' if (has_centrality and has_efficiency and is_connected) else 'PARTIAL'
        print(f"    🎓 Academic metrics: {academic_readiness}")
        
        self.test_results['optimization_readiness'] = {
            'status': 'PASS',
            'optimization_potential': optimization_potential,
            'potential_new_links': potential_new_links,
            'geographic_feasibility': geographic_feasibility,
            'academic_readiness': academic_readiness
        }
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 50)
        print("📋 COMPLETE SYSTEM TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASS')
        
        print(f"Overall: {passed_tests}/{total_tests} tests passed")
        
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            print(f"{status_icon} {test_name.replace('_', ' ').title()}: {result['status']}")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! System ready for Enhanced ResiLink optimization.")
            print("\n🚀 Next steps:")
            print("   1. Keep this network running")
            print("   2. Run: python3 hybrid_resilink_implementation.py --max-cycles 10")
            print("   3. Observe optimization suggestions with rich geographic context")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} tests failed. Please address issues before optimization.")
        
        # Save detailed results
        try:
            os.makedirs('data/results', exist_ok=True)
            with open('data/results/system_test_results.json', 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save test results: {e}")
        
        print(f"\n💾 Detailed results saved to: data/results/system_test_results.json")


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete System Test for Enhanced ResiLink')
    parser.add_argument('--topology', default='real_world_topologies/Geant2012.graphml',
                       help='GraphML topology file to test')
    parser.add_argument('--keep-running', action='store_true',
                       help='Keep network running after test for optimization')
    
    args = parser.parse_args()
    
    # Check if running as root
    if os.geteuid() != 0:
        print("❌ This test must be run as root (for Mininet)")
        print("   sudo python3 test_complete_system.py")
        sys.exit(1)
    
    # Check if topology file exists
    if not os.path.exists(args.topology):
        print(f"❌ Topology file not found: {args.topology}")
        print("\nAvailable topologies:")
        topo_dir = Path("real_world_topologies")
        if topo_dir.exists():
            for f in sorted(topo_dir.glob("*.graphml"))[:10]:
                print(f"   {f}")
        sys.exit(1)
    
    # Run test
    test = CompleteSystemTest()
    
    try:
        success = test.run_complete_test(args.topology)
        
        if success and args.keep_running:
            print("\n🔄 Keeping network running for optimization...")
            print("   Press Ctrl+C to stop")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n⏹️  Stopping network...")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        logger.exception("Test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()