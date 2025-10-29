#!/usr/bin/env python3
"""
Enhanced ResiLink: Mininet Topology Integration
==============================================

Complete Mininet topology setup for Enhanced ResiLink demonstration.
Creates academic-grade network topologies for testing hybrid optimization.

This script provides:
- Multiple topology options (linear, tree, fat-tree, custom)
- Academic topology analysis and validation
- Integration with Enhanced ResiLink SDN controller
- Real-time monitoring and optimization demonstration

Academic Foundation:
- Topology design: Al-Fares et al. (2008) - Fat-tree topologies
- Network analysis: Barabási & Albert (1999) - Scale-free networks
- SDN integration: McKeown et al. (2008) - OpenFlow specification
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import time
import json
import argparse
import sys
import os
import networkx as nx
from typing import Dict, List, Tuple

class AcademicLinearTopo(Topo):
    """
    Linear topology for basic resilience testing.
    
    Academic justification: Simple topology for validating
    basic connectivity and resilience metrics (Albert et al. 2000).
    """
    
    def __init__(self, n_switches=4, n_hosts_per_switch=2, **opts):
        super(AcademicLinearTopo, self).__init__(**opts)
        
        self.n_switches = n_switches
        self.n_hosts_per_switch = n_hosts_per_switch
        
        # Add switches
        switches = []
        for i in range(n_switches):
            switch = self.addSwitch(f's{i+1}')
            switches.append(switch)
        
        # Add linear links between switches
        for i in range(n_switches - 1):
            self.addLink(switches[i], switches[i+1], 
                        bw=1000, delay='1ms', loss=0)  # 1 Gbps links
        
        # Add hosts to each switch
        for i, switch in enumerate(switches):
            for j in range(n_hosts_per_switch):
                host = self.addHost(f'h{i+1}{j+1}', 
                                  ip=f'10.0.{i+1}.{j+1}/24')
                self.addLink(host, switch, 
                           bw=100, delay='0.1ms', loss=0)  # 100 Mbps host links

class AcademicTreeTopo(Topo):
    """
    Tree topology for hierarchical network testing.
    
    Academic justification: Hierarchical structure common in
    enterprise networks (Cisco hierarchical model).
    """
    
    def __init__(self, depth=3, fanout=2, **opts):
        super(AcademicTreeTopo, self).__init__(**opts)
        
        self.depth = depth
        self.fanout = fanout
        
        # Build tree recursively
        self.switch_count = 0
        self.host_count = 0
        
        root = self._add_tree_level(None, 0)
        
    def _add_tree_level(self, parent, level):
        """Recursively add tree levels."""
        self.switch_count += 1
        switch = self.addSwitch(f's{self.switch_count}')
        
        if parent:
            self.addLink(parent, switch, 
                        bw=1000, delay='1ms', loss=0)
        
        if level < self.depth - 1:
            # Add child switches
            for i in range(self.fanout):
                self._add_tree_level(switch, level + 1)
        else:
            # Add hosts at leaf level
            for i in range(self.fanout):
                self.host_count += 1
                host = self.addHost(f'h{self.host_count}',
                                  ip=f'10.0.0.{self.host_count}/24')
                self.addLink(host, switch,
                           bw=100, delay='0.1ms', loss=0)
        
        return switch

class AcademicFatTreeTopo(Topo):
    """
    Fat-tree topology for data center network simulation.
    
    Academic justification: Al-Fares et al. (2008) - "A scalable,
    commodity data center network architecture" - SIGCOMM.
    """
    
    def __init__(self, k=4, **opts):
        super(AcademicFatTreeTopo, self).__init__(**opts)
        
        self.k = k  # Number of ports per switch
        
        # Calculate topology parameters
        self.num_pods = k
        self.num_core = (k // 2) ** 2
        self.num_agg_per_pod = k // 2
        self.num_edge_per_pod = k // 2
        self.num_hosts_per_edge = k // 2
        
        # Add core switches
        core_switches = []
        for i in range(self.num_core):
            switch = self.addSwitch(f'core{i+1}')
            core_switches.append(switch)
        
        # Add pods
        for pod in range(self.num_pods):
            self._add_pod(pod, core_switches)
    
    def _add_pod(self, pod_id, core_switches):
        """Add a single pod to the fat-tree."""
        # Add aggregation switches
        agg_switches = []
        for i in range(self.num_agg_per_pod):
            switch = self.addSwitch(f'agg{pod_id}_{i}')
            agg_switches.append(switch)
        
        # Add edge switches
        edge_switches = []
        for i in range(self.num_edge_per_pod):
            switch = self.addSwitch(f'edge{pod_id}_{i}')
            edge_switches.append(switch)
        
        # Connect aggregation to core
        core_per_agg = len(core_switches) // self.num_agg_per_pod
        for i, agg_switch in enumerate(agg_switches):
            start_core = i * core_per_agg
            for j in range(core_per_agg):
                if start_core + j < len(core_switches):
                    self.addLink(agg_switch, core_switches[start_core + j],
                               bw=1000, delay='1ms', loss=0)
        
        # Connect aggregation to edge
        for agg_switch in agg_switches:
            for edge_switch in edge_switches:
                self.addLink(agg_switch, edge_switch,
                           bw=1000, delay='1ms', loss=0)
        
        # Add hosts to edge switches
        for i, edge_switch in enumerate(edge_switches):
            for j in range(self.num_hosts_per_edge):
                host_id = pod_id * self.num_edge_per_pod * self.num_hosts_per_edge + i * self.num_hosts_per_edge + j + 1
                host = self.addHost(f'h{host_id}',
                                  ip=f'10.{pod_id}.{i}.{j+1}/24')
                self.addLink(host, edge_switch,
                           bw=100, delay='0.1ms', loss=0)

class AcademicStarTopo(Topo):
    """
    Star topology for hub vulnerability testing.
    
    Academic justification: Tests single point of failure scenarios
    and hub vulnerability (Albert et al. 2000).
    """
    
    def __init__(self, n_spokes=4, **opts):
        super(AcademicStarTopo, self).__init__(**opts)
        
        # Central hub switch
        hub = self.addSwitch('s1')
        
        # Add spoke switches
        spokes = []
        for i in range(n_spokes):
            spoke = self.addSwitch(f's{i+2}')
            spokes.append(spoke)
            # Connect to hub
            self.addLink(hub, spoke, bw=1000, delay='1ms', loss=0)
            
            # Add hosts to each spoke
            for j in range(2):
                host = self.addHost(f'h{i+1}{j+1}', ip=f'10.0.{i+1}.{j+1}/24')
                self.addLink(host, spoke, bw=100, delay='0.1ms', loss=0)

class AcademicRingTopo(Topo):
    """
    Ring topology for path diversity testing.
    
    Academic justification: Tests moderate resilience with dual paths
    and small-world optimization (Watts & Strogatz 1998).
    """
    
    def __init__(self, n_switches=6, **opts):
        super(AcademicRingTopo, self).__init__(**opts)
        
        # Add switches in ring
        switches = []
        for i in range(n_switches):
            switch = self.addSwitch(f's{i+1}')
            switches.append(switch)
        
        # Create ring connections
        for i in range(n_switches):
            next_switch = (i + 1) % n_switches
            self.addLink(switches[i], switches[next_switch], 
                        bw=1000, delay='1ms', loss=0)
        
        # Add hosts (every other switch)
        for i in range(0, n_switches, 2):
            for j in range(2):
                host = self.addHost(f'h{i+1}{j+1}', ip=f'10.0.{i+1}.{j+1}/24')
                self.addLink(host, switches[i], bw=100, delay='0.1ms', loss=0)

class AcademicGridTopo(Topo):
    """
    2D Grid topology for mesh resilience testing.
    
    Academic justification: Tests 2D mesh networks common in
    parallel computing and data centers (Dally & Towles 2004).
    """
    
    def __init__(self, rows=3, cols=3, **opts):
        super(AcademicGridTopo, self).__init__(**opts)
        
        self.rows = rows
        self.cols = cols
        
        # Add switches in grid
        switches = {}
        for i in range(rows):
            for j in range(cols):
                switch_name = f's{i*cols + j + 1}'
                switch = self.addSwitch(switch_name)
                switches[(i, j)] = switch
        
        # Add horizontal links
        for i in range(rows):
            for j in range(cols - 1):
                self.addLink(switches[(i, j)], switches[(i, j+1)],
                           bw=1000, delay='1ms', loss=0)
        
        # Add vertical links
        for i in range(rows - 1):
            for j in range(cols):
                self.addLink(switches[(i, j)], switches[(i+1, j)],
                           bw=1000, delay='1ms', loss=0)
        
        # Add hosts to corner and center switches
        host_positions = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
        if rows % 2 == 1 and cols % 2 == 1:  # Add center if odd dimensions
            host_positions.append((rows//2, cols//2))
        
        host_counter = 1
        for pos in host_positions:
            if pos in switches:
                for k in range(2):
                    host = self.addHost(f'h{host_counter}', 
                                      ip=f'10.0.0.{host_counter}/24')
                    self.addLink(host, switches[pos], 
                               bw=100, delay='0.1ms', loss=0)
                    host_counter += 1

class AcademicDisconnectedTopo(Topo):
    """
    Disconnected components topology for connectivity testing.
    
    Academic justification: Tests fundamental connectivity algorithms
    and component bridging capabilities.
    """
    
    def __init__(self, n_components=3, switches_per_component=2, **opts):
        super(AcademicDisconnectedTopo, self).__init__(**opts)
        
        switch_counter = 1
        host_counter = 1
        
        for comp in range(n_components):
            # Create switches for this component
            comp_switches = []
            for i in range(switches_per_component):
                switch = self.addSwitch(f's{switch_counter}')
                comp_switches.append(switch)
                switch_counter += 1
            
            # Connect switches within component (linear)
            for i in range(len(comp_switches) - 1):
                self.addLink(comp_switches[i], comp_switches[i+1],
                           bw=1000, delay='1ms', loss=0)
            
            # Add hosts to first and last switch of component
            for switch in [comp_switches[0], comp_switches[-1]]:
                host = self.addHost(f'h{host_counter}', 
                                  ip=f'10.0.{comp+1}.{host_counter}/24')
                self.addLink(host, switch, bw=100, delay='0.1ms', loss=0)
                host_counter += 1

class AcademicBridgeTopo(Topo):
    """
    Bridge topology for critical link testing.
    
    Academic justification: Tests bridge identification and bypass
    creation (Tarjan 1972 bridge-finding algorithm).
    """
    
    def __init__(self, **opts):
        super(AcademicBridgeTopo, self).__init__(**opts)
        
        # Left cluster
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3')
        
        # Bridge switch
        s4 = self.addSwitch('s4')
        
        # Right cluster
        s5 = self.addSwitch('s5')
        s6 = self.addSwitch('s6')
        s7 = self.addSwitch('s7')
        
        # Left cluster connections (triangle)
        self.addLink(s1, s2, bw=1000, delay='1ms', loss=0)
        self.addLink(s2, s3, bw=1000, delay='1ms', loss=0)
        self.addLink(s3, s1, bw=1000, delay='1ms', loss=0)
        
        # Bridge connections (critical links)
        self.addLink(s2, s4, bw=1000, delay='1ms', loss=0)  # Critical bridge
        self.addLink(s4, s5, bw=1000, delay='1ms', loss=0)  # Critical bridge
        
        # Right cluster connections (triangle)
        self.addLink(s5, s6, bw=1000, delay='1ms', loss=0)
        self.addLink(s6, s7, bw=1000, delay='1ms', loss=0)
        self.addLink(s7, s5, bw=1000, delay='1ms', loss=0)
        
        # Add hosts
        hosts_switches = [s1, s3, s6, s7]
        for i, switch in enumerate(hosts_switches):
            host = self.addHost(f'h{i+1}', ip=f'10.0.0.{i+1}/24')
            self.addLink(host, switch, bw=100, delay='0.1ms', loss=0)

class RealWorldTopo(Topo):
    """
    Real-world topology loader for Internet Topology Zoo data.
    
    Academic justification: Testing on real network topologies
    provides essential validation for practical deployment.
    """
    
    def __init__(self, topology_file=None, **opts):
        super(RealWorldTopo, self).__init__(**opts)
        
        if topology_file is None:
            raise ValueError("topology_file parameter required for real-world topologies")
        
        # Load topology from JSON file
        import json
        with open(topology_file, 'r') as f:
            topo_data = json.load(f)
        
        # Add switches
        switches = {}
        for node in topo_data['nodes']:
            switches[node] = self.addSwitch(node)
        
        # Add links
        for src, dst in topo_data['edges']:
            self.addLink(switches[src], switches[dst], 
                        bw=1000, delay='1ms', loss=0)
        
        # Add hosts to high-degree nodes
        if 'centrality_analysis' in topo_data:
            degree_centrality = topo_data['centrality_analysis']['degree_centrality']
            high_degree_nodes = sorted(degree_centrality.keys(), 
                                     key=lambda x: degree_centrality[x], reverse=True)
            
            # Add hosts to top 25% of nodes by degree
            num_host_nodes = max(2, len(high_degree_nodes) // 4)
            host_counter = 1
            
            for i, node in enumerate(high_degree_nodes[:num_host_nodes]):
                for j in range(2):  # 2 hosts per selected switch
                    host = self.addHost(f'h{host_counter}', 
                                      ip=f'10.0.{i+1}.{j+1}/24')
                    self.addLink(host, switches[node], 
                               bw=100, delay='0.1ms', loss=0)
                    host_counter += 1

class AcademicCustomTopo(Topo):
    """
    Custom topology for specific resilience testing scenarios.
    
    Academic justification: Allows testing of specific network
    properties and resilience characteristics.
    """
    
    def __init__(self, topology_config=None, **opts):
        super(AcademicCustomTopo, self).__init__(**opts)
        
        if topology_config is None:
            # Default: Small network with interesting properties
            topology_config = {
                'switches': ['s1', 's2', 's3', 's4', 's5'],
                'links': [
                    ('s1', 's2'), ('s2', 's3'), ('s3', 's4'),
                    ('s4', 's5'), ('s5', 's1'), ('s2', 's4')  # Creates cycles
                ],
                'hosts': {
                    's1': ['h1', 'h2'],
                    's3': ['h3', 'h4'],
                    's5': ['h5', 'h6']
                }
            }
        
        self.topology_config = topology_config
        
        # Add switches
        for switch_name in topology_config['switches']:
            self.addSwitch(switch_name)
        
        # Add switch-to-switch links
        for src, dst in topology_config['links']:
            self.addLink(src, dst, bw=1000, delay='1ms', loss=0)
        
        # Add hosts
        host_counter = 1
        for switch_name, host_list in topology_config.get('hosts', {}).items():
            for i, host_name in enumerate(host_list):
                self.addHost(host_name, ip=f'10.0.0.{host_counter}/24')
                self.addLink(host_name, switch_name, 
                           bw=100, delay='0.1ms', loss=0)
                host_counter += 1

class MininetAcademicDemo:
    """
    Academic demonstration using Mininet with Enhanced ResiLink integration.
    
    Provides complete workflow:
    1. Topology creation with academic justification
    2. SDN controller integration
    3. Network analysis and monitoring
    4. Resilience optimization demonstration
    """
    
    def __init__(self, controller_ip='127.0.0.1', controller_port=6653):
        self.controller_ip = controller_ip
        self.controller_port = controller_port
        self.net = None
        self.topology_analysis = {}
    
    def create_network(self, topology_type='linear', **topo_params):
        """
        Create Mininet network with specified topology.
        
        Args:
            topology_type: Type of topology ('linear', 'tree', 'fat_tree', 'custom')
            **topo_params: Parameters for topology creation
        """
        print(f"🏗️  Creating {topology_type} topology...")
        
        # Select topology class
        if topology_type == 'linear':
            topo = AcademicLinearTopo(**topo_params)
        elif topology_type == 'tree':
            topo = AcademicTreeTopo(**topo_params)
        elif topology_type == 'fat_tree':
            topo = AcademicFatTreeTopo(**topo_params)
        elif topology_type == 'star':
            topo = AcademicStarTopo(**topo_params)
        elif topology_type == 'ring':
            topo = AcademicRingTopo(**topo_params)
        elif topology_type == 'grid':
            topo = AcademicGridTopo(**topo_params)
        elif topology_type == 'disconnected':
            topo = AcademicDisconnectedTopo(**topo_params)
        elif topology_type == 'bridge':
            topo = AcademicBridgeTopo(**topo_params)
        elif topology_type == 'real_world':
            topo = RealWorldTopo(**topo_params)
        elif topology_type == 'custom':
            topo = AcademicCustomTopo(**topo_params)
        else:
            raise ValueError(f"Unknown topology type: {topology_type}")
        
        # Create network with remote controller
        self.net = Mininet(
            topo=topo,
            controller=RemoteController('c0', ip=self.controller_ip, port=self.controller_port),
            switch=OVSSwitch,
            link=TCLink,
            autoSetMacs=True,
            autoStaticArp=True
        )
        
        print(f"✅ Network created with {len(self.net.switches)} switches and {len(self.net.hosts)} hosts")
        
        # Analyze topology
        self._analyze_topology()
        
        return self.net
    
    def start_network(self):
        """Start the Mininet network."""
        if not self.net:
            raise RuntimeError("Network not created. Call create_network() first.")
        
        print("🚀 Starting network...")
        self.net.start()
        
        # Wait for controller connection
        print("⏳ Waiting for controller connection...")
        time.sleep(5)
        
        # Test connectivity
        print("🔍 Testing initial connectivity...")
        self._test_connectivity()
        
        print("✅ Network started successfully")
    
    def stop_network(self):
        """Stop and cleanup the Mininet network."""
        if self.net:
            print("🛑 Stopping network...")
            self.net.stop()
            self.net = None
            print("✅ Network stopped")
    
    def run_interactive_demo(self):
        """Run interactive demonstration with CLI."""
        if not self.net:
            raise RuntimeError("Network not started. Call start_network() first.")
        
        print("\n🎮 Starting interactive demo...")
        print("Available commands:")
        print("  - pingall: Test connectivity between all hosts")
        print("  - iperf: Run bandwidth test between hosts")
        print("  - dump: Show network information")
        print("  - links: Show link information")
        print("  - exit: Exit CLI")
        print("\n💡 The Enhanced ResiLink controller is monitoring the network!")
        print("💡 Access metrics at: http://localhost:8080/enhanced/metrics")
        
        CLI(self.net)
    
    def run_automated_demo(self, duration=300):
        """
        Run automated demonstration with traffic generation.
        
        Args:
            duration: Duration of demo in seconds
        """
        if not self.net:
            raise RuntimeError("Network not started. Call start_network() first.")
        
        print(f"🤖 Running automated demo for {duration} seconds...")
        
        # Generate background traffic
        self._generate_background_traffic()
        
        # Monitor network periodically
        start_time = time.time()
        while time.time() - start_time < duration:
            print(f"📊 Monitoring network... ({int(time.time() - start_time)}s elapsed)")
            
            # Test connectivity
            self._test_connectivity()
            
            # Show network stats
            self._show_network_stats()
            
            # Wait before next iteration
            time.sleep(30)
        
        print("✅ Automated demo completed")
    
    def _analyze_topology(self):
        """Analyze network topology using graph theory."""
        # Build NetworkX graph
        G = nx.Graph()
        
        # Add switches as nodes
        for switch in self.net.switches:
            G.add_node(switch.name, type='switch')
        
        # Add hosts as nodes
        for host in self.net.hosts:
            G.add_node(host.name, type='host')
        
        # Add links as edges
        for link in self.net.links:
            node1, node2 = link.intf1.node.name, link.intf2.node.name
            G.add_edge(node1, node2)
        
        # Calculate academic metrics
        self.topology_analysis = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'switches': len(self.net.switches),
            'hosts': len(self.net.hosts),
            'density': nx.density(G),
            'is_connected': nx.is_connected(G),
            'diameter': nx.diameter(G) if nx.is_connected(G) else float('inf'),
            'average_clustering': nx.average_clustering(G),
            'academic_properties': {
                'algebraic_connectivity': nx.algebraic_connectivity(G) if nx.is_connected(G) else 0.0,
                'global_efficiency': nx.global_efficiency(G),
                'robustness_estimate': self._estimate_robustness(G)
            }
        }
        
        print(f"📊 Topology Analysis:")
        print(f"   Nodes: {self.topology_analysis['nodes']} "
              f"(Switches: {self.topology_analysis['switches']}, "
              f"Hosts: {self.topology_analysis['hosts']})")
        print(f"   Edges: {self.topology_analysis['edges']}")
        print(f"   Density: {self.topology_analysis['density']:.3f}")
        print(f"   Connected: {self.topology_analysis['is_connected']}")
        print(f"   Algebraic Connectivity: {self.topology_analysis['academic_properties']['algebraic_connectivity']:.3f}")
        print(f"   Global Efficiency: {self.topology_analysis['academic_properties']['global_efficiency']:.3f}")
    
    def _estimate_robustness(self, G):
        """Estimate network robustness (Albert et al. 2000)."""
        if G.number_of_nodes() < 2:
            return 0.0
        
        # Simple robustness estimate: fraction of nodes that can be removed
        # before network becomes disconnected
        temp_G = G.copy()
        nodes_removed = 0
        total_nodes = temp_G.number_of_nodes()
        
        # Remove highest degree nodes iteratively
        while nx.is_connected(temp_G) and temp_G.number_of_nodes() > 1:
            degrees = dict(temp_G.degree())
            if not degrees:
                break
            
            highest_degree_node = max(degrees.keys(), key=lambda x: degrees[x])
            temp_G.remove_node(highest_degree_node)
            nodes_removed += 1
            
            # Prevent infinite loop
            if nodes_removed > total_nodes // 2:
                break
        
        return nodes_removed / total_nodes
    
    def _test_connectivity(self):
        """Test network connectivity."""
        if not self.net:
            return
        
        print("🔍 Testing connectivity...")
        
        # Ping between random host pairs
        hosts = self.net.hosts
        if len(hosts) >= 2:
            h1, h2 = hosts[0], hosts[1]
            result = self.net.ping([h1, h2], timeout='1')
            
            if result == 0:
                print("   ✅ Connectivity test passed")
            else:
                print("   ❌ Connectivity test failed")
        else:
            print("   ⚠️  Not enough hosts for connectivity test")
    
    def _generate_background_traffic(self):
        """Generate background traffic for realistic network conditions."""
        hosts = self.net.hosts
        
        if len(hosts) < 2:
            print("   ⚠️  Not enough hosts for traffic generation")
            return
        
        print("🚦 Generating background traffic...")
        
        # Start iperf servers on some hosts
        for i in range(0, len(hosts), 2):
            host = hosts[i]
            host.cmd('iperf -s &')
        
        # Start iperf clients on other hosts
        for i in range(1, len(hosts), 2):
            if i < len(hosts) - 1:
                client = hosts[i]
                server = hosts[i - 1]
                server_ip = server.IP()
                client.cmd(f'iperf -c {server_ip} -t 3600 &')  # Run for 1 hour
        
        print("   ✅ Background traffic started")
    
    def _show_network_stats(self):
        """Show current network statistics."""
        print("📈 Network Statistics:")
        
        # Show switch information
        for switch in self.net.switches:
            print(f"   Switch {switch.name}: {len(switch.intfs)} interfaces")
        
        # Show host information
        for host in self.net.hosts:
            print(f"   Host {host.name}: IP {host.IP()}")
    
    def save_topology_info(self, filename='topology_info.json'):
        """Save topology information for analysis."""
        topology_info = {
            'timestamp': time.time(),
            'topology_analysis': self.topology_analysis,
            'network_info': {
                'switches': [s.name for s in self.net.switches] if self.net else [],
                'hosts': [h.name for h in self.net.hosts] if self.net else [],
                'links': [(link.intf1.node.name, link.intf2.node.name) 
                         for link in self.net.links] if self.net else []
            },
            'academic_foundation': {
                'topology_theory': 'Graph theory analysis (Fiedler 1973, Albert et al. 2000)',
                'connectivity_metrics': 'Algebraic connectivity, global efficiency',
                'robustness_analysis': 'Targeted attack simulation (Albert et al. 2000)'
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(topology_info, f, indent=2, default=str)
            print(f"💾 Topology information saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save topology info: {e}")

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Enhanced ResiLink Mininet Topology Demo')
    
    parser.add_argument('--topology', choices=['linear', 'tree', 'fat_tree', 'star', 'ring', 'grid', 'disconnected', 'bridge', 'custom', 'real_world'],
                       default='linear', help='Topology type (default: linear)')
    parser.add_argument('--controller-ip', default='127.0.0.1',
                       help='Controller IP address (default: 127.0.0.1)')
    parser.add_argument('--controller-port', type=int, default=6653,
                       help='Controller port (default: 6653)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive CLI demo')
    parser.add_argument('--duration', type=int, default=300,
                       help='Duration for automated demo in seconds (default: 300)')
    
    # Topology-specific parameters
    parser.add_argument('--switches', type=int, default=4,
                       help='Number of switches for linear topology (default: 4)')
    parser.add_argument('--hosts-per-switch', type=int, default=2,
                       help='Hosts per switch for linear topology (default: 2)')
    parser.add_argument('--depth', type=int, default=3,
                       help='Depth for tree topology (default: 3)')
    parser.add_argument('--fanout', type=int, default=2,
                       help='Fanout for tree topology (default: 2)')
    parser.add_argument('--k', type=int, default=4,
                       help='K parameter for fat-tree topology (default: 4)')
    
    # New topology parameters
    parser.add_argument('--spokes', type=int, default=4,
                       help='Number of spokes for star topology (default: 4)')
    parser.add_argument('--ring-size', type=int, default=6,
                       help='Number of switches in ring topology (default: 6)')
    parser.add_argument('--rows', type=int, default=3,
                       help='Number of rows for grid topology (default: 3)')
    parser.add_argument('--cols', type=int, default=3,
                       help='Number of columns for grid topology (default: 3)')
    parser.add_argument('--components', type=int, default=3,
                       help='Number of disconnected components (default: 3)')
    parser.add_argument('--real-world-file', 
                       help='JSON file with real-world topology data')
    
    args = parser.parse_args()
    
    # Set Mininet log level
    setLogLevel('info')
    
    print("🌐 Enhanced ResiLink Mininet Topology Demo")
    print("=" * 50)
    print(f"Topology: {args.topology}")
    print(f"Controller: {args.controller_ip}:{args.controller_port}")
    
    # Create demo instance
    demo = MininetAcademicDemo(
        controller_ip=args.controller_ip,
        controller_port=args.controller_port
    )
    
    try:
        # Prepare topology parameters
        topo_params = {}
        if args.topology == 'linear':
            topo_params = {
                'n_switches': args.switches,
                'n_hosts_per_switch': args.hosts_per_switch
            }
        elif args.topology == 'tree':
            topo_params = {
                'depth': args.depth,
                'fanout': args.fanout
            }
        elif args.topology == 'fat_tree':
            topo_params = {'k': args.k}
        elif args.topology == 'star':
            topo_params = {'n_spokes': args.spokes}
        elif args.topology == 'ring':
            topo_params = {'n_switches': args.ring_size}
        elif args.topology == 'grid':
            topo_params = {'rows': args.rows, 'cols': args.cols}
        elif args.topology == 'disconnected':
            topo_params = {'n_components': args.components, 'switches_per_component': 2}
        elif args.topology == 'bridge':
            topo_params = {}  # Bridge topology has fixed structure
        elif args.topology == 'real_world':
            if not args.real_world_file:
                raise ValueError("--real-world-file required for real_world topology")
            topo_params = {'topology_file': args.real_world_file}
        
        # Create and start network
        demo.create_network(args.topology, **topo_params)
        demo.start_network()
        
        # Save topology information
        demo.save_topology_info()
        
        print("\n🎯 Network is ready for Enhanced ResiLink optimization!")
        print("💡 Access controller metrics at: http://localhost:8080/enhanced/metrics")
        print("💡 Run optimization with: python3 examples/complete_sdn_integration_demo.py")
        
        # Run demo
        if args.interactive:
            demo.run_interactive_demo()
        else:
            demo.run_automated_demo(duration=args.duration)
    
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
    finally:
        demo.stop_network()
        print("🏁 Demo completed")

if __name__ == '__main__':
    # Check if running as root (required for Mininet)
    if os.geteuid() != 0:
        print("❌ This script must be run as root (use sudo)")
        print("   sudo python3 examples/mininet_topology_demo.py")
        sys.exit(1)
    
    main()