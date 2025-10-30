#!/usr/bin/env python3
"""
Simple Mininet Script for GraphML Topologies
============================================

Converts GraphML files from Internet Topology Zoo into Mininet networks
and connects to the updated SDN controller.

Usage:
    sudo python3 mininet_graphml_topology.py real_world_topologies/Bellcanada.graphml
    sudo python3 mininet_graphml_topology.py --graphml-file real_world_topologies/Geant2012.graphml --duration 300
"""

import xml.etree.ElementTree as ET
import json
import argparse
import sys
import os
from pathlib import Path

from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import time


class GraphMLTopologyParser:
    """Parser for GraphML topology files from Internet Topology Zoo."""
    
    def __init__(self, graphml_file):
        self.graphml_file = Path(graphml_file)
        self.nodes = {}
        self.edges = []
        self.network_info = {}
        
    def parse(self):
        """Parse the GraphML file and extract topology information."""
        try:
            tree = ET.parse(self.graphml_file)
            root = tree.getroot()
            
            # Define namespace
            ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
            
            # Extract network metadata
            graph = root.find('.//graphml:graph', ns)
            if graph is not None:
                for data in graph.findall('graphml:data', ns):
                    key = data.get('key')
                    if key == 'd3':  # Network name
                        self.network_info['name'] = data.text
                    elif key == 'd1':  # GeoLocation
                        self.network_info['location'] = data.text
                    elif key == 'd2':  # GeoExtent
                        self.network_info['extent'] = data.text
            
            # Parse nodes
            for node in root.findall('.//graphml:node', ns):
                node_id = node.get('id')
                node_data = {'id': node_id}
                
                for data in node.findall('graphml:data', ns):
                    key = data.get('key')
                    if key == 'd35':  # label/name
                        node_data['label'] = data.text
                    elif key == 'd30':  # Latitude
                        node_data['latitude'] = float(data.text) if data.text else 0.0
                    elif key == 'd34':  # Longitude
                        node_data['longitude'] = float(data.text) if data.text else 0.0
                    elif key == 'd31':  # Country
                        node_data['country'] = data.text
                    elif key == 'd32':  # Type
                        node_data['type'] = data.text
                
                self.nodes[node_id] = node_data
            
            # Parse edges
            for edge in root.findall('.//graphml:edge', ns):
                source = edge.get('source')
                target = edge.get('target')
                
                edge_data = {
                    'source': source,
                    'target': target
                }
                
                for data in edge.findall('graphml:data', ns):
                    key = data.get('key')
                    if key == 'd36':  # LinkType
                        edge_data['link_type'] = data.text
                    elif key == 'd37':  # LinkLabel
                        edge_data['link_label'] = data.text
                
                self.edges.append(edge_data)
            
            info(f"✅ Parsed GraphML: {len(self.nodes)} nodes, {len(self.edges)} edges\n")
            return True
            
        except Exception as e:
            info(f"❌ Failed to parse GraphML file: {e}\n")
            return False
    
    def to_json(self, output_file=None):
        """Convert parsed topology to JSON format."""
        topology_data = {
            'network_info': self.network_info,
            'nodes': self.nodes,
            'edges': self.edges,
            'stats': {
                'node_count': len(self.nodes),
                'edge_count': len(self.edges)
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(topology_data, f, indent=2)
            info(f"💾 Topology saved to {output_file}\n")
        
        return topology_data


class GraphMLMininetTopology:
    """Creates Mininet topology from parsed GraphML data."""
    
    def __init__(self, parser, controller_ip='127.0.0.1', controller_port=6653):
        self.parser = parser
        self.controller_ip = controller_ip
        self.controller_port = controller_port
        self.net = None
        self.switches = {}
        self.hosts = {}
        
    def create_topology(self):
        """Create the Mininet topology."""
        info("🌐 Creating Mininet topology from GraphML data...\n")
        
        # Create network with remote controller
        self.net = Mininet(
            controller=None,
            switch=OVSSwitch,
            link=TCLink,
            autoSetMacs=True,
            autoStaticArp=True
        )
        
        # Add controller
        controller = self.net.addController(
            'c0',
            controller=RemoteController,
            ip=self.controller_ip,
            port=self.controller_port
        )
        
        # Create switches for each node
        info("🔧 Creating switches...\n")
        for node_id, node_data in self.parser.nodes.items():
            switch_name = f"s{node_id}"
            label = node_data.get('label', f'Node{node_id}')
            
            switch = self.net.addSwitch(
                switch_name,
                dpid=f"{int(node_id)+1:016x}"  # Ensure unique DPID
            )
            self.switches[node_id] = switch
            info(f"   Switch {switch_name}: {label}\n")
        
        # Create links between switches based on edges
        info("🔗 Creating links...\n")
        for edge in self.parser.edges:
            source_id = edge['source']
            target_id = edge['target']
            
            if source_id in self.switches and target_id in self.switches:
                source_switch = self.switches[source_id]
                target_switch = self.switches[target_id]
                
                # Add link with some basic parameters
                self.net.addLink(
                    source_switch,
                    target_switch,
                    bw=100,  # 100 Mbps default
                    delay='5ms',
                    loss=0
                )
                
                source_label = self.parser.nodes[source_id].get('label', f'Node{source_id}')
                target_label = self.parser.nodes[target_id].get('label', f'Target{target_id}')
                info(f"   Link: {source_label} <-> {target_label}\n")
        
        # Add a few hosts for testing connectivity
        info("🖥️  Adding test hosts...\n")
        host_count = min(4, len(self.switches))  # Add up to 4 hosts
        switch_ids = list(self.switches.keys())[:host_count]
        
        for i, switch_id in enumerate(switch_ids):
            host_name = f"h{i+1}"
            switch = self.switches[switch_id]
            
            host = self.net.addHost(
                host_name,
                ip=f"10.0.0.{i+1}/24",
                mac=f"00:00:00:00:00:{i+1:02x}"
            )
            
            self.net.addLink(host, switch)
            self.hosts[host_name] = host
            
            switch_label = self.parser.nodes[switch_id].get('label', f'Node{switch_id}')
            info(f"   Host {host_name} -> {switch_label}\n")
        
        info(f"✅ Topology created: {len(self.switches)} switches, {len(self.hosts)} hosts\n")
        return True
    
    def start_network(self):
        """Start the Mininet network."""
        info("🚀 Starting network...\n")
        self.net.start()
        
        # Wait for controller connection
        info("⏳ Waiting for switches to connect to controller...\n")
        time.sleep(5)
        
        # Test connectivity
        info("🔍 Testing basic connectivity...\n")
        if len(self.hosts) >= 2:
            host_list = list(self.hosts.values())
            result = self.net.ping([host_list[0], host_list[1]], timeout=10)
            if result == 0:
                info("✅ Basic connectivity test passed\n")
            else:
                info("⚠️  Basic connectivity test failed (this is normal initially)\n")
        
        return True
    
    def run_cli(self):
        """Run the Mininet CLI."""
        info("🎮 Starting Mininet CLI...\n")
        info("Available commands:\n")
        info("  pingall - Test connectivity between all hosts\n")
        info("  iperf - Test bandwidth between hosts\n")
        info("  dump - Show network information\n")
        info("  exit - Exit CLI\n\n")
        
        CLI(self.net)
    
    def stop_network(self):
        """Stop the Mininet network."""
        if self.net:
            info("🛑 Stopping network...\n")
            self.net.stop()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create Mininet topology from GraphML file')
    parser.add_argument('graphml_file', nargs='?', 
                       help='Path to GraphML file (e.g., real_world_topologies/Bellcanada.graphml)')
    parser.add_argument('--graphml-file', help='Path to GraphML file (alternative)')
    parser.add_argument('--controller-ip', default='127.0.0.1', 
                       help='Controller IP address (default: 127.0.0.1)')
    parser.add_argument('--controller-port', type=int, default=6653,
                       help='Controller port (default: 6653)')
    parser.add_argument('--duration', type=int, help='Run for specified seconds then exit')
    parser.add_argument('--save-json', help='Save topology as JSON file')
    
    args = parser.parse_args()
    
    # Determine GraphML file
    graphml_file = args.graphml_file or args.graphml_file
    if not graphml_file:
        print("❌ Please specify a GraphML file")
        print("Usage: sudo python3 mininet_graphml_topology.py real_world_topologies/Bellcanada.graphml")
        sys.exit(1)
    
    if not os.path.exists(graphml_file):
        print(f"❌ GraphML file not found: {graphml_file}")
        sys.exit(1)
    
    # Check if running as root
    if os.geteuid() != 0:
        print("❌ This script must be run as root (use sudo)")
        sys.exit(1)
    
    setLogLevel('info')
    
    try:
        # Parse GraphML file
        info(f"📖 Parsing GraphML file: {graphml_file}\n")
        graphml_parser = GraphMLTopologyParser(graphml_file)
        
        if not graphml_parser.parse():
            sys.exit(1)
        
        # Save JSON if requested
        if args.save_json:
            graphml_parser.to_json(args.save_json)
        
        # Display network information
        network_name = graphml_parser.network_info.get('name', 'Unknown Network')
        location = graphml_parser.network_info.get('location', 'Unknown Location')
        info(f"🌐 Network: {network_name}\n")
        info(f"📍 Location: {location}\n")
        info(f"📊 Topology: {len(graphml_parser.nodes)} nodes, {len(graphml_parser.edges)} edges\n\n")
        
        # Create and start Mininet topology
        mininet_topo = GraphMLMininetTopology(
            graphml_parser, 
            args.controller_ip, 
            args.controller_port
        )
        
        if not mininet_topo.create_topology():
            sys.exit(1)
        
        if not mininet_topo.start_network():
            sys.exit(1)
        
        # Run for specified duration or start CLI
        if args.duration:
            info(f"⏰ Running for {args.duration} seconds...\n")
            time.sleep(args.duration)
        else:
            mininet_topo.run_cli()
        
    except KeyboardInterrupt:
        info("\n⏹️  Interrupted by user\n")
    except Exception as e:
        info(f"\n💥 Error: {e}\n")
    finally:
        # Cleanup
        if 'mininet_topo' in locals():
            mininet_topo.stop_network()
        
        # Clean up Mininet
        os.system('mn -c > /dev/null 2>&1')


if __name__ == '__main__':
    main()