#!/usr/bin/env python3
"""
Simple Mininet Network Creator
=============================

Creates a Mininet network from JSON topology file and runs indefinitely.
Designed to work with hybrid_resilink_implementation.py via Ryu controller.

Usage:
    sudo python3 examples/simple_mininet.py real_world_topologies/geant_topology.json
"""

import os
import sys
import json
import time
import signal
from pathlib import Path

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.util import dumpNodeConnections


class SimpleMininet:
    """Simple Mininet network from JSON topology."""
    
    def __init__(self, controller_ip='127.0.0.1', controller_port=6653):
        self.controller_ip = controller_ip
        self.controller_port = controller_port
        self.net = None
        
        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self._cleanup)
        signal.signal(signal.SIGTERM, self._cleanup)
    
    def _cleanup(self, signum, frame):
        """Clean shutdown on signal."""
        info("*** Shutting down network...\n")
        if self.net:
            self.net.stop()
        sys.exit(0)
    
    def create_network(self, topology_file):
        """Create network from JSON topology file."""
        info(f"*** Loading topology from {topology_file}\n")
        
        # Load topology
        with open(topology_file, 'r') as f:
            topo_data = json.load(f)
        
        # Create network with remote controller
        controller = RemoteController('c0', ip=self.controller_ip, port=self.controller_port)
        
        self.net = Mininet(
            controller=controller,
            switch=OVSSwitch,
            autoSetMacs=True,
            autoStaticArp=True
        )
        
        # Add controller
        self.net.addController(controller)
        
        # Create switches
        info("*** Adding switches\n")
        switches = {}
        nodes = topo_data.get('nodes', [])
        
        for i, node in enumerate(nodes):
            node_id = node if isinstance(node, str) else node.get('id', f'node_{i}')
            switch_name = f's{i+1}'
            switch = self.net.addSwitch(switch_name, dpid=f'{i+1:016x}')
            switches[node_id] = switch
            info(f"Added {switch_name} for {node_id}\n")
        
        # Add hosts
        info("*** Adding hosts\n")
        for i, (node_id, switch) in enumerate(switches.items()):
            for j in range(2):  # 2 hosts per switch
                host_name = f'h{i*2+j+1}'
                host = self.net.addHost(host_name, ip=f'10.0.{i+1}.{j+1}/24')
                self.net.addLink(host, switch)
                info(f"Added {host_name} to {switch.name}\n")
        
        # Add links between switches
        info("*** Adding links\n")
        edges = topo_data.get('edges', [])
        for edge in edges:
            if isinstance(edge, list) and len(edge) >= 2:
                src_id, dst_id = edge[0], edge[1]
                
                if src_id in switches and dst_id in switches:
                    src_switch = switches[src_id]
                    dst_switch = switches[dst_id]
                    self.net.addLink(src_switch, dst_switch)
                    info(f"Added link {src_switch.name} <-> {dst_switch.name}\n")
        
        info(f"*** Created network: {len(switches)} switches, "
             f"{len(self.net.hosts)} hosts, {len(edges)} links\n")
    
    def start_network(self):
        """Start the network and wait."""
        info("*** Starting network\n")
        self.net.start()
        
        info("*** Waiting for controller connection...\n")
        time.sleep(3)
        
        info("*** Testing connectivity\n")
        self.net.pingAll()
        
        info("*** Network ready!\n")
        info("*** Controller API available at: http://localhost:8080\n")
        info("*** Run your optimization in another terminal:\n")
        info("    python3 hybrid_resilink_implementation.py --max-cycles 10\n")
        info("*** Press Ctrl+C to stop\n")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            info("*** Stopping network\n")
            self.net.stop()


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: sudo python3 examples/simple_mininet.py <topology.json>")
        print("Example: sudo python3 examples/simple_mininet.py real_world_topologies/geant_topology.json")
        sys.exit(1)
    
    topology_file = sys.argv[1]
    
    if not os.path.exists(topology_file):
        print(f"Error: Topology file not found: {topology_file}")
        sys.exit(1)
    
    # Set log level
    setLogLevel('info')
    
    print("🌐 Simple Mininet Network Creator")
    print("=" * 40)
    print(f"Topology: {topology_file}")
    print("Controller: 127.0.0.1:6653")
    print()
    
    # Create and start network
    mininet = SimpleMininet()
    mininet.create_network(topology_file)
    mininet.start_network()


if __name__ == '__main__':
    # Check root
    if os.geteuid() != 0:
        print("❌ This script must be run as root")
        print("   sudo python3 examples/simple_mininet.py <topology.json>")
        sys.exit(1)
    
    main()