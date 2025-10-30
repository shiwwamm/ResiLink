#!/usr/bin/env python3
"""
Persistent Mininet Demo for Enhanced ResiLink
============================================

A long-running Mininet topology that maintains network state and generates
continuous traffic for resilience optimization experiments.

Key Features:
- Runs indefinitely until manually stopped
- Continuous background traffic generation
- Real-world topology support
- Academic-grade monitoring and logging
- Graceful shutdown handling

Usage:
    sudo python3 examples/persistent_mininet_demo.py \
        --topology real_world \
        --real-world-file real_world_topologies/geant_topology.json

Academic Foundation:
- Traffic modeling: Paxson & Floyd (1995) - Wide-area traffic characteristics
- Network simulation: Fall & Varadhan (2000) - ns-2 network simulator
- SDN evaluation: McKeown et al. (2008) - OpenFlow specification
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import threading
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel
from mininet.util import dumpNodeConnections
import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('persistent_mininet.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PersistentMininetDemo:
    """
    Persistent Mininet demonstration for Enhanced ResiLink experiments.
    
    Maintains network topology and generates continuous traffic for
    long-running resilience optimization experiments.
    """
    
    def __init__(self, controller_ip='127.0.0.1', controller_port=6653):
        self.controller_ip = controller_ip
        self.controller_port = controller_port
        self.net = None
        self.topology_analysis = {}
        self.traffic_threads = []
        self.running = True
        self.traffic_stats = {
            'flows_started': 0,
            'bytes_transferred': 0,
            'connections_active': 0
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Persistent Mininet Demo initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        self.stop_network()
        sys.exit(0)
    
    def create_real_world_topology(self, topology_file):
        """Create network from real-world topology JSON file."""
        logger.info(f"Loading real-world topology from {topology_file}")
        
        try:
            with open(topology_file, 'r') as f:
                topo_data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load topology file: {e}")
        
        # Create Mininet network with remote controller
        controller = RemoteController(
            'c0',
            ip=self.controller_ip,
            port=self.controller_port
        )
        
        self.net = Mininet(
            controller=controller,
            switch=OVSSwitch,
            link=TCLink,
            autoSetMacs=True,
            autoStaticArp=True
        )
        
        # Add controller
        self.net.addController(controller)
        
        # Create switches from topology data
        switches = {}
        nodes = topo_data.get('nodes', [])
        
        for i, node in enumerate(nodes):
            node_id = node.get('id', f'node_{i}')
            switch_name = f's{i+1}'
            switch = self.net.addSwitch(switch_name, dpid=f'{i+1:016x}')
            switches[node_id] = switch
            logger.debug(f"Added switch {switch_name} for node {node_id}")
        
        # Add hosts (2 per switch for traffic generation)
        hosts = {}
        for i, (node_id, switch) in enumerate(switches.items()):
            for j in range(2):
                host_name = f'h{i*2+j+1}'
                host = self.net.addHost(host_name, ip=f'10.0.{i+1}.{j+1}/24')
                self.net.addLink(host, switch)
                hosts[f'{node_id}_h{j}'] = host
                logger.debug(f"Added host {host_name} to switch {switch.name}")
        
        # Add links from topology data
        links = topo_data.get('links', [])
        for link in links:
            src_id = link.get('source')
            dst_id = link.get('target')
            
            if src_id in switches and dst_id in switches:
                src_switch = switches[src_id]
                dst_switch = switches[dst_id]
                
                # Add link with realistic parameters
                self.net.addLink(
                    src_switch, dst_switch,
                    bw=1000,  # 1 Gbps
                    delay='5ms',
                    loss=0.01,  # 1% packet loss
                    use_htb=True
                )
                logger.debug(f"Added link {src_switch.name} <-> {dst_switch.name}")
        
        # Store topology info
        self.topology_analysis = {
            'name': topo_data.get('name', 'Unknown'),
            'nodes': len(switches),
            'hosts': len(hosts),
            'links': len(links),
            'switches': switches,
            'hosts': hosts
        }
        
        logger.info(f"Created topology '{self.topology_analysis['name']}' with "
                   f"{self.topology_analysis['nodes']} switches, "
                   f"{self.topology_analysis['hosts']} hosts, "
                   f"{self.topology_analysis['links']} links")
    
    def start_network(self):
        """Start the Mininet network."""
        if not self.net:
            raise RuntimeError("Network not created. Call create_*_topology() first.")
        
        logger.info("Starting Mininet network...")
        self.net.start()
        
        # Wait for controller connection
        logger.info("Waiting for controller connection...")
        time.sleep(5)
        
        # Test connectivity
        logger.info("Testing initial connectivity...")
        result = self.net.pingAll()
        logger.info(f"Initial ping test: {result}% packet loss")
        
        # Dump connections for debugging
        logger.info("Network connections:")
        dumpNodeConnections(self.net.hosts)
        dumpNodeConnections(self.net.switches)
        
        logger.info("✅ Network started successfully")
    
    def start_continuous_traffic(self):
        """Start continuous background traffic generation."""
        logger.info("Starting continuous traffic generation...")
        
        hosts = list(self.net.hosts)
        if len(hosts) < 2:
            logger.warning("Not enough hosts for traffic generation")
            return
        
        # Start iperf servers on even-numbered hosts
        for i in range(0, len(hosts), 2):
            host = hosts[i]
            port = 5001 + i
            cmd = f'iperf -s -p {port} -i 30 > /tmp/iperf_server_{host.name}.log 2>&1 &'
            host.cmd(cmd)
            logger.debug(f"Started iperf server on {host.name}:{port}")
        
        # Start traffic generation threads
        self.traffic_threads = []
        
        # HTTP-like traffic (short bursts)
        http_thread = threading.Thread(target=self._generate_http_traffic, daemon=True)
        http_thread.start()
        self.traffic_threads.append(http_thread)
        
        # Bulk transfer traffic (long flows)
        bulk_thread = threading.Thread(target=self._generate_bulk_traffic, daemon=True)
        bulk_thread.start()
        self.traffic_threads.append(bulk_thread)
        
        # Interactive traffic (small packets)
        interactive_thread = threading.Thread(target=self._generate_interactive_traffic, daemon=True)
        interactive_thread.start()
        self.traffic_threads.append(interactive_thread)
        
        logger.info(f"✅ Started {len(self.traffic_threads)} traffic generation threads")
    
    def _generate_http_traffic(self):
        """Generate HTTP-like traffic patterns."""
        hosts = list(self.net.hosts)
        
        while self.running:
            try:
                # Select random client and server
                import random
                client = random.choice(hosts)
                server = random.choice([h for h in hosts if h != client])
                
                # Short burst transfer (simulating web page)
                size = random.randint(100, 1000)  # KB
                duration = random.randint(1, 5)   # seconds
                
                cmd = f'iperf -c {server.IP()} -t {duration} -n {size}K > /dev/null 2>&1 &'
                client.cmd(cmd)
                
                self.traffic_stats['flows_started'] += 1
                self.traffic_stats['bytes_transferred'] += size * 1024
                
                # Wait between requests (think time)
                time.sleep(random.uniform(2, 10))
                
            except Exception as e:
                logger.error(f"HTTP traffic generation error: {e}")
                time.sleep(5)
    
    def _generate_bulk_traffic(self):
        """Generate bulk transfer traffic."""
        hosts = list(self.net.hosts)
        
        while self.running:
            try:
                import random
                client = random.choice(hosts)
                server = random.choice([h for h in hosts if h != client])
                
                # Long transfer (simulating file download)
                duration = random.randint(30, 120)  # 30s to 2min
                
                cmd = f'iperf -c {server.IP()} -t {duration} > /dev/null 2>&1 &'
                client.cmd(cmd)
                
                self.traffic_stats['flows_started'] += 1
                
                # Wait before starting next bulk transfer
                time.sleep(random.uniform(60, 180))
                
            except Exception as e:
                logger.error(f"Bulk traffic generation error: {e}")
                time.sleep(10)
    
    def _generate_interactive_traffic(self):
        """Generate interactive traffic (small packets)."""
        hosts = list(self.net.hosts)
        
        while self.running:
            try:
                import random
                
                # Ping traffic for connectivity
                for _ in range(3):
                    if not self.running:
                        break
                    
                    src = random.choice(hosts)
                    dst = random.choice([h for h in hosts if h != src])
                    
                    # Small ping burst
                    cmd = f'ping -c 5 -i 0.2 {dst.IP()} > /dev/null 2>&1 &'
                    src.cmd(cmd)
                
                time.sleep(30)  # Every 30 seconds
                
            except Exception as e:
                logger.error(f"Interactive traffic generation error: {e}")
                time.sleep(15)
    
    def monitor_network(self):
        """Continuously monitor network status."""
        logger.info("Starting network monitoring...")
        
        while self.running:
            try:
                # Count active connections
                active_connections = 0
                for host in self.net.hosts:
                    result = host.cmd('netstat -an | grep ESTABLISHED | wc -l')
                    active_connections += int(result.strip())
                
                self.traffic_stats['connections_active'] = active_connections
                
                # Log status every 60 seconds
                logger.info(f"Network Status - Flows: {self.traffic_stats['flows_started']}, "
                           f"Active Connections: {active_connections}, "
                           f"Data Transferred: {self.traffic_stats['bytes_transferred']//1024//1024} MB")
                
                # Test connectivity periodically
                if self.traffic_stats['flows_started'] % 10 == 0:
                    logger.info("Testing network connectivity...")
                    loss = self.net.pingAll()
                    if loss > 10:
                        logger.warning(f"High packet loss detected: {loss}%")
                
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Network monitoring error: {e}")
                time.sleep(30)
    
    def run_persistent_demo(self):
        """Run the persistent demo indefinitely."""
        logger.info("🚀 Starting persistent Mininet demo...")
        logger.info("💡 Network will run until manually stopped (Ctrl+C)")
        logger.info("💡 Access controller at: http://localhost:8080/v1.0/topology/switches")
        
        # Start traffic generation
        self.start_continuous_traffic()
        
        # Start monitoring in background
        monitor_thread = threading.Thread(target=self.monitor_network, daemon=True)
        monitor_thread.start()
        
        # Save topology info
        self.save_topology_info()
        
        logger.info("✅ Demo is running! Press Ctrl+C to stop gracefully.")
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            self.running = False
    
    def save_topology_info(self, filename='persistent_topology_info.json'):
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
            'traffic_stats': self.traffic_stats,
            'academic_foundation': {
                'traffic_modeling': 'Paxson & Floyd (1995) - Wide-area traffic characteristics',
                'network_simulation': 'Fall & Varadhan (2000) - ns-2 network simulator',
                'sdn_evaluation': 'McKeown et al. (2008) - OpenFlow specification'
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(topology_info, f, indent=2, default=str)
            logger.info(f"💾 Topology information saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save topology info: {e}")
    
    def stop_network(self):
        """Stop the Mininet network gracefully."""
        if self.net:
            logger.info("Stopping network gracefully...")
            
            # Stop traffic generation
            self.running = False
            
            # Kill any remaining processes
            for host in self.net.hosts:
                host.cmd('pkill -f iperf')
                host.cmd('pkill -f ping')
            
            # Stop network
            self.net.stop()
            logger.info("✅ Network stopped")
        else:
            logger.info("Network was not running")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Persistent Mininet Demo for Enhanced ResiLink')
    
    parser.add_argument('--topology', choices=['real_world'], default='real_world',
                       help='Topology type (currently only real_world supported)')
    parser.add_argument('--real-world-file', required=True,
                       help='JSON file with real-world topology data')
    parser.add_argument('--controller-ip', default='127.0.0.1',
                       help='Controller IP address (default: 127.0.0.1)')
    parser.add_argument('--controller-port', type=int, default=6653,
                       help='Controller port (default: 6653)')
    parser.add_argument('--interactive', action='store_true',
                       help='Drop to CLI instead of running persistent demo')
    
    args = parser.parse_args()
    
    # Set Mininet log level
    setLogLevel('info')
    
    print("🌐 Enhanced ResiLink Persistent Mininet Demo")
    print("=" * 50)
    print(f"Topology: {args.topology}")
    print(f"Topology File: {args.real_world_file}")
    print(f"Controller: {args.controller_ip}:{args.controller_port}")
    print(f"Mode: {'Interactive CLI' if args.interactive else 'Persistent Demo'}")
    
    # Verify topology file exists
    if not os.path.exists(args.real_world_file):
        print(f"❌ Topology file not found: {args.real_world_file}")
        sys.exit(1)
    
    # Create demo instance
    demo = PersistentMininetDemo(
        controller_ip=args.controller_ip,
        controller_port=args.controller_port
    )
    
    try:
        # Create and start network
        demo.create_real_world_topology(args.real_world_file)
        demo.start_network()
        
        print("\n🎯 Network is ready for Enhanced ResiLink optimization!")
        print("💡 Run optimization with:")
        print("   python3 hybrid_resilink_implementation.py --max-cycles 10 --cycle-interval 30")
        
        # Run demo
        if args.interactive:
            print("\n🖥️  Dropping to Mininet CLI...")
            CLI(demo.net)
        else:
            demo.run_persistent_demo()
    
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        logger.exception("Demo failed with exception")
    finally:
        demo.stop_network()
        print("🏁 Demo completed")


if __name__ == '__main__':
    # Check if running as root (required for Mininet)
    if os.geteuid() != 0:
        print("❌ This script must be run as root (use sudo)")
        print("   sudo python3 examples/persistent_mininet_demo.py --real-world-file <file>")
        sys.exit(1)
    
    main()