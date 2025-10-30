#!/usr/bin/env python3
"""
Enhanced Mininet Builder for Rich Topologies
===========================================

Builds Mininet topologies from enhanced network data with geographic
and link characteristics preserved.

Usage:
    builder = MininetBuilder()
    builder.build_from_enhanced_network(enhanced_network)
    builder.start_network()
"""

import os
import sys
import time
import signal
import logging
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.util import dumpNodeConnections

from core.enhanced_topology_parser import EnhancedNetwork, EnhancedTopologyParser

logger = logging.getLogger(__name__)


class MininetBuilder:
    """Build Mininet networks from enhanced topology data."""
    
    def __init__(self, controller_ip='127.0.0.1', controller_port=6653):
        self.controller_ip = controller_ip
        self.controller_port = controller_port
        self.net = None
        self.enhanced_network = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._cleanup)
        signal.signal(signal.SIGTERM, self._cleanup)
        
        logger.info("Mininet Builder initialized")
    
    def _cleanup(self, signum, frame):
        """Clean shutdown on signal."""
        info("*** Shutting down network...\n")
        if self.net:
            self.net.stop()
        sys.exit(0)
    
    def build_from_enhanced_network(self, enhanced_network: EnhancedNetwork):
        """Build Mininet network from enhanced network data."""
        self.enhanced_network = enhanced_network
        
        info(f"*** Building network: {enhanced_network.metadata.name}\n")
        info(f"*** Geographic extent: {enhanced_network.metadata.geo_extent}\n")
        
        # Create controller
        controller = RemoteController('c0', ip=self.controller_ip, port=self.controller_port)
        
        # Create Mininet network
        self.net = Mininet(
            controller=controller,
            switch=OVSSwitch,
            link=TCLink,
            autoSetMacs=True,
            autoStaticArp=True
        )
        
        # Add controller
        self.net.addController(controller)
        
        # Add switches
        info("*** Adding switches\n")
        switches = {}
        for i, (node_id, node_meta) in enumerate(enhanced_network.nodes.items()):
            switch_name = f's{i+1}'
            dpid = f'{i+1:016x}'
            
            switch = self.net.addSwitch(switch_name, dpid=dpid)
            switches[node_id] = switch
            
            info(f"Added {switch_name} for {node_meta.label} ({node_meta.country})\n")
        
        # Add hosts (2 per switch for traffic generation)
        info("*** Adding hosts\n")
        host_count = 0
        for i, (node_id, switch) in enumerate(switches.items()):
            for j in range(2):
                host_count += 1
                host_name = f'h{host_count}'
                host_ip = f'10.0.{i+1}.{j+1}/24'
                
                host = self.net.addHost(host_name, ip=host_ip)
                self.net.addLink(host, switch)
                
                info(f"Added {host_name} ({host_ip}) to {switch.name}\n")
        
        # Add links between switches with characteristics
        info("*** Adding links with characteristics\n")
        for edge_key, edge_meta in enhanced_network.edges.items():
            src_id, dst_id = edge_key
            
            if src_id in switches and dst_id in switches:
                src_switch = switches[src_id]
                dst_switch = switches[dst_id]
                
                # Determine link parameters from metadata
                link_params = self._get_link_parameters(edge_meta)
                
                self.net.addLink(src_switch, dst_switch, **link_params)
                
                info(f"Added link {src_switch.name} <-> {dst_switch.name}")
                if edge_meta.link_speed:
                    info(f" ({edge_meta.link_speed})")
                if edge_meta.geographic_distance:
                    info(f" [{edge_meta.geographic_distance:.0f} km]")
                info("\n")
        
        info(f"*** Network built: {len(switches)} switches, {host_count} hosts, {len(enhanced_network.edges)} links\n")
        
        return self.net
    
    def _get_link_parameters(self, edge_meta):
        """Convert edge metadata to Mininet link parameters."""
        params = {}
        
        # Bandwidth from link speed
        if edge_meta.link_speed_raw:
            # Convert to Mbps
            bw_mbps = edge_meta.link_speed_raw / 1e6
            params['bw'] = min(bw_mbps, 10000)  # Cap at 10 Gbps
        else:
            params['bw'] = 1000  # Default 1 Gbps
        
        # Delay from geographic distance
        if edge_meta.geographic_distance:
            # Approximate delay: ~5ms per 1000km + base latency
            delay_ms = max(1, edge_meta.geographic_distance * 0.005 + 1)
            params['delay'] = f'{delay_ms:.1f}ms'
        else:
            params['delay'] = '1ms'
        
        # Loss rate (small random loss for realism)
        params['loss'] = 0.01  # 0.01% packet loss
        
        # Use HTB for traffic shaping
        params['use_htb'] = True
        
        return params
    
    def start_network(self):
        """Start the Mininet network."""
        if not self.net:
            raise RuntimeError("Network not built. Call build_from_enhanced_network() first.")
        
        info("*** Starting network\n")
        self.net.start()
        
        info("*** Waiting for controller connection...\n")
        time.sleep(3)
        
        info("*** Testing connectivity\n")
        result = self.net.pingAll()
        
        if result == 0:
            info("*** ✅ Perfect connectivity!\n")
        elif result < 20:
            info(f"*** ⚠️  Good connectivity ({result}% loss)\n")
        else:
            info(f"*** ❌ Poor connectivity ({result}% loss)\n")
        
        # Display network information
        self._display_network_info()
        
        return result
    
    def _display_network_info(self):
        """Display comprehensive network information."""
        if not self.enhanced_network:
            return
        
        info("*** Network Information\n")
        info(f"Name: {self.enhanced_network.metadata.name}\n")
        info(f"Type: {self.enhanced_network.metadata.network_type}\n")
        info(f"Location: {self.enhanced_network.metadata.geo_location}\n")
        info(f"Date: {self.enhanced_network.metadata.date_obtained}\n")
        
        if 'num_countries' in self.enhanced_network.geographic_analysis:
            info(f"Countries: {self.enhanced_network.geographic_analysis['num_countries']}\n")
        
        if 'link_distances' in self.enhanced_network.geographic_analysis:
            distances = self.enhanced_network.geographic_analysis['link_distances']
            info(f"Avg link distance: {distances['mean']:.1f} km\n")
            info(f"Total network span: {distances['total']:.0f} km\n")
        
        info(f"Switches: {len(self.net.switches)}\n")
        info(f"Hosts: {len(self.net.hosts)}\n")
        info(f"Links: {len(self.net.links)}\n")
        
        info("*** Controller API endpoints:\n")
        info("  All topology: http://localhost:8080/v1.0/topology/all\n")
        info("  Switches: http://localhost:8080/v1.0/topology/switches\n")
        info("  Links: http://localhost:8080/v1.0/topology/links\n")
        info("  Hosts: http://localhost:8080/v1.0/topology/hosts\n")
        info("  Stats: http://localhost:8080/v1.0/stats/controller\n")
        info("\n")
    
    def run_interactive(self):
        """Run interactive CLI."""
        if not self.net:
            raise RuntimeError("Network not started")
        
        info("*** Ready for Enhanced ResiLink optimization!\n")
        info("*** Run optimization in another terminal:\n")
        info("    python3 hybrid_resilink_implementation.py --max-cycles 10\n")
        info("\n")
        info("*** Entering interactive CLI (type 'exit' to quit)\n")
        
        CLI(self.net)
    
    def run_persistent(self):
        """Run persistently until interrupted."""
        if not self.net:
            raise RuntimeError("Network not started")
        
        info("*** Network running persistently\n")
        info("*** Press Ctrl+C to stop\n")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            info("*** Stopping network\n")
            self.net.stop()
    
    def stop_network(self):
        """Stop the network."""
        if self.net:
            info("*** Stopping network\n")
            self.net.stop()
            self.net = None


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Mininet Builder')
    parser.add_argument('topology_file', help='GraphML topology file')
    parser.add_argument('--interactive', action='store_true', help='Run interactive CLI')
    parser.add_argument('--controller-ip', default='127.0.0.1', help='Controller IP')
    parser.add_argument('--controller-port', type=int, default=6653, help='Controller port')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.topology_file):
        print(f"❌ Topology file not found: {args.topology_file}")
        sys.exit(1)
    
    setLogLevel('info')
    
    print("🌐 Enhanced Mininet Builder")
    print("=" * 40)
    print(f"Topology: {args.topology_file}")
    print(f"Controller: {args.controller_ip}:{args.controller_port}")
    print()
    
    try:
        # Parse topology
        parser = EnhancedTopologyParser()
        enhanced_network = parser.parse_graphml(args.topology_file)
        
        # Build and start network
        builder = MininetBuilder(args.controller_ip, args.controller_port)
        builder.build_from_enhanced_network(enhanced_network)
        builder.start_network()
        
        # Run mode
        if args.interactive:
            builder.run_interactive()
        else:
            builder.run_persistent()
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        logger.exception("Builder failed")
    finally:
        if 'builder' in locals():
            builder.stop_network()
        print("🏁 Done")


if __name__ == "__main__":
    if os.geteuid() != 0:
        print("❌ Must run as root: sudo python3 sdn/mininet_builder.py <topology.graphml>")
        sys.exit(1)
    
    main()