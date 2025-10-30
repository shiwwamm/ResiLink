#!/usr/bin/env python3
"""
Test Mininet Network - 4 Switches, 1 Host Each
==============================================

Simple test network for Enhanced ResiLink setup verification.
Creates a linear topology: h1-s1-s2-s3-s4-h4 with additional hosts h2, h3.

Usage:
    sudo python3 examples/test_mininet.py
"""

import os
import sys
import time
import signal

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.util import dumpNodeConnections


class TestMininet:
    """Simple 4-switch test network."""
    
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
    
    def create_network(self):
        """Create simple 4-switch test network."""
        info("*** Creating test network: 4 switches, 1 host each\n")
        
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
        
        # Add switches
        info("*** Adding switches\n")
        s1 = self.net.addSwitch('s1', dpid='0000000000000001')
        s2 = self.net.addSwitch('s2', dpid='0000000000000002')
        s3 = self.net.addSwitch('s3', dpid='0000000000000003')
        s4 = self.net.addSwitch('s4', dpid='0000000000000004')
        
        # Add hosts (1 per switch)
        info("*** Adding hosts\n")
        h1 = self.net.addHost('h1', ip='10.0.1.1/24')
        h2 = self.net.addHost('h2', ip='10.0.2.1/24')
        h3 = self.net.addHost('h3', ip='10.0.3.1/24')
        h4 = self.net.addHost('h4', ip='10.0.4.1/24')
        
        # Connect hosts to switches
        info("*** Connecting hosts to switches\n")
        self.net.addLink(h1, s1)
        self.net.addLink(h2, s2)
        self.net.addLink(h3, s3)
        self.net.addLink(h4, s4)
        
        # Connect switches in a line: s1-s2-s3-s4
        info("*** Connecting switches\n")
        self.net.addLink(s1, s2)
        self.net.addLink(s2, s3)
        self.net.addLink(s3, s4)
        
        info("*** Network topology created:\n")
        info("    h1 -- s1 -- s2 -- s3 -- s4 -- h4\n")
        info("           |     |     |\n")
        info("          h2    h3    (h4 connected above)\n")
        info("\n")
    
    def start_network(self):
        """Start the network and run tests."""
        info("*** Starting network\n")
        self.net.start()
        
        info("*** Waiting for controller connection...\n")
        time.sleep(3)
        
        info("*** Dumping network connections\n")
        dumpNodeConnections(self.net.hosts)
        dumpNodeConnections(self.net.switches)
        
        info("*** Testing connectivity\n")
        result = self.net.pingAll()
        info(f"*** Ping test result: {result}% packet loss\n")
        
        if result == 0:
            info("*** ✅ Network connectivity: PERFECT\n")
        elif result < 50:
            info("*** ⚠️  Network connectivity: PARTIAL\n")
        else:
            info("*** ❌ Network connectivity: POOR\n")
        
        info("*** Network Information:\n")
        info(f"    Switches: {len(self.net.switches)}\n")
        info(f"    Hosts: {len(self.net.hosts)}\n")
        info(f"    Links: {len(self.net.links)}\n")
        info("\n")
        
        info("*** Controller API endpoints:\n")
        info("    Switches: http://localhost:8080/v1.0/topology/switches\n")
        info("    Links:    http://localhost:8080/v1.0/topology/links\n")
        info("    Hosts:    http://localhost:8080/v1.0/topology/hosts\n")
        info("\n")
        
        info("*** Ready for Enhanced ResiLink testing!\n")
        info("*** Run in another terminal:\n")
        info("    python3 hybrid_resilink_implementation.py --max-cycles 5 --cycle-interval 10\n")
        info("\n")
        info("*** Press Ctrl+C to stop, or 'i' for interactive CLI\n")
        
        # Wait for user input
        try:
            while True:
                user_input = input().strip().lower()
                if user_input == 'i':
                    info("*** Entering interactive CLI mode\n")
                    CLI(self.net)
                    break
                elif user_input == 'q' or user_input == 'quit':
                    break
                else:
                    info("*** Commands: 'i' for CLI, 'q' to quit, Ctrl+C to stop\n")
        except KeyboardInterrupt:
            pass
        
        info("*** Stopping network\n")
        self.net.stop()
    
    def run_quick_test(self):
        """Run a quick connectivity test."""
        info("*** Quick connectivity test\n")
        
        if len(self.net.hosts) >= 2:
            h1, h2 = self.net.hosts[0], self.net.hosts[1]
            info(f"*** Testing {h1.name} -> {h2.name}\n")
            
            result = h1.cmd(f'ping -c 3 {h2.IP()}')
            if 'time=' in result:
                info("*** ✅ Basic connectivity: OK\n")
            else:
                info("*** ❌ Basic connectivity: FAILED\n")
                info(f"*** Ping output: {result}\n")


def main():
    """Main function."""
    # Set log level
    setLogLevel('info')
    
    print("🧪 Enhanced ResiLink Test Network")
    print("=" * 40)
    print("Topology: 4 switches, 1 host each")
    print("Controller: 127.0.0.1:6653")
    print("Layout: h1-s1-s2-s3-s4-h4 (+ h2, h3)")
    print()
    print("💡 Make sure Ryu controller is running:")
    print("   ryu-manager ryu.app.simple_switch_13 ryu.app.rest_topology ryu.app.ofctl_rest --observe-links --wsapi-host 0.0.0.0 --wsapi-port 8080")
    print()
    
    # Create and start network
    test_net = TestMininet()
    test_net.create_network()
    test_net.start_network()


if __name__ == '__main__':
    # Check root
    if os.geteuid() != 0:
        print("❌ This script must be run as root")
        print("   sudo python3 examples/test_mininet.py")
        sys.exit(1)
    
    main()