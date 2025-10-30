#!/usr/bin/env python3
"""
Test Controller Only
===================

Simple test to verify the SDN controller is working with a basic Mininet network.
"""

import os
import sys
import time
import signal
from pathlib import Path

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info

class SimpleNetworkTest:
    """Simple network test with basic topology."""
    
    def __init__(self):
        self.net = None
        signal.signal(signal.SIGINT, self._cleanup)
        signal.signal(signal.SIGTERM, self._cleanup)
    
    def _cleanup(self, signum, frame):
        """Clean shutdown."""
        info("*** Shutting down...\n")
        if self.net:
            self.net.stop()
        sys.exit(0)
    
    def create_simple_network(self):
        """Create a simple 4-switch network."""
        info("*** Creating simple test network\n")
        
        # Create controller
        controller = RemoteController('c0', ip='127.0.0.1', port=6653)
        
        # Create network
        self.net = Mininet(
            controller=controller,
            switch=OVSSwitch,
            autoSetMacs=True,
            autoStaticArp=True
        )
        
        # Add controller
        self.net.addController(controller)
        
        # Add switches
        s1 = self.net.addSwitch('s1', dpid='0000000000000001')
        s2 = self.net.addSwitch('s2', dpid='0000000000000002')
        s3 = self.net.addSwitch('s3', dpid='0000000000000003')
        s4 = self.net.addSwitch('s4', dpid='0000000000000004')
        
        # Add hosts
        h1 = self.net.addHost('h1', ip='10.0.1.1/24')
        h2 = self.net.addHost('h2', ip='10.0.2.1/24')
        h3 = self.net.addHost('h3', ip='10.0.3.1/24')
        h4 = self.net.addHost('h4', ip='10.0.4.1/24')
        
        # Connect hosts to switches
        self.net.addLink(h1, s1)
        self.net.addLink(h2, s2)
        self.net.addLink(h3, s3)
        self.net.addLink(h4, s4)
        
        # Connect switches in a line: s1-s2-s3-s4
        self.net.addLink(s1, s2)
        self.net.addLink(s2, s3)
        self.net.addLink(s3, s4)
        
        info("*** Network topology: h1-s1-s2-s3-s4-h4 (+ h2, h3)\n")
    
    def start_and_test(self):
        """Start network and test connectivity."""
        info("*** Starting network\n")
        self.net.start()
        
        info("*** Waiting for controller connection...\n")
        time.sleep(5)
        
        info("*** Testing connectivity\n")
        result = self.net.pingAll()
        
        if result == 0:
            info("*** ✅ Perfect connectivity!\n")
        elif result < 50:
            info(f"*** ⚠️  Partial connectivity ({result}% loss)\n")
        else:
            info(f"*** ❌ Poor connectivity ({result}% loss)\n")
        
        info("*** Network ready! Test API endpoints:\n")
        info("    curl http://localhost:8080/v1.0/topology/switches\n")
        info("    curl http://localhost:8080/v1.0/topology/links\n")
        info("    curl http://localhost:8080/v1.0/topology/hosts\n")
        info("\n")
        
        if result == 0:
            info("*** Network is working! You can now run:\n")
            info("    python3 hybrid_resilink_implementation.py --max-cycles 3\n")
        
        info("*** Press Ctrl+C to stop, or type 'exit' to quit\n")
        
        try:
            CLI(self.net)
        except KeyboardInterrupt:
            pass
        
        info("*** Stopping network\n")
        self.net.stop()

def main():
    """Main test function."""
    if os.geteuid() != 0:
        print("❌ This test must be run as root (for Mininet)")
        print("   sudo python3 test_controller_only.py")
        sys.exit(1)
    
    setLogLevel('info')
    
    print("🧪 Simple Controller Test")
    print("=" * 30)
    print("Make sure controller is running:")
    print("  ryu-manager sdn/basic_controller.py ryu.app.rest_topology ryu.app.ofctl_rest --observe-links --wsapi-host 0.0.0.0 --wsapi-port 8080")
    print()
    
    test = SimpleNetworkTest()
    test.create_simple_network()
    test.start_and_test()

if __name__ == "__main__":
    main()