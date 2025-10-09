#!/usr/bin/env python

"""
Mininet topology script based on Bell Canada GraphML file.

This script reads a GraphML file, creates a Mininet topology with switches
representing nodes and links representing edges. It adds exactly 1 host to each switch
with assigned IP addresses. It uses a remote custom controller (assumed to be
running on localhost:6653, e.g., Ryu or Floodlight). After starting, it runs
pingAll to generate some initial traffic.

Usage:
    sudo python this_script.py --graphml path_to_graphml.xml

Make sure to install networkx: pip install networkx
Run a controller separately, e.g., ryu-manager ryu.app.simple_switch_13
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info

import networkx as nx
import argparse

class GraphMLTopo(Topo):
    """Custom topology from GraphML file."""

    def __init__(self, graphml_file, **opts):
        """Create topology from GraphML."""
        super(GraphMLTopo, self).__init__(**opts)
        self.build_from_graphml(graphml_file)

    def build_from_graphml(self, graphml_file):
        """Build the topology using networkx to parse GraphML."""
        G = nx.read_graphml(graphml_file)

        # Add switches for each node (node ids are strings like '0', '1', etc.)
        switch_map = {}
        for node in G.nodes():
            switch_name = 's' + str(node)
            switch_map[node] = self.addSwitch(switch_name)

        # Add links for each edge
        for u, v in G.edges():
            self.addLink(switch_map[u], switch_map[v])

        # Add exactly 1 host to each switch with IP addresses (all in 10.0.0.0/24 for L2 compatibility)
        host_counter = 1
        for node in G.nodes():
            switch = switch_map[node]
            host_name = 'h%s' % node
            ip = '10.0.0.%s/24' % host_counter
            h = self.addHost(host_name, ip=ip)
            self.addLink(switch, h)
            host_counter += 1

def run(graphml_file):
    """Create and run the Mininet network."""
    topo = GraphMLTopo(graphml_file)
    net = Mininet(topo=topo, controller=None)  # No built-in controller

    # Add a remote custom controller
    net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6653)

    info('*** Starting network\n')
    net.start()
    
    CLI(net)
    info('*** Stopping network\n')
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')

    parser = argparse.ArgumentParser(description='Mininet topology from GraphML')
    parser.add_argument('--graphml', required=True, help='Path to GraphML file')
    args = parser.parse_args()

    info('*** Loading topology from %s\n' % args.graphml)
    run(args.graphml)
