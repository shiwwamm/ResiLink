#!/usr/bin/env python


from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info

import networkx as nx
import argparse
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points in kilometers."""
    # Earth's radius in kilometers
    R = 6371.0
    
    # Convert latitude and longitude to radians
    lat1_rad = math.radians(float(lat1))
    lon1_rad = math.radians(float(lon1))
    lat2_rad = math.radians(float(lat2))
    lon2_rad = math.radians(float(lon2))
    
    # Differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance

def normalize_weights(distances, min_range=0, max_range=1):
    """Normalize a list of distances to the range [min_range, max_range]."""
    if not distances:
        return []
    d_min = min(distances)
    d_max = max(distances)
    if d_max == d_min:
        # Avoid division by zero; return equal weights (e.g., 1.0)
        return [1.0] * len(distances)
    normalized = [(max_range - min_range) * (d - d_min) / (d_max - d_min) + min_range for d in distances]
    return normalized

class GraphMLTopo(Topo):
    """Custom topology from GraphML file with normalized edge weights based on geographical distance."""

    def __init__(self, graphml_file, **opts):
        """Create topology from GraphML."""
        super(GraphMLTopo, self).__init__(**opts)
        self.build_from_graphml(graphml_file)

    def build_from_graphml(self, graphml_file):
        """Build the topology using networkx to parse GraphML."""
        G = nx.read_graphml(graphml_file, node_type=str)

        # Add switches for each node (node ids are strings like '0', '1', etc.)
        switch_map = {}
        for node in G.nodes():
            switch_name = 's' + str(node)
            switch_map[node] = self.addSwitch(switch_name)

        # Collect all distances for normalization
        edge_distances = []
        edge_list = []
        for u, v in G.edges():
            # Get latitude and longitude for both nodes
            lat1 = G.nodes[u]['Latitude']
            lon1 = G.nodes[u]['Longitude']
            lat2 = G.nodes[v]['Latitude']
            lon2 = G.nodes[v]['Longitude']
            
            # Calculate distance
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            edge_distances.append(distance)
            edge_list.append((u, v))

        # Normalize distances to [0, 1]
        normalized_weights = normalize_weights(edge_distances)

        # Add links with normalized weights
        for (u, v), norm_weight, distance in zip(edge_list, normalized_weights, edge_distances):
            # Calculate delay based on distance (assuming 200,000 km/s signal speed)
            delay_ms = (distance / 200000) * 1000
            delay_str = f"{delay_ms:.2f}ms"
            
            # Add link with normalized weight, original distance, and delay
            self.addLink(
                switch_map[u], 
                switch_map[v], 
                weight=norm_weight, 
                distance_km=distance, 
                delay=delay_str
            )
            info(f"Added link {switch_map[u]} - {switch_map[v]}: distance={distance:.2f} km, normalized_weight={norm_weight:.4f}, delay={delay_str}\n")

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
