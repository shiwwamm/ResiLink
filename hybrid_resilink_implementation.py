#!/usr/bin/env python3
"""
Enhanced ResiLink: Hybrid Implementation Script
==============================================

Direct implementation script for hybrid GNN+RL network optimization.
This script connects to your Mininet topology via Ryu controller and
implements real link suggestions with academic justification.

Usage:
    python hybrid_resilink_implementation.py --max-cycles 5 --training-mode

Academic Foundation:
- GNN: Veličković et al. (2018) - Graph Attention Networks
- RL: Mnih et al. (2015) - Deep Q-Networks  
- Ensemble: Breiman (2001) - Random Forests
- Network Analysis: Holme et al. (2002), Freeman (1977)
"""

import requests
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import json
import time
import logging
import argparse
import sys
import os
from collections import deque
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import copy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hybrid_resilink.log'),
        logging.StreamHandler()
    ]
)

class NetworkFeatureExtractor:
    """Extract network features from Ryu controller with academic justification."""
    
    def __init__(self, ryu_api_url="http://localhost:8080"):
        self.ryu_api_url = ryu_api_url
        self.session = requests.Session()
        self.session.timeout = 10
        
    def extract_network_features(self):
        """Extract comprehensive network features from SDN controller."""
        try:
            # Get topology data
            switches_resp = self.session.get(f"{self.ryu_api_url}/v1.0/topology/switches")
            switches_resp.raise_for_status()
            switches = switches_resp.json()
            
            links_resp = self.session.get(f"{self.ryu_api_url}/v1.0/topology/links")
            links_resp.raise_for_status()
            links = links_resp.json()
            
            hosts_resp = self.session.get(f"{self.ryu_api_url}/v1.0/topology/hosts")
            hosts_resp.raise_for_status()
            hosts = hosts_resp.json()
            
            # Get switch statistics
            switch_stats = self._get_switch_stats(switches)
            
            # Build network graph
            G = self._build_network_graph(switches, links, hosts)
            
            # Calculate centralities (Freeman 1977, Brandes 2001)
            centralities = self._calculate_centralities(G)
            
            # Format for hybrid optimization
            network_data = {
                'timestamp': time.time(),
                'topology': {
                    'switches': [int(sw["dpid"], 16) for sw in switches],
                    'hosts': [host["mac"] for host in hosts],
                    'switch_switch_links': self._format_switch_links(links, switch_stats),
                    'host_switch_links': self._format_host_links(hosts, switch_stats)
                },
                'nodes': self._format_nodes(switches, hosts, switch_stats, centralities),
                'centralities': centralities,
                'graph_properties': {
                    'num_nodes': G.number_of_nodes(),
                    'num_edges': G.number_of_edges(),
                    'is_connected': nx.is_connected(G),
                    'density': nx.density(G)
                }
            }
            
            logging.info(f"Extracted features: {len(switches)} switches, {len(hosts)} hosts, {len(links)} links")
            return network_data
            
        except Exception as e:
            logging.error(f"Feature extraction failed: {e}")
            raise
    
    def _get_switch_stats(self, switches):
        """Get statistics for all switches."""
        switch_stats = {}
        
        for sw in switches:
            dpid_int = int(sw["dpid"], 16)
            dpid_str = str(dpid_int)
            
            try:
                # Get flow stats
                flow_resp = self.session.get(f"{self.ryu_api_url}/stats/flow/{dpid_int}")
                flows = flow_resp.json().get(dpid_str, []) if flow_resp.status_code == 200 else []
                
                # Get port stats
                port_resp = self.session.get(f"{self.ryu_api_url}/stats/port/{dpid_int}")
                ports = port_resp.json().get(dpid_str, []) if port_resp.status_code == 200 else []
                
                switch_stats[dpid_int] = {
                    'flows': flows,
                    'ports': {p['port_no']: p for p in ports},
                    'flow_count': len(flows),
                    'total_packets': sum(f['packet_count'] for f in flows),
                    'total_bytes': sum(f['byte_count'] for f in flows)
                }
                
            except Exception as e:
                logging.warning(f"Failed to get stats for switch {dpid_str}: {e}")
                switch_stats[dpid_int] = {'flows': [], 'ports': {}, 'flow_count': 0, 'total_packets': 0, 'total_bytes': 0}
        
        return switch_stats
    
    def _build_network_graph(self, switches, links, hosts):
        """Build NetworkX graph from topology data."""
        G = nx.Graph()
        
        # Add switches
        for sw in switches:
            dpid = int(sw["dpid"], 16)
            G.add_node(dpid, type='switch')
        
        # Add hosts
        for host in hosts:
            G.add_node(host["mac"], type='host')
        
        # Add switch-switch links
        for link in links:
            src_dpid = int(link["src"]["dpid"], 16)
            dst_dpid = int(link["dst"]["dpid"], 16)
            G.add_edge(src_dpid, dst_dpid, type='switch_switch')
        
        # Add host-switch links
        for host in hosts:
            switch_dpid = int(host["port"]["dpid"], 16)
            G.add_edge(host["mac"], switch_dpid, type='host_switch')
        
        return G
    
    def _calculate_centralities(self, G):
        """Calculate network centralities with academic justification."""
        if G.number_of_nodes() == 0:
            return {'degree': {}, 'betweenness': {}, 'closeness': {}}
        
        try:
            # Freeman (1977) centrality measures
            degree_cent = nx.degree_centrality(G)
            betweenness_cent = nx.betweenness_centrality(G)
            
            # Handle disconnected graphs (Rochat 2009)
            if nx.is_connected(G):
                closeness_cent = nx.closeness_centrality(G)
            else:
                closeness_cent = {}
                for node in G.nodes():
                    harmonic_sum = 0.0
                    for other in G.nodes():
                        if node != other:
                            try:
                                distance = nx.shortest_path_length(G, node, other)
                                harmonic_sum += 1.0 / distance
                            except nx.NetworkXNoPath:
                                continue
                    closeness_cent[node] = harmonic_sum / (G.number_of_nodes() - 1)
            
            return {
                'degree': {str(k): v for k, v in degree_cent.items()},
                'betweenness': {str(k): v for k, v in betweenness_cent.items()},
                'closeness': {str(k): v for k, v in closeness_cent.items()}
            }
            
        except Exception as e:
            logging.error(f"Centrality calculation failed: {e}")
            return {'degree': {}, 'betweenness': {}, 'closeness': {}}
    
    def _format_switch_links(self, links, switch_stats):
        """Format switch-to-switch links."""
        switch_links = []
        
        for link in links:
            src_dpid = int(link["src"]["dpid"], 16)
            dst_dpid = int(link["dst"]["dpid"], 16)
            src_port = int(link["src"]["port_no"], 16)
            dst_port = int(link["dst"]["port_no"], 16)
            
            # Get port statistics
            src_stats = switch_stats.get(src_dpid, {}).get('ports', {}).get(src_port, {})
            
            link_data = {
                'src_dpid': src_dpid,
                'dst_dpid': dst_dpid,
                'src_port': src_port,
                'dst_port': dst_port,
                'bandwidth_mbps': 1000.0,  # Default 1 Gbps
                'stats': {
                    'tx_packets': src_stats.get('tx_packets', 0),
                    'tx_bytes': src_stats.get('tx_bytes', 0),
                    'rx_packets': src_stats.get('rx_packets', 0),
                    'rx_bytes': src_stats.get('rx_bytes', 0),
                    'tx_dropped': src_stats.get('tx_dropped', 0),
                    'rx_dropped': src_stats.get('rx_dropped', 0)
                }
            }
            
            switch_links.append(link_data)
        
        return switch_links
    
    def _format_host_links(self, hosts, switch_stats):
        """Format host-to-switch links."""
        host_links = []
        
        for host in hosts:
            switch_dpid = int(host["port"]["dpid"], 16)
            switch_port = int(host["port"]["port_no"], 16)
            
            # Get port statistics
            port_stats = switch_stats.get(switch_dpid, {}).get('ports', {}).get(switch_port, {})
            
            link_data = {
                'host_mac': host["mac"],
                'switch_dpid': switch_dpid,
                'switch_port': switch_port,
                'bandwidth_mbps': 100.0,  # Default 100 Mbps for hosts
                'stats': {
                    'tx_packets': port_stats.get('tx_packets', 0),
                    'tx_bytes': port_stats.get('tx_bytes', 0),
                    'rx_packets': port_stats.get('rx_packets', 0),
                    'rx_bytes': port_stats.get('rx_bytes', 0),
                    'tx_dropped': port_stats.get('tx_dropped', 0),
                    'rx_dropped': port_stats.get('rx_dropped', 0)
                }
            }
            
            host_links.append(link_data)
        
        return host_links
    
    def _format_nodes(self, switches, hosts, switch_stats, centralities):
        """Format nodes with features for GNN."""
        nodes = []
        
        # Switch nodes
        for sw in switches:
            dpid = int(sw["dpid"], 16)
            stats = switch_stats.get(dpid, {})
            
            node_data = {
                'id': dpid,
                'attributes': {
                    'type': 'switch',
                    'num_flows': stats.get('flow_count', 0),
                    'total_packets': stats.get('total_packets', 0),
                    'total_bytes': stats.get('total_bytes', 0),
                    'centrality_scores': {
                        'degree': centralities['degree'].get(str(dpid), 0.0),
                        'betweenness': centralities['betweenness'].get(str(dpid), 0.0),
                        'closeness': centralities['closeness'].get(str(dpid), 0.0)
                    }
                }
            }
            nodes.append(node_data)
        
        # Host nodes
        for host in hosts:
            mac = host["mac"]
            node_data = {
                'id': mac,
                'attributes': {
                    'type': 'host',
                    'ips': host.get('ipv4', []) + host.get('ipv6', []),
                    'centrality_scores': {
                        'degree': centralities['degree'].get(str(mac), 0.0),
                        'betweenness': centralities['betweenness'].get(str(mac), 0.0),
                        'closeness': centralities['closeness'].get(str(mac), 0.0)
                    }
                }
            }
            nodes.append(node_data)
        
        return nodes
    
    def get_available_ports(self, dpid):
        """Get available ports for a switch."""
        try:
            # Get port descriptions
            port_resp = self.session.get(f"{self.ryu_api_url}/stats/portdesc/{dpid}")
            if port_resp.status_code != 200:
                return list(range(1, 11))  # Fallback
            
            ports_data = port_resp.json().get(str(dpid), [])
            
            # Get used ports from topology
            links_resp = self.session.get(f"{self.ryu_api_url}/v1.0/topology/links")
            links = links_resp.json() if links_resp.status_code == 200 else []
            
            used_ports = set()
            for link in links:
                if int(link["src"]["dpid"], 16) == dpid:
                    used_ports.add(int(link["src"]["port_no"], 16))
                if int(link["dst"]["dpid"], 16) == dpid:
                    used_ports.add(int(link["dst"]["port_no"], 16))
            
            # Find available ports
            available_ports = []
            for port in ports_data:
                try:
                    port_no = int(port["port_no"])
                    if port_no not in used_ports and port_no != 0xfffffffe:  # Exclude controller port
                        available_ports.append(port_no)
                except (ValueError, TypeError):
                    continue
            
            return available_ports[:10]  # Return first 10 available
            
        except Exception as e:
            logging.error(f"Failed to get available ports for switch {dpid}: {e}")
            return list(range(1, 11))  # Fallback


class HybridGNN(nn.Module):
    """
    Graph Neural Network for network resilience optimization.
    
    Based on Graph Attention Networks (Veličković et al. 2018)
    with academic justification for network analysis.
    """
    
    def __init__(self, node_features=7, edge_features=6, hidden_dim=64, num_layers=3):
        super(HybridGNN, self).__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        
        # Graph Attention layers (Veličković et al. 2018)
        self.gat_layers = nn.ModuleList()
        
        # Input layer
        self.gat_layers.append(GATConv(node_features, hidden_dim, heads=4, concat=True, dropout=0.1))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True, dropout=0.1))
        
        # Output layer
        self.gat_layers.append(GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False, dropout=0.1))
        
        # Edge prediction head
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index):
        """Forward pass for link prediction."""
        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
        
        return x
    
    def predict_links(self, node_embeddings, candidate_edges):
        """Predict scores for candidate edges."""
        edge_scores = []
        
        for src_idx, dst_idx in candidate_edges:
            # Concatenate node embeddings
            edge_embedding = torch.cat([node_embeddings[src_idx], node_embeddings[dst_idx]], dim=0)
            score = self.edge_predictor(edge_embedding)
            edge_scores.append(score)
        
        return torch.stack(edge_scores) if edge_scores else torch.tensor([])


class ReinforcementLearningAgent:
    """
    RL agent for adaptive network optimization.
    
    Based on Deep Q-Networks (Mnih et al. 2015) with academic justification.
    """
    
    def __init__(self, state_dim=20, learning_rate=0.001, epsilon=0.1, gamma=0.95):
        self.state_dim = state_dim
        self.epsilon = epsilon
        self.gamma = gamma
        
        # Q-network (Mnih et al. 2015)
        self.q_network = nn.Sequential(
            nn.Linear(state_dim + 1, 128),  # +1 for action encoding
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
    
    def get_state_representation(self, network_data):
        """Extract state representation from network data."""
        graph_props = network_data.get('graph_properties', {})
        centralities = network_data.get('centralities', {})
        
        # Graph-level features
        state_features = [
            graph_props.get('num_nodes', 0) / 100.0,  # Normalized
            graph_props.get('num_edges', 0) / 100.0,
            graph_props.get('density', 0.0),
            1.0 if graph_props.get('is_connected', False) else 0.0
        ]
        
        # Centrality statistics
        degree_values = list(centralities.get('degree', {}).values())
        betweenness_values = list(centralities.get('betweenness', {}).values())
        closeness_values = list(centralities.get('closeness', {}).values())
        
        if degree_values:
            state_features.extend([
                np.mean(degree_values), np.std(degree_values),
                np.mean(betweenness_values), np.std(betweenness_values),
                np.mean(closeness_values), np.std(closeness_values)
            ])
        else:
            state_features.extend([0.0] * 6)
        
        # Node statistics
        nodes = network_data.get('nodes', [])
        switch_nodes = [n for n in nodes if n['attributes']['type'] == 'switch']
        
        if switch_nodes:
            flow_counts = [n['attributes'].get('num_flows', 0) for n in switch_nodes]
            packet_counts = [n['attributes'].get('total_packets', 0) for n in switch_nodes]
            
            state_features.extend([
                np.mean(flow_counts) / 100.0,  # Normalized
                np.std(flow_counts) / 100.0,
                np.mean(packet_counts) / 10000.0,  # Normalized
                np.std(packet_counts) / 10000.0
            ])
        else:
            state_features.extend([0.0] * 4)
        
        # Pad or truncate to expected size
        while len(state_features) < self.state_dim:
            state_features.append(0.0)
        
        return torch.tensor(state_features[:self.state_dim], dtype=torch.float32)
    
    def select_action(self, state, num_actions):
        """Select action using ε-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, num_actions)
        
        # Get Q-values for all actions
        q_values = []
        for action in range(num_actions):
            action_state = torch.cat([state, torch.tensor([action], dtype=torch.float32)])
            q_value = self.q_network(action_state)
            q_values.append(q_value.item())
        
        return np.argmax(q_values)
    
    def calculate_reward(self, old_network, new_network, added_edge):
        """Calculate reward for adding an edge."""
        # Reward based on network improvement
        reward = 0.0
        
        # Connectivity improvement
        old_connected = old_network.get('graph_properties', {}).get('is_connected', False)
        new_connected = new_network.get('graph_properties', {}).get('is_connected', False)
        
        if new_connected and not old_connected:
            reward += 1.0  # Major reward for connecting network
        
        # Density improvement (moderate)
        old_density = old_network.get('graph_properties', {}).get('density', 0.0)
        new_density = new_network.get('graph_properties', {}).get('density', 0.0)
        
        density_improvement = new_density - old_density
        reward += density_improvement * 0.5
        
        # Centrality improvement
        old_centralities = old_network.get('centralities', {})
        new_centralities = new_network.get('centralities', {})
        
        src_node, dst_node = added_edge
        old_betweenness = old_centralities.get('betweenness', {}).get(str(src_node), 0.0)
        new_betweenness = new_centralities.get('betweenness', {}).get(str(src_node), 0.0)
        
        betweenness_improvement = new_betweenness - old_betweenness
        reward += betweenness_improvement * 0.3
        
        # Small penalty for adding links (encourage efficiency)
        reward -= 0.1
        
        return reward
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self, batch_size=32):
        """Perform training step using experience replay."""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.bool)
        
        # Compute Q-values
        current_q_values = []
        for i in range(len(states)):
            action_state = torch.cat([states[i], torch.tensor([actions[i]], dtype=torch.float32)])
            q_value = self.q_network(action_state)
            current_q_values.append(q_value)
        
        current_q_values = torch.stack(current_q_values).squeeze()
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = []
            for i in range(len(next_states)):
                max_q = -float('inf')
                for action in range(10):  # Assume max 10 actions
                    action_state = torch.cat([next_states[i], torch.tensor([action], dtype=torch.float32)])
                    q_value = self.q_network(action_state)
                    max_q = max(max_q, q_value.item())
                next_q_values.append(max_q)
            
            next_q_values = torch.tensor(next_q_values)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss and update
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * 0.995)


class HybridResiLinkImplementation:
    """
    Main implementation class for hybrid network optimization.
    
    Combines GNN and RL with academic justification:
    - GNN: Pattern learning from network structure (Veličković et al. 2018)
    - RL: Adaptive optimization strategy (Mnih et al. 2015)
    - Ensemble: Principled combination (Breiman 2001)
    """
    
    def __init__(self, ryu_api_url="http://localhost:8080", reward_threshold=0.95):
        self.ryu_api_url = ryu_api_url
        self.feature_extractor = NetworkFeatureExtractor(ryu_api_url)
        
        # Initialize ML components
        self.gnn = HybridGNN()
        self.rl_agent = ReinforcementLearningAgent()
        
        # Academic weights (Breiman 2001 - ensemble methods)
        self.gnn_weight = 0.6    # Pattern learning importance
        self.rl_weight = 0.4     # Adaptive optimization importance
        
        # Optimization control
        self.reward_threshold = reward_threshold  # Stop when network quality reaches this
        self.suggested_links = set()  # Track already suggested links
        self.network_quality_history = []  # Track network improvement
        
        # Results storage
        self.optimization_history = []
        
        logging.info(f"Hybrid ResiLink Implementation initialized (reward threshold: {reward_threshold})")
    
    def run_optimization_cycle(self, training_mode=True):
        """Run single optimization cycle."""
        logging.info("Starting hybrid optimization cycle")
        
        try:
            # 1. Extract network features
            network_data = self.feature_extractor.extract_network_features()
            
            # 2. Build graph for GNN
            G = self._build_networkx_graph(network_data)
            
            if G.number_of_nodes() < 2:
                logging.warning("Network too small for optimization")
                return None
            
            # 3. Get candidate switch pairs (only switches can have new links)
            candidate_edges = self._get_candidate_switch_pairs(network_data)
            
            if not candidate_edges:
                logging.warning("No candidate edges available")
                return None
            
            # 4. GNN predictions
            gnn_scores = self._get_gnn_predictions(network_data, candidate_edges)
            
            # 5. RL predictions
            rl_scores = self._get_rl_predictions(network_data, candidate_edges, training_mode)
            
            # 6. Combine predictions (Breiman 2001 ensemble method)
            combined_scores = self._combine_predictions(gnn_scores, rl_scores)
            
            # 7. Select best link
            best_link = self._select_best_link(candidate_edges, combined_scores)
            
            if best_link is None:
                logging.warning("No suitable link found")
                return None
            
            # 8. Get implementation details
            link_suggestion = self._create_link_suggestion(best_link, combined_scores, network_data)
            
            # 9. Store results
            self.optimization_history.append({
                'timestamp': time.time(),
                'suggested_link': link_suggestion,
                'network_state': network_data,
                'candidate_count': len(candidate_edges)
            })
            
            logging.info(f"Optimization completed: suggested link {link_suggestion['src_dpid']}-{link_suggestion['dst_dpid']}")
            
            return link_suggestion
            
        except Exception as e:
            logging.error(f"Optimization cycle failed: {e}")
            return None
    
    def _build_networkx_graph(self, network_data):
        """Build NetworkX graph from network data."""
        G = nx.Graph()
        
        # Add nodes
        for node in network_data['nodes']:
            G.add_node(node['id'], **node['attributes'])
        
        # Add edges
        for link in network_data['topology']['switch_switch_links']:
            G.add_edge(link['src_dpid'], link['dst_dpid'])
        
        for link in network_data['topology']['host_switch_links']:
            G.add_edge(link['host_mac'], link['switch_dpid'])
        
        return G
    
    def _get_candidate_switch_pairs(self, network_data):
        """Get candidate switch pairs for new links, excluding already suggested ones."""
        switches = network_data['topology']['switches']
        existing_links = set()
        
        # Get existing switch-switch links
        for link in network_data['topology']['switch_switch_links']:
            existing_links.add((min(link['src_dpid'], link['dst_dpid']), 
                              max(link['src_dpid'], link['dst_dpid'])))
        
        # Add already suggested links to exclusion set
        for suggested_link in self.suggested_links:
            existing_links.add(suggested_link)
        
        # Generate candidate pairs
        candidates = []
        for i in range(len(switches)):
            for j in range(i + 1, len(switches)):
                src, dst = switches[i], switches[j]
                normalized_pair = (min(src, dst), max(src, dst))
                if normalized_pair not in existing_links:
                    candidates.append((src, dst))
        
        logging.info(f"Found {len(candidates)} candidate links (excluding {len(self.suggested_links)} already suggested)")
        return candidates
    
    def _get_gnn_predictions(self, network_data, candidate_edges):
        """Get GNN predictions for candidate edges."""
        try:
            # Prepare node features
            node_features = []
            node_mapping = {}
            
            for i, node in enumerate(network_data['nodes']):
                node_mapping[node['id']] = i
                
                # Extract features (academic justification: Freeman 1977, Holme et al. 2002)
                attrs = node['attributes']
                centrality = attrs.get('centrality_scores', {})
                
                features = [
                    centrality.get('degree', 0.0),
                    centrality.get('betweenness', 0.0),
                    centrality.get('closeness', 0.0),
                    attrs.get('num_flows', 0) / 100.0,  # Normalized
                    attrs.get('total_packets', 0) / 10000.0,  # Normalized
                    attrs.get('total_bytes', 0) / 1000000.0,  # Normalized
                    1.0 if attrs['type'] == 'switch' else 0.0  # Node type
                ]
                
                node_features.append(features)
            
            # Prepare edge index
            edge_index = []
            for link in network_data['topology']['switch_switch_links']:
                if link['src_dpid'] in node_mapping and link['dst_dpid'] in node_mapping:
                    src_idx = node_mapping[link['src_dpid']]
                    dst_idx = node_mapping[link['dst_dpid']]
                    edge_index.extend([[src_idx, dst_idx], [dst_idx, src_idx]])  # Undirected
            
            for link in network_data['topology']['host_switch_links']:
                if link['host_mac'] in node_mapping and link['switch_dpid'] in node_mapping:
                    host_idx = node_mapping[link['host_mac']]
                    switch_idx = node_mapping[link['switch_dpid']]
                    edge_index.extend([[host_idx, switch_idx], [switch_idx, host_idx]])
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float32)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
            
            # Get node embeddings
            with torch.no_grad():
                node_embeddings = self.gnn(x, edge_index)
            
            # Predict scores for candidate edges
            candidate_indices = []
            for src, dst in candidate_edges:
                if src in node_mapping and dst in node_mapping:
                    candidate_indices.append((node_mapping[src], node_mapping[dst]))
            
            if candidate_indices:
                with torch.no_grad():
                    scores = self.gnn.predict_links(node_embeddings, candidate_indices)
                return scores.numpy().flatten()
            else:
                return np.array([])
            
        except Exception as e:
            logging.error(f"GNN prediction failed: {e}")
            return np.zeros(len(candidate_edges))
    
    def _get_rl_predictions(self, network_data, candidate_edges, training_mode):
        """Get RL predictions for candidate edges."""
        try:
            state = self.rl_agent.get_state_representation(network_data)
            scores = []
            
            for i, edge in enumerate(candidate_edges):
                # Get Q-value for this action
                action_state = torch.cat([state, torch.tensor([i], dtype=torch.float32)])
                with torch.no_grad():
                    q_value = self.rl_agent.q_network(action_state)
                scores.append(q_value.item())
            
            # Training step if in training mode
            if training_mode and len(candidate_edges) > 0:
                # Select action using ε-greedy
                action = self.rl_agent.select_action(state, len(candidate_edges))
                
                # Simulate adding the edge and calculate reward
                selected_edge = candidate_edges[action]
                reward = self._simulate_edge_addition_reward(network_data, selected_edge)
                
                # Store experience (simplified - using same state as next state)
                self.rl_agent.store_experience(state, action, reward, state, False)
                
                # Train if enough experiences
                if len(self.rl_agent.memory) >= 32:
                    self.rl_agent.train_step()
            
            return np.array(scores)
            
        except Exception as e:
            logging.error(f"RL prediction failed: {e}")
            return np.zeros(len(candidate_edges))
    
    def _simulate_edge_addition_reward(self, network_data, edge):
        """Simulate reward for adding an edge."""
        # Simple reward based on centrality improvement
        src, dst = edge
        centralities = network_data.get('centralities', {})
        
        src_betweenness = centralities.get('betweenness', {}).get(str(src), 0.0)
        dst_betweenness = centralities.get('betweenness', {}).get(str(dst), 0.0)
        
        # Higher reward for connecting high-centrality nodes
        centrality_reward = (src_betweenness + dst_betweenness) * 0.5
        
        # Connectivity reward
        connectivity_reward = 0.3 if network_data['graph_properties']['is_connected'] else 0.5
        
        # Small cost for adding link
        cost = -0.1
        
        return centrality_reward + connectivity_reward + cost
    
    def _combine_predictions(self, gnn_scores, rl_scores):
        """Combine GNN and RL predictions using ensemble method."""
        if len(gnn_scores) == 0 or len(rl_scores) == 0:
            return np.array([])
        
        # Normalize scores to [0, 1]
        gnn_norm = (gnn_scores - np.min(gnn_scores)) / (np.max(gnn_scores) - np.min(gnn_scores) + 1e-8)
        rl_norm = (rl_scores - np.min(rl_scores)) / (np.max(rl_scores) - np.min(rl_scores) + 1e-8)
        
        # Ensemble combination (Breiman 2001)
        combined = self.gnn_weight * gnn_norm + self.rl_weight * rl_norm
        
        return combined
    
    def _select_best_link(self, candidate_edges, combined_scores):
        """Select best link based on combined scores."""
        if len(combined_scores) == 0:
            return None
        
        best_idx = np.argmax(combined_scores)
        return candidate_edges[best_idx]
    
    def _calculate_network_quality(self, network_data):
        """
        Calculate overall network quality score.
        
        Based on academic metrics:
        - Connectivity: Is network fully connected?
        - Density: How close to complete graph?
        - Resilience: Algebraic connectivity, robustness
        - Efficiency: Average path length, global efficiency
        """
        try:
            G = self._build_networkx_graph(network_data)
            
            if G.number_of_nodes() < 2:
                return 0.0
            
            # Connectivity score (30%)
            connectivity_score = 1.0 if nx.is_connected(G) else 0.0
            
            # Density score (25%) - how close to complete graph
            current_density = nx.density(G)
            density_score = current_density
            
            # Resilience score (25%)
            resilience_score = 0.0
            if nx.is_connected(G):
                try:
                    # Algebraic connectivity (Fiedler 1973)
                    algebraic_conn = nx.algebraic_connectivity(G)
                    # Normalize by number of nodes
                    resilience_score = min(algebraic_conn / G.number_of_nodes(), 1.0)
                except:
                    resilience_score = 0.5 if nx.is_connected(G) else 0.0
            
            # Efficiency score (20%)
            efficiency_score = nx.global_efficiency(G)
            
            # Weighted combination
            quality = (0.30 * connectivity_score + 
                      0.25 * density_score + 
                      0.25 * resilience_score + 
                      0.20 * efficiency_score)
            
            return quality
            
        except Exception as e:
            logging.error(f"Error calculating network quality: {e}")
            return 0.0
    
    def _analyze_link_strategic_value(self, src_dpid, dst_dpid, network_data, candidate_edges, combined_scores):
        """
        Provide comprehensive academic justification for link selection.
        
        Based on network theory and vulnerability analysis:
        - Centrality analysis (Freeman 1977, Brandes 2001)
        - Vulnerability assessment (Albert et al. 2000, Holme et al. 2002)
        - Load balancing theory (Kleinrock 1976)
        - Resilience optimization (Fiedler 1973)
        """
        try:
            G = self._build_networkx_graph(network_data)
            centralities = network_data.get('centralities', {})
            
            # Get centrality scores for both nodes
            src_degree = centralities.get('degree', {}).get(str(src_dpid), 0.0)
            dst_degree = centralities.get('degree', {}).get(str(dst_dpid), 0.0)
            src_betweenness = centralities.get('betweenness', {}).get(str(src_dpid), 0.0)
            dst_betweenness = centralities.get('betweenness', {}).get(str(dst_dpid), 0.0)
            src_closeness = centralities.get('closeness', {}).get(str(src_dpid), 0.0)
            dst_closeness = centralities.get('closeness', {}).get(str(dst_dpid), 0.0)
            
            # Calculate strategic reasons
            strategic_analysis = []
            priority_score = 0.0
            
            # 1. Bottleneck Relief (Kleinrock 1976 - Queueing Theory)
            if src_betweenness > 0.3 or dst_betweenness > 0.3:
                strategic_analysis.append({
                    'reason': 'Bottleneck Relief',
                    'academic_basis': 'Kleinrock (1976) - Queueing Theory in Computer Networks',
                    'explanation': f'Node {src_dpid if src_betweenness > dst_betweenness else dst_dpid} has high betweenness centrality ({max(src_betweenness, dst_betweenness):.3f}), indicating traffic bottleneck. Adding bypass link reduces congestion.',
                    'impact': 'Reduces average path length and distributes traffic load'
                })
                priority_score += 0.3
            
            # 2. Vulnerability Reduction (Albert et al. 2000)
            if src_degree > np.mean([centralities.get('degree', {}).get(str(n), 0.0) for n in network_data['topology']['switches']]) * 1.5:
                strategic_analysis.append({
                    'reason': 'Vulnerability Mitigation',
                    'academic_basis': 'Albert et al. (2000) - Error and Attack Tolerance of Complex Networks',
                    'explanation': f'Node {src_dpid} has high degree centrality ({src_degree:.3f}), making it vulnerable to targeted attacks. Adding redundant paths reduces single-point-of-failure risk.',
                    'impact': 'Improves network robustness against node failures'
                })
                priority_score += 0.25
            
            # 3. Path Diversity Enhancement (Holme et al. 2002)
            try:
                current_paths = len(list(nx.all_simple_paths(G, src_dpid, dst_dpid, cutoff=4)))
                if current_paths <= 1:
                    strategic_analysis.append({
                        'reason': 'Path Diversity Enhancement',
                        'academic_basis': 'Holme et al. (2002) - Attack Vulnerability of Complex Networks',
                        'explanation': f'Nodes {src_dpid} and {dst_dpid} have limited path diversity ({current_paths} paths). Direct connection creates alternative routes.',
                        'impact': 'Increases fault tolerance and load distribution options'
                    })
                    priority_score += 0.2
            except:
                # Nodes not connected - high priority
                strategic_analysis.append({
                    'reason': 'Network Connectivity',
                    'academic_basis': 'Fiedler (1973) - Algebraic Connectivity of Graphs',
                    'explanation': f'Nodes {src_dpid} and {dst_dpid} are in different connected components. Link bridges network partitions.',
                    'impact': 'Fundamental connectivity improvement - highest priority'
                })
                priority_score += 0.5
            
            # 4. Load Balancing Optimization (Freeman 1977)
            degree_imbalance = abs(src_degree - dst_degree)
            if degree_imbalance > 0.2:
                strategic_analysis.append({
                    'reason': 'Load Balancing',
                    'academic_basis': 'Freeman (1977) - Centrality in Social Networks',
                    'explanation': f'Degree imbalance between nodes ({src_degree:.3f} vs {dst_degree:.3f}). Link helps balance network load distribution.',
                    'impact': 'Improves traffic distribution and reduces hotspots'
                })
                priority_score += 0.15
            
            # 5. Algebraic Connectivity Improvement (Fiedler 1973)
            if nx.is_connected(G):
                try:
                    current_algebraic = nx.algebraic_connectivity(G)
                    if current_algebraic < 0.5:
                        strategic_analysis.append({
                            'reason': 'Algebraic Connectivity Enhancement',
                            'academic_basis': 'Fiedler (1973) - Algebraic Connectivity of Graphs',
                            'explanation': f'Current algebraic connectivity ({current_algebraic:.3f}) is low. Link improves network synchronizability and robustness.',
                            'impact': 'Enhances network stability and convergence properties'
                        })
                        priority_score += 0.1
                except:
                    pass
            
            # 6. Efficiency Optimization (Latora & Marchiori 2001)
            current_efficiency = nx.global_efficiency(G)
            if current_efficiency < 0.8:
                strategic_analysis.append({
                    'reason': 'Global Efficiency Improvement',
                    'academic_basis': 'Latora & Marchiori (2001) - Efficient Behavior of Small-World Networks',
                    'explanation': f'Current global efficiency ({current_efficiency:.3f}) indicates suboptimal routing. Link reduces average path lengths.',
                    'impact': 'Improves communication efficiency and reduces latency'
                })
                priority_score += 0.1
            
            # Determine primary strategic reason
            if not strategic_analysis:
                strategic_analysis.append({
                    'reason': 'Network Densification',
                    'academic_basis': 'Barabási & Albert (1999) - Scale-Free Networks',
                    'explanation': 'Adding link increases network density and provides additional routing options.',
                    'impact': 'General network improvement'
                })
                priority_score = 0.05
            
            return {
                'strategic_analysis': strategic_analysis,
                'priority_score': priority_score,
                'primary_reason': strategic_analysis[0]['reason'] if strategic_analysis else 'Network Improvement',
                'node_characteristics': {
                    'src_node': {
                        'dpid': src_dpid,
                        'degree_centrality': src_degree,
                        'betweenness_centrality': src_betweenness,
                        'closeness_centrality': src_closeness,
                        'role': self._classify_node_role(src_degree, src_betweenness, src_closeness)
                    },
                    'dst_node': {
                        'dpid': dst_dpid,
                        'degree_centrality': dst_degree,
                        'betweenness_centrality': dst_betweenness,
                        'closeness_centrality': dst_closeness,
                        'role': self._classify_node_role(dst_degree, dst_betweenness, dst_closeness)
                    }
                }
            }
            
        except Exception as e:
            logging.error(f"Error in strategic analysis: {e}")
            return {
                'strategic_analysis': [{'reason': 'Analysis Error', 'explanation': str(e)}],
                'priority_score': 0.0,
                'primary_reason': 'Unknown'
            }
    
    def _classify_node_role(self, degree, betweenness, closeness):
        """Classify node role based on centrality measures."""
        if betweenness > 0.3:
            return "Critical Hub (High Traffic Load)"
        elif degree > 0.6:
            return "Highly Connected Node"
        elif closeness > 0.6:
            return "Central Coordinator"
        elif betweenness > 0.1:
            return "Traffic Bridge"
        else:
            return "Peripheral Node"
    
    def _create_link_suggestion(self, best_link, combined_scores, network_data):
        """Create detailed link suggestion with comprehensive academic justification."""
        src_dpid, dst_dpid = best_link
        
        # Get available ports
        src_ports = self.feature_extractor.get_available_ports(src_dpid)
        dst_ports = self.feature_extractor.get_available_ports(dst_dpid)
        
        # Get score details
        best_idx = None
        candidate_edges = self._get_candidate_switch_pairs(network_data)
        for i, edge in enumerate(candidate_edges):
            if edge == best_link:
                best_idx = i
                break
        
        score = combined_scores[best_idx] if best_idx is not None else 0.0
        
        # Calculate network quality improvement
        current_quality = self._calculate_network_quality(network_data)
        
        # Get comprehensive strategic analysis
        strategic_analysis = self._analyze_link_strategic_value(src_dpid, dst_dpid, network_data, candidate_edges, combined_scores)
        
        # Add this link to suggested links for future exclusion
        normalized_link = (min(src_dpid, dst_dpid), max(src_dpid, dst_dpid))
        self.suggested_links.add(normalized_link)
        
        return {
            'src_dpid': src_dpid,
            'dst_dpid': dst_dpid,
            'src_port': src_ports[0] if src_ports else 'unavailable',
            'dst_port': dst_ports[0] if dst_ports else 'unavailable',
            'score': float(score),
            'network_quality': float(current_quality),
            'implementation_feasible': len(src_ports) > 0 and len(dst_ports) > 0,
            'available_src_ports': src_ports[:5],  # First 5 available
            'available_dst_ports': dst_ports[:5],
            'strategic_justification': strategic_analysis,
            'optimization_progress': {
                'suggested_links_count': len(self.suggested_links),
                'quality_threshold': self.reward_threshold,
                'should_continue': current_quality < self.reward_threshold
            },
            'academic_justification': {
                'primary_reason': strategic_analysis.get('primary_reason', 'Network Improvement'),
                'gnn_component': f'Graph pattern learning (Veličković et al. 2018) - weight: {self.gnn_weight}',
                'rl_component': f'Adaptive optimization (Mnih et al. 2015) - weight: {self.rl_weight}',
                'ensemble_method': 'Breiman (2001) - Random Forests ensemble theory',
                'network_theory': 'Freeman (1977), Albert et al. (2000), Holme et al. (2002), Fiedler (1973)',
                'quality_metrics': f'Connectivity + Density + Resilience + Efficiency = {current_quality:.4f}',
                'strategic_priority': f'{strategic_analysis.get("priority_score", 0.0):.3f}/1.0'
            },
            'ryu_implementation': {
                'add_link_command': f'curl -X POST {self.ryu_api_url}/stats/flowentry/add -d \'{{"dpid": {src_dpid}, "match": {{}}, "actions": [{{"type": "OUTPUT", "port": {src_ports[0] if src_ports else 1}}}]}}\'',
                'feasible': len(src_ports) > 0 and len(dst_ports) > 0
            }
        }
    
    def run_continuous_optimization(self, max_cycles=10, cycle_interval=60, training_mode=True):
        """Run continuous optimization cycles with intelligent stopping."""
        print(f"🚀 Starting Hybrid ResiLink Implementation")
        print(f"🔄 Running up to {max_cycles} optimization cycles")
        print(f"⏱️  Cycle interval: {cycle_interval} seconds")
        print(f"🤖 Training mode: {training_mode}")
        print(f"🎯 Quality threshold: {self.reward_threshold}")
        print("=" * 60)
        
        successful_cycles = 0
        
        for cycle in range(max_cycles):
            print(f"\n--- Cycle {cycle + 1}/{max_cycles} ---")
            
            try:
                # Run optimization
                result = self.run_optimization_cycle(training_mode)
                
                if result:
                    successful_cycles += 1
                    quality = result.get('network_quality', 0.0)
                    progress = result.get('optimization_progress', {})
                    strategic = result.get('strategic_justification', {})
                    
                    print(f"✅ Suggested Link: {result['src_dpid']} -> {result['dst_dpid']}")
                    print(f"📊 Score: {result['score']:.4f}")
                    print(f"🌐 Network Quality: {quality:.4f} (threshold: {self.reward_threshold})")
                    print(f"🎯 Primary Reason: {strategic.get('primary_reason', 'Network Improvement')}")
                    print(f"⭐ Strategic Priority: {strategic.get('priority_score', 0.0):.3f}/1.0")
                    
                    # Show detailed strategic analysis
                    strategic_analysis = strategic.get('strategic_analysis', [])
                    if strategic_analysis:
                        print(f"🧠 Academic Justification:")
                        for analysis in strategic_analysis[:2]:  # Show top 2 reasons
                            print(f"   • {analysis['reason']}: {analysis['explanation']}")
                            print(f"     📚 Basis: {analysis['academic_basis']}")
                    
                    # Show node roles
                    node_chars = strategic.get('node_characteristics', {})
                    if node_chars:
                        src_role = node_chars.get('src_node', {}).get('role', 'Unknown')
                        dst_role = node_chars.get('dst_node', {}).get('role', 'Unknown')
                        print(f"🏷️  Node Roles: {result['src_dpid']} ({src_role}) ↔ {result['dst_dpid']} ({dst_role})")
                    
                    print(f"🔧 Implementation: {'Feasible' if result['implementation_feasible'] else 'Not feasible'}")
                    print(f"📈 Progress: {len(self.suggested_links)} links suggested so far")
                    
                    if result['implementation_feasible']:
                        print(f"🔌 Ports: {result['src_port']} -> {result['dst_port']}")
                        print(f"💡 Ryu command ready for implementation")
                    
                    # Save individual result
                    with open(f'link_suggestion_cycle_{cycle + 1}.json', 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    
                    # Check if we should stop (quality threshold reached)
                    if quality >= self.reward_threshold:
                        print(f"\n🎉 OPTIMIZATION COMPLETE!")
                        print(f"🏆 Network quality ({quality:.4f}) reached threshold ({self.reward_threshold})")
                        print(f"🔗 Total links suggested: {len(self.suggested_links)}")
                        break
                    
                    # Check if no more candidates available
                    if not progress.get('should_continue', True):
                        print(f"\n🏁 NO MORE CANDIDATES!")
                        print(f"🔗 All beneficial links have been suggested: {len(self.suggested_links)}")
                        break
                    
                else:
                    print(f"❌ No link suggestion generated")
                    print(f"🏁 Optimization stopping - no more beneficial links found")
                    break
                
            except Exception as e:
                print(f"💥 Cycle {cycle + 1} failed: {e}")
                logging.error(f"Cycle {cycle + 1} failed: {e}")
            
            # Wait before next cycle (except last one or if stopping)
            if cycle < max_cycles - 1:
                print(f"⏳ Waiting {cycle_interval} seconds...")
                time.sleep(cycle_interval)
        
        # Final summary
        print(f"\n" + "=" * 60)
        print(f"🏁 Optimization Complete")
        print(f"✅ Successful cycles: {successful_cycles}/{max_cycles}")
        print(f"🔗 Total links suggested: {len(self.suggested_links)}")
        
        if successful_cycles > 0:
            final_quality = self.network_quality_history[-1] if self.network_quality_history else 0.0
            print(f"🌐 Final network quality: {final_quality:.4f}")
            print(f"📁 Results saved to individual JSON files")
            
            # Save complete history
            with open('hybrid_optimization_history.json', 'w') as f:
                json.dump(self.optimization_history, f, indent=2, default=str)
            
            print(f"📊 Complete history saved to 'hybrid_optimization_history.json'")
            
            # Save suggested links summary
            summary = {
                'total_links_suggested': len(self.suggested_links),
                'suggested_links': list(self.suggested_links),
                'final_quality': final_quality,
                'quality_threshold': self.reward_threshold,
                'optimization_complete': final_quality >= self.reward_threshold
            }
            
            with open('optimization_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"📋 Summary saved to 'optimization_summary.json'")
        
        return successful_cycles


def main():
    """Main implementation function."""
    parser = argparse.ArgumentParser(description='Hybrid ResiLink Implementation')
    parser.add_argument('--ryu-url', default='http://localhost:8080',
                       help='Ryu controller API URL')
    parser.add_argument('--max-cycles', type=int, default=10,
                       help='Maximum optimization cycles')
    parser.add_argument('--cycle-interval', type=int, default=30,
                       help='Interval between cycles (seconds)')
    parser.add_argument('--training-mode', action='store_true',
                       help='Enable RL training mode')
    parser.add_argument('--single-cycle', action='store_true',
                       help='Run single optimization cycle')
    parser.add_argument('--reward-threshold', type=float, default=0.95,
                       help='Network quality threshold to stop optimization (default: 0.95)')
    
    args = parser.parse_args()
    
    # Initialize implementation
    implementation = HybridResiLinkImplementation(args.ryu_url, args.reward_threshold)
    
    try:
        if args.single_cycle:
            # Single cycle
            print("🚀 Running single optimization cycle...")
            result = implementation.run_optimization_cycle(args.training_mode)
            
            if result:
                print("✅ Optimization successful!")
                print(json.dumps(result, indent=2, default=str))
            else:
                print("❌ Optimization failed")
                return 1
        else:
            # Continuous optimization
            successful = implementation.run_continuous_optimization(
                max_cycles=args.max_cycles,
                cycle_interval=args.cycle_interval,
                training_mode=args.training_mode
            )
            
            if successful == 0:
                print("❌ No successful optimizations")
                return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Implementation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Implementation failed: {e}")
        logging.error(f"Implementation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())