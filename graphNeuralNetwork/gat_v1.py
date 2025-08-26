import json
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from sklearn.preprocessing import StandardScaler
import numpy as np

# GAT Layer modified for batching and edge features
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, alpha=0.2):
        super(GATLayer, self).__init__()
        self.lin = Linear(in_dim, out_dim)
        self.att = Linear(2 * out_dim + edge_dim, 1)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj, edge_attr):
        # x: [B, N, in_dim]
        # adj: [B, N, N]
        # edge_attr: [B, N, N, edge_dim]
        B, N, _ = x.shape
        h = self.lin(x)  # [B, N, out_dim]

        # Prepare for attention: cat(h_i, h_j, e_ij)
        h_i = h.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, out]
        h_j = h.unsqueeze(1).repeat(1, N, 1, 1)  # [B, N, N, out]
        a_input = torch.cat([h_i, h_j, edge_attr], dim=-1)  # [B, N, N, 2*out + edge]

        e = self.leakyrelu(self.att(a_input).squeeze(-1))  # [B, N, N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)  # Normalize over neighbors (dim=2 for PyTorch <2.0, but -1 is col)

        h_prime = torch.matmul(attention, h)  # [B, N, out]
        return F.elu(h_prime)

# Function to build data for one snapshot
def build_graph_data(entry):
    topology = entry['topology']
    centralities = entry['centralities']
    delta_centralities = entry['delta_centralities']
    nodes_data = {str(node['id']): node['attributes'] for node in entry['nodes']}

    # Build graph (undirected)
    G = nx.Graph()
    for node_id in nodes_data:
        G.add_node(node_id, **nodes_data[node_id])

    # Add edges with attributes
    all_links = topology['switch_switch_links'] + topology['host_switch_links']
    for link in all_links:
        if 'host_mac' in link:
            src = link['host_mac']
            dst = str(link['switch_dpid'])
        else:
            src = str(link['src_dpid'])
            dst = str(link['dst_dpid'])
        G.add_edge(src, dst, **link)

    # Node list and mapping
    node_list = list(G.nodes())
    num_nodes = len(node_list)
    node_id_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Node features (12 dims)
    features = []
    for node_id in node_list:
        deg = centralities.get('degree', {}).get(node_id, 0.0)
        bet = centralities.get('betweenness', {}).get(node_id, 0.0)
        clo = centralities.get('closeness', {}).get(node_id, 0.0)
        d_deg = delta_centralities.get('degree', {}).get(node_id, 0.0)
        d_bet = delta_centralities.get('betweenness', {}).get(node_id, 0.0)
        d_clo = delta_centralities.get('closeness', {}).get(node_id, 0.0)
        base_f = [deg, bet, clo, d_deg, d_bet, d_clo]

        attrs = nodes_data.get(node_id, {})
        if attrs.get('type') == 'switch':
            switch_f = [
                attrs.get('num_flows', 0.0),
                attrs.get('total_packets', 0.0),
                attrs.get('total_bytes', 0.0),
                attrs.get('avg_flow_duration', 0.0),
                attrs.get('delta_total_packets', 0.0),
                attrs.get('delta_total_bytes', 0.0)
            ]
        else:
            switch_f = [0.0] * 6
        f = base_f + switch_f
        features.append(f)

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features) if np.any(features) else features  # Avoid nan if all zero
    x = torch.tensor(features, dtype=torch.float)

    # Adjacency matrix (binary)
    adj_np = nx.to_numpy_array(G)
    adj = torch.tensor(adj_np, dtype=torch.float)
    adj += torch.eye(num_nodes)  # Self-loops

    # Edge attr dense [num_nodes, num_nodes, 17]
    edge_dim = 17
    edge_attr_dense = torch.zeros(num_nodes, num_nodes, edge_dim)
    stats_keys = ['tx_packets', 'tx_bytes', 'tx_dropped', 'rx_packets', 'rx_bytes', 'rx_dropped', 'duration_sec']
    delta_keys = ['delta_tx_packets', 'delta_tx_bytes', 'delta_tx_dropped', 'delta_rx_packets', 'delta_rx_bytes', 'delta_rx_dropped', 'tx_packet_rate_pps', 'tx_byte_rate_bps', 'packet_loss_rate']
    for src, dst, link_data in G.edges(data=True):
        i = node_id_to_idx[src]
        j = node_id_to_idx[dst]
        stats = link_data.get('stats', {})
        d_stats = link_data.get('delta_stats', {})
        bandwidth = link_data.get('bandwidth_mbps', 0.0)
        e_f = [bandwidth] + [stats.get(k, 0.0) for k in stats_keys] + [d_stats.get(k, 0.0) for k in delta_keys]
        edge_attr_dense[i, j] = torch.tensor(e_f)
        edge_attr_dense[j, i] = torch.tensor(e_f)  # Symmetric for undirected

    # Normalize edge_attr? Optional, skip for now as mostly 0

    return x, adj, edge_attr_dense, node_list, num_nodes

# Load data (the list of entries)
with open('network_features.json', 'r') as f:
    data_list = json.load(f)  

# Build data for each snapshot
x_list = []
adj_list = []
edge_attr_list = []
node_lists = []
num_nodes_list = []
for entry in data_list:
    x, adj, edge_attr, node_list, num_nodes = build_graph_data(entry)
    x_list.append(x)
    adj_list.append(adj)
    edge_attr_list.append(edge_attr)
    node_lists.append(node_list)
    num_nodes_list.append(num_nodes)

# Batch by padding
B = len(data_list)
max_N = max(num_nodes_list)
in_dim = x_list[0].shape[1]  # 12
edge_dim = edge_attr_list[0].shape[-1]  # 17

x_batched = torch.zeros(B, max_N, in_dim)
adj_batched = torch.zeros(B, max_N, max_N)
edge_attr_batched = torch.zeros(B, max_N, max_N, edge_dim)

for b in range(B):
    N = num_nodes_list[b]
    x_batched[b, :N] = x_list[b]
    adj_batched[b, :N, :N] = adj_list[b]
    edge_attr_batched[b, :N, :N] = edge_attr_list[b]

# Model
gat = GATLayer(in_dim=in_dim, out_dim=8, edge_dim=edge_dim)

# Forward
embeddings_batched = gat(x_batched, adj_batched, edge_attr_batched)

# Output per snapshot (slice to real nodes)
for b in range(B):
    N = num_nodes_list[b]
    emb = embeddings_batched[b, :N]
    print(f"Snapshot {b+1} ({data_list[b]['timestamp']}):")
    for i, node in enumerate(node_lists[b]):
        print(f"{node}: {emb[i].tolist()}")
