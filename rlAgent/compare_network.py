import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
import networkx.algorithms.community as nx_comm

def build_graph(data):
    G = nx.Graph()
    # Add nodes with attributes, converting node IDs to strings
    for node in data['nodes']:
        node_id = str(node['id'])  # Convert all node IDs to strings
        G.add_node(node_id, **node['attributes'])
    
    # Add switch-switch links (undirected, avoid duplicates)
    added_edges = set()
    for link in data['topology']['switch_switch_links']:
        src = str(link['src_dpid'])  # Convert to string
        dst = str(link['dst_dpid'])  # Convert to string
        edge = tuple(sorted([src, dst]))
        if edge not in added_edges:
            G.add_edge(src, dst, bandwidth=link['bandwidth_mbps'], stats=link['stats'])
            added_edges.add(edge)
    
    # Add host-switch links
    for link in data['topology']['host_switch_links']:
        host = str(link['host_mac'])  # Already a string, but ensure consistency
        sw = str(link['switch_dpid'])  # Convert to string
        G.add_edge(host, sw, bandwidth=link['bandwidth_mbps'], stats=link['stats'])
    
    return G

def compute_metrics(G, data):
    metrics = {}
    
    # Basic topological metrics
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['average_degree'] = sum(dict(G.degree()).values()) / metrics['num_nodes']
    metrics['density'] = nx.density(G)
    metrics['diameter'] = nx.diameter(G) if nx.is_connected(G) else float('inf')
    metrics['avg_shortest_path'] = nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')
    metrics['clustering_coeff'] = nx.average_clustering(G)
    metrics['assortativity'] = nx.degree_assortativity_coefficient(G)
    
    # Connectivity metrics
    metrics['node_connectivity'] = nx.node_connectivity(G)
    metrics['edge_connectivity'] = nx.edge_connectivity(G)
    try:
        metrics['algebraic_connectivity'] = nx.algebraic_connectivity(G)
    except nx.NetworkXError:
        metrics['algebraic_connectivity'] = 0.0  # Handle disconnected graphs
    
    try:
        metrics['average_node_connectivity'] = nx.average_node_connectivity(G)
    except nx.NetworkXError:
        metrics['average_node_connectivity'] = 0.0  # Handle disconnected graphs
    
    # Community and modularity
    try:
        communities = nx_comm.greedy_modularity_communities(G)
        metrics['modularity'] = nx_comm.modularity(G, communities)
        metrics['num_communities'] = len(communities)
    except Exception as e:
        print(f"Warning: Failed to compute modularity: {e}")
        metrics['modularity'] = 0.0
        metrics['num_communities'] = 1  # Assume single community if fails
    
    # Centralities
    metrics['degree_centrality'] = nx.degree_centrality(G)
    metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
    metrics['closeness_centrality'] = nx.closeness_centrality(G)
    try:
        metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        metrics['eigenvector_centrality'] = {n: 0.0 for n in G.nodes()}  # Fallback
    
    # Redundancy metrics (average number of node-disjoint paths between hosts)
    hosts = [n for n in G.nodes() if G.nodes[n]['type'] == 'host']
    if len(hosts) > 1:
        path_diversities = []
        for i in range(len(hosts)):
            for j in range(i+1, len(hosts)):
                try:
                    num_disjoint_paths = len(list(nx.node_disjoint_paths(G, hosts[i], hosts[j])))
                    path_diversities.append(num_disjoint_paths)
                except nx.NetworkXNoPath:
                    path_diversities.append(0)
        metrics['avg_path_diversity'] = np.mean(path_diversities) if path_diversities else 0
    else:
        metrics['avg_path_diversity'] = 0
    
    # SDN-specific metrics from node attributes (switches)
    switch_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'switch']
    if switch_nodes:
        metrics['avg_num_flows'] = np.mean([G.nodes[n]['num_flows'] for n in switch_nodes])
        metrics['avg_total_packets'] = np.mean([G.nodes[n]['total_packets'] for n in switch_nodes])
        metrics['avg_total_bytes'] = np.mean([G.nodes[n]['total_bytes'] for n in switch_nodes])
        metrics['avg_flow_duration'] = np.mean([G.nodes[n]['avg_flow_duration'] for n in switch_nodes])
    
    # SDN-specific from links (aggregate stats)
    link_stats = [e[2]['stats'] for e in G.edges(data=True) if 'stats' in e[2]]
    if link_stats:
        metrics['total_tx_packets'] = sum(s['tx_packets'] for s in link_stats)
        metrics['total_rx_packets'] = sum(s['rx_packets'] for s in link_stats)
        metrics['total_tx_bytes'] = sum(s['tx_bytes'] for s in link_stats)
        metrics['total_rx_bytes'] = sum(s['rx_bytes'] for s in link_stats)
        metrics['total_tx_dropped'] = sum(s['tx_dropped'] for s in link_stats)
        metrics['total_rx_dropped'] = sum(s['rx_dropped'] for s in link_stats)
        metrics['avg_packet_loss_rate'] = (
            (metrics['total_tx_dropped'] + metrics['total_rx_dropped']) /
            (metrics['total_tx_packets'] + metrics['total_rx_packets'] + 1e-10) * 100
        )  # Avoid division by zero
    
    # Controller metrics
    metrics['controller_cpu'] = data['controller_data']['cpu_percent']
    metrics['controller_memory'] = data['controller_data']['memory_percent']
    
    # Fault tolerance: Number of single points of failure (articulation points)
    try:
        metrics['num_articulation_points'] = len(list(nx.articulation_points(G)))
    except nx.NetworkXError:
        metrics['num_articulation_points'] = 0  # Handle if not applicable
    
    return metrics

def plot_network(G, title):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    node_colors = ['blue' if G.nodes[n]['type'] == 'switch' else 'red' for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=10)
    plt.title(f'Network Visualization - {title}')
    plt.savefig(f'network_{title.lower()}.png')
    plt.close()

def plot_centrality(cent_dict, title, metric_name):
    plt.figure(figsize=(12, 6))
    nodes = [str(n) for n in cent_dict.keys()]
    values = list(cent_dict.values())
    plt.bar(nodes, values, color='#1f77b4')
    plt.title(f'{metric_name} - {title}')
    plt.xlabel('Nodes')
    plt.ylabel(f'{metric_name} Value')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'{metric_name.lower().replace(" ", "_")}_{title.lower()}.png')
    plt.close()

def plot_robustness(G, title):
    H = copy.deepcopy(G)
    n = H.number_of_nodes()
    frac_removed = []
    frac_giant = []
    nodes_remaining = list(H.nodes())
    
    while len(nodes_remaining) > 1:
        try:
            between = nx.betweenness_centrality(H)
            nodes_sorted = sorted(nodes_remaining, key=lambda x: between.get(x, 0), reverse=True)
            if not nodes_sorted:
                break
            node_to_remove = nodes_sorted[0]
            H.remove_node(node_to_remove)
            nodes_remaining.remove(node_to_remove)
            
            removed_frac = (n - H.number_of_nodes()) / n
            frac_removed.append(removed_frac)
            
            if H.number_of_nodes() == 0:
                frac_giant.append(0)
                break
            else:
                if nx.is_connected(H):
                    giant_frac = 1.0
                else:
                    largest_cc = max(nx.connected_components(H), key=len)
                    giant_frac = len(largest_cc) / H.number_of_nodes()
                frac_giant.append(giant_frac)
        except nx.NetworkXError:
            break
    
    plt.figure(figsize=(8, 6))
    plt.plot(frac_removed, frac_giant, marker='o', color='#1f77b4')
    plt.title(f'Robustness Curve - {title}')
    plt.xlabel('Fraction of Nodes Removed (Targeted by Betweenness)')
    plt.ylabel('Fraction of Nodes in Giant Component')
    plt.grid(True)
    plt.savefig(f'robustness_{title.lower()}.png')
    plt.close()

def plot_scalar_metrics(metrics_before, metrics_after):
    scalar_metrics = [
        'average_degree', 'density', 'diameter', 'avg_shortest_path',
        'clustering_coeff', 'assortativity', 'node_connectivity', 'edge_connectivity',
        'algebraic_connectivity', 'average_node_connectivity', 'modularity',
        'num_communities', 'avg_path_diversity', 'num_articulation_points',
        'avg_num_flows', 'avg_total_packets', 'avg_total_bytes', 'avg_flow_duration',
        'total_tx_packets', 'total_rx_packets', 'total_tx_bytes', 'total_rx_bytes',
        'total_tx_dropped', 'total_rx_dropped', 'avg_packet_loss_rate',
        'controller_cpu', 'controller_memory'
    ]
    
    for metric in scalar_metrics:
        if metric in metrics_before and metric in metrics_after:
            plt.figure(figsize=(8, 6))
            labels = ['Before', 'After']
            values = [metrics_before[metric], metrics_after[metric]]
            plt.bar(labels, values, color=['#1f77b4', '#ff7f0e'])
            plt.title(f'{metric.replace("_", " ").title()} Comparison')
            plt.ylabel(metric.replace("_", " ").title())
            plt.tight_layout()
            plt.savefig(f'{metric.lower()}_comparison.png')
            plt.close()

def main():
    # Load data
    try:
        with open('network_features.json', 'r') as f:
            data_before = json.load(f)
        with open('final_network_features.json', 'r') as f:
            data_after = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return
    
    # Build graphs
    G_before = build_graph(data_before)
    G_after = build_graph(data_after)
    
    # Compute metrics
    metrics_before = compute_metrics(G_before, data_before)
    metrics_after = compute_metrics(G_after, data_after)
    
    # Print scalar metrics for comparison
    print("Scalar Metrics Comparison:")
    scalar_metrics = [
        'num_nodes', 'num_edges', 'average_degree', 'density', 'diameter', 'avg_shortest_path',
        'clustering_coeff', 'assortativity', 'node_connectivity', 'edge_connectivity',
        'algebraic_connectivity', 'average_node_connectivity', 'modularity', 'num_communities',
        'avg_path_diversity', 'num_articulation_points', 'avg_num_flows', 'avg_total_packets',
        'avg_total_bytes', 'avg_flow_duration', 'total_tx_packets', 'total_rx_packets',
        'total_tx_bytes', 'total_rx_bytes', 'total_tx_dropped', 'total_rx_dropped',
        'avg_packet_loss_rate', 'controller_cpu', 'controller_memory'
    ]
    for m in scalar_metrics:
        if m in metrics_before and m in metrics_after:
            print(f"{m}: Before = {metrics_before[m]}, After = {metrics_after[m]}")
    
    # Generate network visualizations
    plot_network(G_before, 'Before')
    plot_network(G_after, 'After')
    
    # Generate centrality plots (separate for before and after)
    centralities = ['Degree Centrality', 'Betweenness Centrality', 'Closeness Centrality', 'Eigenvector Centrality']
    cent_keys = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 'eigenvector_centrality']
    for cent_name, cent_key in zip(centralities, cent_keys):
        plot_centrality(metrics_before[cent_key], 'Before', cent_name)
        plot_centrality(metrics_after[cent_key], 'After', cent_name)
    
    # Generate robustness curves (separate for before and after)
    plot_robustness(G_before, 'Before')
    plot_robustness(G_after, 'After')
    
    # Generate scalar metric comparison plots
    plot_scalar_metrics(metrics_before, metrics_after)

if __name__ == '__main__':
    main()
