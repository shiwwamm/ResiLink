import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy

def build_graph(data):
    G = nx.Graph()
    # Add nodes with attributes
    for node in data['nodes']:
        G.add_node(node['id'], **node['attributes'])
    
    # Add switch-switch links (undirected, avoid duplicates)
    added_edges = set()
    for link in data['topology']['switch_switch_links']:
        src = link['src_dpid']
        dst = link['dst_dpid']
        edge = tuple(sorted([src, dst]))
        if edge not in added_edges:
            G.add_edge(src, dst, bandwidth=link['bandwidth_mbps'])
            added_edges.add(edge)
    
    # Add host-switch links
    for link in data['topology']['host_switch_links']:
        host = link['host_mac']
        sw = link['switch_dpid']
        G.add_edge(host, sw, bandwidth=link['bandwidth_mbps'])
    
    return G

def compute_metrics(G):
    metrics = {}
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['average_degree'] = sum(dict(G.degree()).values()) / metrics['num_nodes']
    metrics['diameter'] = nx.diameter(G) if nx.is_connected(G) else float('inf')
    metrics['avg_shortest_path'] = nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf')
    metrics['clustering_coeff'] = nx.average_clustering(G)
    metrics['assortativity'] = nx.degree_assortativity_coefficient(G)
    metrics['node_connectivity'] = nx.node_connectivity(G)
    metrics['edge_connectivity'] = nx.edge_connectivity(G)
    metrics['algebraic_connectivity'] = nx.algebraic_connectivity(G)
    
    # Centralities
    metrics['degree_centrality'] = nx.degree_centrality(G)
    metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
    metrics['closeness_centrality'] = nx.closeness_centrality(G)
    
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
    nodes = [str(n) for n in cent_dict.keys()]  # Convert to str for labels
    values = list(cent_dict.values())
    plt.bar(nodes, values)
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
        # Compute current betweenness for adaptive attack
        between = nx.betweenness_centrality(H)
        # Sort remaining nodes by betweenness descending
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
    
    plt.figure(figsize=(8, 6))
    plt.plot(frac_removed, frac_giant, marker='o')
    plt.title(f'Robustness Curve - {title}')
    plt.xlabel('Fraction of Nodes Removed (Targeted by Betweenness)')
    plt.ylabel('Fraction of Nodes in Giant Component')
    plt.grid(True)
    plt.savefig(f'robustness_{title.lower()}.png')
    plt.close()

def main():
    # Load data
    with open('network_features.json', 'r') as f:
        data_before = json.load(f)
    with open('final_network_features.json', 'r') as f:
        data_after = json.load(f)
    
    # Build graphs
    G_before = build_graph(data_before)
    G_after = build_graph(data_after)
    
    # Compute metrics
    metrics_before = compute_metrics(G_before)
    metrics_after = compute_metrics(G_after)
    
    # Print scalar metrics for comparison
    print("Scalar Metrics Comparison:")
    scalar_metrics = [
        'num_nodes', 'num_edges', 'average_degree', 'diameter', 'avg_shortest_path',
        'clustering_coeff', 'assortativity', 'node_connectivity', 'edge_connectivity',
        'algebraic_connectivity'
    ]
    for m in scalar_metrics:
        print(f"{m}: Before = {metrics_before[m]}, After = {metrics_after[m]}")
    
    # Generate network visualizations
    plot_network(G_before, 'Before')
    plot_network(G_after, 'After')
    
    # Generate centrality plots (separate for before and after)
    centralities = ['Degree Centrality', 'Betweenness Centrality', 'Closeness Centrality']
    cent_keys = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']
    for cent_name, cent_key in zip(centralities, cent_keys):
        plot_centrality(metrics_before[cent_key], 'Before', cent_name)
        plot_centrality(metrics_after[cent_key], 'After', cent_name)
    
    # Generate robustness curves (separate for before and after)
    plot_robustness(G_before, 'Before')
    plot_robustness(G_after, 'After')

if __name__ == '__main__':
    main()
