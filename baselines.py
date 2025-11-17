# baselines.py — FULL VERSION WITH 7 BASELINES (November 17, 2025)
# All respect degree constraint: max 8 total degree → max 4 new links per node

import networkx as nx
import random
import numpy as np
from collections import Counter
try:
    from community import community_louvain  # pip install python-louvain
except ImportError:
    community_louvain = None

def random_topology(G, max_links=10):
    """Random valid links"""
    G = G.copy()
    candidates = [(u,v) for u,v in nx.non_edges(G) if G.degree(u)<8 and G.degree(v)<8]
    random.shuffle(candidates)
    added = []
    for u,v in candidates[:max_links]:
        if G.degree(u)<8 and G.degree(v)<8:
            G.add_edge(u,v)
            added.append((u,v))
    return G, added

def degree_topology(G, max_links=10):
    """Connect lowest-degree nodes"""
    G = G.copy()
    nodes = sorted(G.nodes(), key=G.degree)
    added = []
    i = 0
    while len(added) < max_links and i + 1 < len(nodes):
        u, v = nodes[i], nodes[i+1]
        if not G.has_edge(u,v) and G.degree(u)<8 and G.degree(v)<8:
            G.add_edge(u,v)
            added.append((u,v))
        i += 2
    return G, added

def betweenness_topology(G, max_links=10):
    """Connect high-betweenness nodes"""
    G = G.copy()
    bc = nx.betweenness_centrality(G)
    candidates = [(u,v) for u,v in nx.non_edges(G) if G.degree(u)<8 and G.degree(v)<8]
    scores = [(bc[u] + bc[v], u, v) for u,v in candidates]
    scores.sort(reverse=True)
    added = []
    for _, u, v in scores[:max_links]:
        if G.degree(u)<8 and G.degree(v)<8:
            G.add_edge(u,v)
            added.append((u,v))
    return G, added

def greedy_topology(G, max_links=10):
    """Myopic min-cut improvement"""
    G = G.copy()
    orig_cut = nx.stoer_wagner(G)[0]
    candidates = [(u,v) for u,v in nx.non_edges(G) if G.degree(u)<8 and G.degree(v)<8]
    gains = []
    for u,v in candidates:
        G.add_edge(u,v)
        new_cut = nx.stoer_wagner(G)[0]
        gain = new_cut - orig_cut
        gains.append((gain, u, v))
        G.remove_edge(u,v)
    gains.sort(reverse=True)
    added = []
    for gain, u, v in gains[:max_links]:
        if G.degree(u)<8 and G.degree(v)<8:
            G.add_edge(u,v)
            added.append((u,v))
    return G, added

def shortest_path_greedy(G, max_links=10):
    """Reduce average shortest path length"""
    G = G.copy()
    orig_aspl = nx.average_shortest_path_length(G)
    candidates = [(u,v) for u,v in nx.non_edges(G) if G.degree(u)<8 and G.degree(v)<8]
    gains = []
    for u,v in candidates:
        G.add_edge(u,v)
        try:
            new_aspl = nx.average_shortest_path_length(G)
            gain = orig_aspl - new_aspl
        except:
            gain = 0
        gains.append((gain, u, v))
        G.remove_edge(u,v)
    gains.sort(reverse=True)
    added = []
    for gain, u, v in gains[:max_links]:
        if G.degree(u)<8 and G.degree(v)<8:
            G.add_edge(u,v)
            added.append((u,v))
    return G, added

def eigenvector_greedy(G, max_links=10):
    """Connect high eigenvector centrality nodes"""
    G = G.copy()
    try:
        ec = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        ec = nx.degree_centrality(G)  # fallback
    nodes = sorted(ec, key=ec.get, reverse=True)
    added = []
    i = 0
    while len(added) < max_links and i + 1 < len(nodes):
        u, v = nodes[i], nodes[i+1]
        if not G.has_edge(u,v) and G.degree(u)<8 and G.degree(v)<8:
            G.add_edge(u,v)
            added.append((u,v))
        i += 1
    return G, added

def community_bridge(G, max_links=10):
    """Bridge largest Louvain communities"""
    if community_louvain is None:
        print("python-louvain not installed → skipping community_bridge")
        return G.copy(), []
    G = G.copy()
    partition = community_louvain.best_partition(G)
    comm_count = Counter(partition.values())
    top_comms = [c for c, _ in comm_count.most_common(10)]
    added = []
    attempts = 0
    while len(added) < max_links and attempts < 1000:
        c1, c2 = random.sample(top_comms, 2)
        nodes1 = [n for n in G.nodes() if partition[n] == c1]
        nodes2 = [n for n in G.nodes() if partition[n] == c2]
        u = random.choice(nodes1)
        v = random.choice(nodes2)
        if not G.has_edge(u,v) and G.degree(u)<8 and G.degree(v)<8:
            try:
                if nx.shortest_path_length(G, u, v) > 2:
                    G.add_edge(u,v)
                    added.append((u,v))
            except:
                pass
        attempts += 1
    return G, added

# Export list for benchmark.py
BASELINES = [
    ("Random", random_topology),
    ("Degree", degree_topology),
    ("Betweenness", betweenness_topology),
    ("Greedy", greedy_topology),
    ("ShortestPath", shortest_path_greedy),
    ("Eigenvector", eigenvector_greedy),
    ("CommunityBridge", community_bridge),
]