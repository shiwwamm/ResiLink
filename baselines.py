# baselines.py — FULLY FIXED (November 16, 2025)
import networkx as nx
import random
import numpy as np

def random_topology(G, max_links=10):
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
    G = G.copy()
    nodes = sorted(G.nodes(), key=G.degree)
    added = []
    i = 0
    while len(added) < max_links and i + 1 < len(nodes):  # ← SAFE BOUNDS
        u, v = nodes[i], nodes[i+1]
        if not G.has_edge(u,v) and G.degree(u)<8 and G.degree(v)<8:
            G.add_edge(u,v)
            added.append((u,v))
        i += 2
    return G, added

def betweenness_topology(G, max_links=10):
    G = G.copy()
    candidates = [(u,v) for u,v in nx.non_edges(G) if G.degree(u)<8 and G.degree(v)<8]
    bc = nx.betweenness_centrality(G)
    scores = [(bc[u] + bc[v], u, v) for u,v in candidates]
    scores.sort(reverse=True)
    added = []
    for _, u, v in scores[:max_links]:
        if G.degree(u)<8 and G.degree(v)<8:
            G.add_edge(u,v)
            added.append((u,v))
    return G, added

def greedy_topology(G, max_links=10):
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
