 # baselines.py
import networkx as nx
import numpy as np
import random

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
    for i in range(0, len(nodes), 2):
        if len(added) >= max_links: break
        u, v = nodes[i], nodes[i+1]
        if not G.has_edge(u,v) and G.degree(u)<8 and G.degree(v)<8:
            G.add_edge(u,v)
            added.append((u,v))
    return G, added