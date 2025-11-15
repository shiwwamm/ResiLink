# baselines.py — Baseline methods for network optimization
import networkx as nx
import random

def greedy_topology(G, max_links=10):
    """Greedy: iteratively add link with highest min-cut gain"""
    G = G.copy()
    added = []
    
    for _ in range(max_links):
        current_cut = nx.stoer_wagner(G)[0]
        candidates = [(u, v) for u, v in nx.non_edges(G) 
                     if G.degree(u) < 8 and G.degree(v) < 8]
        
        if not candidates:
            break
        
        best_gain = -float('inf')
        best_edge = None
        
        # Sample subset for efficiency (full evaluation is too slow)
        sample_size = min(50, len(candidates))
        sampled = random.sample(candidates, sample_size)
        
        for u, v in sampled:
            G.add_edge(u, v)
            new_cut = nx.stoer_wagner(G)[0]
            gain = new_cut - current_cut
            if gain > best_gain:
                best_gain = gain
                best_edge = (u, v)
            G.remove_edge(u, v)
        
        if best_edge:
            G.add_edge(*best_edge)
            added.append(best_edge)
    
    return G, added

def random_topology(G, max_links=10, seed=42):
    """Random: add random valid links"""
    G = G.copy()
    random.seed(seed)
    
    candidates = [(u, v) for u, v in nx.non_edges(G) 
                 if G.degree(u) < 8 and G.degree(v) < 8]
    random.shuffle(candidates)
    
    added = []
    for u, v in candidates[:max_links]:
        if G.degree(u) < 8 and G.degree(v) < 8:
            G.add_edge(u, v)
            added.append((u, v))
    
    return G, added

def degree_topology(G, max_links=10):
    """Degree-based: connect low-degree nodes"""
    G = G.copy()
    added = []
    
    while len(added) < max_links:
        # Get nodes sorted by degree
        nodes = sorted(G.nodes(), key=lambda n: G.degree(n))
        
        # Try to connect two lowest-degree nodes
        found = False
        for i in range(len(nodes)):
            if len(added) >= max_links:
                break
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if (not G.has_edge(u, v) and 
                    G.degree(u) < 8 and G.degree(v) < 8):
                    G.add_edge(u, v)
                    added.append((u, v))
                    found = True
                    break
            if found:
                break
        
        if not found:
            break
    
    return G, added

def betweenness_topology(G, max_links=10):
    """Betweenness-based: connect high-betweenness nodes"""
    G = G.copy()
    betweenness = nx.betweenness_centrality(G)
    
    # Sort nodes by betweenness
    nodes = sorted(G.nodes(), key=lambda n: betweenness[n], reverse=True)
    
    added = []
    for i in range(len(nodes)):
        if len(added) >= max_links:
            break
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if (not G.has_edge(u, v) and 
                G.degree(u) < 8 and G.degree(v) < 8):
                G.add_edge(u, v)
                added.append((u, v))
                if len(added) >= max_links:
                    break
    
    return G, added