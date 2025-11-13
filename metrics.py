#!/usr/bin/env python3
import networkx as nx
import numpy as np

def citation_accuracy_proxy(G: nx.Graph) -> float:
    """
    GraphRARE uses node-classification accuracy as utility.
    We approximate it with density × 100 (scaled to %).  In a real
    comparison you would train a GCN on the *optimized* graph.
    """
    return nx.density(G) * 100.0

def throughput_proxy(G: nx.Graph) -> float:
    """NeuroPlan – bisection bandwidth (Gbps)"""
    if nx.is_connected(G):
        return float(nx.stoer_wagner(G)[0])
    return 0.0

def utilization_score(G: nx.Graph) -> float:
    """DRL-GS – 1 / diameter (higher = better)"""
    if nx.is_connected(G):
        return 1.0 / nx.diameter(G)
    return 0.0

def coverage_proxy(G: nx.Graph) -> float:
    """WSN paper – density (proxy for area coverage)"""
    return nx.density(G)

def delay_reduction_proxy(G: nx.Graph) -> float:
    """MARL WMN – diameter (smaller = better) → % reduction vs original"""
    orig = nx.read_graphml(f"benchmarks/{G.name}.graphml")
    if nx.is_connected(G) and nx.is_connected(orig):
        red = (nx.diameter(orig) - nx.diameter(G)) / nx.diameter(orig) * 100.0
        return max(red, 0.0)
    return 0.0

def default_network_metrics(G: nx.Graph) -> dict:
    """ResiLink core metrics (always computed)"""
    if not nx.is_connected(G):
        return {"density":0,"diameter":999,"bisection":0,"alg_conn":0}
    return {
        "density": nx.density(G),
        "diameter": nx.diameter(G),
        "bisection": nx.stoer_wagner(G)[0],
        "alg_conn": nx.algebraic_connectivity(G)
    }