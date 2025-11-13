#!/usr/bin/env python3
"""
Fix GraphML files for ResiLink v2:
- String node IDs ("n0", "n1")
- Add label attribute
- Remove self-loops / parallel edges
- Normalize capacity field
- Ensure connected
"""
import networkx as nx
import argparse
from pathlib import Path
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def fix_graphml(input_path: Path, output_path: Path):
    G = nx.read_graphml(input_path)
    log.info(f"Loaded {input_path.name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 1. String node IDs
    mapping = {n: f"n{n}" if not str(n).startswith('n') else str(n) for n in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    # 2. Add label
    for n in G.nodes():
        G.nodes[n]['label'] = n

    # 3. Clean edges
    G.remove_edges_from(nx.selfloop_edges(G))
    if not nx.is_simple_graph(G):
        G = nx.Graph(G)  # drop parallel

    # 4. Normalize capacity
    for u, v, d in G.edges(data=True):
        cap = d.get('capacity', '10 Gbps')
        cap = str(cap).strip().lower()
        if 'gbps' in cap:
            d['capacity'] = cap.replace('gbps', '').strip() + ' Gbps'
        elif 'mbps' in cap:
            d['capacity'] = cap.replace('mbps', '').strip() + ' Mbps'
        else:
            d['capacity'] = '10 Gbps'

    # 5. Ensure connected (minimal tree)
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components)-1):
            c1, c2 = components[i], components[i+1]
            n1, n2 = next(iter(c1)), next(iter(c2))
            G.add_edge(n1, n2, capacity='10 Gbps')
        log.info("  Added tree edges to connect components")

    # Save
    nx.write_graphml(G, output_path)
    log.info(f"  Fixed → {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="GraphML files or glob")
    args = parser.parse_args()

    for pattern in args.files:
        for path in Path(".").glob(pattern):
            if path.suffix != ".graphml": continue
            out_path = path.parent / f"fixed_{path.name}"
            fix_graphml(path, out_path)
            # Overwrite original for benchmark
            out_path.replace(path)

if __name__ == "__main__":
    main()