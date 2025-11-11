# =============================
# File: hybrid_resilink_implementation_mod.py
# =============================
# Drop-in enhanced wrapper that imports your original implementation from
# /mnt/data/hybrid_resilink_implementation.py and overrides the critical parts:
# - Real metric recomputation after adding links
# - Candidate pruning (hop>=2, optional top-P by betweenness sum)
# - Feature enrichment (current-flow betweenness, k-core)
# - Simple CLI for one-off graphml and eval bench entry points

import sys, os
sys.path.append("/mnt/data")
import math
import numpy as np
import networkx as nx
import random
import argparse
import csv
import glob

# Import the user's original implementation as a base
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "hybrid_resilink_impl_orig", 
    "hybrid_resilink_implementation.py"
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

BaseImpl = _impl.HybridResiLinkImplementation if hasattr(_impl, "HybridResiLinkImplementation") else None
if BaseImpl is None:
    raise RuntimeError("Could not find HybridResiLinkImplementation in the original file.")

class HybridResiLinkImplementation(BaseImpl):
    """
    Enhanced version:
      - Real metric recomputation after adding links
      - Candidate pruning (hop>=2, optional top-P by (betw_u+betw_v))
      - Feature enrichment (current-flow betweenness, core number)
    """

    # ------- CENTRALITIES (+extras) -------
    def _calculate_centralities(self, G):
        if G.number_of_nodes() == 0:
            return {'degree': {}, 'betweenness': {}, 'closeness': {}, 'cfbetweenness': {}, 'core': {}}
        try:
            degree_cent = nx.degree_centrality(G)
            betweenness_cent = nx.betweenness_centrality(G)
            # closeness or harmonic closeness for disconnected graphs
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
            # extras
            try:
                cf_bet = nx.current_flow_betweenness_centrality(G, normalized=True)
            except Exception:
                cf_bet = {n: 0.0 for n in G.nodes()}
            try:
                core_num = nx.core_number(G)
            except Exception:
                core_num = {n: 0 for n in G.nodes()}
            return {
                'degree': {str(k): v for k, v in degree_cent.items()},
                'betweenness': {str(k): v for k, v in betweenness_cent.items()},
                'closeness': {str(k): v for k, v in closeness_cent.items()},
                'cfbetweenness': {str(k): v for k, v in cf_bet.items()},
                'core': {str(k): v for k, v in core_num.items()}
            }
        except Exception as e:
            import logging
            logging.error(f"Centrality calculation failed: {e}")
            return {'degree': {}, 'betweenness': {}, 'closeness': {}, 'cfbetweenness': {}, 'core': {}}

    # ------- CANDIDATE GENERATION (with pruning) -------
    def _generate_candidate_links(self, network_data=None, min_hops=2, top_p_by_centrality=None, geo_filter=None):
        switches = network_data['topology']['switches']
        existing_links = set()
        for link in network_data['topology']['switch_switch_links']:
            existing_links.add((min(link['src_dpid'], link['dst_dpid']), 
                              max(link['src_dpid'], link['dst_dpid'])))
        for suggested_link in getattr(self, 'suggested_links', set()):
            existing_links.add(suggested_link)

        # Build graph - try to use existing graph first
        G = None
        if 'network_graph' in network_data and isinstance(network_data['network_graph'], nx.Graph):
            G = network_data['network_graph']
        else:
            try:
                G = self._build_networkx_graph(network_data)
            except Exception as e:
                # Fallback: build simple graph from topology
                G = nx.Graph()
                for switch in switches:
                    G.add_node(switch)
                for link in network_data['topology']['switch_switch_links']:
                    G.add_edge(link['src_dpid'], link['dst_dpid'])

        # Raw candidates
        candidates = []
        for i in range(len(switches)):
            for j in range(i + 1, len(switches)):
                src, dst = switches[i], switches[j]
                pair = (min(src, dst), max(src, dst))
                if pair not in existing_links:
                    candidates.append((src, dst))

        # Prune by hop distance
        pruned = []
        for (u, v) in candidates:
            if G is not None:
                try:
                    d = nx.shortest_path_length(G, u, v)
                    if d < min_hops:
                        continue
                except Exception:
                    pass
            pruned.append((u, v))

        # Optional: keep only top-P by (betw(u)+betw(v))
        if top_p_by_centrality is not None and G is not None:
            cen = network_data.get('centralities', {})
            btw = cen.get('betweenness', {})
            def score(pair):
                u, v = pair
                return float(btw.get(str(u), 0.0)) + float(btw.get(str(v), 0.0))
            pruned = sorted(pruned, key=score, reverse=True)[:int(top_p_by_centrality)]

        return pruned

    # ------- REAL METRIC RECOMPUTATION -------
    def _create_simulated_final_network(self, initial_state):
        import copy
        simulated_state = copy.deepcopy(initial_state)

        # Build base graph
        G0 = None
        if 'network_graph' in initial_state and isinstance(initial_state['network_graph'], nx.Graph):
            G0 = initial_state['network_graph']
        elif 'topology_snapshot' in initial_state:
            topo = initial_state['topology_snapshot']
            G0 = nx.Graph()
            for n in topo.get('nodes', []):
                G0.add_node(n)
            for e in topo.get('edges', []):
                G0.add_edge(e[0], e[1])

        if G0 is None:
            return simulated_state  # nothing to recompute safely

        G1 = G0.copy()
        for (u, v) in getattr(self, 'suggested_links', set()):
            if not G1.has_edge(u, v):
                G1.add_edge(u, v)

        def _safe_algcon(G):
            try:
                return float(nx.algebraic_connectivity(G)) if nx.is_connected(G) else 0.0
            except Exception:
                return 0.0

        def _safe_aspl(G):
            try:
                H = G if nx.is_connected(G) else G.subgraph(max(nx.connected_components(G), key=len)).copy()
                return float(nx.average_shortest_path_length(H))
            except Exception:
                return None

        def _safe_diam(G):
            try:
                H = G if nx.is_connected(G) else G.subgraph(max(nx.connected_components(G), key=len)).copy()
                return int(nx.diameter(H))
            except Exception:
                return None

        def _safe_eff(G):
            try:
                return float(nx.global_efficiency(G))
            except Exception:
                return None

        nodes = G1.number_of_nodes(); edges = G1.number_of_edges()
        density = (2 * edges) / (nodes * (nodes - 1)) if nodes > 1 else 0.0
        simulated_state['basic_properties'] = {'nodes': nodes, 'edges': edges, 'density': density}

        alg0 = _safe_algcon(G0); alg1 = _safe_algcon(G1)
        aspl0 = _safe_aspl(G0);  aspl1 = _safe_aspl(G1)
        diam0 = _safe_diam(G0);  diam1 = _safe_diam(G1)
        eff0 = _safe_eff(G0);    eff1 = _safe_eff(G1)

        simulated_state['metric_deltas'] = {
            'delta_lambda2': (alg1 - alg0) if alg0 is not None and alg1 is not None else None,
            'delta_global_efficiency': (eff1 - eff0) if eff0 is not None and eff1 is not None else None,
            'delta_aspl': (None if aspl0 is None or aspl1 is None else (aspl1 - aspl0)),
            'delta_diameter': (None if diam0 is None or diam1 is None else (diam1 - diam0)),
        }
        simulated_state['metrics_recomputed'] = True
        return simulated_state


# ---------- Utilities for evaluation ----------

def build_network_data_from_graph(G, impl):
    """Build network_data structure from a NetworkX graph."""
    # Create switch list with integer IDs
    switches = list(range(1, G.number_of_nodes() + 1))
    id_map = {n: i + 1 for i, n in enumerate(G.nodes())}
    
    # Build switch-switch links
    switch_switch_links = []
    for u, v in G.edges():
        switch_switch_links.append({
            'src_dpid': id_map[u],
            'dst_dpid': id_map[v],
            'src_port': 1,
            'dst_port': 1,
            'bandwidth_mbps': 1000.0,
            'stats': {
                'tx_packets': 0, 'tx_bytes': 0,
                'rx_packets': 0, 'rx_bytes': 0,
                'tx_dropped': 0, 'rx_dropped': 0
            }
        })
    
    # Calculate centralities directly on the graph
    centralities = impl._calculate_centralities(G)
    
    # Build nodes list with attributes
    nodes = []
    for node_id in switches:
        nodes.append({
            'id': node_id,
            'attributes': {
                'type': 'switch',
                'num_flows': 0,
                'total_packets': 0,
                'total_bytes': 0,
                'centrality_scores': {
                    'degree': centralities['degree'].get(str(node_id), 0.0),
                    'betweenness': centralities['betweenness'].get(str(node_id), 0.0),
                    'closeness': centralities['closeness'].get(str(node_id), 0.0)
                }
            }
        })
    
    # Build network data structure
    network_data = {
        'timestamp': 0,
        'topology': {
            'switches': switches,
            'hosts': [],
            'switch_switch_links': switch_switch_links,
            'host_switch_links': []
        },
        'nodes': nodes,
        'centralities': centralities,
        'graph_properties': {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'is_connected': nx.is_connected(G),
            'density': nx.density(G)
        },
        'network_graph': G  # Store the original graph
    }
    
    return network_data


def run_eval_bench(graphml_inputs, out_csv="/mnt/data/eval_bench_full.csv"):
    paths = []
    for item in graphml_inputs:
        if os.path.isdir(item):
            paths += glob.glob(os.path.join(item, "*.graphml"))
        else:
            paths.append(item)

    impl = HybridResiLinkImplementation("http://localhost:8080", 0.95, True, None)
    rows = []
    rng = np.random.default_rng(123)

    def recompute_deltas(G0, edges_to_add):
        G1 = G0.copy()
        for u, v in edges_to_add:
            if not G1.has_edge(u, v):
                G1.add_edge(u, v)
        def _safe_algcon(G):
            try:
                return float(nx.algebraic_connectivity(G)) if nx.is_connected(G) else 0.0
            except Exception:
                return 0.0
        def _safe_aspl(G):
            try:
                H = G if nx.is_connected(G) else G.subgraph(max(nx.connected_components(G), key=len)).copy()
                return float(nx.average_shortest_path_length(H))
            except Exception:
                return None
        def _safe_diam(G):
            try:
                H = G if nx.is_connected(G) else G.subgraph(max(nx.connected_components(G), key=len)).copy()
                return int(nx.diameter(H))
            except Exception:
                return None
        def _safe_eff(G):
            try:
                return float(nx.global_efficiency(G))
            except Exception:
                return None
        return {
            'delta_lambda2': _safe_algcon(G1) - _safe_algcon(G0),
            'delta_global_efficiency': _safe_eff(G1) - _safe_eff(G0),
            'delta_aspl': (None if _safe_aspl(G0) is None or _safe_aspl(G1) is None else _safe_aspl(G1) - _safe_aspl(G0)),
            'delta_diameter': (None if _safe_diam(G0) is None or _safe_diam(G1) is None else _safe_diam(G1) - _safe_diam(G0)),
        }

    def degree_baseline(G):
        deg = dict(G.degree())
        nodes = sorted(G.nodes(), key=lambda n: deg[n], reverse=True)
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if not G.has_edge(u, v):
                    return (u, v)
        return None

    def betweenness_baseline(G):
        btw = nx.betweenness_centrality(G)
        nodes = sorted(G.nodes(), key=lambda n: btw[n], reverse=True)
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if not G.has_edge(u, v):
                    return (u, v)
        return None

    def random_baseline(G):
        nodes = list(G.nodes())
        for _ in range(2000):
            u = rng.choice(nodes)
            v = rng.choice(nodes)
            if u != v and not G.has_edge(u, v):
                return (u, v)
        return None

    for path in paths:
        try:
            G = nx.read_graphml(path)
            if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
                G = nx.Graph(G)
            network_data = build_network_data_from_graph(G, impl)
            candidates = impl._generate_candidate_links(network_data=network_data, min_hops=2, top_p_by_centrality=200)
            if not candidates:
                continue
            gnn_scores = impl._get_gnn_predictions(network_data, candidates)
            rl_scores = impl._get_rl_predictions(network_data, candidates, training_mode=False)
            if gnn_scores is None or len(gnn_scores)==0:
                gnn_scores = np.zeros(len(candidates))
            if rl_scores is None or len(rl_scores)==0:
                rl_scores = np.zeros(len(candidates))
            ens = 0.6*gnn_scores + 0.4*rl_scores
            best_idx = int(np.argmax(ens))
            best_edge = candidates[best_idx]

            deg_edge = degree_baseline(G) or best_edge
            btw_edge = betweenness_baseline(G) or best_edge
            rnd_edge = random_baseline(G) or best_edge

            ours = recompute_deltas(G, [best_edge])
            degd = recompute_deltas(G, [deg_edge])
            btwd = recompute_deltas(G, [btw_edge])
            rndd = recompute_deltas(G, [rnd_edge])

            rows.append({
                'graphml': os.path.basename(path),
                'ours_edge': str(best_edge),
                **{f'ours_{k}': v for k,v in ours.items()},
                'deg_edge': str(deg_edge), **{f'deg_{k}': v for k,v in degd.items()},
                'btw_edge': str(btw_edge), **{f'btw_{k}': v for k,v in btwd.items()},
                'rnd_edge': str(rnd_edge), **{f'rnd_{k}': v for k,v in rndd.items()},
            })
        except Exception as e:
            print(f"[bench] Failed on {path}: {e}")

    if rows:
        out = out_csv
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Saved eval bench to {out}")
    else:
        print("No results written (no rows).")


def main():
    ap = argparse.ArgumentParser(description="Enhanced Hybrid ResiLink Runner")
    ap.add_argument("--eval-bench", type=str, help="Directory or comma-separated GraphML list")
    ap.add_argument("--graphml-file", type=str, help="Single GraphML for a one-off cycle")
    args = ap.parse_args()

    if args.eval_bench:
        items = []
        if os.path.isdir(args.eval_bench):
            items.append(args.eval_bench)
        else:
            items += [p.strip() for p in args.eval_bench.split(",")]
        run_eval_bench(items)
        return

    if args.graphml_file:
        G = nx.read_graphml(args.graphml_file)
        if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
            G = nx.Graph(G)
        impl = HybridResiLinkImplementation("http://localhost:8080", 0.95, True, args.graphml_file)
        network_data = build_network_data_from_graph(G, impl)
        candidates = impl._generate_candidate_links(network_data=network_data, min_hops=2, top_p_by_centrality=200)
        gnn_scores = impl._get_gnn_predictions(network_data, candidates)
        rl_scores = impl._get_rl_predictions(network_data, candidates, training_mode=False)
        if not candidates:
            print("No candidate edges.")
            return
        if gnn_scores is None or len(gnn_scores)==0:
            gnn_scores = np.zeros(len(candidates))
        if rl_scores is None or len(rl_scores)==0:
            rl_scores = np.zeros(len(candidates))
        ens = 0.6*gnn_scores + 0.4*rl_scores
        best_idx = int(np.argmax(ens))
        best_edge = candidates[best_idx]
        impl.suggested_links = {(min(best_edge[0], best_edge[1]), max(best_edge[0], best_edge[1]))}
        initial_state = {
            'basic_properties': {'nodes': G.number_of_nodes(), 'edges': G.number_of_edges(), 'density': 0.0},
            'overall_quality': 0.5,
            'path_metrics': {'global_efficiency': nx.global_efficiency(G)},
            'resilience_score': 0.5,
            'robustness_metrics': {'random_failure_threshold': 0.5, 'targeted_attack_threshold': 0.5},
            'network_graph': nx.Graph(G)
        }
        sim_state = impl._create_simulated_final_network(initial_state)
        print("Chosen edge:", best_edge)
        print("Metric deltas:", sim_state.get('metric_deltas', {}))
        return

    print("Nothing to do. Use --eval-bench or --graphml-file.")

if __name__ == "__main__":
    main()
