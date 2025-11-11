# =============================
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