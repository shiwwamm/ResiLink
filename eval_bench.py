import argparse as _argparse
except Exception:
return None
return {
'delta_lambda2': _safe_algcon(G1) - _safe_algcon(G0),
'delta_global_efficiency': _safe_eff(G1) - _safe_eff(G0),
'delta_aspl': (None if _safe_aspl(G0) is None or _safe_aspl(G1) is None else _safe_aspl(G1) - _safe_aspl(G0)),
'delta_diameter': (None if _safe_diam(G0) is None or _safe_diam(G1) is None else _safe_diam(G1) - _safe_diam(G0)),
}




def _deg_baseline(G):
deg = dict(G.degree())
nodes = sorted(G.nodes(), key=lambda n: deg[n], reverse=True)
for i in range(len(nodes)):
for j in range(i+1, len(nodes)):
u, v = nodes[i], nodes[j]
if not G.has_edge(u, v):
return (u, v)
return None




def _btw_baseline(G):
btw = _nx.betweenness_centrality(G)
nodes = sorted(G.nodes(), key=lambda n: btw[n], reverse=True)
for i in range(len(nodes)):
for j in range(i+1, len(nodes)):
u, v = nodes[i], nodes[j]
if not G.has_edge(u, v):
return (u, v)
return None




def _rnd_baseline(G, rng):
nodes = list(G.nodes())
for _ in range(2000):
u = rng.choice(nodes)
v = rng.choice(nodes)
if u != v and not G.has_edge(u, v):
return (u, v)
return None




def main():
ap = _argparse.ArgumentParser()
ap.add_argument("--inputs", required=True, help="Directory or comma-separated .graphml list")
args = ap.parse_args()


paths = []
if _os.path.isdir(args.inputs):
paths = _glob.glob(_os.path.join(args.inputs, "*.graphml"))
else:
for p in args.inputs.split(","):
p = p.strip()
if _os.path.isdir(p):
paths += _glob.glob(_os.path.join(p, "*.graphml"))
else:
paths.append(p)


impl = _Impl("http://localhost:8080", 0.95, True, None)
rows = []
rng = _np.random.default_rng(123)


for path in paths:
try:
G = _nx.read_graphml(path)
if isinstance(G, (_nx.DiGraph, _nx.MultiDiGraph)):
G = _nx.Graph(G)
network_data = _build(G, impl)
candidates = impl._generate_candidate_links(network_data=network_data, min_hops=2, top_p_by_centrality=200)
if not candidates:
continue
gnn_scores = impl._get_gnn_predictions(network_data, candidates)
rl_scores = impl._get_rl_predictions(network_data, candidates, training_mode=False)
if gnn_scores is None or len(gnn_scores)==0:
gnn_scores = _np.zeros(len(candidates))
if rl_scores is None or len(rl_scores)==0:
rl_scores = _np.zeros(len(candidates))
ens = 0.6*gnn_scores + 0.4*rl_scores
best_idx = int(_np.argmax(ens))
best_edge = candidates[best_idx]


deg_edge = _deg_baseline(G) or best_edge
btw_edge = _btw_baseline(G) or best_edge
rnd_edge = _rnd_baseline(G, rng) or best_edge


ours = _recompute_deltas(G, [best_edge])
degd = _recompute_deltas(G, [deg_edge])
btwd = _recompute_deltas(G, [btw_edge])
rndd = _recompute_deltas(G, [rnd_edge])


rows.append({
'graphml': _os.path.basename(path),
'ours_edge': str(best_edge),
**{f'ours_{k}': v for k,v in ours.items()},
'deg_edge': str(deg_edge), **{f'deg_{k}': v for k,v in degd.items()},
'btw_edge': str(btw_edge), **{f'btw_{k}': v for k,v in btwd.items()},
'rnd_edge': str(rnd_edge), **{f'rnd_{k}': v for k,v in rndd.items()},
})
except Exception as e:
print(f"[eval] Failed on {path}: {e}")
continue


out = "/mnt/data/eval_bench_full.csv"
if rows:
with open(out, "w", newline="") as f:
writer = _csv.DictWriter(f, fieldnames=rows[0].keys())
writer.writeheader()
for r in rows:
writer.writerow(r)
print(f"Saved: {out}")
else:
print("No rows to save.")


if __name__ == "__main__":
main()