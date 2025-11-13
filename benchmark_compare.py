#!/usr/bin/env python3
"""
Runs ResiLink v2 on every paper-topology, computes the paper-specific metric,
and prints a LaTeX/Markdown table with statistical significance.
"""
import json, networkx as nx, pandas as pd, numpy as np, scipy.stats as stats
from pathlib import Path
from metrics import *
from run_one_topology import optimise

BENCH = Path("benchmarks")
OUT   = Path("benchmarks_results")
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# 1. Mapping: topology → paper → metric function → reported value
# ----------------------------------------------------------------------
bench_cfg = {
    # citation graphs (GraphRARE)
    "cora.graphml":      {"paper":"GraphRARE", "metric":citation_accuracy_proxy,
                          "reported":85.7, "unit":"% accuracy"},
    "citeseer.graphml":  {"paper":"GraphRARE", "metric":citation_accuracy_proxy,
                          "reported":75.4, "unit":"% accuracy"},
    "pubmed.graphml":    {"paper":"GraphRARE", "metric":citation_accuracy_proxy,
                          "reported":82.3, "unit":"% accuracy"},

    # WANs (NeuroPlan)
    "Nsfnet.graphml":    {"paper":"NeuroPlan", "metric":throughput_proxy,
                          "reported":25.0, "unit":"Gbps"},
    "Geant2012.graphml": {"paper":"NeuroPlan", "metric":throughput_proxy,
                          "reported":30.0, "unit":"Gbps"},

    # ISP backbone (DRL-GS)
    "AttMpls.graphml":   {"paper":"DRL-GS",   "metric":utilization_score,
                          "reported":0.85, "unit":"utilization"},

    # WSN / WMN (proxy)
    "WSN100.graphml":    {"paper":"WSN-RL",  "metric":coverage_proxy,
                          "reported":0.95, "unit":"coverage"},
    "WMN30.graphml":     {"paper":"MARL-WMN","metric":delay_reduction_proxy,
                          "reported":40.0, "unit":"% delay reduction"},
}

# ----------------------------------------------------------------------
# 2. Run ResiLink v2 on each topology (5 added links)
# ----------------------------------------------------------------------
rows = []
for topo_file, cfg in bench_cfg.items():
    topo_path = BENCH / topo_file
    if not topo_path.exists():
        print(f"[SKIP] {topo_file} missing")
        continue

    print(f"\n=== Optimising {topo_file} ===")
    G_opt, hist = optimise(topo_path, steps=5, out_dir=OUT)

    # compute the paper-specific metric
    resilink_val = cfg["metric"](G_opt)

    # store core ResiLink metrics as well
    core = default_network_metrics(G_opt)

    # statistical runs (10 seeds)
    resilink_vals = []
    for seed in range(10):
        np.random.seed(seed); torch.manual_seed(seed)
        G_opt, _ = optimise(topo_path, steps=5, out_dir=OUT / f"seed{seed}")
        resilink_vals.append(cfg["metric"](G_opt))
    mean, std = np.mean(resilink_vals), np.std(resilink_vals)

    # t-test against reported value
    reported_vals = np.full(10, cfg["reported"])
    t, p = stats.ttest_ind(resilink_vals, reported_vals)

    rows.append({
        "Topology": topo_file.split(".")[0].upper(),
        "Paper": cfg["paper"],
        "Reported": f"{cfg['reported']:.3f} {cfg['unit']}",
        "ResiLink": f"{mean:.3f} ± {std:.3f} {cfg['unit']}",
        "Improvement": f"{(mean-cfg['reported'])/cfg['reported']*100:+.1f}%",
        "p-value": f"{p:.3e}",
        "Core Metrics": f"dens={core['density']:.3f}, diam={core['diameter']}, bis={core['bisection']}"
    })

# ----------------------------------------------------------------------
# 3. Print tables
# ----------------------------------------------------------------------
df = pd.DataFrame(rows)
print("\n=== COMPARISON TABLE ===")
print(df[["Topology","Paper","Reported","ResiLink","Improvement","p-value"]].to_markdown(index=False))

# LaTeX
latex = df[["Topology","Paper","Reported","ResiLink","Improvement","p-value"]].to_latex(
    index=False, escape=False, column_format="llccccc")
(Path("comparison_table.tex")).write_text(latex)

print("\nLaTeX table written to comparison_table.tex")