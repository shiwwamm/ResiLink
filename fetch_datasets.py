#!/usr/bin/env python3
import os, tarfile, requests, networkx as nx
from pathlib import Path

OUT = Path("benchmarks")
OUT.mkdir(exist_ok=True)

def download(url, dest):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f: f.write(r.content)

# ---------- citation graphs (GraphRARE) ----------
for name in ["cora", "citeseer", "pubmed"]:
    tar = OUT / f"{name}.tar.gz"
    if not tar.exists():
        download(f"https://linqs-data.soe.ucsc.edu/public/lbc/{name}.tgz", tar)
    with tarfile.open(tar) as tf:
        tf.extractall(OUT)
    # edges are in <name>/<name>.cites
    G = nx.read_edgelist(OUT / name / f"{name}.cites")
    nx.write_graphml(G, OUT / f"{name}.graphml")

# ---------- WAN topologies (NeuroPlan, DRL-GS, …) ----------
zoo = {
    "Nsfnet":      "https://raw.githubusercontent.com/nsg-ethz/synet/master/examples/topozoo_original/Nsfnet.graphml",
    "Geant2012":   "https://topology-zoo.org/files/Geant2012.graphml",
    "AttMpls":     "https://topology-zoo.org/files/AttMpls.graphml",
}
for name, url in zoo.items():
    dest = OUT / f"{name}.graphml"
    if not dest.exists():
        download(url, dest)

# ---------- synthetic WSN / WMN (used as proxy) ----------
G_wsn = nx.random_geometric_graph(100, radius=0.25, seed=42)
nx.write_graphml(G_wsn, OUT / "WSN100.graphml")

G_wmn = nx.grid_2d_graph(6, 5)
nx.write_graphml(G_wmn, OUT / "WMN30.graphml")

print("All datasets are in ./benchmarks/")