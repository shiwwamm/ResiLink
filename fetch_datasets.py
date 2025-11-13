#!/usr/bin/env python3
import os, tarfile, requests, networkx as nx
from pathlib import Path

OUT = Path("benchmarks")
OUT.mkdir(exist_ok=True)

def download(url, dest):
    print(f"Downloading {url}...")
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    with open(dest, "wb") as f: f.write(r.content)
    print(f"✓ Saved to {dest.name}")

# ---------- citation graphs (GraphRARE) ----------
# Cora and Citeseer from linqs
for name in ["cora", "citeseer"]:
    tar = OUT / f"{name}.tar.gz"
    if not tar.exists():
        try:
            download(f"https://linqs-data.soe.ucsc.edu/public/lbc/{name}.tgz", tar)
        except Exception as e:
            print(f"Failed to download {name}: {e}")
            continue
    try:
        with tarfile.open(tar) as tf:
            tf.extractall(OUT)
        # edges are in <name>/<name>.cites
        G = nx.read_edgelist(OUT / name / f"{name}.cites")
        nx.write_graphml(G, OUT / f"{name}.graphml")
        print(f"✓ {name}.graphml created")
    except Exception as e:
        print(f"Failed to process {name}: {e}")

# PubMed from alternative source (Pubmed-Diabetes dataset)
pubmed_tar = OUT / "pubmed.tar.gz"
if not (OUT / "pubmed.graphml").exists():
    try:
        print("Trying alternative PubMed source...")
        download("https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz", pubmed_tar)
        import shutil
        with tarfile.open(pubmed_tar) as tf:
            tf.extractall(OUT / "pubmed_temp")
        # Find the .cites file (may be named differently)
        pubmed_dir = OUT / "pubmed_temp"
        cites_files = list(pubmed_dir.rglob("*.cites"))
        if not cites_files:
            # Try looking for edge list files with different extensions
            cites_files = list(pubmed_dir.rglob("*edges*")) + list(pubmed_dir.rglob("*.edgelist"))
        
        if cites_files:
            G = nx.read_edgelist(cites_files[0])
            nx.write_graphml(G, OUT / "pubmed.graphml")
            print("✓ pubmed.graphml created")
        else:
            raise FileNotFoundError("No .cites or edge file found in archive")
        
        # Cleanup
        shutil.rmtree(OUT / "pubmed_temp", ignore_errors=True)
    except Exception as e:
        print(f"Failed to download PubMed from alternative source: {e}")
        print("Will try PyTorch Geometric fallback...")

# ---------- WAN topologies (NeuroPlan, DRL-GS, …) ----------
zoo = {
    "Nsfnet":      "https://raw.githubusercontent.com/nsg-ethz/synet/master/examples/topozoo_original/Nsfnet.graphml",
    "Geant2012":   "https://topology-zoo.org/files/Geant2012.graphml",
    "AttMpls":     "https://topology-zoo.org/files/AttMpls.graphml",
}
for name, url in zoo.items():
    dest = OUT / f"{name}.graphml"
    if not dest.exists():
        try:
            download(url, dest)
        except Exception as e:
            print(f"Failed to download {name}: {e}")
    else:
        print(f"✓ {name}.graphml already exists")

# ---------- synthetic WSN / WMN (used as proxy) ----------
print("Generating synthetic topologies...")
G_wsn = nx.random_geometric_graph(100, radius=0.25, seed=42)
# Remove position attributes that GraphML doesn't support
for node in G_wsn.nodes():
    G_wsn.nodes[node].clear()
nx.write_graphml(G_wsn, OUT / "WSN100.graphml")
print("✓ WSN100.graphml created")

G_wmn = nx.grid_2d_graph(6, 5)
# Convert tuple node IDs to strings for GraphML compatibility
G_wmn = nx.convert_node_labels_to_integers(G_wmn, label_attribute="grid_pos")
# Remove any non-serializable attributes
for node in G_wmn.nodes():
    if "grid_pos" in G_wmn.nodes[node]:
        del G_wmn.nodes[node]["grid_pos"]
nx.write_graphml(G_wmn, OUT / "WMN30.graphml")
print("✓ WMN30.graphml created")

print(f"\n✓ All datasets are in ./benchmarks/")
print(f"Total files: {len(list(OUT.glob('*.graphml')))} GraphML files")

# Optional: Download PubMed via PyTorch Geometric if available
if not (OUT / "pubmed.graphml").exists():
    try:
        print("\nAttempting to download PubMed via PyTorch Geometric...")
        from torch_geometric.datasets import Planetoid
        from torch_geometric.utils import to_networkx
        
        dataset = Planetoid(root=str(OUT / "pubmed_pyg"), name='PubMed')
        data = dataset[0]
        G = to_networkx(data, to_undirected=True)
        nx.write_graphml(G, OUT / "pubmed.graphml")
        print("✓ pubmed.graphml created via PyTorch Geometric")
    except ImportError:
        print("PyTorch Geometric not installed. Skipping PubMed.")
    except Exception as e:
        print(f"Could not fetch PubMed via PyG: {e}")