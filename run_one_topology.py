#!/usr/bin/env python3
import subprocess, json, networkx as nx, os, shutil
from pathlib import Path

def optimise(graphml: Path, steps: int = 5, out_dir: Path = Path("tmp_v2")):
    """
    Calls hybrid_resilink_v2.py and returns the final graph.
    """
    out_dir.mkdir(exist_ok=True)
    cmd = [
        "python3", "hybrid_resilink_v2.py",
        str(graphml),
        "--max-steps", str(steps),
        "--episodes", "1500",          # enough for convergence on small nets
        "--reward-type", "graphrare"
    ]
    subprocess.run(cmd, check=True, cwd=".")

    # v2 writes into ./resilink_v2_results
    hist_path = Path("resilink_v2_results/history.json")
    hist = json.loads(hist_path.read_text())

    G = nx.read_graphml(graphml)
    for step in hist:
        src, dst = step["added_link"]["src"], step["added_link"]["dst"]
        G.add_edge(src, dst, capacity="10 Gbps")

    # copy result for this run
    run_dir = out_dir / graphml.stem
    run_dir.mkdir(exist_ok=True)
    nx.write_graphml(G, run_dir / "optimized.graphml")
    (run_dir / "history.json").write_text(json.dumps(hist, indent=2))
    return G, hist