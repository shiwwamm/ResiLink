#!/usr/bin/env python3
"""
Topology-Only Analyzer
----------------------
Loads a GraphML **only for its connectivity** (nodes + edges).  
All geographic attributes are discarded.
"""

import logging
from pathlib import Path
import networkx as nx
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class TopologyOnlyAnalyzer:
    def __init__(self):
        self.G: nx.Graph = nx.Graph()
        self.node_labels: Dict[str, str] = {}

    def load_graphml(self, path: str) -> bool:
        try:
            p = Path(path)
            if not p.exists():
                logger.warning(f"File not found: {path}")
                return False

            raw = nx.read_graphml(str(p))

            for nid in raw.nodes():
                nid_str = str(nid)
                attrs = raw.nodes[nid]
                label = attrs.get("label") or attrs.get("Label") or nid_str
                self.G.add_node(nid_str)
                self.node_labels[nid_str] = label

            for u, v, edata in raw.edges(data=True):
                u_str, v_str = str(u), str(v)
                cap = edata.get("LinkLabel") or edata.get("capacity") or "unknown"
                self.G.add_edge(u_str, v_str, capacity=cap)

            logger.info(f"Loaded topology-only graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
            return True

        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return False

    def candidate_pairs(self) -> List[Tuple[str, str]]:
        nodes = list(self.G.nodes())
        return [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :] if not self.G.has_edge(u, v)]

    def add_link(self, u: str, v: str, capacity: str = "unknown"):
        self.G.add_edge(u, v, capacity=capacity)