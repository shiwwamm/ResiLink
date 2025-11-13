#!/usr/bin/env python3
"""
Batch GraphML Cleaner for ResiLink
----------------------------------
Fixes common GraphML issues so that hybrid_resilink_implementation.py
can parse any topology from the Topology Zoo or similar sources.

Usage:
    python3 fix_graphml_batch.py path/to/files/*.graphml
    python3 fix_graphml_batch.py path/to/folder/ --recursive
"""

import argparse
import sys
import os
import shutil
from pathlib import Path
from lxml import etree
from typing import List, Set, Dict
import logging

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
NAMESPACE = "http://graphml.graphdrawing.org/xmlns"
NSMAP = {None: NAMESPACE}
ET = etree.ElementTree
Q = etree.QName

# Capacity string normalization
CAPACITY_MAP = {
    "< 10Gbps": "< 10 Gbps",
    "< 2.5Gbps": "< 2.5 Gbps",
    "< 1Gbps": "< 1 Gbps",
    "< 155Mbps": "< 155 Mbps",
    "< 622Mbps": "< 622 Mbps",
}

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
def normalize_capacity(text: str) -> str:
    """Normalize capacity strings."""
    return CAPACITY_MAP.get(text.strip(), text.strip())

def make_node_id(old_id: str) -> str:
    """Convert numeric ID to 'n0', 'n1', etc."""
    return f"n{old_id}"

def make_edge_id(idx: int) -> str:
    """Generate sequential edge ID."""
    return f"e{idx}"

# --------------------------------------------------------------------------- #
# Core Fixer
# --------------------------------------------------------------------------- #
def fix_graphml_file(input_path: Path, output_path: Path, backup: bool = True) -> bool:
    """
    Fix a single GraphML file.
    Returns True if changes were made.
    """
    try:
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(str(input_path), parser)
        root = tree.getroot()

        if root.tag != Q(NAMESPACE, "graphml").text:
            log.error(f"{input_path.name}: Root is not <graphml>")
            return False

        changed = False

        # --- 1. Deduplicate <key> elements ---
        seen_keys: Set[str] = set()
        keys_to_remove = []
        for key_elem in root.findall(f".//{{{NAMESPACE}}}key"):
            key_id = key_elem.get("id")
            if key_id in seen_keys:
                keys_to_remove.append(key_elem)
                changed = True
            else:
                seen_keys.add(key_id)

        for k in keys_to_remove:
            root.remove(k)
            log.debug(f"Removed duplicate key id={k.get('id')}")

        # --- 2. Fix nodes: ensure string IDs ---
        node_map: Dict[str, str] = {}  # old_id -> new_id
        for node in root.findall(f".//{{{NAMESPACE}}}node"):
            old_id = node.get("id")
            if old_id is None:
                log.error(f"{input_path.name}: Node missing id")
                return False
            new_id = make_node_id(old_id)
            if new_id != old_id:
                node.set("id", new_id)
                node_map[old_id] = new_id
                changed = True

        # --- 3. Fix edges: add id, normalize capacity, remap source/target ---
        edge_elems = root.findall(f".//{{{NAMESPACE}}}edge")
        for idx, edge in enumerate(edge_elems):
            # Add missing id
            if edge.get("id") is None:
                edge.set("id", make_edge_id(idx))
                changed = True

            # Remap source/target using node_map
            src = edge.get("source")
            tgt = edge.get("target")
            if src in node_map:
                edge.set("source", node_map[src])
                changed = True
            if tgt in node_map:
                edge.set("target", node_map[tgt])
                changed = True

            # Normalize capacity in <data key="..."> (usually LinkLabel or capacity)
            for data in edge.findall(f".//{{{NAMESPACE}}}data"):
                key = data.get("key")
                if key and "LinkLabel" in key or "capacity" in key.lower():
                    old_text = data.text or ""
                    new_text = normalize_capacity(old_text)
                    if new_text != old_text:
                        data.text = new_text
                        changed = True

        # --- 4. Write output ---
        if changed:
            if backup and output_path.exists():
                backup_path = output_path.with_suffix(output_path.suffix + ".bak")
                shutil.copy2(output_path, backup_path)
                log.info(f"Backup created: {backup_path.name}")

            tree.write(
                str(output_path),
                pretty_print=True,
                xml_declaration=True,
                encoding="utf-8"
            )
            log.info(f"Fixed: {input_path.name} → {output_path.name}")
        else:
            log.info(f"No changes needed: {input_path.name}")

        return changed

    except etree.XMLSyntaxError as e:
        log.error(f"XML syntax error in {input_path.name}: {e}")
        return False
    except Exception as e:
        log.error(f"Failed to process {input_path.name}: {e}")
        return False

# --------------------------------------------------------------------------- #
# Batch Processor
# --------------------------------------------------------------------------- #
def process_files(file_patterns: List[str], recursive: bool = False, dry_run: bool = False):
    """
    Process all GraphML files matching patterns.
    """
    paths = []
    for pattern in file_patterns:
        p = Path(pattern)
        if p.is_dir():
            glob_pattern = "**/*.graphml" if recursive else "*.graphml"
            paths.extend(p.glob(glob_pattern))
        else:
            paths.extend(Path().glob(pattern) if "*" in pattern else [p])

    paths = [p.resolve() for p in paths if p.suffix.lower() in {".graphml", ".xml"}]
    if not paths:
        log.warning("No GraphML files found.")
        return

    log.info(f"Found {len(paths)} GraphML file(s) to process.")

    for input_path in paths:
        output_path = input_path  # overwrite same file
        if dry_run:
            log.info(f"[DRY RUN] Would fix: {input_path}")
            continue
        fix_graphml_file(input_path, output_path)

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Batch fix GraphML files for ResiLink compatibility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="GraphML files or glob patterns (e.g. 'topologies/*.graphml')"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively search directories"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without writing"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak files"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    process_files(
        file_patterns=args.files,
        recursive=args.recursive,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()