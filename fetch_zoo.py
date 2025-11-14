#!/usr/bin/env python3
# fetch_zoo.py
import os, urllib.request, zipfile
from pathlib import Path

url = "http://www.topology-zoo.org/files/archive.zip"
zip_path = "zoo.zip"
out_dir = Path("real_world_topologies")
out_dir.mkdir(exist_ok=True)

if not zip_path.exists():
    print("Downloading Topology Zoo (~120MB)...")
    urllib.request.urlretrieve(url, zip_path)

print("Extracting 8 key topologies...")
with zipfile.ZipFile(zip_path) as z:
    for name in z.namelist():
        if any(topo in name for topo in ["Abilene", "Aarnet", "Geant", "Nsfnet", "TataNld", "Chinanet", "BtNorthAmerica", "BellCanada"]):
            z.extract(name, out_dir)

print("Done → real_world_topologies/")
print("Run: python3 run_all.py")