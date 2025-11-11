# ResiLink Module Setup Guide

## Overview

This document explains the module structure and dependencies for the Hybrid ResiLink implementation.

## Module Structure

```
ResiLink/
├── hybrid_resilink_implementation.py    # Main implementation
├── geographic_network_analyzer.py       # Geographic constraint analysis
├── core/
│   └── enhanced_topology_parser.py      # GraphML/GML topology parser
├── real_world_topologies/               # Network topology files
│   ├── *.graphml                        # GraphML format topologies
│   └── *.gml                            # GML format topologies
└── requirements.txt                     # Python dependencies
```

## Required Modules

### 1. `geographic_network_analyzer.py`

**Purpose**: Analyzes network topology with geographic constraints for realistic link feasibility assessment.

**Key Classes**:
- `GeographicConstraints`: Defines geographic constraints (max distance, costs, penalties)
- `GeographicNetworkAnalyzer`: Analyzes link feasibility based on geographic data

**Academic Foundation**:
- Haversine formula for geographic distance calculation
- Network geography: Knight et al. (2011) - Internet Topology Zoo
- Cost modeling: Realistic fiber deployment costs

**Usage**:
```python
from geographic_network_analyzer import GeographicNetworkAnalyzer, GeographicConstraints

# Create constraints
constraints = GeographicConstraints(
    max_distance_km=2000.0,
    cost_per_km=1000.0
)

# Create analyzer
analyzer = GeographicNetworkAnalyzer(constraints)

# Load geographic data from GraphML
analyzer.load_graphml_geography('real_world_topologies/Geant2012.graphml')

# Analyze link feasibility
result = analyzer.analyze_link_feasibility('node1', 'node2')
print(f"Feasible: {result['feasible']}")
print(f"Distance: {result['distance_km']} km")
print(f"Cost: ${result['cost_estimate']:,.0f}")
```

### 2. `core/enhanced_topology_parser.py`

**Purpose**: Comprehensive parser for Internet Topology Zoo GraphML/GML files with rich metadata extraction.

**Key Classes**:
- `NodeMetadata`: Rich metadata for network nodes (coordinates, country, type)
- `EdgeMetadata`: Rich metadata for network edges (speed, distance, cost)
- `NetworkMetadata`: Network-level metadata (name, type, geographic extent)
- `EnhancedNetwork`: Complete network representation
- `EnhancedTopologyParser`: Main parser class

**Academic Foundation**:
- Graph parsing: NetworkX (Hagberg et al. 2008)
- Geographic analysis: Haversine formula for distances
- Network characterization: Knight et al. (2011) - Internet Topology Zoo

**Usage**:
```python
from core.enhanced_topology_parser import EnhancedTopologyParser

# Create parser
parser = EnhancedTopologyParser()

# Parse a GraphML file
network = parser.parse_graphml('real_world_topologies/Geant2012.graphml')

# Access network data
print(f"Network: {network.metadata.name}")
print(f"Nodes: {len(network.nodes)}")
print(f"Edges: {len(network.edges)}")
print(f"Countries: {network.geographic_analysis['num_countries']}")

# Access academic metrics
metrics = network.academic_metrics
print(f"Density: {metrics['basic_properties']['density']:.3f}")
print(f"Global Efficiency: {metrics['path_metrics']['global_efficiency']:.3f}")
```

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

Run the test script to verify all modules are working:

```bash
python3 test_imports.py
```

Expected output:
```
Testing imports...
✓ Testing geographic_network_analyzer...
  ✓ GeographicNetworkAnalyzer imported
  ✓ GeographicConstraints imported

✓ Testing core.enhanced_topology_parser...
  ✓ EnhancedTopologyParser imported
  ✓ NodeMetadata imported
  ✓ EdgeMetadata imported

✓ Testing standard libraries...
  ✓ networkx imported
  ✓ numpy imported
  ✓ torch imported

✅ All imports successful!
```

## Common Issues

### Issue 1: Missing PyTorch Geometric

**Error**: `ModuleNotFoundError: No module named 'torch_geometric'`

**Solution**:
```bash
pip install torch-geometric torch-scatter torch-sparse
```

### Issue 2: Missing NetworkX

**Error**: `ModuleNotFoundError: No module named 'networkx'`

**Solution**:
```bash
pip install networkx>=3.0
```

### Issue 3: Geographic Data Not Loading

**Error**: `No geographic data available`

**Solution**: Ensure you're using GraphML files from the Internet Topology Zoo that include geographic metadata (Latitude, Longitude, Country attributes).

## Module Dependencies

### `geographic_network_analyzer.py` depends on:
- `math` (standard library)
- `logging` (standard library)
- `dataclasses` (standard library)
- `networkx` (for GraphML parsing)

### `core/enhanced_topology_parser.py` depends on:
- `networkx` (for graph operations)
- `numpy` (for numerical calculations)
- `math` (standard library)
- `logging` (standard library)
- `dataclasses` (standard library)
- `json` (standard library)

### `hybrid_resilink_implementation.py` depends on:
- All of the above
- `torch` and `torch_geometric` (for GNN)
- `requests` (for Ryu API)
- `matplotlib` and `seaborn` (for visualization)
- `sklearn` (for preprocessing)

## Academic References

1. **Freeman, L. C. (1977)**. "A set of measures of centrality based on betweenness." *Sociometry*, 40(1), 35-41.

2. **Albert, R., Jeong, H., & Barabási, A. L. (2000)**. "Error and attack tolerance of complex networks." *Nature*, 406(6794), 378-382.

3. **Latora, V., & Marchiori, M. (2001)**. "Efficient behavior of small-world networks." *Physical Review Letters*, 87(19), 198701.

4. **Watts, D. J., & Strogatz, S. H. (1998)**. "Collective dynamics of 'small-world' networks." *Nature*, 393(6684), 440-442.

5. **Fiedler, M. (1973)**. "Algebraic connectivity of graphs." *Czechoslovak Mathematical Journal*, 23(2), 298-305.

6. **Knight, S., Nguyen, H. X., Falkner, N., Bowden, R., & Roughan, M. (2011)**. "The internet topology zoo." *IEEE Journal on Selected Areas in Communications*, 29(9), 1765-1775.

7. **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018)**. "Graph attention networks." *ICLR*.

8. **Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015)**. "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.

## Testing

Run the comprehensive test suite:

```bash
# Test imports only
python3 test_imports.py

# Test with a real topology file
python3 hybrid_resilink_implementation.py \
    --simulation-mode \
    --max-cycles 3 \
    --graphml-file real_world_topologies/Abilene.graphml
```

## Support

For issues or questions:
1. Check this documentation
2. Run `test_imports.py` to diagnose import issues
3. Verify all dependencies are installed: `pip list | grep -E "torch|networkx|numpy"`
4. Check Python version: `python3 --version` (requires Python 3.8+)
