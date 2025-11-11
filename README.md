# Enhanced ResiLink: Network Resilience Optimization

## Overview

Enhanced ResiLink is a network resilience optimization system that combines Graph Neural Networks (GNN) and Reinforcement Learning (RL) with real-world topology data to intelligently suggest network improvements.

**Key Innovation**: Integrates 260+ real-world network topologies from the Internet Topology Zoo with geographic coordinates, link speeds, and cost modeling.

## Features

- **Hybrid AI**: GNN + RL ensemble for network optimization
- **Real-World Data**: 260+ topologies with geographic coordinates and link speeds
- **Academic Metrics**: Comprehensive network analysis with proper citations
- **SDN Integration**: Working controller and realistic network simulation
- **Cost Analysis**: Geographic distance and implementation cost modeling

## Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install SDN/Mininet (Ubuntu/Debian)
pip install ryu
sudo apt-get install mininet
```

### Run Optimization
```bash
# Start SDN Controller
ryu-manager sdn/working_controller.py --observe-links --wsapi-host 0.0.0.0 --wsapi-port 8080

# In another terminal: Run optimization
python3 hybrid_resilink_implementation.py --max-cycles 10
```

## Dataset

### Internet Topology Zoo (260+ Networks)
- **Research Networks**: GÉANT, Internet2, ARPANET
- **Commercial ISPs**: AT&T, Sprint, Cogent
- **Geographic Data**: Latitude/longitude coordinates
- **Link Characteristics**: Speeds (1-100 Gbps), fiber/copper types
- **Temporal Evolution**: Network snapshots across decades

## System Architecture

### Core Components
- `hybrid_resilink_implementation.py` - Main optimization engine (GNN + RL)
- `core/enhanced_topology_parser.py` - GraphML/GML parser with geographic features
- `sdn/working_controller.py` - SDN controller with REST API
- `sdn/mininet_builder.py` - Network builder for Mininet simulation

### Academic Foundation
- **GNN**: Graph Attention Networks (Veličković et al. 2018)
- **RL**: Deep Q-Networks (Mnih et al. 2015)
- **Network Metrics**: Freeman (1977), Albert et al. (2000), Holme et al. (2002)
- **Geographic Analysis**: Haversine distance, cost modeling

## Metrics

### Network Analysis
- Robustness: Random/targeted failure tolerance
- Centrality: Degree, betweenness, closeness with Gini coefficients
- Resilience: Combined scoring with algebraic connectivity
- Small-World: Clustering vs path length analysis

### Geographic Features
- Distance calculations using Haversine formula
- Implementation cost modeling (distance + speed based)
- Multi-country network analysis
- Link feasibility assessment

## Usage Examples

### Basic Optimization
```bash
python3 hybrid_resilink_implementation.py --max-cycles 5
```

### Advanced Configuration
```bash
python3 hybrid_resilink_implementation.py \
    --max-cycles 10 \
    --cycle-interval 30 \
    --training-mode \
    --reward-threshold 0.85
```

### Test Specific Topology
```bash
# Test with GÉANT network
sudo python3 sdn/mininet_builder.py real_world_topologies/Geant2012.graphml

# Test with Internet2 network  
sudo python3 sdn/mininet_builder.py real_world_topologies/Abilene.graphml
```

## Repository Structure

```
enhanced_resilink/
├── hybrid_resilink_implementation.py    # Main optimization engine (GNN + RL)
├── core/
│   └── enhanced_topology_parser.py      # GraphML/GML parser with geographic features
├── sdn/
│   ├── working_controller.py            # SDN controller with REST API
│   └── mininet_builder.py               # Mininet network builder
├── real_world_topologies/               # 260+ topology files (GraphML/GML)
│   ├── Geant2012.graphml               # European research network
│   ├── Abilene.graphml                 # Internet2 predecessor
│   ├── Arpanet*.graphml                # Historical ARPANET evolution
│   └── ...                             # 250+ more networks
├── requirements.txt                     # Python dependencies
├── LICENSE                             # MIT license
└── README.md                           # This file
```

## Applications

### Research
- Network resilience studies with real-world validation
- Geographic network science with actual coordinates
- Algorithm benchmarking on diverse topology types
- Cost-benefit analysis with realistic implementation costs

### Industry
- Network planning with geographic and cost constraints
- Infrastructure optimization based on real network characteristics
- Disaster recovery planning with resilience analysis
- Investment prioritization using cost-effectiveness metrics

## Expected Results

- 15-30% resilience improvement on real-world networks
- Geographic feasibility assessment for practical deployment
- Cost optimization within budget constraints
- Multi-objective optimization balancing resilience, cost, and feasibility

## License

MIT License - see LICENSE file for details.

## Citation

If you use Enhanced ResiLink in your research, please cite the relevant academic papers referenced in the code for GNN, RL, and network science metrics.