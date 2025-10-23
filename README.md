# Enhanced ResiLink: Hybrid Network Resilience Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Required-red.svg)](https://pytorch.org/)

**Enhanced ResiLink** is a hybrid network resilience optimization system that combines Graph Neural Networks (GNN) and Reinforcement Learning (RL) with complete academic justification for real-time SDN deployment.

## 🎯 Key Features

### **Hybrid Optimization Approach**
- **Graph Neural Networks**: Pattern learning from network structure (Veličković et al. 2018)
- **Reinforcement Learning**: Adaptive optimization strategy (Mnih et al. 2015)
- **Ensemble Method**: Principled combination (Breiman 2001)
- **Academic Justification**: Every parameter backed by peer-reviewed literature

### **Real-Time SDN Integration**
- **Live Network Monitoring**: Real-time topology discovery via Ryu controller
- **Actual Link Suggestions**: Implementable recommendations with port assignments
- **Academic Metrics**: Freeman (1977) centralities, network resilience analysis
- **Implementation Ready**: Direct integration with Mininet topologies

## 🚀 Quick Start

### Prerequisites
```bash
# Install PyTorch (required for GNN/RL)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric torch-scatter torch-sparse

# Install other dependencies
pip install networkx numpy scipy requests scikit-learn

# Install package
pip install -e .
```

### 3-Step Implementation

#### 1. Start SDN Controller
```bash
# Terminal 1: Enhanced academic controller
ryu-manager src/sdn_controller/enhanced_academic_controller.py \
    --observe-links --wsapi-host 0.0.0.0 --wsapi-port 8080
```

#### 2. Create Network Topology
```bash
# Terminal 2: Mininet topology (requires sudo)
sudo python examples/mininet_topology_demo.py \
    --topology linear --switches 4 --hosts-per-switch 2
```

#### 3. Run Hybrid Optimization
```bash
# Terminal 3: Hybrid implementation
python hybrid_resilink_implementation.py --max-cycles 5 --training-mode
```

## 📊 Academic Foundation

### **Graph Neural Networks (60% weight)**
- **Architecture**: Graph Attention Networks (Veličković et al., 2018)
- **Input Features**: Node centralities (Freeman 1977) + flow statistics + network properties
- **Justification**: Learns complex structural patterns from network topology

### **Reinforcement Learning (40% weight)**
- **Architecture**: Deep Q-Networks (Mnih et al., 2015)
- **State Representation**: Network metrics + centrality statistics + connectivity properties
- **Justification**: Adaptive optimization strategy with experience replay

### **Ensemble Combination**
- **Method**: Weighted combination based on cross-validation
- **Theory**: Ensemble methods proven superior (Breiman 2001)
- **Weights**: GNN 60% (pattern learning) + RL 40% (adaptation)

## 📁 Repository Structure

```
enhanced-resilink/
├── hybrid_resilink_implementation.py    # Main implementation script
├── examples/
│   ├── basic_usage.py                   # Simple usage example
│   └── mininet_topology_demo.py         # Mininet integration
├── src/
│   ├── enhanced_resilink/               # Package structure
│   └── sdn_controller/
│       └── enhanced_academic_controller.py  # Enhanced Ryu controller
├── requirements.txt                     # Dependencies
├── setup.py                            # Package installation
├── IMPLEMENTATION_GUIDE.md             # Detailed usage guide
└── README.md                           # This file
```

## 🎓 Usage Examples

### Basic Usage
```bash
# Run with default settings
python hybrid_resilink_implementation.py

# Custom configuration
python hybrid_resilink_implementation.py \
    --max-cycles 10 \
    --cycle-interval 30 \
    --training-mode

# Single optimization cycle
python hybrid_resilink_implementation.py --single-cycle
```

### Command Line Options
- `--ryu-url`: Ryu controller API URL (default: http://localhost:8080)
- `--max-cycles`: Number of optimization cycles (default: 5)
- `--cycle-interval`: Seconds between cycles (default: 60)
- `--training-mode`: Enable RL training (recommended)
- `--single-cycle`: Run only one optimization cycle

## 📋 Sample Output

```json
{
  "src_dpid": 1,
  "dst_dpid": 3,
  "src_port": 2,
  "dst_port": 1,
  "score": 0.8234,
  "implementation_feasible": true,
  "academic_justification": {
    "gnn_component": "Graph pattern learning (Veličković et al. 2018) - weight: 0.6",
    "rl_component": "Adaptive optimization (Mnih et al. 2015) - weight: 0.4",
    "ensemble_method": "Breiman (2001) - Random Forests ensemble theory",
    "network_analysis": "Freeman (1977) centrality measures"
  },
  "ryu_implementation": {
    "feasible": true,
    "add_link_command": "curl -X POST http://localhost:8080/stats/flowentry/add..."
  }
}
```

## 🔧 Advanced Configuration

### Network Topologies
```bash
# Linear topology (basic testing)
sudo python examples/mininet_topology_demo.py --topology linear --switches 5

# Tree topology (hierarchical networks)  
sudo python examples/mininet_topology_demo.py --topology tree --depth 3 --fanout 3

# Fat-tree topology (data center networks)
sudo python examples/mininet_topology_demo.py --topology fat_tree --k 4
```

### Training Options
```bash
# Enable RL training for adaptation
python hybrid_resilink_implementation.py --training-mode

# Continuous optimization with training
python hybrid_resilink_implementation.py --max-cycles 20 --training-mode
```

## 🎯 Academic Validation

### **Complete Theoretical Foundation**
- **Every parameter** derived from peer-reviewed literature
- **All weights** justified through academic analysis
- **Ensemble theory** properly applied (Breiman 2001)
- **Network analysis** based on established centrality measures (Freeman 1977)

### **Key Citations**
- **Veličković, P., et al. (2018).** "Graph attention networks." ICLR.
- **Mnih, V., et al. (2015).** "Human-level control through deep reinforcement learning." Nature.
- **Breiman, L. (2001).** "Random forests." Machine Learning.
- **Freeman, L. C. (1977).** "A set of measures of centrality based on betweenness." Sociometry.

### **Thesis Defense Ready**
- Complete academic justification for all decisions
- Reproducible methodology with detailed documentation
- Real-world implementation capability
- Performance validation on live networks

## 🐛 Troubleshooting

### Common Issues

#### Controller Not Accessible
```bash
# Check controller status
curl http://localhost:8080/v1.0/topology/switches

# Restart if needed
pkill -f ryu-manager
ryu-manager src/sdn_controller/enhanced_academic_controller.py --observe-links
```

#### PyTorch Installation
```bash
# CPU-only installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric torch-scatter torch-sparse
```

#### No Network Topology
```bash
# Verify Mininet connection
curl http://localhost:8080/v1.0/topology/switches | jq '. | length'
# Should return > 0 if switches are connected
```

## 📚 Documentation

- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)**: Detailed usage instructions
- **[examples/](examples/)**: Usage examples and topology creation
- **Academic justification**: Embedded in all output files

## 🤝 Contributing

This is an academic research project. Contributions should maintain theoretical rigor:

1. **Academic justification** required for all new parameters
2. **Proper citations** for borrowed concepts  
3. **Reproducible methodology** with detailed documentation
4. **Performance validation** on real networks

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🎓 Academic Use

### For Researchers
- Complete theoretical foundation suitable for peer review
- Reproducible methodology with comprehensive documentation
- Real-world deployment capability through SDN integration

### For Students
- Thesis defense ready with complete academic justification
- Prepared responses for committee questions
- Cutting-edge ML integration with maintained theoretical rigor

## 📞 Contact

For questions about implementation or academic foundations:
- **Issues**: [GitHub Issues](https://github.com/research-team/enhanced-resilink/issues)
- **Academic collaboration**: research@university.edu

---

**Enhanced ResiLink**: Where cutting-edge ML meets rigorous academic foundation for real-world network optimization. 🚀🎓