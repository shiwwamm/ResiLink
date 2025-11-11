# ResiLink Quick Start Guide

## ✅ Module Status

All required modules are now in place:

- ✅ `geographic_network_analyzer.py` - **COMPLETE**
- ✅ `core/enhanced_topology_parser.py` - **COMPLETE**  
- ✅ `hybrid_resilink_implementation.py` - **COMPLETE**

## 🚀 Getting Started

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- networkx (graph operations)
- numpy, scipy (numerical computing)
- torch, torch-geometric (machine learning)
- scikit-learn (preprocessing)
- requests (API communication)
- matplotlib, seaborn, pandas (visualization)

### Step 2: Verify Installation

```bash
python3 diagnose.py
```

You should see:
```
✅ ALL CHECKS PASSED
```

### Step 3: Run Your First Optimization

#### Option A: Simulation Mode (No SDN Controller Required)

```bash
python3 hybrid_resilink_implementation.py \
    --simulation-mode \
    --max-cycles 5 \
    --training-mode
```

#### Option B: With Real Topology File

```bash
python3 hybrid_resilink_implementation.py \
    --simulation-mode \
    --max-cycles 5 \
    --graphml-file real_world_topologies/Abilene.graphml
```

#### Option C: With Ryu Controller (Advanced)

```bash
# First, start Ryu controller in another terminal
ryu-manager ryu.app.simple_switch_13 ryu.app.ofctl_rest

# Then run ResiLink
python3 hybrid_resilink_implementation.py \
    --ryu-url http://localhost:8080 \
    --max-cycles 10 \
    --training-mode
```

## 📊 What to Expect

The implementation will:

1. **Extract Network Features** - Analyze current topology
2. **Calculate Metrics** - Compute centrality, efficiency, robustness
3. **GNN Analysis** - Use Graph Neural Networks to identify patterns
4. **RL Optimization** - Apply Reinforcement Learning for adaptive decisions
5. **Suggest Links** - Recommend new links with academic justification
6. **Geographic Analysis** - Assess feasibility based on distance and cost

### Sample Output

```
🚀 Starting Hybrid ResiLink Implementation
🔄 Running up to 5 optimization cycles
⏱️  Cycle interval: 60 seconds
🤖 Training mode: True
🎯 Quality threshold: 0.95

--- Cycle 1/5 ---
✅ Suggested Link: 1 -> 5
📊 Score: 0.8234
🌐 Network Quality: 0.7123 (threshold: 0.95)
🎯 Primary Reason: Bottleneck Relief
⭐ Strategic Priority: 0.750/1.0

🧠 Academic Justification:
   • Bottleneck Relief: Node 1 has high betweenness centrality (0.456)
     📚 Basis: Kleinrock (1976) - Queueing Theory in Computer Networks
   • Path Diversity Enhancement: Nodes 1 and 5 have limited path diversity
     📚 Basis: Holme et al. (2002) - Attack Vulnerability of Complex Networks

🌍 Geographic Analysis:
   📍 New York (USA) ↔ Chicago (USA)
   📏 Distance: 1145 km (national)
   🔗 Link Type: long_distance_terrestrial
   💰 Cost Estimate: $1,717,500

🔧 Implementation: ✅ Feasible
🔌 Ports: 3 -> 2
```

## 📁 Output Files

After running, you'll get:

- `link_suggestion_cycle_1.json` - Detailed suggestion for cycle 1
- `link_suggestion_cycle_2.json` - Detailed suggestion for cycle 2
- ... (one per cycle)
- `hybrid_optimization_history.json` - Complete optimization history
- `optimization_summary.json` - Summary of all suggestions
- `hybrid_resilink.log` - Detailed execution log

## 🔍 Understanding the Output

### Network Quality Score (0-1)
- **< 0.5**: Poor network structure
- **0.5-0.7**: Moderate network quality
- **0.7-0.85**: Good network quality
- **0.85-0.95**: Very good network quality
- **> 0.95**: Excellent network quality (optimization target)

### Strategic Priority Score (0-1)
- **< 0.2**: Low priority improvement
- **0.2-0.4**: Moderate priority
- **0.4-0.6**: High priority
- **> 0.6**: Critical improvement

### Feasibility
- **✅ Feasible**: Link can be implemented (ports available, geographic constraints met)
- **❌ Not feasible**: Issues with ports or geographic constraints

## 🎓 Academic Foundation

The implementation is based on:

1. **Graph Neural Networks** (Veličković et al. 2018)
   - Pattern learning from network structure
   - Graph Attention Networks (GAT)

2. **Reinforcement Learning** (Mnih et al. 2015)
   - Adaptive optimization strategy
   - Deep Q-Networks (DQN)

3. **Ensemble Methods** (Breiman 2001)
   - Combining GNN and RL predictions
   - Weighted voting (60% GNN, 40% RL)

4. **Network Theory**
   - Centrality measures (Freeman 1977)
   - Robustness analysis (Albert et al. 2000)
   - Efficiency metrics (Latora & Marchiori 2001)
   - Algebraic connectivity (Fiedler 1973)

## 🛠️ Troubleshooting

### "No module named 'networkx'"
```bash
pip install networkx
```

### "No module named 'torch_geometric'"
```bash
pip install torch-geometric torch-scatter torch-sparse
```

### "No geographic data available"
Use GraphML files from Internet Topology Zoo that include coordinates:
```bash
python3 hybrid_resilink_implementation.py \
    --simulation-mode \
    --graphml-file real_world_topologies/Geant2012.graphml
```

### "Connection refused" (Ryu API)
Make sure Ryu controller is running:
```bash
ryu-manager ryu.app.simple_switch_13 ryu.app.ofctl_rest
```

Or use simulation mode:
```bash
python3 hybrid_resilink_implementation.py --simulation-mode
```

## 📚 Further Reading

- `MODULE_SETUP.md` - Detailed module documentation
- `MODULES_FIXED.md` - What was fixed and why
- `requirements.txt` - Complete dependency list
- `README.md` - Project overview

## 🎯 Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Run diagnostics: `python3 diagnose.py`
3. ✅ Test simulation: `python3 hybrid_resilink_implementation.py --simulation-mode --max-cycles 3`
4. ✅ Analyze results: Check generated JSON files
5. ✅ Implement suggestions: Use Ryu API commands from output

## 💡 Tips

- Start with `--max-cycles 3` for quick testing
- Use `--simulation-mode` if you don't have a Ryu controller
- Check `hybrid_resilink.log` for detailed execution information
- Geographic constraints are automatically applied when using GraphML files
- The optimization stops early if network quality reaches the threshold

## ✨ Features

- ✅ Hybrid GNN+RL optimization
- ✅ Geographic feasibility analysis
- ✅ Cost estimation
- ✅ Academic justification for every suggestion
- ✅ Comprehensive network metrics
- ✅ Real-time network quality tracking
- ✅ Intelligent stopping criteria
- ✅ Detailed logging and visualization

Happy optimizing! 🚀
