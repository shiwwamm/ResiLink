# ResiLink Repository Analysis

## Overview
**ResiLink** is a network topology optimization system that uses Graph Neural Networks (GNNs) and Reinforcement Learning (PPO) to improve network resilience by strategically adding links to real-world network topologies.

**Primary Goal**: Increase network throughput and resilience (measured by min-cut capacity) through intelligent link placement.

---

## Repository Structure

```
ResiLink/
├── Core RL System
│   ├── env.py                    # Gymnasium environment for graph optimization
│   ├── gnn.py                    # GraphSAGE encoder (2-layer)
│   ├── train.py                  # GNN feature extractor with attention pooling
│   └── run_all.py                # Main training & optimization pipeline
│
├── Evaluation & Baselines
│   ├── benchmark.py              # Comprehensive benchmark suite
│   ├── baselines.py              # Baseline methods (Random, Greedy, Degree, Betweenness)
│   ├── debug_env.py              # Environment debugging & testing
│   └── test_shapes.py            # Shape compatibility tests
│
├── Data Management
│   ├── fetch_zoo.py              # Download Topology Zoo datasets
│   ├── fetch_datasets.py         # Fetch citation graphs & synthetic networks
│   └── fix_graphml_batch.py      # GraphML normalization utility
│
├── Data & Results
│   ├── real_world_topologies/    # 261 real network topologies (.graphml)
│   └── optimized_graphs/         # Output directory for optimized graphs
│
└── Cache
    └── __pycache__/              # Python bytecode cache
```

---

## Core Components

### 1. Environment (`env.py`)
**Type**: Custom Gymnasium environment for graph optimization

**State Space**:
- Node features (7 dims): degree, betweenness, clustering, core indicator, high-degree flag, local connectivity, inverse eccentricity
- Edge connectivity: Padded to max possible edges
- Candidate links: Top 200 prioritized candidates

**Action Space**: Discrete(200) - select from candidate links

**Reward Function**:
- **Min-cut improvement**: 10× multiplier for increasing min-cut
- **Utility improvement**: Weighted combination of min-cut, algebraic connectivity, degree balance
- **Exploration bonus**: 0.01 for trying new placements

**Utility Function**:
```
U = 2.0 × min_cut + 0.5 × λ₂ - 0.1 × degree_variance + 5.0 (if min_cut ≥ 2)
```

**Key Features**:
- **Smart candidate selection**: Prioritizes partition-bridging links (+100 priority)
- **Plateau detection**: Stops if improvement < 0.005 for 8 consecutive steps
- **Degree constraints**: Max 8 connections per node, max 2 new links per node per episode
- **Max steps**: 20 link additions per episode

### 2. GNN Architecture (`gnn.py` + `train.py`)

**GNN Encoder** (GraphSAGE):
- Input: 7-dim node features
- Layer 1: 7 → 64 (hidden)
- Layer 2: 64 → 32 (output)

**Feature Extractor** (with attention):
- Attention pooling over 200 candidate embeddings
- Projects to 128-dim feature vector
- Feeds into PPO policy/value networks

**Policy Network**:
- Actor (pi): [128, 64] → action logits
- Critic (vf): [128, 64] → value estimate

### 3. Training Pipeline (`run_all.py`)

**PPO Hyperparameters**:
- Total timesteps: 20,000
- Learning rate: 3e-4
- Batch size: 64
- N-steps: 2048
- N-epochs: 10
- GAE lambda: 0.95
- Clip range: 0.2
- Entropy coefficient: 0.01

**Process**:
1. Load topology from GraphML
2. Create vectorized environment
3. Train PPO agent (20k steps)
4. Run inference (deterministic policy)
5. Save optimized graph + link list + metrics

**Current Configuration**:
- Processes 8 topologies: Abilene, Aarnet, Geant2012, Nsfnet, TataNld, Chinanet, BtNorthAmerica, BellCanada
- Outputs to `optimized_graphs/`

### 4. Baseline Methods (`baselines.py`)

**Random**: Random link selection (baseline)
**Degree**: Connect low-degree nodes (load balancing)
**Betweenness**: Connect high-betweenness nodes (target bottlenecks)
**Greedy**: Iteratively add link with highest min-cut gain (samples 50 candidates for efficiency)

### 5. Benchmark Suite (`benchmark.py`)

Compares RL-GNN against all baselines on 4 topologies:
- Aarnet, Abilene, Nsfnet, Geant2012
- Outputs CSV + formatted table
- Tracks: original throughput, final throughput, gain %, links added, time

---

## Dataset

**Source**: Topology Zoo (http://www.topology-zoo.org/)
**Count**: 261 real-world network topologies
**Format**: GraphML (normalized with string node IDs, capacity fields, connectivity)

**Key Topologies**:
- ISP networks (Aarnet, Abilene, Geant, etc.)
- Research networks (Nsfnet, etc.)
- Commercial networks (Chinanet, BtNorthAmerica, etc.)

**Characteristics**:
- Nodes: 3-145 (varies by topology)
- Many have min-cut = 1 (bottleneck links)
- Sparse connectivity (typical degree 2-4)

---

## Recent Improvements

### Fixed Issues:
1. ✅ **Shape mismatch errors** - Proper feature extractor with attention pooling
2. ✅ **Zero reward problem** - Added min-cut improvement bonus (10×) + exploration bonus
3. ✅ **Poor candidate selection** - Now prioritizes partition-bridging links
4. ✅ **Weak utility function** - Increased min-cut weight (2.0×) with bonus
5. ✅ **Aggressive plateau** - Relaxed to 8 steps with 0.005 threshold
6. ✅ **Inefficient greedy baseline** - Now samples 50 candidates instead of all

### Performance Gains:
**Before fixes**:
- Aarnet: 10→20 Gbps (+100%)
- Most topologies: 0% gain

**After fixes**:
- Abilene: 20→50 Gbps (+150%)
- Aarnet: 10→40 Gbps (+300%)
- Geant2012: 10→20 Gbps (+100%)

---

## Key Algorithms

### Candidate Selection Strategy:
```python
Priority = partition_bonus + betweenness_score × 10 + shortest_path

Where:
- partition_bonus = 100 if link bridges min-cut partition, else 0
- betweenness_score = sum of node betweenness centralities
- shortest_path = distance between nodes in current graph
```

This ensures links that bridge bottlenecks get highest priority.

### Reward Shaping:
```python
if min_cut_gain > 0:
    reward = 10.0 × min_cut_gain + utility_gain
elif utility_gain > 0:
    reward = utility_gain
else:
    reward = 0.01  # exploration bonus
```

Heavy emphasis on min-cut improvements to guide learning.

---

## Usage

### Quick Start:
```bash
# Download topologies
python3 fetch_zoo.py

# Run optimization on 8 topologies
python3 run_all.py

# Run comprehensive benchmark
python3 benchmark.py

# Debug environment behavior
python3 debug_env.py

# Test shape compatibility
python3 test_shapes.py
```

### Output Files:
- `optimized_graphs/opt_<topology>.graphml` - Optimized network
- `optimized_graphs/links_<topology>.txt` - Added links
- `optimized_graphs/results.csv` - Performance metrics
- `results/benchmark.csv` - Benchmark comparison

---

## Dependencies

**Core**:
- `networkx` - Graph operations & algorithms
- `torch` - Deep learning framework
- `torch_geometric` - GNN layers (GraphSAGE)
- `stable-baselines3` - PPO implementation
- `gymnasium` - RL environment interface
- `numpy` - Numerical operations

**Optional**:
- `requests` - Dataset downloading
- `tarfile`, `zipfile` - Archive extraction

---

## Strengths

1. **Domain-aware design**: Incorporates network topology knowledge (min-cut, betweenness, partitions)
2. **Smart candidate selection**: Prioritizes bottleneck-bridging links
3. **Effective reward shaping**: Strong signal for min-cut improvements
4. **Comprehensive evaluation**: Multiple baselines + benchmark suite
5. **Large dataset**: 261 real-world topologies
6. **Reproducible**: Fixed seeds, deterministic inference

---

## Limitations & Future Work

### Current Limitations:
1. **No documentation** - Missing README, requirements.txt, usage guide
2. **CPU-only training** - No GPU support configured
3. **Fixed hyperparameters** - No config file for easy tuning
4. **Limited scalability** - Attention over 200 candidates may not scale to very large graphs
5. **Single objective** - Only optimizes min-cut (could add latency, cost, etc.)
6. **No model checkpointing** - Retrains from scratch each time

### Potential Improvements:
1. **Multi-objective optimization** - Balance throughput, latency, cost
2. **Transfer learning** - Pre-train on multiple topologies, fine-tune on target
3. **Hierarchical RL** - High-level strategy + low-level link selection
4. **Constraint handling** - Budget limits, geographic constraints, equipment availability
5. **Online learning** - Adapt to changing network conditions
6. **Explainability** - Visualize why certain links were chosen
7. **GPU acceleration** - Speed up training 10-100×
8. **Hyperparameter tuning** - Automated search (Optuna, Ray Tune)

---

## Research Context

**Related Work**:
- **NeuroPlan** (Xu et al.) - RL for WAN traffic engineering
- **DRL-GS** (Chen et al.) - Deep RL for graph structure optimization
- **GraphRARE** (Zhao et al.) - Graph rewiring for robustness
- **Topology Zoo** - Real-world network topology dataset

**Novel Contributions**:
1. GNN-based feature extraction for network optimization
2. Partition-aware candidate selection
3. Min-cut focused reward shaping
4. Comprehensive baseline comparison

---

## Metrics & Evaluation

**Primary Metric**: Min-cut capacity (network throughput bottleneck)

**Secondary Metrics**:
- Algebraic connectivity (λ₂) - Overall connectivity
- Degree variance - Load balance
- Number of links added - Resource efficiency
- Training time - Computational cost

**Evaluation Protocol**:
1. Train on topology for 20k timesteps
2. Run deterministic inference
3. Compare final min-cut vs original
4. Report throughput gain % and links added

---

## Code Quality

**Strengths**:
- Clean separation of concerns (env, model, training, evaluation)
- Type hints in some places
- Reasonable variable naming
- Modular design

**Areas for Improvement**:
- Missing docstrings in many functions
- No unit tests
- No logging framework (just print statements)
- No error recovery (crashes on bad input)
- Hardcoded paths and constants

---

## Conclusion

ResiLink is a well-designed research prototype that successfully applies GNN-based RL to network topology optimization. The recent fixes have dramatically improved performance, with 100-300% throughput gains on test topologies.

The system demonstrates strong domain knowledge integration (min-cut partitions, betweenness centrality) and effective reward shaping. The comprehensive baseline comparison and large dataset make it suitable for research publication.

Main gaps are documentation, configurability, and production-readiness features (logging, error handling, checkpointing). With these additions, it could transition from research prototype to practical tool.

**Overall Assessment**: Strong research prototype with excellent results, needs polish for broader use.
