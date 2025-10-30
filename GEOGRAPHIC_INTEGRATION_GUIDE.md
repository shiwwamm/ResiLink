# Geographic Integration Guide

## 🌍 Overview

The hybrid ResiLink implementation now includes geographic awareness using real-world data from Internet Topology Zoo GraphML files. This makes link suggestions more realistic by considering physical distance constraints, international boundaries, and infrastructure costs.

## 🚀 New Features

### 1. Geographic Data Extraction
- Extracts latitude/longitude from GraphML files
- Identifies countries and node types
- Calculates great-circle distances between nodes

### 2. Distance-Based Feasibility
- **Maximum Distance**: 2000km default limit for terrestrial links
- **International Penalty**: 30% feasibility reduction for cross-border links
- **Submarine Cable Threshold**: 500km triggers submarine cable requirements
- **Cost Estimation**: Distance-based cost calculations

### 3. Link Classification
- **Metropolitan**: < 200km within same region
- **Regional**: 200-1000km domestic
- **Long-haul Domestic**: > 1000km same country
- **International Terrestrial**: Cross-border < 500km
- **International Submarine**: Cross-border > 500km

## 📋 Usage Examples

### Basic Usage with Geographic Constraints
```bash
# Start controller
ryu-manager ryu.app.ofctl_rest ryu.app.rest_topology sdn/updated_controller.py --observe-links

# Start Mininet with Bell Canada topology
sudo python3 mininet_graphml_topology.py real_world_topologies/Bellcanada.graphml

# Run optimization with geographic constraints
python3 hybrid_resilink_implementation.py \
    --max-cycles 5 \
    --training-mode \
    --graphml-file real_world_topologies/Bellcanada.graphml
```

### Test Geographic Features
```bash
# Test standalone geographic analyzer
python3 geographic_network_analyzer.py

# Test integration
python3 test_geographic_integration.py
```

### Custom Geographic Constraints
You can modify constraints in `geographic_network_analyzer.py`:
```python
constraints = GeographicConstraints(
    max_distance_km=1500.0,        # Stricter distance limit
    international_penalty=0.1,     # Heavy international penalty
    submarine_cable_threshold=300.0, # Lower submarine threshold
    cost_per_km=1500.0            # Higher cost per km
)
```

## 📊 Sample Output

### With Geographic Constraints:
```
✅ Suggested Link: 12 -> 29
📊 Score: 0.8234
🌐 Network Quality: 0.3657 (threshold: 0.95)
🎯 Primary Reason: Vulnerability Mitigation

🌍 Geographic Analysis:
   📍 New York, United States ↔ Toronto, Canada
   📏 Distance: 352 km (Regional)
   🔗 Link Type: International Terrestrial
   💰 Cost Estimate: $528,000
   
🔧 Implementation: ✅ Feasible
🔌 Ports: 4 -> 7
💡 Ryu command ready for implementation
```

### Geographically Infeasible Link:
```
✅ Suggested Link: 31 -> 24
📊 Score: 0.1245
🌐 Network Quality: 0.3657 (threshold: 0.95)

🌍 Geographic Analysis:
   📍 Seattle, United States ↔ St John's, Canada
   📏 Distance: 4,127 km (Intercontinental)
   🔗 Link Type: International Submarine
   💰 Cost Estimate: $18,571,500
   ❌ Geographic Issue: Distance (4127km) exceeds maximum (2000km)
   
🔧 Implementation: ❌ Not feasible
   🌍 Geographic Issue: Distance/location constraints
```

## 🎯 Benefits

### 1. Realistic Constraints
- No more suggestions for impossible 5000km links
- Considers physical infrastructure limitations
- Accounts for international regulatory complexity

### 2. Cost Awareness
- Estimates implementation costs based on distance
- Factors in submarine cable requirements
- Considers international link complexity

### 3. Strategic Prioritization
- Favors shorter, more feasible links
- Balances network improvement with practical constraints
- Provides clear justification for infeasible suggestions

## 🔧 Available GraphML Topologies

Each topology provides different geographic scenarios:

### Continental Networks
- **Bellcanada.graphml**: Canada + some US nodes (good for testing international links)
- **AttMpls.graphml**: AT&T US network (domestic long-haul testing)
- **UsCarrier.graphml**: US carrier network

### International Networks  
- **Geant2012.graphml**: European research network (multi-country)
- **Globalcenter.graphml**: Global ISP network (intercontinental)

### Regional Networks
- **Abilene.graphml**: Internet2 network (regional US)
- **Cesnet.graphml**: Czech Republic network (small country)

## 🧪 Testing Scenarios

### Test 1: Short Distance Links
```bash
# Should be feasible with low cost
python3 hybrid_resilink_implementation.py --single-cycle --graphml-file real_world_topologies/Cesnet.graphml
```

### Test 2: International Links
```bash
# Should have penalties but be feasible
python3 hybrid_resilink_implementation.py --single-cycle --graphml-file real_world_topologies/Geant2012.graphml
```

### Test 3: Long Distance Links
```bash
# Should be infeasible due to distance
python3 hybrid_resilink_implementation.py --single-cycle --graphml-file real_world_topologies/Globalcenter.graphml
```

## 🔍 Debugging Geographic Issues

### Check Geographic Data Loading
```bash
# Verify GraphML parsing
python3 geographic_network_analyzer.py
```

### Inspect Distance Calculations
```python
from geographic_network_analyzer import GeographicNetworkAnalyzer
analyzer = GeographicNetworkAnalyzer()
analyzer.load_graphml_geography(Path("real_world_topologies/Bellcanada.graphml"))
distance = analyzer.calculate_distance("0", "47")  # Node IDs
print(f"Distance: {distance:.0f} km")
```

### Test Feasibility Analysis
```python
analysis = analyzer.analyze_link_feasibility("0", "47")
print(f"Feasible: {analysis['feasible']}")
print(f"Reason: {analysis['reason']}")
```

## 📈 Impact on Optimization

### Before Geographic Integration:
- All links considered equally feasible
- No distance constraints
- Unrealistic suggestions (e.g., 5000km links)

### After Geographic Integration:
- Distance-based feasibility scoring
- Realistic cost estimates
- Prioritizes practical implementations
- Clear geographic context for decisions

The geographic integration makes ResiLink suggestions much more practical and implementable in real-world scenarios!