# GraphML Mininet Integration

This integration allows you to create Mininet topologies directly from GraphML files (like those from Internet Topology Zoo) and connect them to your updated SDN controller.

## Files Created

1. **`mininet_graphml_topology.py`** - Main script that parses GraphML files and creates Mininet topologies
2. **`test_graphml_mininet.py`** - Test script that demonstrates the complete workflow
3. **`sdn/updated_controller.py`** - Your updated SDN controller with STP support

## Quick Start

### 1. Make scripts executable
```bash
chmod +x mininet_graphml_topology.py test_graphml_mininet.py
```

### 2. Test with Bell Canada topology
```bash
sudo python3 test_graphml_mininet.py
```

### 3. Manual usage with any GraphML file
```bash
# Start the controller in one terminal
ryu-manager ryu.app.ofctl_rest ryu.app.rest_topology sdn/updated_controller.py --observe-links

# In another terminal, start Mininet with GraphML topology
sudo python3 mininet_graphml_topology.py real_world_topologies/Bellcanada.graphml
```

## Usage Examples

### Basic Usage
```bash
# Create topology from GraphML file
sudo python3 mininet_graphml_topology.py real_world_topologies/Bellcanada.graphml

# Save topology as JSON for later use
sudo python3 mininet_graphml_topology.py real_world_topologies/Geant2012.graphml --save-json geant_topology.json

# Run for specific duration (useful for automated testing)
sudo python3 mininet_graphml_topology.py real_world_topologies/Abilene.graphml --duration 300
```

### Advanced Usage
```bash
# Use custom controller settings
sudo python3 mininet_graphml_topology.py real_world_topologies/Bellcanada.graphml \
    --controller-ip 192.168.1.100 \
    --controller-port 6653

# Combine with your ResiLink testing
sudo python3 mininet_graphml_topology.py real_world_topologies/Bellcanada.graphml --duration 600 &
sleep 30  # Wait for network to stabilize
python3 hybrid_resilink_implementation.py --max-cycles 5
```

## Features

### GraphML Parser
- Extracts network topology from Internet Topology Zoo GraphML files
- Preserves node metadata (names, locations, types)
- Handles edge information (link types, labels)
- Converts to JSON format for easy integration

### Mininet Integration
- Creates OpenFlow switches for each network node
- Establishes links based on GraphML edge data
- Adds test hosts for connectivity verification
- Connects to your updated SDN controller

### Controller Compatibility
- Works with your `sdn/updated_controller.py`
- Supports Spanning Tree Protocol (STP)
- Provides REST API for network inspection
- Handles loop-free topology management

## Network Information

When you run the script, you'll see output like:
```
📖 Parsing GraphML file: real_world_topologies/Bellcanada.graphml
✅ Parsed GraphML: 48 nodes, 64 edges
🌐 Network: Bell Canada
📍 Location: Canada, USA
📊 Topology: 48 nodes, 64 edges

🔧 Creating switches...
   Switch s0: Cold Lake
   Switch s1: Grande Prairie
   Switch s2: Edmonton
   ...

🔗 Creating links...
   Link: Cold Lake <-> Edmonton
   Link: Grande Prairie <-> Edmonton
   ...

🖥️  Adding test hosts...
   Host h1 -> Cold Lake
   Host h2 -> Grande Prairie
   ...
```

## Integration with ResiLink

After starting the network, you can run your ResiLink optimization:

```bash
# Terminal 1: Start controller
ryu-manager ryu.app.ofctl_rest ryu.app.rest_topology sdn/updated_controller.py --observe-links

# Terminal 2: Start Mininet (keep running)
sudo python3 mininet_graphml_topology.py real_world_topologies/Bellcanada.graphml

# Terminal 3: Run ResiLink optimization
python3 hybrid_resilink_implementation.py --training-mode --max-cycles 8
```

## Troubleshooting

### Controller Connection Issues
```bash
# Check if controller REST API is running
curl http://localhost:8080/v1.0/topology/switches

# Check flow stats API
curl http://localhost:8080/stats/flow/1

# Check Mininet switches
sudo ovs-vsctl show
```

### Network Discovery Issues
```bash
# In Mininet CLI, check connectivity
mininet> pingall
mininet> dump
```

### Large Topology Performance
For large topologies (>30 nodes), consider:
- Reducing the number of test hosts
- Using shorter test durations
- Monitoring system resources

## Available GraphML Topologies

The `real_world_topologies/` directory contains many real-world network topologies:
- **Bellcanada.graphml** - Bell Canada network (48 nodes)
- **Geant2012.graphml** - European research network
- **Abilene.graphml** - Internet2 Abilene network
- **AttMpls.graphml** - AT&T MPLS network
- And many more...

Each represents a real ISP or research network topology from the Internet Topology Zoo dataset.