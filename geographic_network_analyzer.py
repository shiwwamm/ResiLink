#!/usr/bin/env python3
"""
Geographic Network Analyzer for Hybrid ResiLink
==============================================

Integrates geographic information from GraphML files to make
network optimization decisions based on physical constraints.

Features:
- Extract geographic data from Internet Topology Zoo GraphML files
- Calculate great-circle distances between nodes
- Determine feasibility based on distance constraints
- Provide geographic context for link suggestions
"""

import xml.etree.ElementTree as ET
import math
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class GeographicNode:
    """Represents a network node with geographic information."""
    id: str
    label: str
    country: str
    latitude: float
    longitude: float
    node_type: str
    internal: bool = True

@dataclass
class GeographicConstraints:
    """Geographic constraints for network feasibility."""
    max_distance_km: float = 2000.0  # Maximum feasible link distance
    cross_country_penalty: float = 0.5  # Penalty for cross-country links
    international_penalty: float = 0.3  # Penalty for international links
    submarine_cable_threshold: float = 500.0  # Distance requiring submarine cables
    cost_per_km: float = 1000.0  # Cost per kilometer (USD)

class GeographicNetworkAnalyzer:
    """Analyzes network topology with geographic constraints."""
    
    def __init__(self, constraints: Optional[GeographicConstraints] = None):
        self.constraints = constraints or GeographicConstraints()
        self.nodes: Dict[str, GeographicNode] = {}
        self.distance_cache: Dict[Tuple[str, str], float] = {}
        
    def load_graphml_geography(self, graphml_file: Path) -> bool:
        """Load geographic information from GraphML file."""
        try:
            tree = ET.parse(graphml_file)
            root = tree.getroot()
            
            # Define namespace
            ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
            
            # Parse nodes with geographic data
            for node in root.findall('.//graphml:node', ns):
                node_id = node.get('id')
                node_data = {'id': node_id}
                
                for data in node.findall('graphml:data', ns):
                    key = data.get('key')
                    if key == 'd35':  # label/name
                        node_data['label'] = data.text or f'Node{node_id}'
                    elif key == 'd30':  # Latitude
                        try:
                            node_data['latitude'] = float(data.text) if data.text else 0.0
                        except ValueError:
                            node_data['latitude'] = 0.0
                    elif key == 'd34':  # Longitude
                        try:
                            node_data['longitude'] = float(data.text) if data.text else 0.0
                        except ValueError:
                            node_data['longitude'] = 0.0
                    elif key == 'd31':  # Country
                        node_data['country'] = data.text or 'Unknown'
                    elif key == 'd32':  # Type
                        node_data['type'] = data.text or 'Unknown'
                
                # Only add nodes with valid coordinates
                if 'latitude' in node_data and 'longitude' in node_data:
                    if node_data['latitude'] != 0.0 or node_data['longitude'] != 0.0:
                        self.nodes[node_id] = GeographicNode(
                            id=node_id,
                            label=node_data.get('label', f'Node{node_id}'),
                            country=node_data.get('country', 'Unknown'),
                            latitude=node_data['latitude'],
                            longitude=node_data['longitude'],
                            node_type=node_data.get('type', 'Unknown'),
                            internal=True
                        )
            
            logger.info(f"Loaded geographic data for {len(self.nodes)} nodes from {graphml_file.name}")
            return len(self.nodes) > 0
            
        except Exception as e:
            logger.error(f"Failed to load GraphML geography: {e}")
            return False
    
    def calculate_distance(self, node1_id: str, node2_id: str) -> float:
        """Calculate great-circle distance between two nodes in kilometers."""
        # Check cache first
        cache_key = tuple(sorted([node1_id, node2_id]))
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        if node1_id not in self.nodes or node2_id not in self.nodes:
            return float('inf')  # Unknown nodes
        
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        
        # Convert to radians
        lat1, lon1 = math.radians(node1.latitude), math.radians(node1.longitude)
        lat2, lon2 = math.radians(node2.latitude), math.radians(node2.longitude)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        earth_radius_km = 6371.0
        distance = earth_radius_km * c
        
        # Cache the result
        self.distance_cache[cache_key] = distance
        return distance
    
    def analyze_link_feasibility(self, src_id: str, dst_id: str) -> Dict:
        """Analyze the geographic feasibility of a link between two nodes."""
        if src_id not in self.nodes or dst_id not in self.nodes:
            return {
                'feasible': False,
                'reason': 'Geographic data not available',
                'distance_km': float('inf'),
                'cost_estimate': float('inf'),
                'feasibility_score': 0.0
            }
        
        src_node = self.nodes[src_id]
        dst_node = self.nodes[dst_id]
        distance = self.calculate_distance(src_id, dst_id)
        
        # Base feasibility checks
        feasible = True
        reasons = []
        feasibility_score = 1.0
        
        # Distance constraint
        if distance > self.constraints.max_distance_km:
            feasible = False
            reasons.append(f'Distance ({distance:.0f}km) exceeds maximum ({self.constraints.max_distance_km:.0f}km)')
            feasibility_score *= 0.1
        
        # Cross-country penalty
        if src_node.country != dst_node.country:
            feasibility_score *= self.constraints.international_penalty
            reasons.append(f'International link: {src_node.country} ↔ {dst_node.country}')
        
        # Submarine cable requirement
        if distance > self.constraints.submarine_cable_threshold:
            feasibility_score *= 0.7  # Submarine cables are more complex
            reasons.append(f'Requires submarine cable (>{self.constraints.submarine_cable_threshold}km)')
        
        # Cost estimation
        base_cost = distance * self.constraints.cost_per_km
        if src_node.country != dst_node.country:
            base_cost *= 1.5  # International links cost more
        if distance > self.constraints.submarine_cable_threshold:
            base_cost *= 3.0  # Submarine cables are expensive
        
        return {
            'feasible': feasible,
            'reason': '; '.join(reasons) if reasons else 'Geographically feasible',
            'distance_km': distance,
            'cost_estimate': base_cost,
            'feasibility_score': feasibility_score,
            'src_location': f"{src_node.label}, {src_node.country}",
            'dst_location': f"{dst_node.label}, {dst_node.country}",
            'link_type': self._classify_link_type(src_node, dst_node, distance),
            'geographic_context': {
                'same_country': src_node.country == dst_node.country,
                'requires_submarine': distance > self.constraints.submarine_cable_threshold,
                'distance_category': self._categorize_distance(distance)
            }
        }
    
    def _classify_link_type(self, src: GeographicNode, dst: GeographicNode, distance: float) -> str:
        """Classify the type of link based on geography."""
        if src.country != dst.country:
            if distance > self.constraints.submarine_cable_threshold:
                return 'International Submarine'
            else:
                return 'International Terrestrial'
        else:
            if distance > 1000:
                return 'Long-haul Domestic'
            elif distance > 200:
                return 'Regional'
            else:
                return 'Metropolitan'
    
    def _categorize_distance(self, distance: float) -> str:
        """Categorize distance for reporting."""
        if distance < 50:
            return 'Local'
        elif distance < 200:
            return 'Regional'
        elif distance < 500:
            return 'National'
        elif distance < 2000:
            return 'Continental'
        else:
            return 'Intercontinental'
    
    def get_feasible_neighbors(self, node_id: str, max_distance: Optional[float] = None) -> List[Tuple[str, float]]:
        """Get all feasible neighbors for a node within distance constraints."""
        if node_id not in self.nodes:
            return []
        
        max_dist = max_distance or self.constraints.max_distance_km
        neighbors = []
        
        for other_id in self.nodes:
            if other_id != node_id:
                distance = self.calculate_distance(node_id, other_id)
                if distance <= max_dist:
                    neighbors.append((other_id, distance))
        
        # Sort by distance
        neighbors.sort(key=lambda x: x[1])
        return neighbors
    
    def generate_geographic_report(self) -> Dict:
        """Generate a comprehensive geographic analysis report."""
        if not self.nodes:
            return {'error': 'No geographic data loaded'}
        
        countries = {}
        total_distances = []
        
        # Analyze by country
        for node in self.nodes.values():
            if node.country not in countries:
                countries[node.country] = []
            countries[node.country].append(node)
        
        # Calculate all pairwise distances
        node_ids = list(self.nodes.keys())
        for i, src_id in enumerate(node_ids):
            for dst_id in node_ids[i+1:]:
                distance = self.calculate_distance(src_id, dst_id)
                total_distances.append(distance)
        
        return {
            'total_nodes': len(self.nodes),
            'countries': {country: len(nodes) for country, nodes in countries.items()},
            'distance_statistics': {
                'min_km': min(total_distances) if total_distances else 0,
                'max_km': max(total_distances) if total_distances else 0,
                'avg_km': sum(total_distances) / len(total_distances) if total_distances else 0,
                'median_km': sorted(total_distances)[len(total_distances)//2] if total_distances else 0
            },
            'feasibility_constraints': {
                'max_distance_km': self.constraints.max_distance_km,
                'international_penalty': self.constraints.international_penalty,
                'submarine_threshold_km': self.constraints.submarine_cable_threshold
            },
            'geographic_coverage': {
                'latitude_range': [
                    min(node.latitude for node in self.nodes.values()),
                    max(node.latitude for node in self.nodes.values())
                ],
                'longitude_range': [
                    min(node.longitude for node in self.nodes.values()),
                    max(node.longitude for node in self.nodes.values())
                ]
            }
        }
    
    def save_geographic_data(self, output_file: Path):
        """Save geographic data to JSON file."""
        data = {
            'nodes': {
                node_id: {
                    'label': node.label,
                    'country': node.country,
                    'latitude': node.latitude,
                    'longitude': node.longitude,
                    'type': node.node_type
                }
                for node_id, node in self.nodes.items()
            },
            'constraints': {
                'max_distance_km': self.constraints.max_distance_km,
                'cross_country_penalty': self.constraints.cross_country_penalty,
                'international_penalty': self.constraints.international_penalty,
                'submarine_cable_threshold': self.constraints.submarine_cable_threshold,
                'cost_per_km': self.constraints.cost_per_km
            },
            'report': self.generate_geographic_report()
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Geographic data saved to {output_file}")


def main():
    """Test the geographic analyzer with Bell Canada topology."""
    analyzer = GeographicNetworkAnalyzer()
    
    # Load Bell Canada topology
    graphml_file = Path("real_world_topologies/Bellcanada.graphml")
    if not graphml_file.exists():
        print(f"❌ GraphML file not found: {graphml_file}")
        return
    
    print("🌍 Geographic Network Analyzer Test")
    print("=" * 40)
    
    # Load geographic data
    if analyzer.load_graphml_geography(graphml_file):
        print(f"✅ Loaded {len(analyzer.nodes)} nodes with geographic data")
        
        # Generate report
        report = analyzer.generate_geographic_report()
        print(f"📊 Countries: {list(report['countries'].keys())}")
        print(f"📏 Distance range: {report['distance_statistics']['min_km']:.0f} - {report['distance_statistics']['max_km']:.0f} km")
        
        # Test some link feasibility
        node_ids = list(analyzer.nodes.keys())[:5]
        print(f"\n🔍 Testing link feasibility:")
        
        for i in range(min(3, len(node_ids)-1)):
            src_id = node_ids[i]
            dst_id = node_ids[i+1]
            
            analysis = analyzer.analyze_link_feasibility(src_id, dst_id)
            src_node = analyzer.nodes[src_id]
            dst_node = analyzer.nodes[dst_id]
            
            feasible_icon = "✅" if analysis['feasible'] else "❌"
            print(f"{feasible_icon} {src_node.label} ↔ {dst_node.label}")
            print(f"   Distance: {analysis['distance_km']:.0f} km")
            print(f"   Type: {analysis['link_type']}")
            print(f"   Score: {analysis['feasibility_score']:.2f}")
            if not analysis['feasible']:
                print(f"   Reason: {analysis['reason']}")
        
        # Save data
        analyzer.save_geographic_data(Path("geographic_analysis.json"))
        print(f"\n💾 Geographic analysis saved to geographic_analysis.json")
        
    else:
        print("❌ Failed to load geographic data")


if __name__ == "__main__":
    main()