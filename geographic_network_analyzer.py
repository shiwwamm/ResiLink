#!/usr/bin/env python3
"""
Geographic Network Analyzer
============================

Analyzes network topology with geographic constraints for realistic
link feasibility assessment.

Academic Foundation:
- Haversine formula for geographic distance calculation
- Network geography: Knight et al. (2011) - Internet Topology Zoo
- Cost modeling: Realistic fiber deployment costs
"""

import math
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GeographicConstraints:
    """Geographic constraints for network link feasibility."""
    max_distance_km: float = 2000.0  # Maximum terrestrial link distance
    cross_country_penalty: float = 0.5  # Penalty for cross-country links
    international_penalty: float = 0.3  # Penalty for international links
    submarine_cable_threshold: float = 500.0  # Distance threshold for submarine cables
    cost_per_km: float = 1000.0  # Base cost per km in USD
    
    def get_distance_penalty(self, distance_km: float) -> float:
        """Calculate penalty based on distance."""
        if distance_km > self.max_distance_km:
            return 0.0  # Infeasible
        elif distance_km > self.submarine_cable_threshold:
            return 0.3  # Submarine cable - expensive
        elif distance_km > 1000:
            return 0.6  # Long distance - moderate penalty
        elif distance_km > 500:
            return 0.8  # Medium distance - small penalty
        else:
            return 1.0  # Short distance - no penalty


class GeographicNetworkAnalyzer:
    """Analyzes network topology with geographic constraints."""
    
    def __init__(self, constraints: Optional[GeographicConstraints] = None):
        self.constraints = constraints or GeographicConstraints()
        self.node_locations = {}  # Store node geographic locations
        
    def load_graphml_geography(self, graphml_path) -> bool:
        """Load geographic data from GraphML file."""
        try:
            import networkx as nx
            from pathlib import Path
            
            if not Path(graphml_path).exists():
                logger.warning(f"GraphML file not found: {graphml_path}")
                return False
            
            G = nx.read_graphml(str(graphml_path))
            
            # Extract node locations
            for node_id, attrs in G.nodes(data=True):
                lat = self._safe_float(attrs.get('Latitude', attrs.get('latitude', 0.0)))
                lon = self._safe_float(attrs.get('Longitude', attrs.get('longitude', 0.0)))
                country = attrs.get('Country', attrs.get('country', 'Unknown'))
                
                if lat != 0.0 or lon != 0.0:
                    self.node_locations[str(node_id)] = {
                        'latitude': lat,
                        'longitude': lon,
                        'country': country,
                        'label': attrs.get('label', attrs.get('Label', str(node_id)))
                    }
            
            logger.info(f"Loaded {len(self.node_locations)} node locations from {graphml_path}")
            return len(self.node_locations) > 0
            
        except Exception as e:
            logger.error(f"Failed to load geographic data: {e}")
            return False
    
    def analyze_link_feasibility(self, src_node: str, dst_node: str) -> Dict:
        """
        Analyze feasibility of a link between two nodes.
        
        Returns comprehensive analysis including:
        - Geographic distance
        - Feasibility score
        - Cost estimate
        - Link type (terrestrial/submarine)
        - Constraints violated
        """
        # Check if we have location data
        if src_node not in self.node_locations or dst_node not in self.node_locations:
            return {
                'feasible': True,  # Assume feasible if no geographic data
                'reason': 'No geographic data available',
                'feasibility_score': 1.0,
                'distance_km': None,
                'cost_estimate': None,
                'link_type': 'unknown',
                'src_location': 'Unknown',
                'dst_location': 'Unknown',
                'geographic_context': {
                    'distance_category': 'unknown',
                    'crosses_borders': False
                }
            }
        
        src_loc = self.node_locations[src_node]
        dst_loc = self.node_locations[dst_node]
        
        # Calculate geographic distance
        distance_km = self._calculate_haversine_distance(
            src_loc['latitude'], src_loc['longitude'],
            dst_loc['latitude'], dst_loc['longitude']
        )
        
        # Determine link type
        link_type = self._determine_link_type(distance_km, src_loc, dst_loc)
        
        # Calculate feasibility score
        feasibility_score = self._calculate_feasibility_score(distance_km, src_loc, dst_loc)
        
        # Determine if feasible
        feasible = feasibility_score > 0.1 and distance_km <= self.constraints.max_distance_km
        
        # Calculate cost estimate
        cost_estimate = self._estimate_link_cost(distance_km, link_type)
        
        # Determine reason if not feasible
        reason = self._get_infeasibility_reason(distance_km, feasibility_score, src_loc, dst_loc)
        
        return {
            'feasible': feasible,
            'reason': reason if not feasible else 'Link is geographically feasible',
            'feasibility_score': feasibility_score,
            'distance_km': distance_km,
            'cost_estimate': cost_estimate,
            'link_type': link_type,
            'src_location': f"{src_loc['label']} ({src_loc['country']})",
            'dst_location': f"{dst_loc['label']} ({dst_loc['country']})",
            'geographic_context': {
                'distance_category': self._categorize_distance(distance_km),
                'crosses_borders': src_loc['country'] != dst_loc['country'],
                'same_country': src_loc['country'] == dst_loc['country']
            }
        }
    
    def _calculate_haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula."""
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return c * r
    
    def _determine_link_type(self, distance_km: float, src_loc: Dict, dst_loc: Dict) -> str:
        """Determine the type of link based on distance and geography."""
        if distance_km > self.constraints.submarine_cable_threshold:
            return 'submarine_cable'
        elif src_loc['country'] != dst_loc['country']:
            return 'international_terrestrial'
        elif distance_km > 500:
            return 'long_distance_terrestrial'
        else:
            return 'regional_terrestrial'
    
    def _calculate_feasibility_score(self, distance_km: float, src_loc: Dict, dst_loc: Dict) -> float:
        """Calculate overall feasibility score (0-1)."""
        # Distance penalty
        distance_score = self.constraints.get_distance_penalty(distance_km)
        
        # Cross-border penalty
        if src_loc['country'] != dst_loc['country']:
            border_penalty = self.constraints.international_penalty
        else:
            border_penalty = 1.0
        
        # Combined score
        feasibility = distance_score * border_penalty
        
        return max(0.0, min(1.0, feasibility))
    
    def _estimate_link_cost(self, distance_km: float, link_type: str) -> float:
        """Estimate implementation cost in USD."""
        base_cost = distance_km * self.constraints.cost_per_km
        
        # Apply multipliers based on link type
        if link_type == 'submarine_cable':
            multiplier = 5.0  # Submarine cables are very expensive
        elif link_type == 'international_terrestrial':
            multiplier = 2.0  # International links have regulatory costs
        elif link_type == 'long_distance_terrestrial':
            multiplier = 1.5  # Long distance has economies of scale
        else:
            multiplier = 1.0  # Regional links are baseline
        
        return base_cost * multiplier
    
    def _categorize_distance(self, distance_km: float) -> str:
        """Categorize distance for human-readable output."""
        if distance_km < 100:
            return 'local'
        elif distance_km < 500:
            return 'regional'
        elif distance_km < 1000:
            return 'national'
        elif distance_km < 2000:
            return 'continental'
        else:
            return 'intercontinental'
    
    def _get_infeasibility_reason(self, distance_km: float, feasibility_score: float, 
                                   src_loc: Dict, dst_loc: Dict) -> str:
        """Get human-readable reason for infeasibility."""
        if distance_km > self.constraints.max_distance_km:
            return f"Distance ({distance_km:.0f} km) exceeds maximum ({self.constraints.max_distance_km:.0f} km)"
        elif feasibility_score < 0.1:
            return "Geographic constraints make this link impractical"
        elif distance_km > self.constraints.submarine_cable_threshold:
            return f"Requires expensive submarine cable ({distance_km:.0f} km)"
        else:
            return "Link is feasible"
    
    def _safe_float(self, value) -> float:
        """Safely convert value to float."""
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def analyze_graph(self, graph) -> Dict:
        """Analyze entire graph for geographic properties."""
        try:
            import networkx as nx
            
            if not isinstance(graph, nx.Graph):
                return {'error': 'Invalid graph object'}
            
            analysis = {
                'total_nodes': graph.number_of_nodes(),
                'total_edges': graph.number_of_edges(),
                'nodes_with_location': len(self.node_locations),
                'geographic_coverage': {}
            }
            
            # Analyze existing edges
            if self.node_locations:
                edge_distances = []
                for src, dst in graph.edges():
                    if str(src) in self.node_locations and str(dst) in self.node_locations:
                        src_loc = self.node_locations[str(src)]
                        dst_loc = self.node_locations[str(dst)]
                        distance = self._calculate_haversine_distance(
                            src_loc['latitude'], src_loc['longitude'],
                            dst_loc['latitude'], dst_loc['longitude']
                        )
                        edge_distances.append(distance)
                
                if edge_distances:
                    import numpy as np
                    analysis['edge_distances'] = {
                        'mean': np.mean(edge_distances),
                        'std': np.std(edge_distances),
                        'min': np.min(edge_distances),
                        'max': np.max(edge_distances)
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing graph: {e}")
            return {'error': str(e)}
