#!/usr/bin/env python3
"""
Enhanced Topology Parser for Internet Topology Zoo
=================================================

Comprehensive parser for GraphML/GML files that extracts rich metadata
including geographic coordinates, link speeds, network characteristics,
and temporal evolution data.

Academic Foundation:
- Graph parsing: NetworkX (Hagberg et al. 2008)
- Geographic analysis: Haversine formula for distances
- Network characterization: Knight et al. (2011) - Internet Topology Zoo

Usage:
    parser = EnhancedTopologyParser()
    network = parser.parse_graphml('real_world_topologies/Geant2012.graphml')
"""

import os
import sys
import json
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import networkx as nx
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NodeMetadata:
    """Rich metadata for network nodes."""
    id: str
    label: str
    country: str
    latitude: float
    longitude: float
    internal: bool
    node_type: str
    geocode_id: Optional[str] = None
    geocode_country: Optional[str] = None


@dataclass
class EdgeMetadata:
    """Rich metadata for network edges."""
    source: str
    target: str
    link_speed: Optional[str] = None
    link_speed_raw: Optional[float] = None
    link_speed_units: Optional[str] = None
    link_type: Optional[str] = None
    link_note: Optional[str] = None
    link_label: Optional[str] = None
    geographic_distance: Optional[float] = None
    cost_estimate: Optional[float] = None


@dataclass
class NetworkMetadata:
    """Rich metadata for the entire network."""
    name: str
    network_type: str
    geo_location: str
    geo_extent: str
    date_obtained: str
    date_year: str
    date_month: Optional[str] = None
    network_date: Optional[str] = None
    source: Optional[str] = None
    creator: Optional[str] = None
    layer: Optional[str] = None
    backbone: bool = False
    commercial: bool = False
    transit: bool = False
    developed: bool = False
    testbed: bool = False


@dataclass
class EnhancedNetwork:
    """Complete network representation with rich metadata."""
    graph: nx.Graph
    nodes: Dict[str, NodeMetadata]
    edges: Dict[Tuple[str, str], EdgeMetadata]
    metadata: NetworkMetadata
    academic_metrics: Dict[str, Any]
    geographic_analysis: Dict[str, Any]


class EnhancedTopologyParser:
    """Enhanced parser for Internet Topology Zoo data."""
    
    def __init__(self, data_dir: str = "real_world_topologies"):
        self.data_dir = Path(data_dir)
        self.parsed_networks = {}
        
        logger.info("Enhanced Topology Parser initialized")
    
    def parse_graphml(self, file_path: str) -> EnhancedNetwork:
        """Parse a GraphML file with full metadata extraction."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"GraphML file not found: {file_path}")
        
        logger.info(f"Parsing GraphML file: {file_path}")
        
        # Load graph with NetworkX
        try:
            G = nx.read_graphml(str(file_path))
        except Exception as e:
            raise ValueError(f"Failed to parse GraphML file: {e}")
        
        # Extract network metadata
        network_metadata = self._extract_network_metadata(G)
        
        # Extract node metadata
        nodes_metadata = self._extract_nodes_metadata(G)
        
        # Extract edge metadata with geographic calculations
        edges_metadata = self._extract_edges_metadata(G, nodes_metadata)
        
        # Calculate academic metrics
        academic_metrics = self._calculate_academic_metrics(G)
        
        # Perform geographic analysis
        geographic_analysis = self._perform_geographic_analysis(nodes_metadata, edges_metadata)
        
        # Create enhanced network object
        enhanced_network = EnhancedNetwork(
            graph=G,
            nodes=nodes_metadata,
            edges=edges_metadata,
            metadata=network_metadata,
            academic_metrics=academic_metrics,
            geographic_analysis=geographic_analysis
        )
        
        logger.info(f"Successfully parsed network: {network_metadata.name}")
        logger.info(f"  Nodes: {len(nodes_metadata)}, Edges: {len(edges_metadata)}")
        logger.info(f"  Geographic extent: {network_metadata.geo_extent}")
        
        return enhanced_network
    
    def _extract_network_metadata(self, G: nx.Graph) -> NetworkMetadata:
        """Extract network-level metadata from graph attributes."""
        graph_attrs = G.graph
        
        return NetworkMetadata(
            name=graph_attrs.get('Network', graph_attrs.get('label', 'Unknown')),
            network_type=graph_attrs.get('Type', 'Unknown'),
            geo_location=graph_attrs.get('GeoLocation', 'Unknown'),
            geo_extent=graph_attrs.get('GeoExtent', 'Unknown'),
            date_obtained=graph_attrs.get('DateObtained', 'Unknown'),
            date_year=graph_attrs.get('DateYear', 'Unknown'),
            date_month=graph_attrs.get('DateMonth'),
            network_date=graph_attrs.get('NetworkDate'),
            source=graph_attrs.get('Source'),
            creator=graph_attrs.get('Creator'),
            layer=graph_attrs.get('Layer'),
            backbone=bool(graph_attrs.get('Backbone', 0)),
            commercial=bool(graph_attrs.get('Commercial', 0)),
            transit=bool(graph_attrs.get('Transit', 0)),
            developed=bool(graph_attrs.get('Developed', 0)),
            testbed=bool(graph_attrs.get('Testbed', 0))
        )
    
    def _extract_nodes_metadata(self, G: nx.Graph) -> Dict[str, NodeMetadata]:
        """Extract rich metadata for all nodes."""
        nodes_metadata = {}
        
        for node_id, attrs in G.nodes(data=True):
            # Handle different node ID formats
            node_str = str(node_id)
            
            # Extract coordinates with fallbacks
            latitude = self._safe_float(attrs.get('Latitude', attrs.get('latitude', 0.0)))
            longitude = self._safe_float(attrs.get('Longitude', attrs.get('longitude', 0.0)))
            
            nodes_metadata[node_str] = NodeMetadata(
                id=node_str,
                label=attrs.get('label', attrs.get('Label', node_str)),
                country=attrs.get('Country', attrs.get('country', 'Unknown')),
                latitude=latitude,
                longitude=longitude,
                internal=bool(attrs.get('Internal', attrs.get('internal', 1))),
                node_type=attrs.get('type', attrs.get('Type', 'Router')),
                geocode_id=attrs.get('geocode_id'),
                geocode_country=attrs.get('geocode_country')
            )
        
        return nodes_metadata
    
    def _extract_edges_metadata(self, G: nx.Graph, nodes_metadata: Dict[str, NodeMetadata]) -> Dict[Tuple[str, str], EdgeMetadata]:
        """Extract rich metadata for all edges with geographic calculations."""
        edges_metadata = {}
        
        for source, target, attrs in G.edges(data=True):
            source_str, target_str = str(source), str(target)
            edge_key = (source_str, target_str)
            
            # Calculate geographic distance
            geographic_distance = self._calculate_geographic_distance(
                nodes_metadata.get(source_str),
                nodes_metadata.get(target_str)
            )
            
            # Extract link speed information
            link_speed_raw = self._safe_float(attrs.get('LinkSpeedRaw'))
            link_speed = attrs.get('LinkSpeed', attrs.get('link_speed'))
            link_speed_units = attrs.get('LinkSpeedUnits', attrs.get('link_speed_units'))
            
            # Estimate implementation cost
            cost_estimate = self._estimate_link_cost(
                geographic_distance, link_speed_raw, link_speed_units
            )
            
            edges_metadata[edge_key] = EdgeMetadata(
                source=source_str,
                target=target_str,
                link_speed=link_speed,
                link_speed_raw=link_speed_raw,
                link_speed_units=link_speed_units,
                link_type=attrs.get('LinkType', attrs.get('link_type')),
                link_note=attrs.get('LinkNote', attrs.get('link_note')),
                link_label=attrs.get('LinkLabel', attrs.get('link_label')),
                geographic_distance=geographic_distance,
                cost_estimate=cost_estimate
            )
        
        return edges_metadata
    
    def _calculate_geographic_distance(self, node1: Optional[NodeMetadata], node2: Optional[NodeMetadata]) -> Optional[float]:
        """Calculate geographic distance between two nodes using Haversine formula."""
        if not node1 or not node2:
            return None
        
        if node1.latitude == 0 and node1.longitude == 0:
            return None
        if node2.latitude == 0 and node2.longitude == 0:
            return None
        
        # Haversine formula
        lat1, lon1 = math.radians(node1.latitude), math.radians(node1.longitude)
        lat2, lon2 = math.radians(node2.latitude), math.radians(node2.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return c * r
    
    def _estimate_link_cost(self, distance: Optional[float], speed_raw: Optional[float], speed_units: Optional[str]) -> Optional[float]:
        """Estimate implementation cost based on distance and link speed."""
        if not distance:
            return None
        
        # Base cost per km (in thousands of USD)
        base_cost_per_km = 10.0
        
        # Speed multiplier
        speed_multiplier = 1.0
        if speed_raw:
            # Higher speeds cost more
            if speed_raw >= 10e9:  # 10 Gbps+
                speed_multiplier = 3.0
            elif speed_raw >= 1e9:  # 1 Gbps+
                speed_multiplier = 2.0
            elif speed_raw >= 100e6:  # 100 Mbps+
                speed_multiplier = 1.5
        
        return distance * base_cost_per_km * speed_multiplier
    
    def _calculate_academic_metrics(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Calculate comprehensive academic network metrics.
        
        Combines original Enhanced ResiLink metrics with new geographic features.
        Based on established network science literature:
        - Freeman (1977): Centrality measures
        - Albert et al. (2000): Robustness analysis  
        - Latora & Marchiori (2001): Efficiency measures
        - Watts & Strogatz (1998): Small-world properties
        - Fiedler (1973): Algebraic connectivity
        """
        try:
            if G.number_of_nodes() < 2:
                return {'error': 'Insufficient nodes for analysis'}
            
            # Basic graph properties
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            density = nx.density(G)
            
            # Connectivity analysis (Erdős & Rényi 1960)
            is_connected = nx.is_connected(G)
            n_components = nx.number_connected_components(G)
            
            # Centrality analysis (Freeman 1977)
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            if is_connected:
                closeness_centrality = nx.closeness_centrality(G)
            else:
                closeness_centrality = {node: 0.0 for node in G.nodes()}
            
            # Centrality statistics with Gini coefficients
            degree_values = list(degree_centrality.values())
            betweenness_values = list(betweenness_centrality.values())
            closeness_values = list(closeness_centrality.values())
            
            centrality_statistics = {
                'degree': {
                    'mean': np.mean(degree_values),
                    'std': np.std(degree_values),
                    'max': np.max(degree_values),
                    'gini': self._calculate_gini_coefficient(degree_values)
                },
                'betweenness': {
                    'mean': np.mean(betweenness_values),
                    'std': np.std(betweenness_values),
                    'max': np.max(betweenness_values),
                    'gini': self._calculate_gini_coefficient(betweenness_values)
                },
                'closeness': {
                    'mean': np.mean(closeness_values),
                    'std': np.std(closeness_values),
                    'max': np.max(closeness_values),
                    'gini': self._calculate_gini_coefficient(closeness_values)
                }
            }
            
            # Path analysis (Dijkstra 1959, Floyd-Warshall)
            if is_connected:
                avg_shortest_path = nx.average_shortest_path_length(G)
                diameter = nx.diameter(G)
                radius = nx.radius(G)
            else:
                avg_shortest_path = float('inf')
                diameter = float('inf')
                radius = float('inf')
            
            # Efficiency measures (Latora & Marchiori 2001)
            global_efficiency = nx.global_efficiency(G)
            local_efficiency = nx.local_efficiency(G)
            
            # Clustering analysis (Watts & Strogatz 1998)
            clustering_coefficient = nx.average_clustering(G)
            transitivity = nx.transitivity(G)
            
            # Robustness analysis (Albert et al. 2000)
            robustness_metrics = self._calculate_robustness_metrics(G)
            
            # Algebraic connectivity (Fiedler 1973)
            if is_connected and n_nodes > 2:
                try:
                    algebraic_connectivity = nx.algebraic_connectivity(G)
                except:
                    algebraic_connectivity = 0.0
            else:
                algebraic_connectivity = 0.0
            
            # Small-world properties (Watts & Strogatz 1998)
            small_world_metrics = self._calculate_small_world_metrics(G)
            
            # Network resilience (Holme et al. 2002)
            resilience_score = self._calculate_network_resilience(G)
            
            # Degree distribution analysis
            degrees = [d for n, d in G.degree()]
            degree_statistics = {
                'mean': np.mean(degrees),
                'std': np.std(degrees),
                'min': np.min(degrees),
                'max': np.max(degrees)
            }
            
            # Academic assessment
            academic_assessment = self._generate_academic_assessment(G, centrality_statistics, robustness_metrics)
            
            return {
                'basic_properties': {
                    'nodes': n_nodes,
                    'edges': n_edges,
                    'density': density,
                    'is_connected': is_connected,
                    'components': n_components
                },
                'path_metrics': {
                    'average_shortest_path': avg_shortest_path,
                    'diameter': diameter,
                    'radius': radius,
                    'global_efficiency': global_efficiency,
                    'local_efficiency': local_efficiency
                },
                'centrality_statistics': centrality_statistics,
                'clustering_metrics': {
                    'average_clustering': clustering_coefficient,
                    'transitivity': transitivity
                },
                'robustness_metrics': robustness_metrics,
                'algebraic_connectivity': algebraic_connectivity,
                'small_world_metrics': small_world_metrics,
                'resilience_score': resilience_score,
                'degree_statistics': degree_statistics,
                'academic_assessment': academic_assessment,
                
                # Legacy format for backward compatibility
                'degree_centrality': degree_centrality,
                'betweenness_centrality': betweenness_centrality,
                'closeness_centrality': closeness_centrality,
                'average_clustering': clustering_coefficient,
                'transitivity': transitivity,
                'node_connectivity': nx.node_connectivity(G),
                'edge_connectivity': nx.edge_connectivity(G)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating academic metrics: {e}")
            return {'error': str(e)}
    
    def _perform_geographic_analysis(self, nodes: Dict[str, NodeMetadata], edges: Dict[Tuple[str, str], EdgeMetadata]) -> Dict[str, Any]:
        """Perform comprehensive geographic analysis."""
        analysis = {}
        
        try:
            # Country distribution
            countries = [node.country for node in nodes.values() if node.country != 'Unknown']
            analysis['countries'] = list(set(countries))
            analysis['num_countries'] = len(set(countries))
            
            # Geographic span
            latitudes = [node.latitude for node in nodes.values() if node.latitude != 0]
            longitudes = [node.longitude for node in nodes.values() if node.longitude != 0]
            
            if latitudes and longitudes:
                analysis['geographic_bounds'] = {
                    'lat_min': min(latitudes),
                    'lat_max': max(latitudes),
                    'lon_min': min(longitudes),
                    'lon_max': max(longitudes)
                }
                
                analysis['geographic_center'] = {
                    'latitude': np.mean(latitudes),
                    'longitude': np.mean(longitudes)
                }
            
            # Link distance analysis
            distances = [edge.geographic_distance for edge in edges.values() if edge.geographic_distance]
            if distances:
                analysis['link_distances'] = {
                    'mean': np.mean(distances),
                    'std': np.std(distances),
                    'min': np.min(distances),
                    'max': np.max(distances),
                    'total': np.sum(distances)
                }
            
            # Cost analysis
            costs = [edge.cost_estimate for edge in edges.values() if edge.cost_estimate]
            if costs:
                analysis['implementation_costs'] = {
                    'mean': np.mean(costs),
                    'std': np.std(costs),
                    'min': np.min(costs),
                    'max': np.max(costs),
                    'total': np.sum(costs)
                }
            
        except Exception as e:
            logger.warning(f"Error in geographic analysis: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _calculate_gini_coefficient(self, values):
        """Calculate Gini coefficient for inequality measurement (Gini 1912)."""
        if not values or len(values) < 2:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0
    
    def _calculate_robustness_metrics(self, G):
        """
        Calculate network robustness metrics (Albert et al. 2000).
        
        Measures network tolerance to random failures and targeted attacks.
        """
        if G.number_of_nodes() < 3:
            return {'random_failure_threshold': 0.0, 'targeted_attack_threshold': 0.0, 'robustness_ratio': 0.0}
        
        # Random failure robustness
        random_threshold = self._simulate_random_failures(G.copy())
        
        # Targeted attack robustness (remove highest degree nodes)
        targeted_threshold = self._simulate_targeted_attacks(G.copy())
        
        return {
            'random_failure_threshold': random_threshold,
            'targeted_attack_threshold': targeted_threshold,
            'robustness_ratio': targeted_threshold / max(random_threshold, 0.001)
        }
    
    def _simulate_random_failures(self, G):
        """Simulate random node failures until network disconnects."""
        if not nx.is_connected(G):
            return 0.0
        
        original_nodes = G.number_of_nodes()
        removed = 0
        
        nodes = list(G.nodes())
        np.random.shuffle(nodes)
        
        for node in nodes:
            G.remove_node(node)
            removed += 1
            
            if not nx.is_connected(G):
                break
        
        return removed / original_nodes
    
    def _simulate_targeted_attacks(self, G):
        """Simulate targeted attacks on highest degree nodes."""
        if not nx.is_connected(G):
            return 0.0
        
        original_nodes = G.number_of_nodes()
        removed = 0
        
        while nx.is_connected(G) and G.number_of_nodes() > 1:
            # Find node with highest degree
            degrees = dict(G.degree())
            target_node = max(degrees, key=degrees.get)
            
            G.remove_node(target_node)
            removed += 1
        
        return removed / original_nodes
    
    def _calculate_small_world_metrics(self, G):
        """Calculate small-world network properties (Watts & Strogatz 1998)."""
        if not nx.is_connected(G) or G.number_of_nodes() < 4:
            return {'small_world_coefficient': 0.0, 'omega': 0.0, 'sigma': 0.0}
        
        try:
            # Small-world coefficient
            clustering = nx.average_clustering(G)
            path_length = nx.average_shortest_path_length(G)
            
            # Compare with random graph
            n = G.number_of_nodes()
            m = G.number_of_edges()
            p = 2 * m / (n * (n - 1))  # Edge probability
            
            # Expected values for random graph
            random_clustering = p
            random_path_length = np.log(n) / np.log(n * p) if n * p > 1 else float('inf')
            
            if random_clustering > 0 and random_path_length < float('inf'):
                small_world_coeff = (clustering / random_clustering) / (path_length / random_path_length)
            else:
                small_world_coeff = 0.0
            
            # Omega and Sigma metrics (Telesford et al. 2011)
            omega = self._calculate_omega(G)
            sigma = (clustering / random_clustering) / (path_length / random_path_length) if random_clustering > 0 and random_path_length < float('inf') else 0.0
            
            return {
                'small_world_coefficient': small_world_coeff,
                'omega': omega,
                'sigma': sigma
            }
            
        except Exception as e:
            logger.error(f"Error calculating small-world metrics: {e}")
            return {'small_world_coefficient': 0.0, 'omega': 0.0, 'sigma': 0.0}
    
    def _calculate_omega(self, G):
        """Calculate omega small-worldness metric (Telesford et al. 2011)."""
        try:
            clustering = nx.average_clustering(G)
            path_length = nx.average_shortest_path_length(G)
            
            # Generate random graph with same degree sequence
            degree_sequence = [d for n, d in G.degree()]
            random_G = nx.configuration_model(degree_sequence)
            random_G = nx.Graph(random_G)  # Remove multi-edges
            random_G.remove_edges_from(nx.selfloop_edges(random_G))  # Remove self-loops
            
            if nx.is_connected(random_G):
                random_path_length = nx.average_shortest_path_length(random_G)
            else:
                random_path_length = path_length
            
            # Generate lattice graph
            n = G.number_of_nodes()
            k = int(2 * G.number_of_edges() / n)  # Average degree
            if k >= 2 and n >= k:
                lattice_G = nx.watts_strogatz_graph(n, k, 0)  # p=0 gives regular lattice
                lattice_clustering = nx.average_clustering(lattice_G)
            else:
                lattice_clustering = clustering
            
            if lattice_clustering > 0 and random_path_length > 0:
                omega = (random_path_length / path_length) - (clustering / lattice_clustering)
            else:
                omega = 0.0
            
            return omega
            
        except Exception as e:
            logger.error(f"Error calculating omega: {e}")
            return 0.0
    
    def _calculate_network_resilience(self, G):
        """
        Calculate comprehensive network resilience score.
        
        Based on multiple resilience measures from literature:
        - Connectivity resilience (Fiedler 1973)
        - Structural resilience (Albert et al. 2000)  
        - Functional resilience (Holme et al. 2002)
        """
        if G.number_of_nodes() < 2:
            return 0.0
        
        # Connectivity resilience (30%)
        if nx.is_connected(G):
            connectivity_resilience = 1.0
            try:
                algebraic_conn = nx.algebraic_connectivity(G)
                connectivity_resilience = min(algebraic_conn / G.number_of_nodes(), 1.0)
            except:
                connectivity_resilience = 0.5
        else:
            connectivity_resilience = 0.0
        
        # Structural resilience (40%) - based on robustness metrics
        robustness = self._calculate_robustness_metrics(G)
        structural_resilience = (robustness['random_failure_threshold'] + robustness['targeted_attack_threshold']) / 2
        
        # Functional resilience (30%) - based on efficiency and clustering
        efficiency_resilience = nx.global_efficiency(G)
        clustering_resilience = nx.average_clustering(G)
        functional_resilience = (efficiency_resilience + clustering_resilience) / 2
        
        # Combined resilience score
        resilience_score = (0.3 * connectivity_resilience + 
                           0.4 * structural_resilience + 
                           0.3 * functional_resilience)
        
        return resilience_score
    
    def _generate_academic_assessment(self, G, centrality_stats, robustness_metrics):
        """Generate academic assessment of network properties."""
        assessment = []
        
        # Connectivity assessment
        if nx.is_connected(G):
            assessment.append("Network is fully connected (Erdős & Rényi 1960)")
        else:
            components = nx.number_connected_components(G)
            assessment.append(f"Network has {components} components - connectivity improvement needed")
        
        # Centrality assessment
        degree_gini = centrality_stats['degree']['gini']
        if degree_gini > 0.5:
            assessment.append("High degree inequality - potential hub vulnerability (Albert et al. 2000)")
        elif degree_gini < 0.3:
            assessment.append("Low degree inequality - distributed load (Freeman 1977)")
        
        # Robustness assessment
        targeted_threshold = robustness_metrics.get('targeted_attack_threshold', 0)
        if targeted_threshold < 0.1:
            assessment.append("Low targeted attack tolerance - vulnerable to hub failures")
        elif targeted_threshold > 0.3:
            assessment.append("High targeted attack tolerance - resilient to hub failures")
        
        # Efficiency assessment
        global_eff = nx.global_efficiency(G)
        if global_eff < 0.3:
            assessment.append("Low global efficiency - suboptimal routing")
        elif global_eff > 0.7:
            assessment.append("High global efficiency - optimal routing (Latora & Marchiori 2001)")
        
        return assessment
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float."""
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def parse_all_graphml_files(self) -> Dict[str, EnhancedNetwork]:
        """Parse all GraphML files in the data directory."""
        graphml_files = list(self.data_dir.glob("*.graphml"))
        
        logger.info(f"Found {len(graphml_files)} GraphML files to parse")
        
        parsed_networks = {}
        
        for file_path in graphml_files:
            try:
                network = self.parse_graphml(file_path)
                network_name = file_path.stem
                parsed_networks[network_name] = network
                
                logger.info(f"✅ Parsed {network_name}: {network.metadata.name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to parse {file_path}: {e}")
        
        logger.info(f"Successfully parsed {len(parsed_networks)} networks")
        return parsed_networks
    
    def save_enhanced_network(self, network: EnhancedNetwork, output_path: str):
        """Save enhanced network to JSON format."""
        output_data = {
            'metadata': asdict(network.metadata),
            'nodes': {k: asdict(v) for k, v in network.nodes.items()},
            'edges': {f"{k[0]}-{k[1]}": asdict(v) for k, v in network.edges.items()},
            'academic_metrics': network.academic_metrics,
            'geographic_analysis': network.geographic_analysis,
            'graph_data': {
                'nodes': list(network.graph.nodes()),
                'edges': list(network.graph.edges())
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Enhanced network saved to {output_path}")


def main():
    """Example usage of the enhanced topology parser."""
    parser = EnhancedTopologyParser()
    
    # Parse a specific network
    try:
        geant_network = parser.parse_graphml("real_world_topologies/Geant2012.graphml")
        
        print(f"\n🌐 Network: {geant_network.metadata.name}")
        print(f"📍 Location: {geant_network.metadata.geo_location}")
        print(f"📊 Nodes: {len(geant_network.nodes)}, Edges: {len(geant_network.edges)}")
        print(f"🌍 Countries: {geant_network.geographic_analysis.get('num_countries', 'Unknown')}")
        
        if 'link_distances' in geant_network.geographic_analysis:
            distances = geant_network.geographic_analysis['link_distances']
            print(f"📏 Avg link distance: {distances['mean']:.1f} km")
        
        # Save enhanced network
        parser.save_enhanced_network(geant_network, "data/processed/geant2012_enhanced.json")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()