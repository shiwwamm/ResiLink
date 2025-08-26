import requests
import networkx as nx
import json
import time
import os

class NetworkFeatureExtractor:
    def __init__(self, ryu_api_url="http://localhost:8080", output_file="network_features.json"):
        self.ryu_api_url = ryu_api_url
        self.output_file = output_file
        self.topo_graph = nx.Graph()  # Undirected graph for topology (switches and hosts)

    def fetch_topology_data(self):
        """Fetch switches, links, and hosts from Ryu REST API."""
        try:
            # Fetch switches
            switches_resp = requests.get(f"{self.ryu_api_url}/v1.0/topology/switches")
            switches_resp.raise_for_status()
            switches = switches_resp.json()

            # Fetch links
            links_resp = requests.get(f"{self.ryu_api_url}/v1.0/topology/links")
            links_resp.raise_for_status()
            links = links_resp.json()

            # Fetch hosts
            hosts_resp = requests.get(f"{self.ryu_api_url}/v1.0/topology/hosts")
            hosts_resp.raise_for_status()
            hosts = hosts_resp.json()

            return switches, links, hosts
        except requests.exceptions.RequestException as e:
            print(f"Error fetching topology data: {e}")
            return [], [], []

    def update_graph(self, switches, links, hosts):
        """Update NetworkX graph with switches, hosts, and links."""
        self.topo_graph.clear()

        # Add switches as nodes
        for switch in switches:
            dpid = int(switch["dpid"], 16)  # Convert hex dpid to int
            self.topo_graph.add_node(dpid, type="switch")

        # Add hosts as nodes
        for host in hosts:
            mac = host["mac"]  # Use MAC as unique identifier for hosts
            self.topo_graph.add_node(mac, type="host")

        # Add switch-switch links as edges
        for link in links:
            src_dpid = int(link["src"]["dpid"], 16)
            dst_dpid = int(link["dst"]["dpid"], 16)
            src_port = link["src"]["port_no"]
            dst_port = link["dst"]["port_no"]
            self.topo_graph.add_edge(src_dpid, dst_dpid,
                                     src_port=src_port, dst_port=dst_port, type="switch-switch")

        # Add host-switch links as edges
        for host in hosts:
            mac = host["mac"]
            switch_dpid = int(host["port"]["dpid"], 16)
            port_no = host["port"]["port_no"]
            self.topo_graph.add_edge(mac, switch_dpid,
                                     host_port="eth0", switch_port=port_no, type="host-switch")

    def extract_features(self):
        """Extract topology and centrality metrics and append to JSON file."""
        switches, links, hosts = self.fetch_topology_data()

        # Update graph with fetched data
        self.update_graph(switches, links, hosts)

        # Prepare topology data
        switch_ids = [int(sw["dpid"], 16) for sw in switches]
        host_macs = [host["mac"] for host in hosts]
        switch_switch_links = [
            {
                "src_dpid": int(link["src"]["dpid"], 16),
                "dst_dpid": int(link["dst"]["dpid"], 16),
                "src_port": link["src"]["port_no"],
                "dst_port": link["dst"]["port_no"]
            } for link in links
        ]
        host_switch_links = [
            {
                "host_mac": host["mac"],
                "switch_dpid": int(host["port"]["dpid"], 16),
                "host_port": "eth0",
                "switch_port": host["port"]["port_no"]
            } for host in hosts
        ]

        # Prepare centrality data
        centralities = {}
        if self.topo_graph.number_of_nodes() > 0:
            centralities["degree"] = nx.degree_centrality(self.topo_graph)
            centralities["betweenness"] = nx.betweenness_centrality(self.topo_graph)
            centralities["closeness"] = nx.closeness_centrality(self.topo_graph)
            # Add more centralities as needed, e.g.:
            # centralities["eigenvector"] = nx.eigenvector_centrality(self.topo_graph)
        else:
            centralities["degree"] = {}
            centralities["betweenness"] = {}
            centralities["closeness"] = {}

        # Structure output data for this update
        update_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "topology": {
                "switches": switch_ids,
                "hosts": host_macs,
                "switch_switch_links": switch_switch_links,
                "host_switch_links": host_switch_links
            },
            "centralities": {
                "degree": {str(k): v for k, v in centralities["degree"].items()},
                "betweenness": {str(k): v for k, v in centralities["betweenness"].items()},
                "closeness": {str(k): v for k, v in centralities["closeness"].items()}
            }
        }

        # Load existing data from file (if it exists)
        existing_data = []
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, "r") as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]  # Convert to list if needed
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error reading existing JSON file: {e}, starting with empty list")

        # Append new data
        existing_data.append(update_data)

        # Write updated data back to file
        try:
            with open(self.output_file, "w") as f:
                json.dump(existing_data, f, indent=4)
            print(f"Network features appended to {self.output_file}")
        except IOError as e:
            print(f"Error writing to JSON file: {e}")

def main():
    extractor = NetworkFeatureExtractor()
    while True:
        extractor.extract_features()
        time.sleep(5)  # Poll every 5 seconds; adjust as needed

if __name__ == "__main__":
    main()
