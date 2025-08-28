import requests
import networkx as nx
import json
import time
import os
import psutil
import datetime

class NetworkFeatureExtractor:
    def __init__(self, ryu_api_url="http://localhost:8080", output_file="network_features.json"):
        self.ryu_api_url = ryu_api_url
        self.output_file = output_file
        self.topo_graph = nx.Graph()

    def fetch_topology_data(self):
        """Fetch switches, links, and hosts from Ryu REST API."""
        try:
            switches_resp = requests.get(f"{self.ryu_api_url}/v1.0/topology/switches")
            switches_resp.raise_for_status()
            switches = switches_resp.json()

            links_resp = requests.get(f"{self.ryu_api_url}/v1.0/topology/links")
            links_resp.raise_for_status()
            links = links_resp.json()

            hosts_resp = requests.get(f"{self.ryu_api_url}/v1.0/topology/hosts")
            hosts_resp.raise_for_status()
            hosts = hosts_resp.json()

            return switches, links, hosts
        except requests.exceptions.RequestException as e:
            print(f"Error fetching topology data: {e}")
            return [], [], []

    def fetch_switch_stats(self, switches, links, hosts):
        """Fetch stats for each switch: aggregate flows, port stats, port desc, and flows."""
        switch_stats = {}
        port_map = {int(sw["dpid"]): set() for sw in switches}
        for link in links:
            src_dpid = int(link["src"]["dpid"])
            dst_dpid = int(link["dst"]["dpid"])
            port_map[src_dpid].add(int(link["src"]["port_no"]))
            port_map[dst_dpid].add(int(link["dst"]["port_no"]))
        for host in hosts:
            switch_dpid = int(host["port"]["dpid"])
            port_map[switch_dpid].add(int(host["port"]["port_no"]))

        for sw in switches:
            dpid_str = sw["dpid"]
            dpid_int = int(dpid_str)

            try:
                agg_resp = requests.get(f"{self.ryu_api_url}/stats/aggregateflow/{dpid_str}")
                agg_resp.raise_for_status()
                aggregate = agg_resp.json().get(dpid_str, [{}])[0]
            except:
                aggregate = {"flow_count": 0, "packet_count": 0, "byte_count": 0}

            try:
                portstats_resp = requests.get(f"{self.ryu_api_url}/stats/port/{dpid_str}")
                portstats_resp.raise_for_status()
                portstats_list = portstats_resp.json().get(dpid_str, [])
                portstats = {int(p["port_no"]): p for p in portstats_list}
            except:
                portstats = {}

            try:
                portdesc = {}
                for port_no in port_map[dpid_int]:
                    try:
                        portdesc_resp = requests.get(f"{self.ryu_api_url}/stats/portdesc/{dpid_str}/{port_no}")
                        portdesc_resp.raise_for_status()
                        portdesc_data = portdesc_resp.json().get(dpid_str, [{}])[0]
                        portdesc[port_no] = portdesc_data
                    except:
                        portdesc[port_no] = {}
            except:
                portdesc = {}

            try:
                flows_resp = requests.get(f"{self.ryu_api_url}/stats/flow/{dpid_str}")
                flows_resp.raise_for_status()
                flows = flows_resp.json().get(dpid_str, [])
            except:
                flows = []

            switch_stats[dpid_int] = {
                "aggregate": aggregate,
                "portstats": portstats,
                "portdesc": portdesc,
                "flows": flows
            }

        return switch_stats

    def fetch_controller_load(self):
        """Fetch controller CPU and memory load using psutil."""
        controller_cpu = 0.0
        controller_mem = 0.0
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = p.info['cmdline']
                if p.info['name'] in ('python', 'python3') and cmdline and 'ryu-manager' in ' '.join(cmdline):
                    p.cpu_percent()
                    time.sleep(0.2)
                    controller_cpu = p.cpu_percent() / psutil.cpu_count()
                    controller_mem = p.memory_percent()
                    return {"cpu_percent": controller_cpu, "memory_percent": controller_mem}
            except:
                pass
        # Fallback to system-wide metrics
        controller_cpu = psutil.cpu_percent(interval=0.2) / psutil.cpu_count()
        controller_mem = psutil.virtual_memory().percent
        return {"cpu_percent": controller_cpu, "memory_percent": controller_mem}

    def update_graph(self, switches, links, hosts):
        """Update NetworkX graph with switches, hosts, and links for linear topology s1-s2-s3-s4."""
        self.topo_graph.clear()
        for switch in switches:
            dpid = int(switch["dpid"])
            self.topo_graph.add_node(dpid, type="switch")
        for host in hosts:
            mac = host["mac"]
            self.topo_graph.add_node(mac, type="host")
        # Enforce linear topology: s1-s2-s3-s4
        valid_links = [(1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)]
        for link in links:
            src_dpid = int(link["src"]["dpid"])
            dst_dpid = int(link["dst"]["dpid"])
            if (src_dpid, dst_dpid) in valid_links:
                src_port = int(link["src"]["port_no"])
                dst_port = int(link["dst"]["port_no"])
                self.topo_graph.add_edge(src_dpid, dst_dpid,
                                         src_port=src_port, dst_port=dst_port, type="switch-switch")
        for host in hosts:
            mac = host["mac"]
            switch_dpid = int(host["port"]["dpid"])
            port_no = int(host["port"]["port_no"])
            self.topo_graph.add_edge(mac, switch_dpid,
                                     host_port="eth0", switch_port=port_no, type="host-switch")

    def extract_features(self):
        """Extract topology, stats, centrality metrics, and append to JSON with deltas."""
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        switches, links, hosts = self.fetch_topology_data()
        switch_stats = self.fetch_switch_stats(switches, links, hosts)
        controller_load = self.fetch_controller_load()
        self.update_graph(switches, links, hosts)

        switch_ids = [int(sw["dpid"]) for sw in switches]
        host_macs = [host["mac"] for host in hosts]
        nodes = []
        for sw_id in switch_ids:
            stats = switch_stats.get(sw_id, {})
            aggregate = stats.get("aggregate", {"flow_count": 0, "packet_count": 0, "byte_count": 0})
            flows = stats.get("flows", [])
            avg_duration = 0.0
            if flows:
                total_dur = sum(f['duration_sec'] + f['duration_nsec'] / 1e9 for f in flows)
                avg_duration = total_dur / len(flows)
            attr = {
                "type": "switch",
                "num_flows": aggregate["flow_count"],
                "total_packets": aggregate["packet_count"],
                "total_bytes": aggregate["byte_count"],
                "avg_flow_duration": avg_duration
            }
            nodes.append({"id": sw_id, "attributes": attr})
        for host in hosts:
            mac = host["mac"]
            ips = host.get("ipv4", []) + host.get("ipv6", [])
            attr = {"type": "host", "ips": ips}
            nodes.append({"id": mac, "attributes": attr})

        switch_switch_links = []
        for link in links:
            src_dpid = int(link["src"]["dpid"])
            dst_dpid = int(link["dst"]["dpid"])
            if (src_dpid, dst_dpid) in [(1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)]:
                src_port = int(link["src"]["port_no"])
                dst_port = int(link["dst"]["port_no"])
                src_stats = switch_stats.get(src_dpid, {}).get("portstats", {}).get(src_port, {})
                dst_stats = switch_stats.get(dst_dpid, {}).get("portstats", {}).get(dst_port, {})
                src_desc = switch_stats.get(src_dpid, {}).get("portdesc", {}).get(src_port, {})
                dst_desc = switch_stats.get(dst_dpid, {}).get("portdesc", {}).get(dst_port, {})
                bandwidth_mbps = min(src_desc.get("curr_speed", 0) / 1000.0, dst_desc.get("curr_speed", 0) / 1000.0)
                link_data = {
                    "src_dpid": src_dpid,
                    "dst_dpid": dst_dpid,
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "bandwidth_mbps": bandwidth_mbps,
                    "stats": {
                        "tx_packets": src_stats.get("tx_packets", 0),
                        "tx_bytes": src_stats.get("tx_bytes", 0),
                        "tx_dropped": src_stats.get("tx_dropped", 0),
                        "rx_packets": dst_stats.get("rx_packets", 0),
                        "rx_bytes": dst_stats.get("rx_bytes", 0),
                        "rx_dropped": dst_stats.get("rx_dropped", 0),
                        "duration_sec": max(src_stats.get("duration_sec", 0), dst_stats.get("duration_sec", 0))
                    }
                }
                switch_switch_links.append(link_data)

        host_switch_links = []
        for host in hosts:
            mac = host["mac"]
            switch_dpid = int(host["port"]["dpid"])
            switch_port = int(host["port"]["port_no"])
            sw_stats = switch_stats.get(switch_dpid, {}).get("portstats", {}).get(switch_port, {})
            sw_desc = switch_stats.get(switch_dpid, {}).get("portdesc", {}).get(switch_port, {})
            bandwidth_mbps = sw_desc.get("curr_speed", 0) / 1000.0
            link_data = {
                "host_mac": mac,
                "switch_dpid": switch_dpid,
                "host_port": "eth0",
                "switch_port": switch_port,
                "bandwidth_mbps": bandwidth_mbps,
                "stats": {
                    "tx_packets": sw_stats.get("tx_packets", 0),
                    "tx_bytes": sw_stats.get("tx_bytes", 0),
                    "tx_dropped": sw_stats.get("tx_dropped", 0),
                    "rx_packets": sw_stats.get("rx_packets", 0),
                    "rx_bytes": sw_stats.get("rx_bytes", 0),
                    "rx_dropped": sw_stats.get("rx_dropped", 0),
                    "duration_sec": sw_stats.get("duration_sec", 0)
                }
            }
            host_switch_links.append(link_data)

        centralities = {}
        if self.topo_graph.number_of_nodes() > 0:
            centralities["degree"] = nx.degree_centrality(self.topo_graph)
            centralities["betweenness"] = nx.betweenness_centrality(self.topo_graph)
            centralities["closeness"] = nx.closeness_centrality(self.topo_graph)
        else:
            centralities["degree"] = {}
            centralities["betweenness"] = {}
            centralities["closeness"] = {}

        update_data = {
            "timestamp": timestamp,
            "controller_data": controller_load,
            "topology": {
                "switches": switch_ids,
                "hosts": host_macs,
                "switch_switch_links": switch_switch_links,
                "host_switch_links": host_switch_links
            },
            "nodes": nodes,
            "centralities": {
                "degree": {str(k): v for k, v in centralities["degree"].items()},
                "betweenness": {str(k): v for k, v in centralities["betweenness"].items()},
                "closeness": {str(k): v for k, v in centralities["closeness"].items()}
            }
        }

        existing_data = self.load_existing_data()
        if existing_data:
            prev = existing_data[-1]
            prev_time = datetime.datetime.strptime(prev["timestamp"], "%Y-%m-%d %H:%M:%S")
            delta_time = (current_time - prev_time).total_seconds()
            update_data["time_delta_sec"] = delta_time

            prev_switches = set(prev["topology"]["switches"])
            curr_switches = set(switch_ids)
            prev_hosts = set(prev["topology"]["hosts"])
            curr_hosts = set(host_macs)
            changes = {
                "added_switches": list(curr_switches - prev_switches),
                "removed_switches": list(prev_switches - curr_switches),
                "added_hosts": list(curr_hosts - prev_hosts),
                "removed_hosts": list(prev_hosts - curr_hosts)
            }
            update_data["changes"] = changes

            delta_centralities = {}
            for c_type in ["degree", "betweenness", "closeness"]:
                delta = {}
                all_nodes = set(update_data["centralities"][c_type].keys()) | set(prev["centralities"][c_type].keys())
                for node in all_nodes:
                    curr_v = update_data["centralities"][c_type].get(node, 0)
                    prev_v = prev["centralities"][c_type].get(node, 0)
                    delta[node] = curr_v - prev_v
                delta_centralities[c_type] = delta
            update_data["delta_centralities"] = delta_centralities

            prev_nodes = {n["id"]: n["attributes"] for n in prev.get("nodes", [])}
            for node in update_data["nodes"]:
                if node["attributes"]["type"] == "switch":
                    id_ = node["id"]
                    if id_ in prev_nodes:
                        prev_attr = prev_nodes[id_]
                        node["attributes"]["delta_total_packets"] = node["attributes"]["total_packets"] - prev_attr.get("total_packets", 0)
                        node["attributes"]["delta_total_bytes"] = node["attributes"]["total_bytes"] - prev_attr.get("total_bytes", 0)

            prev_ss_links = {(l["src_dpid"], l["src_port"], l["dst_dpid"], l["dst_port"]): l.get("stats", {}) for l in prev["topology"].get("switch_switch_links", [])}
            for link in update_data["topology"]["switch_switch_links"]:
                key = (link["src_dpid"], link["src_port"], link["dst_dpid"], link["dst_port"])
                if key in prev_ss_links:
                    prev_stats = prev_ss_links[key]
                    delta_stats = {
                        "delta_tx_packets": link["stats"]["tx_packets"] - prev_stats.get("tx_packets", 0),
                        "delta_tx_bytes": link["stats"]["tx_bytes"] - prev_stats.get("tx_bytes", 0),
                        "delta_tx_dropped": link["stats"]["tx_dropped"] - prev_stats.get("tx_dropped", 0),
                        "delta_rx_packets": link["stats"]["rx_packets"] - prev_stats.get("rx_packets", 0),
                        "delta_rx_bytes": link["stats"]["rx_bytes"] - prev_stats.get("rx_bytes", 0),
                        "delta_rx_dropped": link["stats"]["rx_dropped"] - prev_stats.get("rx_dropped", 0)
                    }
                    if delta_time > 0:
                        delta_stats["tx_packet_rate_pps"] = delta_stats["delta_tx_packets"] / delta_time
                        delta_stats["tx_byte_rate_bps"] = delta_stats["delta_tx_bytes"] * 8 / delta_time
                        delta_stats["packet_loss_rate"] = (max(0, delta_stats["delta_tx_packets"] - delta_stats["delta_rx_packets"]) / delta_stats["delta_tx_packets"]) if delta_stats["delta_tx_packets"] > 0 else 0.0
                    link["delta_stats"] = delta_stats

            prev_hs_links = {(l["host_mac"], l["switch_dpid"], l["switch_port"]): l.get("stats", {}) for l in prev["topology"].get("host_switch_links", [])}
            for link in update_data["topology"]["host_switch_links"]:
                key = (link["host_mac"], link["switch_dpid"], link["switch_port"])
                if key in prev_hs_links:
                    prev_stats = prev_hs_links[key]
                    delta_stats = {
                        "delta_tx_packets": link["stats"]["tx_packets"] - prev_stats.get("tx_packets", 0),
                        "delta_tx_bytes": link["stats"]["tx_bytes"] - prev_stats.get("tx_bytes", 0),
                        "delta_tx_dropped": link["stats"]["tx_dropped"] - prev_stats.get("tx_dropped", 0),
                        "delta_rx_packets": link["stats"]["rx_packets"] - prev_stats.get("rx_packets", 0),
                        "delta_rx_bytes": link["stats"]["rx_bytes"] - prev_stats.get("rx_bytes", 0),
                        "delta_rx_dropped": link["stats"]["rx_dropped"] - prev_stats.get("rx_dropped", 0)
                    }
                    if delta_time > 0:
                        delta_stats["tx_packet_rate_pps"] = delta_stats["delta_tx_packets"] / delta_time
                        delta_stats["tx_byte_rate_bps"] = delta_stats["delta_tx_bytes"] * 8 / delta_time
                        delta_stats["packet_loss_rate"] = (max(0, delta_stats["delta_tx_packets"] - delta_stats["delta_rx_packets"]) / delta_stats["delta_tx_packets"]) if delta_stats["delta_tx_packets"] > 0 else 0.0
                    link["delta_stats"] = delta_stats
        else:
            update_data["time_delta_sec"] = 0.0
            update_data["changes"] = {}
            update_data["delta_centralities"] = {}

        existing_data.append(update_data)
        try:
            with open(self.output_file, "w") as f:
                json.dump(existing_data, f, indent=4)
            print(f"Network features appended to {self.output_file} at {timestamp}")
        except IOError as e:
            print(f"Error writing to JSON file: {e}")

    def load_existing_data(self):
        """Load existing JSON data from file."""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, "r") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else [data]
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error reading existing JSON file: {e}, starting with empty list")
        return []

def main():
    extractor = NetworkFeatureExtractor()
    while True:
        extractor.extract_features()
        time.sleep(5)

if __name__ == "__main__":
    main()
