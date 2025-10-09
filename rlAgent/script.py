import requests
import networkx as nx
import json
import time
import os
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.optim as optim
import logging
import copy

# Setup logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ryu_errors.log'),
        logging.StreamHandler()  # Also print to console
    ]
)

class NetworkFeatureExtractor:
    def __init__(self, ryu_api_url="http://localhost:8080", output_file="network_features.json"):
        self.ryu_api_url = ryu_api_url
        self.output_file = output_file
        self.topo_graph = nx.Graph()
        self.session = requests.Session()
        self.session.timeout = 5  # 5-second timeout for API calls

    def fetch_topology_data(self):
        try:
            switches_resp = self.session.get(f"{self.ryu_api_url}/v1.0/topology/switches")
            switches_resp.raise_for_status()
            switches = switches_resp.json()
            logging.info(f"Fetched switches: {len(switches)} switches")
            if not switches:
                logging.warning("No switches returned from Ryu API")

            links_resp = self.session.get(f"{self.ryu_api_url}/v1.0/topology/links")
            links_resp.raise_for_status()
            links = links_resp.json()
            logging.info(f"Fetched links: {len(links)} links")
            if not links:
                logging.warning("No links returned from Ryu API")

            hosts_resp = self.session.get(f"{self.ryu_api_url}/v1.0/topology/hosts")
            hosts_resp.raise_for_status()
            hosts = hosts_resp.json()
            logging.info(f"Fetched hosts: {len(hosts)} hosts")
            if not hosts:
                logging.warning("No hosts returned from Ryu API")

            return switches, links, hosts
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch topology data: {e}")
            raise

    def fetch_switch_stats(self, switches, links, hosts):
        switch_stats = {}
        port_map = {int(sw["dpid"], 16): set() for sw in switches}
        for link in links:
            src_dpid = int(link["src"]["dpid"], 16)
            dst_dpid = int(link["dst"]["dpid"], 16)
            port_map[src_dpid].add(int(link["src"]["port_no"], 16))
            port_map[dst_dpid].add(int(link["dst"]["port_no"], 16))
        for host in hosts:
            switch_dpid = int(host["port"]["dpid"], 16)
            port_map[switch_dpid].add(int(host["port"]["port_no"], 16))

        for sw in switches:
            dpid_int = int(sw["dpid"], 16)
            dpid_str = str(dpid_int) 

            try:
                agg_resp = self.session.get(f"{self.ryu_api_url}/stats/aggregateflow/{dpid_int}")
                agg_resp.raise_for_status()
                aggregate = agg_resp.json().get(dpid_str, [{}])[0]
            except:
                logging.warning(f"Failed to fetch aggregate stats for switch {dpid_str}, using defaults")
                aggregate = {"flow_count": 0, "packet_count": 0, "byte_count": 0}

            try:
                portstats_resp = self.session.get(f"{self.ryu_api_url}/stats/port/{dpid_int}")
                portstats_resp.raise_for_status()
                portstats_list = portstats_resp.json().get(dpid_str, [])
                portstats = {int(p["port_no"]) if str(p["port_no"]).isdigit() else -1: p for p in portstats_list}
            except:
                logging.warning(f"Failed to fetch port stats for switch {dpid_str}, using defaults")
                portstats = {}

            try:
                portdesc = {}
                for port_no in port_map[dpid_int]:
                    try:
                        portdesc_resp = self.session.get(f"{self.ryu_api_url}/stats/portdesc/{dpid_int}/{port_no}")
                        portdesc_resp.raise_for_status()
                        portdesc_data = portdesc_resp.json().get(dpid_str, [{}])[0]
                        portdesc[port_no] = portdesc_data
                    except:
                        logging.warning(f"Failed to fetch port desc for switch {dpid_str}, port {port_no}")
                        portdesc[port_no] = {}
            except:
                logging.warning(f"Failed to fetch port descriptions for switch {dpid_str}")
                portdesc = {}

            try:
                flows_resp = self.session.get(f"{self.ryu_api_url}/stats/flow/{dpid_int}")
                flows_resp.raise_for_status()
                flows = flows_resp.json().get(dpid_str, [])
            except:
                logging.warning(f"Failed to fetch flows for switch {dpid_str}, using defaults")
                flows = []

            switch_stats[dpid_int] = {
                "aggregate": aggregate,
                "portstats": portstats,
                "portdesc": portdesc,
                "flows": flows
            }
            

        return switch_stats

    def fetch_controller_load(self):
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
        controller_cpu = psutil.cpu_percent(interval=0.2) / psutil.cpu_count()
        controller_mem = psutil.virtual_memory().percent
        return {"cpu_percent": controller_cpu, "memory_percent": controller_mem}

    def fetch_available_ports(self, dpid):
        try:
            portdesc_resp = self.session.get(f"{self.ryu_api_url}/stats/portdesc/{dpid}")
            portdesc_resp.raise_for_status()
            ports = portdesc_resp.json().get(str(dpid), [])
            logging.info(f"Fetched {len(ports)} ports for switch {dpid}")
            # Get used ports from topology
            used_ports = set()
            switches, links, _ = self.fetch_topology_data()
            for link in links:
                if int(link["src"]["dpid"], 16) == dpid:
                    used_ports.add(int(link["src"]["port_no"], 16))
                if int(link["dst"]["dpid"], 16) == dpid:
                    used_ports.add(int(link["dst"]["port_no"], 16))
            # Filter for ports that are up and not used
            available_ports = []
            for p in ports:
                try:
                    port_no = int(p["port_no"], 16)  # Convert port_no to int (handles both string and int)
                    if port_no not in used_ports and p.get("curr_speed", 0) > 0:
                        available_ports.append(port_no)
                except (ValueError, TypeError):
                    logging.warning(f"Invalid port_no {p.get('port_no')} for switch {dpid}, skipping")
            logging.info(f"Available ports for switch {dpid}: {available_ports}")
            return available_ports
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch ports for switch {dpid}: {e}")
            return []

    def update_graph(self, switches, links, hosts):
        self.topo_graph.clear()
        for switch in switches:
            dpid = int(switch["dpid"], 16)
            self.topo_graph.add_node(dpid, type="switch")
        for host in hosts:
            mac = host["mac"]
            self.topo_graph.add_node(mac, type="host")
        for link in links:
            src_dpid = int(link["src"]["dpid"], 16)
            dst_dpid = int(link["dst"]["dpid"], 16)
            src_port = int(link["src"]["port_no"], 16)
            dst_port = int(link["dst"]["port_no"], 16)
            self.topo_graph.add_edge(src_dpid, dst_dpid,
                                     src_port=src_port, dst_port=dst_port, type="switch-switch")
        for host in hosts:
            mac = host["mac"]
            switch_dpid = int(host["port"]["dpid"], 16)
            port_no = int(host["port"]["port_no"], 16)
            self.topo_graph.add_edge(mac, switch_dpid,
                                     host_port="eth0", switch_port=port_no, type="host-switch")
        logging.info(f"Graph updated: {self.topo_graph.number_of_nodes()} nodes, {self.topo_graph.number_of_edges()} edges")
        

    def extract_features(self):
        try:
            switches, links, hosts = self.fetch_topology_data()
        except Exception as e:
            logging.error(f"Failed to fetch topology data in extract_features: {e}")
            raise
        switch_stats = self.fetch_switch_stats(switches, links, hosts)
        print("got switch stats")
        controller_load = self.fetch_controller_load()
        self.update_graph(switches, links, hosts)
        print("updated graph")

        switch_ids = [int(sw["dpid"], 16) for sw in switches]
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
            src_dpid = int(link["src"]["dpid"], 16)
            dst_dpid = int(link["dst"]["dpid"], 16)
            src_port = int(link["src"]["port_no"], 16)
            dst_port = int(link["dst"]["port_no"], 16)
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
                    "rx_packets": src_stats.get("rx_packets", 0),
                    "rx_bytes": src_stats.get("rx_bytes", 0),
                    "rx_dropped": src_stats.get("rx_dropped", 0),
                    "duration_sec": src_stats.get("duration_sec", 0)
                }
            }
            switch_switch_links.append(link_data)

        host_switch_links = []
        for host in hosts:
            mac = host["mac"]
            switch_dpid = int(host["port"]["dpid"], 16)
            switch_port = int(host["port"]["port_no"], 16)
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
            logging.warning("Graph has no nodes, centralities are empty")

        update_data = {
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

        try:
            with open(self.output_file, "w") as f:
                json.dump(update_data, f, indent=4)
            logging.info(f"Network features written to {self.output_file}")
        except IOError as e:
            logging.error(f"Error writing to JSON file: {e}")
            raise

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, alpha=0.2):
        super(GATLayer, self).__init__()
        self.lin = Linear(in_dim, out_dim)
        self.att = Linear(2 * out_dim + edge_dim, 1)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj, edge_attr):
        B, N, _ = x.shape
        h = self.lin(x)
        h_i = h.unsqueeze(2).repeat(1, 1, N, 1)
        h_j = h.unsqueeze(1).repeat(1, N, 1, 1)
        a_input = torch.cat([h_i, h_j, edge_attr], dim=-1)
        e = self.leakyrelu(self.att(a_input).squeeze(-1))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)

def build_graph_data(entry):
    topology = entry['topology']
    centralities = entry['centralities']
    nodes_data = {str(node['id']): node['attributes'] for node in entry['nodes']}
    G = nx.Graph()
    for node_id in nodes_data:
        G.add_node(node_id, **nodes_data[node_id])
    all_links = topology['switch_switch_links'] + topology['host_switch_links']
    for link in all_links:
        if 'host_mac' in link:
            src = link['host_mac']
            dst = str(link['switch_dpid'])
        else:
            src = str(link['src_dpid'])
            dst = str(link['dst_dpid'])
        G.add_edge(src, dst, **link)
    node_list = list(G.nodes())
    num_nodes = len(node_list)
    node_id_to_idx = {node: idx for idx, node in enumerate(node_list)}
    features = []
    for node_id in node_list:
        deg = centralities.get('degree', {}).get(node_id, 0.0)
        bet = centralities.get('betweenness', {}).get(node_id, 0.0)
        clo = centralities.get('closeness', {}).get(node_id, 0.0)
        base_f = [deg, bet, clo]
        attrs = nodes_data.get(node_id, {})
        if attrs.get('type') == 'switch':
            switch_f = [
                attrs.get('num_flows', 0.0),
                attrs.get('total_packets', 0.0),
                attrs.get('total_bytes', 0.0),
                attrs.get('avg_flow_duration', 0.0)
            ]
        else:
            switch_f = [0.0] * 4
        f = base_f + switch_f
        features.append(f)
    scaler = StandardScaler()
    features = scaler.fit_transform(features) if np.any(features) else features
    x = torch.tensor(features, dtype=torch.float)
    adj_np = nx.to_numpy_array(G)
    adj = torch.tensor(adj_np, dtype=torch.float)
    adj += torch.eye(num_nodes)
    edge_dim = 8
    edge_attr_dense = torch.zeros(num_nodes, num_nodes, edge_dim)
    stats_keys = ['tx_packets', 'tx_bytes', 'tx_dropped', 'rx_packets', 'rx_bytes', 'rx_dropped', 'duration_sec']
    for src, dst, link_data in G.edges(data=True):
        i = node_id_to_idx[src]
        j = node_id_to_idx[dst]
        stats = link_data.get('stats', {})
        bandwidth = link_data.get('bandwidth_mbps', 0.0)
        e_f = [bandwidth] + [stats.get(k, 0.0) for k in stats_keys]
        edge_attr_dense[i, j] = torch.tensor(e_f)
        edge_attr_dense[j, i] = torch.tensor(e_f)
    return x, adj, edge_attr_dense, node_list, num_nodes

def generate_embeddings(data_list):
    x_list = []
    adj_list = []
    edge_attr_list = []
    node_lists = []
    num_nodes_list = []
    for entry in data_list:
        x, adj, edge_attr, node_list, num_nodes = build_graph_data(entry)
        x_list.append(x)
        adj_list.append(adj)
        edge_attr_list.append(edge_attr)
        node_lists.append(node_list)
        num_nodes_list.append(num_nodes)
    B = len(data_list)
    max_N = max(num_nodes_list)
    in_dim = x_list[0].shape[1] if x_list else 7
    edge_dim = edge_attr_list[0].shape[-1] if edge_attr_list else 8
    x_batched = torch.zeros(B, max_N, in_dim)
    adj_batched = torch.zeros(B, max_N, max_N)
    edge_attr_batched = torch.zeros(B, max_N, max_N, edge_dim)
    for b in range(B):
        N = num_nodes_list[b]
        x_batched[b, :N] = x_list[b]
        adj_batched[b, :N, :N] = adj_list[b]
        edge_attr_batched[b, :N, :N] = edge_attr_list[b]
    gat = GATLayer(in_dim=in_dim, out_dim=8, edge_dim=edge_dim)
    embeddings_batched = gat(x_batched, adj_batched, edge_attr_batched)
    embeddings = {}
    for b in range(B):
        N = num_nodes_list[b]
        emb = embeddings_batched[b, :N]
        for i, node in enumerate(node_lists[b]):
            embeddings[str(node)] = emb[i].detach().numpy()
    return embeddings

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

def build_graph(network_features):
    G = nx.Graph()
    topology = network_features['topology']
    switches = [str(s) for s in topology['switches']]
    hosts = topology['hosts']
    nodes = switches + hosts
    for node in nodes:
        G.add_node(node)
    for link in topology['switch_switch_links']:
        src, dst = str(link['src_dpid']), str(link['dst_dpid'])
        G.add_edge(src, dst)
    for link in topology['host_switch_links']:
        host, switch = link['host_mac'], str(link['switch_dpid'])
        G.add_edge(host, switch)
    return G, switches, hosts

def get_possible_links(G, switches, hosts):
    possible_links = []
    for i in range(len(switches)):
        for j in range(i + 1, len(switches)):
            u, v = switches[i], switches[j]
            if not G.has_edge(u, v):
                possible_links.append((u, v))
    logging.info(f"Possible links: {len(possible_links)}")
    return possible_links

def get_state(G, embeddings, centralities):
    emb_array = np.array([embeddings.get(node, np.zeros(8)) for node in G.nodes()])
    mean_emb = np.mean(emb_array, axis=0)
    avg_degree = np.mean(list(centralities['degree'].values())) if centralities['degree'] else 0.0
    avg_between = np.mean(list(centralities['betweenness'].values())) if centralities['betweenness'] else 0.0
    avg_close = np.mean(list(centralities['closeness'].values())) if centralities['closeness'] else 0.0
    state = np.concatenate([mean_emb, [avg_degree, avg_between, avg_close]])
    return torch.tensor(state, dtype=torch.float)

def update_centralities(G):
    degree = nx.degree_centrality(G)
    between = nx.betweenness_centrality(G)
    close = nx.closeness_centrality(G)
    return {'degree': degree, 'betweenness': between, 'closeness': close}

def compute_reward(old_state, new_state):
    delta_close = new_state[-1] - old_state[-1]
    delta_between = old_state[-2] - new_state[-2]
    delta_degree = new_state[-3] - old_state[-3]
    cost = -0.1
    return 0.5 * delta_close + 0.3 * delta_degree + 0.2 * delta_between + cost

def add_suggested_link(network_features, suggested_link):
    """
    Add the suggested link to a copy of the network features, creating an updated topology.
    Adds bidirectional link entries to switch_switch_links and recomputes centralities.
    Node attributes remain unchanged, as adding a link doesn't affect them without simulation.
    """
    new_data = copy.deepcopy(network_features)
    
    src_dpid = int(suggested_link['src_dpid'])
    dst_dpid = int(suggested_link['dst_dpid'])
    src_port = int(suggested_link.get('src_port', 0)) if suggested_link.get('src_port') != 'unavailable' else 0
    dst_port = int(suggested_link.get('dst_port', 0)) if suggested_link.get('dst_port') != 'unavailable' else 0
    
    forward_link = {
        "src_dpid": src_dpid,
        "dst_dpid": dst_dpid,
        "src_port": src_port,
        "dst_port": dst_port,
        "bandwidth_mbps": 0.0,
        "stats": {
            "tx_packets": 0,
            "tx_bytes": 0,
            "tx_dropped": 0,
            "rx_packets": 0,
            "rx_bytes": 0,
            "rx_dropped": 0,
            "duration_sec": 0
        }
    }
    
    reverse_link = {
        "src_dpid": dst_dpid,
        "dst_dpid": src_dpid,
        "src_port": dst_port,
        "dst_port": src_port,
        "bandwidth_mbps": 0.0,
        "stats": {
            "tx_packets": 0,
            "tx_bytes": 0,
            "tx_dropped": 0,
            "rx_packets": 0,
            "rx_bytes": 0,
            "rx_dropped": 0,
            "duration_sec": 0
        }
    }
    
    new_data['topology']['switch_switch_links'].append(forward_link)
    new_data['topology']['switch_switch_links'].append(reverse_link)
    
    G = nx.Graph()
    topology = new_data['topology']
    for node in new_data['nodes']:
        G.add_node(str(node['id']), **node['attributes'])
    for link in topology['switch_switch_links']:
        G.add_edge(str(link['src_dpid']), str(link['dst_dpid']), **link)
    for link in topology['host_switch_links']:
        G.add_edge(link['host_mac'], str(link['switch_dpid']), **link)
    
    if G.number_of_nodes() > 0:
        new_data['centralities'] = {
            "degree": {str(k): v for k, v in nx.degree_centrality(G).items()},
            "betweenness": {str(k): v for k, v in nx.betweenness_centrality(G).items()},
            "closeness": {str(k): v for k, v in nx.closeness_centrality(G).items()}
        }
    else:
        new_data['centralities'] = {
            "degree": {},
            "betweenness": {},
            "closeness": {}
        }
    
    return new_data

def train_rl(policy, optimizer, G, embeddings, centralities, action_to_link):
    num_episodes = 50
    total_reward = 0.0
    for episode in range(num_episodes):
        G_current = G.copy()
        curr_centralities = {k: dict(v) for k, v in centralities.items()}
        state = get_state(G_current, embeddings, curr_centralities)
        probs = policy(state.unsqueeze(0))
        action = torch.multinomial(probs, 1).item()
        u, v = action_to_link[action]
        G_new = G_current.copy()
        G_new.add_edge(u, v)
        new_centralities = update_centralities(G_new)
        new_state = get_state(G_new, embeddings, new_centralities)
        reward = compute_reward(state.numpy(), new_state.numpy())
        total_reward += reward
        log_prob = torch.log(probs[0, action])
        loss = -log_prob * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_reward = total_reward / num_episodes
    logging.info(f"RL training completed, average reward: {avg_reward:.4f}")
    return avg_reward

def main(max_cycles=5, break_threshold=0.01):
    extractor = NetworkFeatureExtractor()
    state_dim = 8 + 3
    action_space_size_placeholder = 20
    policy = PolicyNetwork(state_dim, action_space_size_placeholder)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    previous_reward = -np.inf
    G = None
    centralities = {}
    all_suggested_links = []  # To store all suggested links across cycles

    # Ensure network_features.json exists
    if not os.path.exists(extractor.output_file):
        try:
            print(f"{extractor.output_file} does not exist. Generating initial network features...")
            extractor.extract_features()
            logging.info(f"Generated initial {extractor.output_file}")
        except Exception as e:
            logging.error(f"Failed to generate initial network features: {e}")
            print(f"Failed to generate initial network features: {e}")
            return

    # Load original network features
    try:
        with open(extractor.output_file, 'r') as f:
            original_network_features = json.load(f)
        logging.info(f"Loaded original network features from {extractor.output_file}")
    except Exception as e:
        logging.error(f"Error loading original network features: {e}")
        print(f"Failed to load original network features: {e}")
        return

    for cycle in range(max_cycles):
        print(f"\n--- Cycle {cycle + 1} ---")
        try:
            extractor.extract_features()
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            print(f"Failed to extract features: {e}")
            continue
        try:
            with open(extractor.output_file, 'r') as f:
                network_features = json.load(f)
            logging.info(f"Loaded network features from {extractor.output_file}")
        except Exception as e:
            logging.error(f"Error loading network features: {e}")
            print(f"Failed to load network features: {e}")
            continue
        try:
            embeddings = generate_embeddings([network_features])
            logging.info("Generated embeddings successfully")
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            print(f"Failed to generate embeddings: {e}")
            continue
        if G is None:
            G, switches, hosts = build_graph(network_features)
            centralities = network_features['centralities']
            logging.info(f"Built graph with {len(switches)} switches, {len(hosts)} hosts")
        possible_links = get_possible_links(G, [str(s) for s in network_features['topology']['switches']], network_features['topology']['hosts'])
        if not possible_links:
            logging.warning("No new switch-to-switch links available, stopping")
            print("No new switch-to-switch links available, stopping")
            break
        action_space_size = len(possible_links)
        action_to_link = {idx: pair for idx, pair in enumerate(possible_links)}
        if policy.fc3.out_features != action_space_size:
            policy.fc3 = nn.Linear(64, action_space_size)
            optimizer = optim.Adam(policy.parameters(), lr=0.001)
            logging.info(f"Updated policy network for {action_space_size} actions")
        try:
            current_reward = train_rl(policy, optimizer, G, embeddings, centralities, action_to_link)
        except Exception as e:
            logging.error(f"Error in RL training: {e}")
            print(f"Failed RL training: {e}")
            continue
        state = get_state(G, embeddings, centralities)
        probs = policy(state.unsqueeze(0))
        best_action = torch.argmax(probs).item()
        u, v = action_to_link[best_action]
        src_ports = extractor.fetch_available_ports(int(u))
        dst_ports = extractor.fetch_available_ports(int(v))
        src_port = src_ports[0] if src_ports else None
        dst_port = dst_ports[0] if dst_ports else None
        print(f"Suggested new link: {u} - {v} (Probability: {probs[0, best_action]:.4f})")
        link_config = {
            "src_dpid": u,
            "dst_dpid": v,
            "host_mac": None,
            "src_port": src_port if src_port else "unavailable",
            "dst_port": dst_port if dst_port else "unavailable"
        }
        print(f"Ryu-compatible link config: {link_config}")
        
        # Save suggested link to JSON
        try:
            with open(f"suggested_link_cycle_{cycle + 1}.json", "w") as f:
                json.dump(link_config, f, indent=4)
            logging.info(f"Saved suggested link to suggested_link_cycle_{cycle + 1}.json")
        except IOError as e:
            logging.error(f"Error saving suggested link: {e}")
            print(f"Failed to save suggested link: {e}")
            continue
        
        # Store the suggested link for final graph
        all_suggested_links.append(link_config)
        
        # Create updated network features with the new link for this cycle
        try:
            updated_features = add_suggested_link(network_features, link_config)
            output_file = f"updated_network_features_cycle_{cycle + 1}.json"
            with open(output_file, "w") as f:
                json.dump(updated_features, f, indent=4)
            logging.info(f"Updated network features saved to {output_file}")
            print(f"Updated network features saved to {output_file}")
        except Exception as e:
            logging.error(f"Error updating network features: {e}")
            print(f"Failed to update network features: {e}")
            continue
        
        G.add_edge(u, v)
        centralities = update_centralities(G)
        if abs(current_reward - previous_reward) < break_threshold:
            print(f"Break condition met: Reward improvement {abs(current_reward - previous_reward):.4f} < {break_threshold}")
            break
        previous_reward = current_reward
        time.sleep(5)
    
    # After all cycles, create and save the final graph with all suggested links
    try:
        final_features = copy.deepcopy(original_network_features)
        for link_config in all_suggested_links:
            final_features = add_suggested_link(final_features, link_config)
        with open("final_network_features.json", "w") as f:
            json.dump(final_features, f, indent=4)
        logging.info("Final network features with all suggested links saved to final_network_features.json")
        print("Final network features with all suggested links saved to final_network_features.json")
    except Exception as e:
        logging.error(f"Error creating final network features: {e}")
        print(f"Failed to create final network features: {e}")

if __name__ == "__main__":
    main()
