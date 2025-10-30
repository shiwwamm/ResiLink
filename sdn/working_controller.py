#!/usr/bin/env python3
"""
Production SDN Controller for Enhanced ResiLink
==============================================

A robust SDN controller that provides both packet forwarding functionality
and comprehensive REST API for network topology discovery and optimization.

Key Features:
- Learning switch functionality for packet forwarding
- REST API endpoints compatible with Enhanced ResiLink
- Real-time topology monitoring and updates
- Academic-grade logging and metrics collection

Usage:
    ryu-manager sdn/working_controller.py --observe-links --wsapi-host 0.0.0.0 --wsapi-port 8080

Academic Foundation:
- OpenFlow specification: McKeown et al. (2008)
- Learning switch algorithm: Perlman (1985)
- Network topology discovery: Topology Zoo methodology
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, arp, ipv4, icmp
from ryu.topology import event
from ryu.topology.api import get_switch, get_link, get_host
from ryu.app.wsgi import ControllerWSGI, WSGIApplication, route
from ryu.lib import hub

import json
import time
import logging
from collections import defaultdict
from webob import Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkingController(app_manager.RyuApp):
    """Production SDN controller with learning switch and REST API."""
    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}
    
    def __init__(self, *args, **kwargs):
        super(WorkingController, self).__init__(*args, **kwargs)
        
        # Learning switch state
        self.mac_to_port = {}  # {dpid: {mac: port}}
        self.port_stats = defaultdict(dict)  # {dpid: {port: stats}}
        
        # Topology state
        self.switches = {}
        self.links = {}
        self.hosts = {}
        
        # Performance monitoring
        self.packet_count = 0
        self.flow_count = 0
        self.start_time = time.time()
        
        # Setup REST API
        wsgi = kwargs['wsgi']
        wsgi.register(ControllerAPI, {'controller_app': self})
        
        logger.info("🚀 Working SDN Controller initialized")
        logger.info("📡 REST API available at http://localhost:8080")
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle new switch connection."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        dpid = datapath.id
        
        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        
        # Initialize MAC learning table
        self.mac_to_port[dpid] = {}
        
        logger.info(f"🔌 Switch {dpid} connected and configured")
    
    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle_timeout=0, hard_timeout=0):
        """Add a flow entry to the switch."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match, instructions=inst,
                                    idle_timeout=idle_timeout, hard_timeout=hard_timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst,
                                    idle_timeout=idle_timeout, hard_timeout=hard_timeout)
        
        datapath.send_msg(mod)
        self.flow_count += 1
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle packet_in events with learning switch logic."""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        dpid = datapath.id
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        
        # Ignore LLDP packets
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        
        dst = eth.dst
        src = eth.src
        
        self.packet_count += 1
        
        # Learn source MAC address
        if dpid not in self.mac_to_port:
            self.mac_to_port[dpid] = {}
        
        self.mac_to_port[dpid][src] = in_port
        
        # Determine output port
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD
        
        actions = [parser.OFPActionOutput(out_port)]
        
        # Install flow rule if we know the destination
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            
            # Add flow with timeout to prevent table overflow
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id, idle_timeout=60)
                return
            else:
                self.add_flow(datapath, 1, match, actions, idle_timeout=60)
        
        # Send packet out
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
    
    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        """Handle switch entering topology."""
        switch = ev.switch
        dpid = switch.dp.id
        self.switches[dpid] = switch
        
        logger.info(f"📊 Switch {dpid} entered topology")
    
    @set_ev_cls(event.EventSwitchLeave)
    def switch_leave_handler(self, ev):
        """Handle switch leaving topology."""
        switch = ev.switch
        dpid = switch.dp.id
        
        if dpid in self.switches:
            del self.switches[dpid]
        if dpid in self.mac_to_port:
            del self.mac_to_port[dpid]
        
        logger.info(f"📊 Switch {dpid} left topology")
    
    @set_ev_cls(event.EventLinkAdd)
    def link_add_handler(self, ev):
        """Handle link addition."""
        link = ev.link
        src_dpid = link.src.dpid
        dst_dpid = link.dst.dpid
        src_port = link.src.port_no
        dst_port = link.dst.port_no
        
        link_id = f"{src_dpid}-{dst_dpid}"
        self.links[link_id] = link
        
        logger.info(f"🔗 Link added: {src_dpid}:{src_port} ↔ {dst_dpid}:{dst_port}")
    
    @set_ev_cls(event.EventLinkDelete)
    def link_delete_handler(self, ev):
        """Handle link deletion."""
        link = ev.link
        src_dpid = link.src.dpid
        dst_dpid = link.dst.dpid
        
        link_id = f"{src_dpid}-{dst_dpid}"
        if link_id in self.links:
            del self.links[link_id]
        
        logger.info(f"🔗 Link deleted: {src_dpid} ↔ {dst_dpid}")
    
    @set_ev_cls(event.EventHostAdd)
    def host_add_handler(self, ev):
        """Handle host addition."""
        host = ev.host
        self.hosts[host.mac] = host
        
        logger.info(f"🖥️  Host added: {host.mac} on switch {host.port.dpid}:{host.port.port_no}")
    
    def get_topology_data(self):
        """Get current topology data for REST API."""
        # Get fresh topology data from Ryu
        switches = get_switch(self, None)
        links = get_link(self, None)
        hosts = get_host(self, None)
        
        # Format switches with enhanced information
        switch_list = []
        for switch in switches:
            dpid_hex = f'{switch.dp.id:016x}'
            
            # Get port information
            ports = []
            for port in switch.ports:
                port_data = {
                    'port_no': f'{port.port_no:08x}',
                    'hw_addr': port.hw_addr,
                    'name': port.name.decode('utf-8') if port.name else f'port{port.port_no}',
                    'config': port.config,
                    'state': port.state,
                    'curr_speed': port.curr_speed,
                    'max_speed': port.max_speed
                }
                ports.append(port_data)
            
            switch_data = {
                'dpid': dpid_hex,
                'ports': ports,
                'n_buffers': switch.dp.n_buffers,
                'n_tables': switch.dp.n_tables,
                'capabilities': switch.dp.capabilities
            }
            switch_list.append(switch_data)
        
        # Format links with enhanced information
        link_list = []
        for link in links:
            link_data = {
                'src': {
                    'dpid': f'{link.src.dpid:016x}',
                    'port_no': f'{link.src.port_no:08x}',
                    'hw_addr': link.src.hw_addr,
                    'name': link.src.name
                },
                'dst': {
                    'dpid': f'{link.dst.dpid:016x}',
                    'port_no': f'{link.dst.port_no:08x}',
                    'hw_addr': link.dst.hw_addr,
                    'name': link.dst.name
                }
            }
            link_list.append(link_data)
        
        # Format hosts with enhanced information
        host_list = []
        for host in hosts:
            host_data = {
                'mac': host.mac,
                'ipv4': host.ipv4,
                'ipv6': host.ipv6,
                'port': {
                    'dpid': f'{host.port.dpid:016x}',
                    'port_no': f'{host.port.port_no:08x}',
                    'hw_addr': host.port.hw_addr,
                    'name': host.port.name
                }
            }
            host_list.append(host_data)
        
        return {
            'switches': switch_list,
            'links': link_list,
            'hosts': host_list,
            'stats': {
                'packet_count': self.packet_count,
                'flow_count': self.flow_count,
                'uptime': time.time() - self.start_time,
                'switches_count': len(switch_list),
                'links_count': len(link_list),
                'hosts_count': len(host_list)
            }
        }


class ControllerAPI(ControllerWSGI):
    """REST API for the working controller."""
    
    def __init__(self, req, link, data, **config):
        super(ControllerAPI, self).__init__(req, link, data, **config)
        self.controller_app = data['controller_app']
    
    @route('topology', '/v1.0/topology/switches', methods=['GET'])
    def get_switches(self, req, **kwargs):
        """Get all switches."""
        try:
            topology = self.controller_app.get_topology_data()
            return self._json_response(topology['switches'])
        except Exception as e:
            logger.error(f"Error getting switches: {e}")
            return self._json_response({'error': str(e)}, status=500)
    
    @route('topology', '/v1.0/topology/links', methods=['GET'])
    def get_links(self, req, **kwargs):
        """Get all links."""
        try:
            topology = self.controller_app.get_topology_data()
            return self._json_response(topology['links'])
        except Exception as e:
            logger.error(f"Error getting links: {e}")
            return self._json_response({'error': str(e)}, status=500)
    
    @route('topology', '/v1.0/topology/hosts', methods=['GET'])
    def get_hosts(self, req, **kwargs):
        """Get all hosts."""
        try:
            topology = self.controller_app.get_topology_data()
            return self._json_response(topology['hosts'])
        except Exception as e:
            logger.error(f"Error getting hosts: {e}")
            return self._json_response({'error': str(e)}, status=500)
    
    @route('topology', '/v1.0/topology/all', methods=['GET'])
    def get_all_topology(self, req, **kwargs):
        """Get complete topology data."""
        try:
            topology = self.controller_app.get_topology_data()
            return self._json_response(topology)
        except Exception as e:
            logger.error(f"Error getting topology: {e}")
            return self._json_response({'error': str(e)}, status=500)
    
    @route('stats', '/v1.0/stats/controller', methods=['GET'])
    def get_controller_stats(self, req, **kwargs):
        """Get controller performance statistics."""
        try:
            topology = self.controller_app.get_topology_data()
            return self._json_response(topology['stats'])
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return self._json_response({'error': str(e)}, status=500)
    
    def _json_response(self, data, status=200):
        """Create JSON response."""
        body = json.dumps(data, indent=2, default=str)
        return Response(content_type='application/json', body=body, status=status)