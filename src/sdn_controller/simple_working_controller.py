#!/usr/bin/env python3
"""
Simple Working SDN Controller for Enhanced ResiLink
==================================================

A basic learning switch controller that actually forwards packets
while providing the REST API endpoints needed for ResiLink.

This controller:
1. Implements basic learning switch functionality
2. Provides REST API endpoints for topology discovery
3. Works with hybrid_resilink_implementation.py

Usage:
    ryu-manager src/sdn_controller/simple_working_controller.py \
        --observe-links --wsapi-host 0.0.0.0 --wsapi-port 8080
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, arp, ipv4
from ryu.topology import event
from ryu.topology.api import get_switch, get_link, get_host
from ryu.app.wsgi import ControllerWSGI, WSGIApplication, route
from ryu.lib import hub

import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleWorkingController(app_manager.RyuApp):
    """Simple learning switch with REST API for ResiLink."""
    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}
    
    def __init__(self, *args, **kwargs):
        super(SimpleWorkingController, self).__init__(*args, **kwargs)
        
        # MAC learning table: {dpid: {mac: port}}
        self.mac_to_port = {}
        
        # Topology data
        self.switches = {}
        self.links = {}
        self.hosts = {}
        
        # Setup REST API
        wsgi = kwargs['wsgi']
        wsgi.register(SimpleControllerAPI, {'controller_app': self})
        
        logger.info("Simple Working Controller initialized")
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle switch connection."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # Install table-miss flow entry (send unknown packets to controller)
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        
        logger.info(f"Switch {datapath.id} connected")
    
    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        """Add a flow entry to the switch."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match, instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        
        datapath.send_msg(mod)
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle packet_in events (learning switch logic)."""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            # Ignore LLDP packets
            return
        
        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        
        # Initialize MAC table for this switch
        self.mac_to_port.setdefault(dpid, {})
        
        # Learn source MAC
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
            
            # Verify we have a valid buffer_id
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                return
            else:
                self.add_flow(datapath, 1, match, actions)
        
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
        self.switches[switch.dp.id] = switch
        logger.info(f"Switch {switch.dp.id} entered topology")
    
    @set_ev_cls(event.EventSwitchLeave)
    def switch_leave_handler(self, ev):
        """Handle switch leaving topology."""
        switch = ev.switch
        if switch.dp.id in self.switches:
            del self.switches[switch.dp.id]
        logger.info(f"Switch {switch.dp.id} left topology")
    
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
        
        logger.info(f"Link added: {src_dpid}:{src_port} -> {dst_dpid}:{dst_port}")
    
    @set_ev_cls(event.EventLinkDelete)
    def link_delete_handler(self, ev):
        """Handle link deletion."""
        link = ev.link
        src_dpid = link.src.dpid
        dst_dpid = link.dst.dpid
        
        link_id = f"{src_dpid}-{dst_dpid}"
        if link_id in self.links:
            del self.links[link_id]
        
        logger.info(f"Link deleted: {src_dpid} -> {dst_dpid}")
    
    @set_ev_cls(event.EventHostAdd)
    def host_add_handler(self, ev):
        """Handle host addition."""
        host = ev.host
        self.hosts[host.mac] = host
        logger.info(f"Host added: {host.mac} on switch {host.port.dpid}:{host.port.port_no}")
    
    def get_topology_data(self):
        """Get current topology data for REST API."""
        # Get current topology from Ryu
        switches = get_switch(self, None)
        links = get_link(self, None)
        hosts = get_host(self, None)
        
        # Format switches
        switch_list = []
        for switch in switches:
            switch_data = {
                'dpid': f'{switch.dp.id:016x}',
                'ports': []
            }
            
            for port in switch.ports:
                port_data = {
                    'port_no': f'{port.port_no:08x}',
                    'hw_addr': port.hw_addr,
                    'name': port.name.decode('utf-8') if port.name else f'port{port.port_no}'
                }
                switch_data['ports'].append(port_data)
            
            switch_list.append(switch_data)
        
        # Format links
        link_list = []
        for link in links:
            link_data = {
                'src': {
                    'dpid': f'{link.src.dpid:016x}',
                    'port_no': f'{link.src.port_no:08x}'
                },
                'dst': {
                    'dpid': f'{link.dst.dpid:016x}',
                    'port_no': f'{link.dst.port_no:08x}'
                }
            }
            link_list.append(link_data)
        
        # Format hosts
        host_list = []
        for host in hosts:
            host_data = {
                'mac': host.mac,
                'ipv4': host.ipv4,
                'ipv6': host.ipv6,
                'port': {
                    'dpid': f'{host.port.dpid:016x}',
                    'port_no': f'{host.port.port_no:08x}'
                }
            }
            host_list.append(host_data)
        
        return {
            'switches': switch_list,
            'links': link_list,
            'hosts': host_list
        }


class SimpleControllerAPI(ControllerWSGI):
    """REST API for the simple controller."""
    
    def __init__(self, req, link, data, **config):
        super(SimpleControllerAPI, self).__init__(req, link, data, **config)
        self.controller_app = data['controller_app']
    
    @route('topology', '/v1.0/topology/switches', methods=['GET'])
    def get_switches(self, req, **kwargs):
        """Get all switches."""
        try:
            topology = self.controller_app.get_topology_data()
            return self._response(topology['switches'])
        except Exception as e:
            logger.error(f"Error getting switches: {e}")
            return self._response({'error': str(e)}, status=500)
    
    @route('topology', '/v1.0/topology/links', methods=['GET'])
    def get_links(self, req, **kwargs):
        """Get all links."""
        try:
            topology = self.controller_app.get_topology_data()
            return self._response(topology['links'])
        except Exception as e:
            logger.error(f"Error getting links: {e}")
            return self._response({'error': str(e)}, status=500)
    
    @route('topology', '/v1.0/topology/hosts', methods=['GET'])
    def get_hosts(self, req, **kwargs):
        """Get all hosts."""
        try:
            topology = self.controller_app.get_topology_data()
            return self._response(topology['hosts'])
        except Exception as e:
            logger.error(f"Error getting hosts: {e}")
            return self._response({'error': str(e)}, status=500)
    
    def _response(self, data, status=200):
        """Create JSON response."""
        body = json.dumps(data, indent=2)
        return Response(content_type='application/json', body=body, status=status)


# Import Response class
try:
    from webob import Response
except ImportError:
    from ryu.lib.packet import Response