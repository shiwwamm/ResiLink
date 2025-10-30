#!/usr/bin/env python3
"""
Simple Working SDN Controller for Enhanced ResiLink
==================================================

A basic SDN controller that provides packet forwarding functionality
and REST API endpoints compatible with standard Ryu installations.

Usage:
    ryu-manager sdn/simple_controller.py --observe-links --wsapi-host 0.0.0.0 --wsapi-port 8080
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types
from ryu.topology import event
from ryu.topology.api import get_switch, get_link, get_host
from ryu.app.wsgi import WSGIApplication

import json
import time
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleController(app_manager.RyuApp):
    """Simple SDN controller with learning switch and basic REST API."""
    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}
    
    def __init__(self, *args, **kwargs):
        super(SimpleController, self).__init__(*args, **kwargs)
        
        # Learning switch state
        self.mac_to_port = {}  # {dpid: {mac: port}}
        
        # Topology state
        self.switches = {}
        self.links = {}
        self.hosts = {}
        
        # Performance monitoring
        self.packet_count = 0
        self.flow_count = 0
        self.start_time = time.time()
        
        # Setup basic REST API
        wsgi = kwargs['wsgi']
        wsgi.register(SimpleControllerREST, {'controller_app': self})
        
        logger.info("🚀 Simple SDN Controller initialized")
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
    
    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle_timeout=0):
        """Add a flow entry to the switch."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match, instructions=inst,
                                    idle_timeout=idle_timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst,
                                    idle_timeout=idle_timeout)
        
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
        
        # Format switches
        switch_list = []
        for switch in switches:
            dpid_hex = f'{switch.dp.id:016x}'
            
            ports = []
            for port in switch.ports:
                port_data = {
                    'port_no': f'{port.port_no:08x}',
                    'hw_addr': port.hw_addr,
                    'name': port.name.decode('utf-8') if port.name else f'port{port.port_no}',
                    'config': port.config,
                    'state': port.state
                }
                ports.append(port_data)
            
            switch_data = {
                'dpid': dpid_hex,
                'ports': ports
            }
            switch_list.append(switch_data)
        
        # Format links
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
        
        # Format hosts
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


class SimpleControllerREST(object):
    """Simple REST API for the controller."""
    
    def __init__(self, req, link, data, **config):
        super(SimpleControllerREST, self).__init__()
        self.controller_app = data['controller_app']
    
    def switches(self, req, **kwargs):
        """Get all switches."""
        try:
            topology = self.controller_app.get_topology_data()
            body = json.dumps(topology['switches'], indent=2)
            return Response(content_type='application/json', body=body)
        except Exception as e:
            logger.error(f"Error getting switches: {e}")
            body = json.dumps({'error': str(e)})
            return Response(content_type='application/json', body=body, status=500)
    
    def links(self, req, **kwargs):
        """Get all links."""
        try:
            topology = self.controller_app.get_topology_data()
            body = json.dumps(topology['links'], indent=2)
            return Response(content_type='application/json', body=body)
        except Exception as e:
            logger.error(f"Error getting links: {e}")
            body = json.dumps({'error': str(e)})
            return Response(content_type='application/json', body=body, status=500)
    
    def hosts(self, req, **kwargs):
        """Get all hosts."""
        try:
            topology = self.controller_app.get_topology_data()
            body = json.dumps(topology['hosts'], indent=2)
            return Response(content_type='application/json', body=body)
        except Exception as e:
            logger.error(f"Error getting hosts: {e}")
            body = json.dumps({'error': str(e)})
            return Response(content_type='application/json', body=body, status=500)
    
    def topology(self, req, **kwargs):
        """Get complete topology data."""
        try:
            topology = self.controller_app.get_topology_data()
            body = json.dumps(topology, indent=2, default=str)
            return Response(content_type='application/json', body=body)
        except Exception as e:
            logger.error(f"Error getting topology: {e}")
            body = json.dumps({'error': str(e)})
            return Response(content_type='application/json', body=body, status=500)


# Import Response - try different locations for compatibility
try:
    from webob import Response
except ImportError:
    try:
        from webob.response import Response
    except ImportError:
        # Fallback simple response
        class Response:
            def __init__(self, body='', status=200, content_type='text/plain'):
                self.body = body
                self.status = status
                self.content_type = content_type


# Register REST API endpoints manually for compatibility
def create_wsgi_app():
    """Create WSGI application with REST endpoints."""
    from wsgiref.simple_server import make_server
    
    def application(environ, start_response):
        path = environ.get('PATH_INFO', '')
        
        if path == '/v1.0/topology/switches':
            response = controller_instance.rest_api.switches(None)
        elif path == '/v1.0/topology/links':
            response = controller_instance.rest_api.links(None)
        elif path == '/v1.0/topology/hosts':
            response = controller_instance.rest_api.hosts(None)
        elif path == '/v1.0/topology/all':
            response = controller_instance.rest_api.topology(None)
        else:
            response = Response(body='{"error": "Not found"}', status=404, content_type='application/json')
        
        start_response(f'{response.status} OK', [('Content-Type', response.content_type)])
        return [response.body.encode()]
    
    return application


# Global controller instance for REST API
controller_instance = None