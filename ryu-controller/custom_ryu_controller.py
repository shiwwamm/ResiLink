from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.topology.api import get_switch, get_link, get_host
from ryu.app.wsgi import ControllerBase, WSGIApplication, route, Response
from ryu.lib import dpid as dpid_lib
import json

class CustomTopologyController(ControllerBase):
    def __init__(self, req, link, data, **config):
        super(CustomTopologyController, self).__init__(req, link, data, **config)
        self.topology_api_app = data['topology_api_app']
        self.datapaths = data['datapaths']

    @route('custom_topology', '/custom/topology/switches', methods=['GET'])
    def list_switches(self, req, **kwargs):
        return self._switches(req, **kwargs)

    @route('custom_topology', '/custom/topology/switches/{dpid}', methods=['GET'],
           requirements={'dpid': dpid_lib.DPID_PATTERN})
    def get_switch(self, req, **kwargs):
        return self._switches(req, **kwargs)

    @route('custom_topology', '/custom/topology/links', methods=['GET'])
    def list_links(self, req, **kwargs):
        return self._links(req, **kwargs)

    @route('custom_topology', '/custom/topology/links/{dpid}', methods=['GET'],
           requirements={'dpid': dpid_lib.DPID_PATTERN})
    def get_links(self, req, **kwargs):
        return self._links(req, **kwargs)

    @route('custom_topology', '/custom/topology/hosts', methods=['GET'])
    def list_hosts(self, req, **kwargs):
        return self._hosts(req, **kwargs)

    @route('custom_topology', '/custom/topology/hosts/{dpid}', methods=['GET'],
           requirements={'dpid': dpid_lib.DPID_PATTERN})
    def get_hosts(self, req, **kwargs):
        return self._hosts(req, **kwargs)

    @route('custom_stats', '/custom/stats/aggregateflow/{dpid}', methods=['GET'],
           requirements={'dpid': dpid_lib.DPID_PATTERN})
    def get_aggregate_flow_stats(self, req, **kwargs):
        dpid = dpid_lib.str_to_dpid(kwargs['dpid'])
        if dpid not in self.datapaths:
            return Response(status=404, body=json.dumps({"error": "Switch not found"}))

        datapath = self.datapaths[dpid]
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Request aggregate flow stats
        match = parser.OFPMatch()
        req = parser.OFPAggregateStatsRequest(
            datapath=datapath,
            match=match,
            table_id=ofproto.OFPTT_ALL,
            out_port=ofproto.OFPP_ANY,
            out_group=ofproto.OFPG_ANY
        )
        datapath.send_msg(req)

        return Response(content_type='application/json',
                        body=json.dumps({"message": "Aggregate flow stats requested"}))

    def _switches(self, req, **kwargs):
        dpid = None
        if 'dpid' in kwargs:
            dpid = dpid_lib.str_to_dpid(kwargs['dpid'])
        switches = get_switch(self.topology_api_app, dpid)
        body = json.dumps([switch.to_dict() for switch in switches])
        return Response(content_type='application/json', body=body)

    def _links(self, req, **kwargs):
        dpid = None
        if 'dpid' in kwargs:
            dpid = dpid_lib.str_to_dpid(kwargs['dpid'])
        links = get_link(self.topology_api_app, dpid)
        body = json.dumps([link.to_dict() for link in links])
        return Response(content_type='application/json', body=body)

    def _hosts(self, req, **kwargs):
        dpid = None
        if 'dpid' in kwargs:
            dpid = dpid_lib.str_to_dpid(kwargs['dpid'])
        hosts = get_host(self.topology_api_app, dpid)
        body = json.dumps([host.to_dict() for host in hosts])
        return Response(content_type='application/json', body=body)

class CustomRyuController(app_manager.RyuApp):
    _CONTEXTS = {'wsgi': WSGIApplication}
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(CustomRyuController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}
        self.wsgi = kwargs['wsgi']
        self.wsgi.register(CustomTopologyController,
                          {'topology_api_app': self, 'datapaths': self.datapaths})

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        self.datapaths[datapath.id] = datapath

        # Install table-miss flow entry with higher priority
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                         ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 100, match, actions)

    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            command=ofproto.OFPFC_ADD,  # Explicitly set command
            buffer_id=ofproto.OFP_NO_BUFFER,  # Explicitly set buffer_id
            idle_timeout=0,
            hard_timeout=0,
            cookie=0
        )
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocol(ethernet.ethernet)
        if not eth_pkt:
            return

        dst = eth_pkt.dst
        src = eth_pkt.src

        # Learn MAC address
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # Install flow for known destination with higher priority
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            self.add_flow(datapath, 200, match, actions)

        # Send packet out
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data
        )
        datapath.send_msg(out)

    @set_ev_cls(ofp_event.EventOFPAggregateStatsReply, MAIN_DISPATCHER)
    def aggregate_stats_reply_handler(self, ev):
        msg = ev.msg
        body = msg.body
        stats = {
            'dpid': dpid_lib.dpid_to_str(msg.datapath.id),
            'packet_count': body.packet_count,
            'byte_count': body.byte_count,
            'flow_count': body.flow_count
        }
        self.logger.info('AggregateStatsReply: %s', json.dumps(stats, indent=2))
