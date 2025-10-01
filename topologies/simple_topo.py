#!/usr/bin/python

from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info

def myNetwork():
    setLogLevel('info')

    net = Mininet(controller=RemoteController)

    info('*** Adding controller\n')
    net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6653)

    #      h5
    #      |
    # S1 - S2 - S3 - S4 
    # |    |    |    | 
    # h1   h2   h3   h4
    info('*** Adding hosts and switches\n')
    # Add two hosts
    h1 = net.addHost('h1', ip='10.0.0.1/24')
    h2 = net.addHost('h2', ip='10.0.0.2/24')
    h3 = net.addHost('h3', ip='10.0.0.3/24')
    h4 = net.addHost('h4', ip='10.0.0.4/24')
    h5 = net.addHost('h5', ip='10.0.0.5/24')
    h6 = net.addHost('h6', ip='10.0.0.6/24')
    h7 = net.addHost('h7', ip='10.0.0.7/24')

    # Add one switch
    s1 = net.addSwitch('s1')
    s2 = net.addSwitch('s2')
    s3 = net.addSwitch('s3')
    s4 = net.addSwitch('s4')
    s5 = net.addSwitch('s5')
    s6 = net.addSwitch('s6')
    s7 = net.addSwitch('s7')

    info('*** Creating links\n')
    # Link the hosts to the switch
    net.addLink(h1, s1)
    net.addLink(h2, s2)
    net.addLink(h3, s3)
    net.addLink(h4, s4)
    net.addLink(h5, s2)
    net.addLink(h6, s5)
    net.addLink(h7, s6)
    
    net.addLink(s1, s2)
    net.addLink(s2, s3)
    net.addLink(s3, s4)
    net.addLink(s4, s5)
    net.addLink(s5, s6)
    net.addLink(s3, s7)

    info('*** Starting network\n')
    net.start()

    CLI(net)

    info('*** Stopping network\n')
    net.stop()

if __name__ == '__main__':
    myNetwork()
