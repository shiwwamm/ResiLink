from mininet.topo import Topo

class ThreeNodeSwitchTopo(Topo):
    def build(self):
        # Add one switch for each host
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3')

        # Add hosts
        h1 = self.addHost('h1')
        h2 = self.addHost('h2')
        h3 = self.addHost('h3')

        # Connect each host to its switch
        self.addLink(h1, s1)
        self.addLink(h2, s3)
        self.addLink(h3, s3)

        # Connect switches: A–B and B–C
        self.addLink(s1, s2)
        self.addLink(s2, s3)

topos = {
    'threenodes': (lambda: ThreeNodeSwitchTopo())
}
