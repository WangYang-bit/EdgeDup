from Graph import graph_generator as gg
from omegaconf import OmegaConf as oc
from util import config
import networkx as nx

TestConf = oc.structured(config.TestConfig)
GraphConf = oc.structured(config.GraphConfig)


class NetGraph:
    def __init__(self):
        self.nodeNum = TestConf.NODENUM
        self.degree = GraphConf.DEGREE
        self.graph_seed = GraphConf.GRAPH_SEED
        self.node_distance = dict()
        if GraphConf.REAL_GRAPH:
            self.graph = gg.generate_edge_network()
        else:
            self.graph = gg.generate_random_graph(self.nodeNum, self.degree)

        for node in self.graph.nodes:
            self.node_distance[node] = nx.shortest_path_length(self.graph, node)

    def show(self, with_label=False):
        gg.draw_graph(self.graph, with_label)

    def distance(self, node1, node2):
        return self.node_distance[node1][node2]

    def density(self):
        return nx.density(self.graph)


if __name__ == '__main__':
    G = NetGraph()
    G.density()
    print(G.density())
    G.show()

