import networkx as nx
import matplotlib.pyplot as plt

class networkxTest:
    def Test1(self):
        G = nx.random_regular_graph(4,10)
        nx.draw(G,with_labels=True)
        plt.show()

    def Test2(self):
        G = nx.random_tree(10)
        nx.draw(G, with_labels=True)
        plt.show()

    def Test3(self):
        G = nx.random_cograph(4)  # 2^n个点
        nx.draw(G, with_labels=True)
        plt.show()

    def Test4(self):
        G = nx.gnp_random_graph(30, 0.1, seed=10374196)
        Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
        pos = nx.spring_layout(Gcc, seed=1039695)
        nx.draw_networkx_nodes(Gcc, pos, node_size=20)
        nx.draw_networkx_edges(Gcc, pos, alpha=0.4)
        plt.show()

    def Test5(self):
        G = nx.grid_2d_graph(5, 5)
        nx.draw(G, with_labels=True)
        plt.show()

    def Test6(self):
        G = nx.Graph()
        G.add_edge(1, 2)
        G.add_edge(1, 3)
        G.add_edge(1, 5)
        G.add_edge(2, 3)
        G.add_edge(3, 4)
        G.add_edge(4, 5)

        # explicitly set positions
        pos = {1: (0, 0), 2: (-1, 0.3), 3: (2, 0.17), 4: (4, 0.255), 5: (5, 0.03)}

        options = {
            "font_size": 36,
            "node_size": 3000,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 5,
            "width": 5,
        }
        nx.draw_networkx(G, pos, **options)

        # Set margins for the axes so that nodes aren't clipped
        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        plt.show()

    def Test7(self):
        deg = [(1, 0), (1, 0), (1, 0), (2, 0), (1, 0), (2, 1), (0, 1), (0, 1)]
        G = nx.random_clustered_graph(deg)
        nx.draw(G, with_labels=True)
        plt.show()
    def Test8(self):
        G = nx.random_internet_as_graph(20)
        nx.draw(G, with_labels=True)
        plt.show()

    def Test9(self):
        G = nx.dense_gnm_random_graph(4,4)
        nx.draw(G, with_labels=True)
        plt.show()

if __name__ == "__main__":
    T = networkxTest()
    T.Test9()
