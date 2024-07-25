import heapq
import logging
import math

import pandas as pd
from pyproj import Proj, transform
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from omegaconf import OmegaConf as oc
from util import config
from util.config import GraphConfig

logging.getLogger('matplotlib.font_manager').disabled = True

def euclidDistance(X, Y):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(X, Y)]))


def calculateDis(data):
    station_num = len(data)
    distance = [[0.0 for i in range(station_num)] for j in range(station_num)]

    for i in range(station_num):
        edge_i = data.iloc[i]
        for j in range(i, station_num):
            edge_j = data.iloc[j]
            distance[i][j] = euclidDistance([edge_i['easting'], edge_i['northing']],
                                            [edge_j['easting'], edge_j['northing']])
    array = np.array(distance)
    df = pd.DataFrame(array)
    df.to_csv('edge_distance.csv')


def CoordinatesToUTM(data):
    wgs84 = Proj(init='epsg:4326')  

    utm = Proj(init='epsg:32651')  
    eastingList = []
    northingList = []
    for index, row in data.iterrows():
        longitude = row['longitude']
        latitude = row['latitude']
        easting, northing = transform(wgs84, utm, longitude, latitude)
        eastingList.append(easting)
        northingList.append(northing)
    data['easting'] = eastingList  
    data['northing'] = northingList  
    data.to_csv('BaseStationUTM.csv', encoding='utf-8')


def draw_graph(G,with_label):
    pos = nx.spring_layout(G, seed=1)
    nx.draw_networkx(G,pos,with_labels=with_label)
    # nx.draw_networkx_nodes(G, pos, node_size=20)
    # nx.draw_networkx_edges(G, pos, alpha=0.4)
    plt.show()


def generate_random_graph(n, k):
    G = nx.watts_strogatz_graph(n, k, 0.8, seed=GraphConfig.GRAPH_SEED)
    # p = k / (n - 1)
    # G = nx.erdos_renyi_graph(n, p, seed=GraphConfig.GRAPH_SEED)
    #
    # while not nx.is_connected(G):
    #     components = list(nx.connected_components(G))
    #     for i in range(len(components) - 1):
    #         if len(components[i]) > 0 and len(components[i + 1]) > 0:
    #             G.add_edge(components[i].pop(), components[i + 1].pop())

    return G


def generate_edge_network():
    edgeAdj = pd.read_csv('../dataSet/edgeNetwork_adj.csv')
    array = np.array(edgeAdj)
    G = nx.from_numpy_array(array)
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    print(Gcc)
    pos = nx.spring_layout(Gcc, seed=1)
    nx.draw_networkx_nodes(Gcc, pos, node_size=5)
    nx.draw_networkx_edges(Gcc, pos, alpha=0.4)

    degree_sequence = sorted((d for n, d in Gcc.degree()), reverse=True)
    ax2 = plt.figure(figsize=(10, 7)).add_subplot()
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")
    plt.show()


class GraphGenerator:
    def __init__(self):
        self.config = oc.structured(config.GraphConfig)
        self.graph_seed = self.config.GRAPH_SEED
        self.max_link = 6
        self.min_dis = 2000
        self.max_dis = 10000

    def getNetworkAdj(self, data):
        station_num = len(data)
        adj = [[0 for i in range(station_num)] for j in range(station_num)]
        degree = [0 for i in range(station_num)]
        # 求出最近的max_link个的id
        edgeNum = 0
        for i in range(station_num):
            edge_i = data.iloc[i].tolist()[i + 1:station_num]
            min_num_index_list = list(map(edge_i.index, heapq.nsmallest(self.max_link, edge_i)))
            for min_index in min_num_index_list:
                index = min_index + i + 1
                if self.max_dis >= edge_i[min_index] > self.min_dis and degree[index] < self.max_link and degree[
                    i] < self.max_link:
                    adj[i][index] = 1
                    degree[i] += 1
                    degree[index] += 1
                    edgeNum += 1

            # print(i)
        adj = np.array(adj)
        adj = adj + adj.T
        # print(adj.shape)
        print(edgeNum)
        df = pd.DataFrame(adj)
        df.to_csv('edgeNetwork_adj.csv', index=False)


if __name__ == '__main__':
    Generator = GraphGenerator()
    G = generate_random_graph(50, 5)
    draw_graph(G)
