import asyncio
import logging
import json
import random
import socket
import threading
import time
from multiprocessing import Process

import numpy as np
from _socket import SOL_SOCKET, SO_REUSEADDR

from Graph import netgraph as ng
from omegaconf import OmegaConf as oc

from Network import network_utils as nutil
from Network import Message
from Network.network_utils import if_mess_show
from Server import server
from util import request_generator, config
from util.Algorithm import solve_set_cover_with_weights, _to_str, solve_set_cover_and_max_weights, \
    solve_dedup_space_balance

NetworkConf = oc.structured(config.NetWorkConfig)
ServerConf = oc.structured(config.ServerConfig)
TestConf = oc.structured(config.TestConfig)

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)


class Manager:
    def __init__(self):
        netgraph = ng.NetGraph()
        self.hopNum = ServerConf.HOP_NUM
        self.ServerNum = netgraph.nodeNum
        self.netgraph = netgraph
        self.process_list = []
        self.lock = threading.Lock()
        self.ServerList = dict()
        self.Capacity_list = []
        self.connections = dict()

    def install(self, Capacity_list=None):
        ServerList = dict()
        # Port assignment and data init
        init_data_num = 0
        graph = self.netgraph.graph
        logger.info(f'graph seed {self.netgraph.graph_seed}')
        host = socket.gethostname()
        if Capacity_list is None:
            capacity = self.init_edge_server_capacity()
        else:
            capacity = Capacity_list
        self.Capacity_list = capacity
        for node in graph.nodes:
            newServer = server.Node(node, host, 50000 + node, capacity[node])
            ServerList[node] = newServer
        self.ServerList = ServerList

        graph = self.netgraph.graph

        # Network node mapping
        for node in graph.nodes:
            p = self.netgraph.node_distance[node]
            for target, dis in p.items():
                if self.hopNum >= dis > 0:
                    adjNode = server.adjNode(target, dis,
                                             (ServerList[target].server_host,
                                              ServerList[target].server_port),
                                             ServerList[target].cbf.copy())
                    ServerList[node].add_neighbor(adjNode)
            ServerList[node].generate_hcbfTree()

    def init_connection_pool(self):
        for server_id, server in self.ServerList.items():
            address = (server.server_host, server.server_port)
            self.connections[server_id] = nutil.ConnectionPool(address, ServerConf.CONNECTION_POOL_SIZE, server_id)

    def init_edge_server_cache(self, cache):
        init_data_num = 0
        data_kind = set()
        for node in self.getServer_list():
            index = cache[node]
            for data in index:
                data_kind.add(data)
                data_id = _to_str(data)
                # data_id = data
                self.ServerList[node].save_data(data_id, data)
                init_data_num += 1
                if not TestConf.IGNORE_HOT:
                    self.update_neighbor_hbfc(node, data_id, True)
        return init_data_num, len(data_kind)

    def init_edge_server_capacity(self):
        capacity_list = np.random.uniform(low=ServerConf.CAPACITY_MIN, high=ServerConf.CAPACITY_MAX,
                                          size=self.ServerNum)
        return capacity_list

    def get_average_neighbor_num(self):
        total = 0
        for Server in self.ServerList.values():
            total += len(Server.neighbors)
        return total / len(self.ServerList)

    def total_data_num(self):
        total = 0
        for Server in self.ServerList.values():
            total += len(Server.index)
        return total

    def run_edge_server(self):
        for server_id, Server in self.ServerList.items():
            # p_server = Process(target=Server.run, args=())
            p_server = threading.Thread(target=Server.run, args=(), name=f"server {server_id}")
            self.process_list.append(p_server)
            p_server.start()
            # p_server.run()
        self.init_connection_pool()
        ready_message = Message.Message_ready()
        asyncio.run(self.broadcast_message_response(self.getServer_list(), ready_message))

    def getServer_list(self):
        return list(self.ServerList.keys())

    def TestLatency(self, request_list):
        query_hop_list = []
        cloud_time = 0
        false_positive = 0
        cloud_data = set()
        edge_hit_list = []
        for row_index, row in request_list.iterrows():
            data_id = _to_str(row['data_id'])
            # data_id = row['data_id']
            Server = self.ServerList[row['server']]
            place_type, object = Server.search_data(data_id)
            if place_type == 'local':  # data in local
                query_hop_list.append(0)
                edge_hit_list.append(0)
                continue
            if place_type is None and object is not None and TestConf.REGISTER_DEPENDENCY:  # vindex evict
                self.unregister_dependency(Server.id, object)
            result, hop, server_id = Server.find_data_hcbf(data_id)
            if result:
                if self.get_data_from_neighbor(server_id, data_id):
                    query_hop_list.append(hop)
                    edge_hit_list.append(hop)
                else:
                    false_positive += 1
                    query_hop_list.append(20)
                    cloud_time += 1
                    # cloud_data.add((Server.id, row['data_id']))
            else:
                for neighbor in Server.get_neighbors():
                    if self.get_data_from_neighbor(neighbor, data_id):
                        print('error')
                query_hop_list.append(20)
                cloud_time += 1
                # cloud_data.add((Server.id, row['data_id']))

            if ServerConf.CACHE_TYPE != 'CLOUD' and data_id in Server.v_index and Server.v_index[data_id].query_times >= 2:  # data in neighbor
                self.server_cache_data(Server.id, row['data_id'])

            if TestConf.REGISTER_DEPENDENCY:
                self.register_dependency(Server, data_id)
        # print(cloud_data)
        average_latency = np.sum(query_hop_list) / len(query_hop_list)
        edge_average_latency = np.sum(edge_hit_list) / len(edge_hit_list)
        logger.info(f'false_positive: {false_positive}')
        return average_latency, cloud_time, edge_average_latency

    def get_data_from_neighbor(self, server_id, data_id):
        place, data = self.ServerList[server_id].search_data(data_id)
        if place == 'local':
            return True
        return False

    def server_sort_by_neighbor_num(self):
        server_list = list(self.ServerList.values())
        server_list.sort(key=lambda x: len(x.neighbors))
        server_list = [Server.id for Server in server_list]
        return server_list

    def cache_data(self, data_list):
        servers = self.getServer_list()
        cache_data_num = 0
        for server in self.ServerList.values():
            while len(server.index) < server.capacity:
                data = random.choice(data_list)
                data_id = _to_str(data)
                # data_id = data
                if data_id not in server.index:
                    self.server_cache_data(server.id, data)
                    cache_data_num += 1
        return cache_data_num

    def server_cache_data(self, server_id, data):
        Server = self.ServerList[server_id]
        data_id = _to_str(data)
        # data_id = data
        if data_id in Server.v_index:
            deleted_vmeta = Server.delete_virtual_data(data_id)
            if TestConf.REGISTER_DEPENDENCY and deleted_vmeta is not None:
                self.unregister_dependency(server_id, deleted_vmeta)

        evictObject = Server.save_data(data_id, data)
        self.update_neighbor_hbfc(server_id, data_id, True)
        if evictObject is not None:
            evict_data_id = _to_str(evictObject.data_address)
            # evict_data_id = evictObject.data_address
            if TestConf.REGISTER_DEPENDENCY:
                for neighbor in evictObject.get_register_servers():
                    self.ServerList[neighbor].delete_select_server(evict_data_id, server_id)
            self.update_neighbor_hbfc(server_id, evict_data_id, False)

    def update_neighbor_hbfc(self, server_id, data_id, status):
        Server = self.ServerList[server_id]
        if status:
            for neighbor in Server.get_neighbors():
                self.ServerList[neighbor].hcbfTree_insert(Server.neighbors[neighbor].distance, server_id, data_id)
        else:
            for neighbor in Server.get_neighbors():
                self.ServerList[neighbor].hcbfTree_delete(Server.neighbors[neighbor].distance, server_id, data_id)

    def register_dependency(self, Server, data_id):
        update_list = Server.update_select_list(data_id)
        for neighbor in update_list:
            if self.ServerList[neighbor].register_dependency(data_id, Server.id):
                Server.add_select_server(data_id, neighbor)
                nutil.counter.increment_count('register')
                nutil.counter.increment_count('register_response')

    def unregister_dependency(self, server_id, vmeta):
        unregister_list = vmeta.get_select_server_id()
        for neighbor in unregister_list:
            self.ServerList[neighbor].unregister_dependency(vmeta.data_id, server_id)

    def delete_server_data(self, server_id, data_id):
        meta = self.ServerList[server_id].index.get(data_id)
        unselect_list = meta.get_register_servers()
        for neighbor in unselect_list:
            self.ServerList[neighbor].delete_select_server(data_id, server_id)
        self.ServerList[server_id].delete_data(data_id)
        server_neighbor = self.ServerList[server_id].get_neighbors()
        for neighbor in server_neighbor:
            self.ServerList[neighbor].hcbfTree_delete(self.netgraph.distance(server_id, neighbor), server_id, data_id)
        # self.register_dependency(self.ServerList[server_id], data_id)

    def data_dedup(self):
        strategy = {}
        decision_time = 0
        if TestConf.DEDUPLICATE_STRATEGY == 0:
            strategy, decision_time = self.hot_aware_deduplicate_with_cover(ignore_hot=TestConf.IGNORE_HOT)
        elif TestConf.DEDUPLICATE_STRATEGY == 3:
            strategy, decision_time = self.balence_deduplicate_with_cover(1-TestConf.MAX_DEDUPLICATE_RATE, ignore_hot=TestConf.IGNORE_HOT)
        elif TestConf.DEDUPLICATE_STRATEGY == 4:
            strategy, decision_time = self.random_deduplicate_with_cover(1-TestConf.MAX_DEDUPLICATE_RATE)
        elif TestConf.DEDUPLICATE_STRATEGY == 5:
            strategy, decision_time = self.random_deduplicate(TestConf.MAX_DEDUPLICATE_RATE)
        elif TestConf.DEDUPLICATE_STRATEGY == 6:
            strategy, decision_time = self.hot_aware_deduplicate(TestConf.MAX_DEDUPLICATE_RATE)
        else:
            self.hot_aware_deduplicate()
        return strategy, decision_time
    def random_deduplicate(self, ratio=1.0):
        # Information collection
        server_with_data = dict()
        server_need_data = dict()
        total_data = 0
        for server in self.ServerList.values():
            for data_id in server.index.keys():
                if data_id not in server_with_data:
                    server_with_data[data_id] = []
                server_with_data[data_id].append(server.id)
                if data_id not in server_need_data:
                    server_need_data[data_id] = []
                if server.index[data_id].query_times > 0:
                    server_need_data[data_id].append(server.id)
                total_data += 1
            for data_id in server.v_index.keys():
                if data_id not in server_need_data:
                    server_need_data[data_id] = []
                server_need_data[data_id].append(server.id)
        # make random decision
        data_cache = list(server_with_data.keys())
        dedup_count = 0
        strategy = {}
        start_time = time.time()
        while dedup_count/total_data <= ratio and len(data_cache) > 0:
            data_id = random.choice(data_cache)
            servers = server_with_data[data_id]
            if len(servers) == 1:
                data_cache.remove(data_id)
                continue
            delete_server_id = random.choice(servers)
            if data_id not in strategy:
                strategy[data_id] = [delete_server_id]
            else:
                strategy[data_id].append(delete_server_id)
            dedup_count += 1
            server_with_data[data_id].remove(delete_server_id)
        end_time = time.time()
        decision_time = end_time - start_time
        return strategy, decision_time
    def random_deduplicate_with_cover(self, percent=0.0):
        # Information collection
        server_with_data = dict()
        server_need_data = dict()
        for server in self.ServerList.values():
            for data_id in server.index.keys():
                if data_id not in server_with_data:
                    server_with_data[data_id] = []
                server_with_data[data_id].append(server.id)
                if data_id not in server_need_data:
                    server_need_data[data_id] = []

                if server.index[data_id].query_times > 0:
                    server_need_data[data_id].append(server.id)
            for data_id in server.v_index.keys():
                if data_id not in server_need_data:
                    server_need_data[data_id] = []
                server_need_data[data_id].append(server.id)
        # start deduplicate
        data_cache = list(server_with_data.keys())

        random.shuffle(data_cache)
        save_length = int(len(data_cache)*percent)
        top_half_data = [key for key in data_cache[:save_length]]
        for data_id in top_half_data:
            del server_with_data[data_id]
            del server_need_data[data_id]

        strategy = {}
        decision_time = 0
        for data_id, servers in server_with_data.items():
            cover = []
            # weights = np.random.uniform(low=1,high=2, size=len(servers))
            weights = []
            universe_cover = set()
            for server in servers:
                coverage = self.ServerList[server].get_neighbors()
                coverage.append(server)
                cover.append(coverage)
                universe_cover.update(coverage)
            server_need = server_need_data[data_id]
            _server_need = server_need.copy()
            for server in _server_need:
                if server not in universe_cover:
                    server_need.remove(server)
            for i in range(len(servers)):
                weights.append(1)
            # use gurobi find the optimal solution
            # select_set indicates which servers retain that data
            start_time = time.time()
            select_set = solve_set_cover_with_weights(server_need, cover, weights)
            end_time = time.time()
            decision_time += end_time - start_time
            if select_set is None:
                continue
            select_server = [servers[i] for i in select_set]
            delete_servers = list(set(servers) - set(select_server))
            strategy[data_id] = delete_servers
        return strategy, decision_time

    def hot_aware_deduplicate(self, ratio=1.0):
        # Information collection
        server_with_data = dict()
        server_data_weight = dict()
        data_hot = dict()
        server_need_data = dict()
        total_data = 0
        for server in self.ServerList.values():
            for data_id in server.index.keys():
                if data_id not in server_with_data:
                    server_with_data[data_id] = []
                server_with_data[data_id].append(server.id)
                if data_id not in server_data_weight:
                    server_data_weight[data_id] = []
                server_data_weight[data_id].append(server.index[data_id].query_times)
                if data_id not in server_need_data:
                    server_need_data[data_id] = []

                if server.index[data_id].query_times > 0:
                    server_need_data[data_id].append(server.id)
                total_data += 1
            for data_id in server.v_index.keys():
                if data_id not in server_need_data:
                    server_need_data[data_id] = []
                server_need_data[data_id].append(server.id)
        # start deduplicate
        for data_id, weights in server_data_weight.items():
            data_hot[data_id] = sum(weights)

        Per_ten_replica = int(self.ServerNum / 10)

        sorted_data_hot = sorted(data_hot, key=data_hot.get, reverse=True)
        full_replication_list = sorted_data_hot[:int(0.1*len(sorted_data_hot))]
        three_replication_list = sorted_data_hot[int(0.1*len(sorted_data_hot)):int(0.2*len(sorted_data_hot))]
        two_replication_list = sorted_data_hot[int(0.2*len(sorted_data_hot)):int(0.5*len(sorted_data_hot))]
        one_replication_list = sorted_data_hot[int(0.5*len(sorted_data_hot)):]
        dedup_count = 0
        strategy = {}
        start_time = time.time()
        for data_id in one_replication_list:
            servers = server_with_data[data_id]
            if len(servers) <= 1:
                continue
            # 选择weights最高的服务器
            # save_server_id = servers[server_data_weight[data_id].index(max(server_data_weight[data_id]))]
            indices = np.random.choice(range(len(servers)), 1, replace=False)
            save_server_id = [servers[i] for i in indices]
            # if data_hot[data_id] != 0:
            #     servers.remove(save_server_id)
            for server_id in save_server_id:
                servers.remove(server_id)
            if len(servers) != 0:
                strategy[data_id] = servers
                dedup_count += len(servers)
            if dedup_count / total_data >= ratio:
                break

        if dedup_count / total_data >= ratio:
            end_time = time.time()
            decision_time = end_time - start_time
            return strategy, decision_time

        for data_id in two_replication_list:
            servers = server_with_data[data_id]
            # 选择weights最高的两个服务器
            if len(servers) <= 2:
                continue
            # sorted_indices = sorted(range(len(server_data_weight[data_id])), key=lambda i: server_data_weight[data_id][i], reverse=True)
            # top_two_indices = sorted_indices[:2]
            top_two_indices = np.random.choice(range(len(servers)), 2, replace=False)
            save_server_id = [servers[i] for i in top_two_indices]
            # if data_hot[data_id] != 0:
            #     for server_id in save_server_id:
            #         servers.remove(server_id)
            for server_id in save_server_id:
                servers.remove(server_id)
            if len(servers) != 0:
                strategy[data_id] = servers
                dedup_count += len(servers)
            if dedup_count / total_data >= ratio:
                break

        if dedup_count / total_data >= ratio:
            end_time = time.time()
            decision_time = end_time - start_time
            return strategy, decision_time

        for data_id in three_replication_list:
            servers = server_with_data[data_id]
            # 选择weights最高的三个服务器
            if len(servers) < 3*Per_ten_replica:
                continue
            sorted_indices = sorted(range(len(server_data_weight[data_id])), key=lambda i: server_data_weight[data_id][i], reverse=True)
            top_three_indices = sorted_indices[:3*Per_ten_replica]
            # top_three_indices = np.random.choice(range(len(servers)), 3, replace=False)
            save_server_id = [servers[i] for i in top_three_indices]
            # if data_hot[data_id] != 0:
            #     for server_id in save_server_id:
            #         servers.remove(server_id)
            for server_id in save_server_id:
                servers.remove(server_id)
            if len(servers) != 0:
                strategy[data_id] = servers
                dedup_count += len(servers)
            if dedup_count / total_data >= ratio:
                break

        if dedup_count / total_data >= ratio:
            end_time = time.time()
            decision_time = end_time - start_time
            return strategy, decision_time

        for data_id in full_replication_list:
            servers = server_with_data[data_id]
            # 选择weights最高的ServerNum/10个服务器
            if len(servers) < 4*Per_ten_replica:
                continue
            sorted_indices = sorted(range(len(server_data_weight[data_id])),
                                    key=lambda i: server_data_weight[data_id][i], reverse=True)
            indices = sorted_indices[:4*Per_ten_replica]
            # indices = np.random.choice(range(len(servers)), int(self.ServerNum/5), replace=False)
            save_server_id = [servers[i] for i in indices]
            # if data_hot[data_id] != 0:
            #     for server_id in save_server_id:
            #         servers.remove(server_id)
            for server_id in save_server_id:
                servers.remove(server_id)
            if len(servers) != 0:
                strategy[data_id] = servers
                dedup_count += len(servers)
            if dedup_count / total_data >= ratio:
                break

        end_time = time.time()
        decision_time = end_time - start_time

        return strategy, decision_time

    def hot_aware_deduplicate_with_cover(self, percent=0.0, server_list = None, ignore_hot=False):
        # Information collection
        start_time = time.time()
        server_with_data = dict()
        server_data_weight = dict()
        data_hot = dict()
        server_need_data = dict()
        if server_list is None:
            server_list = self.ServerList.values()
        for server in server_list:
            for data_id in server.index.keys():
                if data_id not in server_with_data:
                    server_with_data[data_id] = []
                server_with_data[data_id].append(server.id)
                if data_id not in server_data_weight:
                    server_data_weight[data_id] = []
                server_data_weight[data_id].append(server.index[data_id].query_times)
                if data_id not in server_need_data:
                    server_need_data[data_id] = []

                if server.index[data_id].query_times > 0:
                    server_need_data[data_id].append(server.id)
            for data_id in server.v_index.keys():
                if data_id not in server_need_data:
                    server_need_data[data_id] = []
                server_need_data[data_id].append(server.id)

        if ignore_hot:
            for data_id, need_server in server_need_data.items():
                server_need_data[data_id] = self.getServer_list()

        # start deduplicate
        for data_id, weights in server_data_weight.items():
            data_hot[data_id] = sum(weights)

        sorted_data_hot = sorted(data_hot.items(), key=lambda item: item[1], reverse=True)
        # 计算前一半的元素数量
        save_length = int(len(sorted_data_hot)*percent)
        # 获取前一半的键
        top_half_data = [key for key, value in sorted_data_hot[:save_length]]
        for data_id in top_half_data:
            del server_with_data[data_id]
            del server_data_weight[data_id]
            del server_need_data[data_id]
        decision_time = 0
        strategy = {}
        for data_id, servers in server_with_data.items():
            cover = []
            universe_cover = set()
            weights = server_data_weight[data_id]

            if ignore_hot:
                weights = np.random.uniform(low=1, high=2, size=len(servers))

            # max_value = max(weights)
            # weights_p = [max_value - x for x in weights]
            # weights_p = [1 / (x + 1) * 100 for x in weights]
            for server in servers:
                coverage = self.ServerList[server].get_neighbors()
                coverage.append(server)
                cover.append(coverage)
                universe_cover.update(coverage)
            server_need = server_need_data[data_id]
            _server_need = server_need.copy()
            for server in _server_need:
                if server not in universe_cover:
                    server_need.remove(server)

            # use gurobi find the optimal solution
            # select_set indicates which servers retain that data

            # select_set = solve_set_cover_with_weights(server_need, cover, weights_p)
            select_set = solve_set_cover_and_max_weights(server_need, cover, weights)

            if select_set is None:
                continue
            select_server = [servers[i] for i in select_set]
            delete_servers = list(set(servers) - set(select_server))
            strategy[data_id] = delete_servers
        end_time = time.time()
        decision_time += end_time - start_time
        return strategy, decision_time

    def balence_deduplicate_with_cover(self, percent=0.0,ignore_hot = False):
        start_time = time.time()
        # Information collection
        server_with_data = dict()
        server_need_data = dict()
        for server in self.ServerList.values():
            for data_id in server.index.keys():
                if data_id not in server_with_data:
                    server_with_data[data_id] = []
                server_with_data[data_id].append(server.id)
                if data_id not in server_need_data:
                    server_need_data[data_id] = []

                if server.index[data_id].query_times > 0:
                    server_need_data[data_id].append(server.id)
            for data_id in server.v_index.keys():
                if data_id not in server_need_data:
                    server_need_data[data_id] = []
                server_need_data[data_id].append(server.id)

        if ignore_hot:
            for data_id, need_server in server_need_data.items():
                server_need_data[data_id] = self.getServer_list()

        data_need_list = list(server_need_data.keys())
        for data_need in data_need_list:
            if data_need not in server_with_data:
                del server_need_data[data_need]

        # start deduplicate
        data_cache = list(server_with_data.keys())
        random.shuffle(data_cache)
        save_length = int(len(data_cache) * percent)
        top_half_data = [key for key in data_cache[:save_length]]
        for data_id in top_half_data:
            del server_with_data[data_id]
            # del server_need_data[data_id]

        decision_time = 0
        remain_data_num = [0 for i in range(self.ServerNum)]
        strategy = {}
        for Server in self.ServerList.values():
            remain_data_num[Server.id] = len(Server.index)
        for data_id, servers in server_with_data.items():
            cover = []
            server_need = set()
            for server in servers:
                coverage = self.ServerList[server].get_neighbors()
                coverage.append(server)
                cover.append(coverage)
                server_need.update(coverage)
            server_need = list(server_need)
            # server_need = server_need_data[data_id]
            # use gurobi find the optimal solution
            # select_set indicates which servers retain that data
            select_set = solve_dedup_space_balance(server_need, cover, servers, remain_data_num, self.Capacity_list, TestConf.ALPHA)


            if select_set is None:
                continue
            select_server = [servers[i] for i in select_set]
            delete_servers = list(set(servers) - set(select_server))
            strategy[data_id] = delete_servers
            for delete_server in delete_servers:
                remain_data_num[delete_server] -= 1

        end_time = time.time()
        decision_time += end_time - start_time
        return strategy, decision_time

    def strategy_execute(self, strategy):
        dedup_count = 0
        for data_id, delete_servers in strategy.items():
            for delete_server_id in delete_servers:
                self.ServerList[delete_server_id].delete_data(data_id)
                dedup_count += 1
                server_neighbor = self.ServerList[delete_server_id].get_neighbors()
                for neighbor in server_neighbor:
                    self.ServerList[neighbor].hcbfTree_delete(self.netgraph.distance(delete_server_id, neighbor)
                                                              , delete_server_id
                                                              , data_id)
        return dedup_count

    def disDeduplicate(self, server_id):
        dedup_server = self.ServerList[server_id]
        neighbors = dedup_server.get_neighbors()
        nutil.counter.add_count('data_heat', len(neighbors))
        nutil.counter.add_count('data_heat_response', len(neighbors))
        dedup_server_list = []

        for neighbor in neighbors:
            dedup_server_list.append(self.ServerList[neighbor])
        dedup_server_list.append(dedup_server)
        strategy, decision_time = self.hot_aware_deduplicate_with_cover(TestConf.DIS_DEDUP_RATE, dedup_server_list)



        server_deletion = {}
        for data_id, delete_servers in strategy.items():
            # server_with_data = self.ServerList[server_id].neighbors_with_data(data_id)
            for delete_server_id in delete_servers:
                if delete_server_id not in server_deletion:
                    server_deletion[delete_server_id] = []
                server_deletion[delete_server_id].append(data_id)
        nutil.counter.add_count('data_predelete', len(server_deletion))

        for server_id, data_list in server_deletion.items():
            _data_list = data_list.copy()
            for data_id in _data_list:
                if data_id not in self.ServerList[server_id].index:
                    data_list.remove(data_id)
                else:
                    self.ServerList[server_id].index[data_id].pre_delete = True

        # decision_delete_num = 0
        # for server_id, data_list in server_deletion.items():
        #     print('server_id:', server_id, 'data_list:', [self.ServerList[server_id].index[data_id].data_address for data_id in data_list])
        #     decision_delete_num += len(data_list)
        # print('decision_delete_num:', decision_delete_num)

        dedup_count = 0
        server_deleted = {}
        for server_id, data_list in server_deletion.items():
            if TestConf.DEDUPLICATE_STRATEGY == 1:
                permissions = self.get_delete_permission_basic(server_id, data_list)
                # permissions_dependency = self.get_delete_permission_dependency(server_id, data_list)
            elif TestConf.DEDUPLICATE_STRATEGY == 2:
                permissions = self.get_delete_permission_dependency(server_id, data_list)
            elif TestConf.DEDUPLICATE_STRATEGY == 7:
                permissions = self.get_delete_permission_LDI(server_id, data_list)
            elif TestConf.DEDUPLICATE_STRATEGY == 8:
                permissions = self.get_delete_permission_CDI(server_id, data_list)
            else:
                logger.error('No such dis deduplicate mod')
                break
            for data_id, permission in permissions.items():
                if all(permission):
                    self.delete_server_data(server_id, data_id)
                    if server_id not in server_deleted:
                        server_deleted[server_id] = []
                    server_deleted[server_id].append(data_id)
                    dedup_count += 1
                else:
                    # if TestConf.DEDUPLICATE_STRATEGY == 1 and all(permissions_dependency[data_id]):
                    #     print(server_id, data_id)
                    self.ServerList[server_id].index[data_id].pre_delete = False
                    nutil.counter.increment_count('data_delete_failed_count')
                    if TestConf.DEDUPLICATE_STRATEGY == 2 or TestConf.DEDUPLICATE_STRATEGY == 8:
                        for neighbor in self.ServerList[server_id].index[data_id].get_register_servers():
                            nutil.counter.increment_count('data_delete_cancel')
                            if data_id in self.ServerList[neighbor].v_index:
                                self.ServerList[neighbor].v_index[data_id].set_select_server_safe(server_id)

        for server_id, data_list in server_deleted.items():
            for data_id in data_list:
                if self.ServerList[server_id].v_index[data_id].query_times > 0:
                    self.register_dependency(self.ServerList[server_id], data_id)
        nutil.counter.add_count('dedup_count', dedup_count)
        return dedup_count, decision_time


    def get_delete_permission_basic(self, server_id, data_list):
        permissions = {}
        for data_id in data_list:
            permissions[data_id] = []
            need_data_server = []
            # if server_id == 2 and data_id == 'c81e728d9d4c2f636f067f89cc14862c':
            #     print('c81e728d9d4c2f636f067f89cc14862c')
            for neighbor_id in self.ServerList[server_id].get_neighbors():
                nutil.counter.increment_count('delete_permission')
                neighbor = self.ServerList[neighbor_id]
                permission = False
                if data_id in neighbor.index:
                    permission = True
                elif data_id not in neighbor.v_index:
                    permission = True
                elif neighbor.v_index[data_id].query_times > 0:
                    need_data_server.append(neighbor_id)
                    neighbors_with_data = neighbor.neighbors_with_data(data_id)
                    if neighbors_with_data.count(server_id) > 0:
                        neighbors_with_data.remove(server_id)
                    safe = {}
                    for request_server in neighbors_with_data:
                        nutil.counter.increment_count('data_check')
                        meta = self.ServerList[request_server].index.get(data_id)
                        if meta is not None and not meta.pre_delete:
                            safe[request_server] = True
                        else:
                            safe[request_server] = False
                        nutil.counter.increment_count('data_check_response')
                    for if_safe in safe.values():
                        if if_safe:
                            permission = True
                            break
                else:
                    permission = True
                permissions[data_id].append(permission)
                nutil.counter.increment_count('delete_permission_response')
            # meta = self.ServerList[server_id].index.get(data_id)
            # if meta is None:
            #     continue
            # permission_server_list = meta.get_register_servers()
            # if len(need_data_server) != len(permission_server_list):
            #     print('error')
        return permissions

    def get_delete_permission_dependency(self, server_id, data_list):
        permissions = {}
        for data_id in data_list:
            meta = self.ServerList[server_id].index.get(data_id)
            if meta is None:
                continue
            permissions[data_id] = []
            permission_server_list = meta.get_register_servers()
            for neighbor_id in permission_server_list:
                nutil.counter.increment_count('delete_permission')
                neighbor = self.ServerList[neighbor_id]
                permission = False
                if data_id in neighbor.index:
                    permission = True
                else:
                    vmeta = neighbor.v_index.get(data_id)
                    if vmeta is not None and vmeta.get_permission(server_id):
                        permission = True
                permissions[data_id].append(permission)
                nutil.counter.increment_count('delete_permission_response')
        return permissions

    def get_delete_permission_LDI(self, server_id, data_list):
        permissions = {}
        for data_id in data_list:
            meta = self.ServerList[server_id].index.get(data_id)
            if meta is None:
                continue
            permissions[data_id] = []
            permission_server_list = meta.get_register_servers()
            for neighbor_id in permission_server_list:
                nutil.counter.increment_count('delete_permission')
                neighbor = self.ServerList[neighbor_id]
                permission = False
                if data_id in neighbor.index or data_id not in neighbor.v_index:
                    permission = True
                else:
                    neighbors_with_data = neighbor.neighbors_with_data(data_id)
                    if neighbors_with_data.count(server_id) > 0:
                        neighbors_with_data.remove(server_id)
                    safe = {}
                    for request_server in neighbors_with_data:
                        nutil.counter.increment_count('data_check')
                        meta = self.ServerList[request_server].index.get(data_id)
                        if meta is not None and not meta.pre_delete:
                            safe[request_server] = True
                        else:
                            safe[request_server] = False
                        nutil.counter.increment_count('data_check_response')
                    for if_safe in safe.values():
                        if if_safe:
                            permission = True
                            break
                permissions[data_id].append(permission)
                nutil.counter.increment_count('delete_permission_response')
        return permissions

    def get_delete_permission_CDI(self, server_id, data_list):
        permissions = {}
        for data_id in data_list:
            meta = self.ServerList[server_id].index.get(data_id)
            if meta is None:
                continue
            permissions[data_id] = []
            permission_server_list = self.ServerList[server_id].get_neighbors()
            for neighbor_id in permission_server_list:
                nutil.counter.increment_count('delete_permission')
                neighbor = self.ServerList[neighbor_id]
                permission = False
                if neighbor_id not in meta.get_register_servers():
                    permission = True
                else:
                    vmeta = neighbor.v_index.get(data_id)
                    if vmeta is not None and vmeta.get_permission(server_id):
                        permission = True
                permissions[data_id].append(permission)
                nutil.counter.increment_count('delete_permission_response')
        return permissions


    def TestLatency_by_net(self, request_list):
        logger.warning('start test latency')
        query_hop_list = []
        cloud_time = 0
        for row_index, row in request_list.iterrows():
            data_id = row['data_id']
            query_mess = Message.Message_Query(data_id)
            Server = self.ServerList[row['server']]
            response = asyncio.run(self.request(Server.id, query_mess))
            if response is not None and response.status:
                query_hop_list.append(response.trans_hop)
            else:
                query_hop_list.append(20)
                cloud_time += 1
        average_latency = np.sum(query_hop_list) / len(query_hop_list)
        return average_latency, cloud_time

    def close_servers(self):
        for server_id, Server in self.ServerList.items():
            close_message = Message.Message_close_server(server_id)
            address = (Server.server_host, Server.server_port)
            response = asyncio.run(self.request(server_id, close_message))
            if response is None:
                logger.error(f'server {server_id} close response error')
            else:
                count = response.count
                for key, item in count.items():
                    nutil.counter.add_count(key, item)

    def start_server_deduplicate(self, server_id):
        message = Message.Message_Data_Deduplicate(server_id)
        asyncio.run(self.post(server_id, message))

    async def request(self, server_id, mess):
        cnn_pool = self.connections.get(server_id)
        if cnn_pool is None:
            client_thread = threading.Thread(target=self.get_connection_pool, args=(
                (self.ServerList[server_id].server_host, self.ServerList[server_id].server_port), server_id),
                                             name=f"server manager connect_thread")
            client_thread.start()
            client_thread.join()
            cnn_pool = self.connections[server_id]

        if if_mess_show(mess):
            logger.info(f"server manager request {mess} to server {server_id}")
        response = cnn_pool.request(mess)
        if if_mess_show(response):
            logger.info(f"server manager receive {response} from server {server_id}")
        return response

    async def post(self, server_id, mess):
        cnn_pool = self.connections.get(server_id)
        if cnn_pool is None:
            client_thread = threading.Thread(target=self.get_connection_pool, args=(
                (self.ServerList[server_id].server_host, self.ServerList[server_id].server_port), server_id),
                                             name=f"server manager connect_thread")
            client_thread.start()
            client_thread.join()
            cnn_pool = self.connections[server_id]

        if if_mess_show(mess):
            logger.info(f"server manager request {mess} to server {server_id}")
        cnn_pool.send(mess)

    def get_connection_pool(self, address, target_id):
        return nutil.ConnectionPool(address, ServerConf.CONNECTION_POOL_SIZE, target_id)

    async def broadcast_message_response(self, servers, message):
        if nutil.if_mess_show(message):
            logger.info(f'server manager broadcast {message} to {servers}')
        # 使用 asyncio.gather 并发执行异步请求
        responses = await asyncio.gather(
            *[self.request(server, message) for server in servers])
        return responses

    async def broadcast_message(self, servers, message):
        if nutil.if_mess_show(message):
            logger.info(f'server manager broadcast {message} to {servers}')
        # 使用 asyncio.gather 并发执行异步请求
        await asyncio.gather(*[self.post(server, message) for server in servers])


if __name__ == "__main__":
    SM = Manager()
    SM.install()
    print(SM.get_average_neighbor_num())
    print(SM.netgraph.density())
    # # SM.netgraph.show(with_label=True)
    # request_list = pd.read_csv('../request.csv')
    # log.info('去重前：')
    # SM.TestLatency(request_list)
    # SM.random_deduplicate_with_cover()
    # print('去重后：')
    # SM.TestLatency(request_list)
    # unio_set = set()
    # cover = [1 ,2, 3, 4, 5, 6, 7, 8, 9, 10]
    # cover2 = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14]
    # unio_set.update(cover)
    # unio_set.update(cover2)
