# this is the code file for node and how they handle different kinds of message
import asyncio
import copy
import socket
import logging
import threading
import time

import numpy as np
from omegaconf import OmegaConf as oc

from Network.network_utils import if_mess_show
from Server.Cache import CacheFactory
from util import config
from Network import Message
from Network import network_utils as nutil
from Server import pybloom2
from util.Algorithm import _to_str, solve_set_cover_with_weights, solve_set_cover_and_max_weights

NetworkConf = oc.structured(config.NetWorkConfig)
ServerConf = oc.structured(config.ServerConfig)
GraphConfig = oc.structured(config.GraphConfig)
TestConf = oc.structured(config.TestConfig)
logger = logging.getLogger("logger")



def process_heat_responses(responses):
    data_heat = dict()
    server_cover = dict()
    data_need = []
    for response in responses:
        if response.cover is not None:
            data_heat[response.server_id] = response.heat
            server_cover[response.server_id] = response.cover
        if response.need:
            data_need.append(response.server_id)
    return data_heat, server_cover, data_need


class adjNode:
    def __init__(self, id, distance, address, cbf=None):
        self.id = id
        self.distance = distance
        self.address = address
        self.cbf: pybloom2.CountingBloomFilter = cbf


class register:
    def __init__(self, server_id, safe=False, clock=10):
        self.server_id = server_id
        self.safe = safe
        self.clock = clock
        self.select_score = 0


class metaData:
    def __init__(self, address):
        self.data_address = address
        self.query_times = 0
        self.pre_delete = False
        self.register_servers = dict()

    def add_register_server(self, server_id):
        register_server = register(server_id)
        self.register_servers[server_id] = register_server

    def delete_register_server(self, server_id):
        if server_id in self.register_servers:
            del self.register_servers[server_id]

    def get_register_servers(self):
        return list(self.register_servers.keys())

class v_metaData:
    def __init__(self, data_id=None, query_times=1):
        self.data_id = data_id
        self.query_times = query_times
        self.select_server = dict()
        self.lock = threading.Lock()

    def get_select_server_id(self):
        return list(self.select_server.keys())

    def unregister_server(self, server_id):
        if server_id in self.select_server:
            del self.select_server[server_id]

    def get_permission(self, server_id):
        if server_id in self.select_server:
            self.select_server[server_id].safe = False
        safe_list = [key for key, obj in self.select_server.items() if obj.safe]
        return len(safe_list) > 0


    def set_select_server_safe(self,server_id):
        if server_id in self.select_server:
            self.select_server[server_id].safe = True

    def get_safe_server(self):
        if not self.select_server:
            return None
        min_distance_server = min(self.select_server, key=lambda k: self.select_server[k].select_score)
        return min_distance_server


class Node:
    def __init__(self, id: str, server_host: str, server_port: str, capacity):
        self.id = id
        self.server_host = server_host
        self.server_port = server_port
        self.capacity = capacity
        self.hop = ServerConf.HOP_NUM
        self.lock = threading.Lock()
        self.cbf_element_num = max(ServerConf.ELEMENT_NUM,ServerConf.ELEMENT_NUM*self.hop*GraphConfig.DEGREE)
        self.start = False
        self.ready = False
        self.thread_pool = []
        self.counter = nutil.Counter()

        self.adjList = dict()
        self.neighbors = dict()
        # self.index = dict()
        self.index = CacheFactory.create_cache(ServerConf.CACHE_TYPE, self.capacity)
        self.v_index = CacheFactory.create_cache(ServerConf.CACHE_TYPE, 3000)
        # self.request_filter = pybloom2.CountingBloomFilter(error_rate=0.001, element_num=10000)
        self.connections = dict()
        self.cbf = pybloom2.CountingBloomFilter(error_rate=0.001, element_num=self.cbf_element_num)


    def search_data(self, data_id):
        meta = self.index.search(data_id)
        if meta is not None:
            self.index[data_id].query_times += 1
            return 'local', meta.data_address
        vmeta = self.v_index.search(data_id)
        if vmeta is not None:
            self.v_index[data_id].query_times += 1
            return 'neighbor', vmeta.get_safe_server()
        # elif self.request_filter.exists(data_id):
        #     evictObject = self.v_index.put(data_id, v_metaData(data_id))
        #     return None, evictObject
        # else:
        #     self.request_filter.add(data_id)
        else:
            evictObject = self.v_index.put(data_id, v_metaData(data_id))
            return None, evictObject
        # return None, None

    def save_data(self, data_id ,data):
        if data_id in self.index:
            logger.debug(f"data {data} is exist")
            return

        evictObject = self.index.put(data_id,metaData(data))
        self.cbf.add(data_id)
        if self.start and not TestConf.SIMULATION:
            logger.info(f"server {self.id} save data {data}")
            update_message = Message.Message_hbfc_update(self.id, data_id, True)
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.broadcast_message(self.get_neighbors(), update_message))
        # if evictObject is not None:
        #     evictVirtueData = self.add_virtual_data(data_id, evictObject.query_times)
        #     return evictVirtueData
        return evictObject

    def delete_data(self, data_id):
        evitObject = None
        with self.lock:
            delete_update = False
            if data_id in self.index:
                evictObject = self.add_virtual_data(data_id, self.index[data_id].query_times)
                del self.index[data_id]
                self.cbf.delete(data_id)
                delete_update = True
                self.counter.increment_count('total_delete')
        if self.start and delete_update:
            logger.debug(f"server {self.id} delete data {data_id}")
            update_message = Message.Message_hbfc_update(self.id, data_id, False)
            asyncio.run(self.broadcast_message(self.get_neighbors(), update_message))

        # if evictObject is not None:
        #     pass

    def add_virtual_data(self, data_id, query_times = 1):
        vmeta = v_metaData(data_id, query_times)
        evictObject = self.v_index.put(data_id, vmeta)
        return evictObject

    def delete_virtual_data(self, data_id):
        vmeta = self.v_index[data_id]
        del self.v_index[data_id]
        return vmeta

    def add_neighbor(self, adjNode):
        if adjNode.distance not in self.adjList:
            self.adjList[adjNode.distance] = dict()
        self.adjList[adjNode.distance][adjNode.id] = adjNode
        self.neighbors[adjNode.id] = adjNode

    def get_neighbors(self):
        return list(self.neighbors.keys())

    def generate_hcbfTree(self):
        sum_bits_of_cbf, second_level_sum_bits = [], []
        root_filter = pybloom2.CountingBloomFilter(error_rate=0.001, element_num=self.cbf_element_num)
        level_filters = dict()
        root_count = 0
        for j in range(self.hop):
            level_count = 0
            if j+1 not in self.adjList:
                break
            near_nodes: dict = self.adjList[j + 1]
            sum_bits_of_cbf.clear()
            level_filters[j + 1] = pybloom2.CountingBloomFilter(error_rate=0.001,
                                                                element_num=self.cbf_element_num)
            for key, item in near_nodes.items():
                sum_bits_of_cbf.append(copy.deepcopy(item.cbf.each_bit))
                level_count += item.cbf.count
            second_level_sum_bits.append(copy.deepcopy(np.sum(np.array(sum_bits_of_cbf), axis=0)))
            level_filters[j + 1].each_bit = copy.deepcopy(np.sum(np.array(sum_bits_of_cbf), axis=0))
            level_filters[j + 1].count = level_count
            root_count += level_count
        if len(second_level_sum_bits) > 0:
            root_filter.each_bit = copy.deepcopy(np.sum(second_level_sum_bits, axis=0))
            root_filter.count = root_count
        self.adjList[0] = level_filters
        self.adjList['root'] = root_filter

    def hcbfTree_delete(self, hop, server_id, data_id):
        if self.hcbfTree_find(hop, server_id, data_id):
            self.adjList[hop][server_id].cbf.delete(data_id)
            self.adjList[0][hop].delete(data_id)
            self.adjList['root'].delete(data_id)

    def hcbfTree_insert(self, hop, server_id, data_id):
        if not self.hcbfTree_find(hop, server_id, data_id):
            self.adjList[hop][server_id].cbf.add(data_id)
            self.adjList[0][hop].add(data_id)
            self.adjList['root'].add(data_id)
            logger.debug(f'hcbf insert update for {server_id}.{data_id} {self.hcbfTree_find(hop, server_id, data_id)}')

    def hcbfTree_find(self, hop, server_id, data_id):
        return self.adjList[hop][server_id].cbf.exists(data_id)

    def find_data_hcbf(self, data):
        result = False
        server_id = -1
        hop = -1
        root_filter: pybloom2.CountingBloomFilter = self.adjList['root']
        if root_filter.exists(data):
            for hop, lever_filter in self.adjList[0].items():
                if lever_filter.exists(data):
                    for key, adjnode in self.adjList[hop].items():
                        if adjnode.cbf.exists(data):
                            result = True
                            server_id = adjnode.id
                            return result, hop, server_id
        return result, hop, server_id

    def neighbors_with_data(self, data):
        result = []
        root_filter: pybloom2.CountingBloomFilter = self.adjList['root']
        if root_filter.exists(data):
            for hop, lever_filter in self.adjList[0].items():
                if lever_filter.exists(data):
                    for key, adjnode in self.adjList[hop].items():
                        if adjnode.cbf.exists(data):
                            result.append(adjnode.id)
        return result

    def update_select_list(self, data_id):
        if data_id not in self.v_index:
            return []
        vmeta = self.v_index[data_id]
        registered_list = vmeta.get_select_server_id()
        neighbors = self.neighbors_with_data(data_id)
        need_register_list = list(set(neighbors) - set(registered_list))
        if len(need_register_list) > 0:
            if TestConf.SIMULATION:
                return need_register_list
            register_message = Message.Message_register(self.id, data_id)
            responses = asyncio.run(self.broadcast_message_response(need_register_list, register_message))
            for response in responses:
                if response is not None and response.state:
                    self.add_select_server(data_id, response.server_id)
        return []

    # server_id register to establish dependency for data_id
    def register_dependency(self, data_id, server_id):
        meta = self.index.get(data_id)
        if meta is not None and not meta.pre_delete:
            meta.add_register_server(server_id)
            return True
        return False

    def unregister_dependency(self, data_id, server_id):
        meta = self.index.get(data_id)
        if meta is not None:
            meta.delete_register_server(server_id)

    # record had establish dependency to server_id for data_id
    def add_select_server(self, data_id, server_id):
        if data_id in self.v_index:
            register_node = register(server_id, True)
            register_node.select_score = self.neighbors[server_id].distance
            self.v_index[data_id].select_server[server_id] = register_node
            return True
        return False

    def delete_select_server(self,data_id, server_id):
        if data_id in self.v_index:
            self.v_index[data_id].unregister_server(server_id)
            return True
        return False

    #start the server listen thread and keep run it
    def run(self):
        listen_thread = threading.Thread(target=self.listen,args=())
        listen_thread.start()
        self.init_connection_pool()
        self.ready = True
        listen_thread.join()
        logger.info(f'server {self.id} close')

    #init the connection pool for each neighbor
    def init_connection_pool(self):
        for nid,neighbor in self.neighbors.items():
            self.connections[nid] = nutil.ConnectionPool(neighbor.address,ServerConf.CONNECTION_POOL_SIZE,nid)

    #listen the message from client and other servers
    def listen(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.server_host, self.server_port))
        server.listen(5)
        logger.info(f"Server listening on {self.server_host}:{self.server_port}...")
        self.start = True
        while self.start:
            client, addr = server.accept()
            client_thread = threading.Thread(target=self.handle_message, args=(client,))
            self.thread_pool.append(client_thread)
            client_thread.start()


    def handle_message(self, client):  # this function need to be run by a thread so that it can be run forever
        while self.start:
            try:
                message, error = nutil.recive_mess(client)
                if nutil.if_mess_show(message):
                    logger.debug(f"server {self.id} receive {str(message)} from {client.getpeername()}")
                if not message:
                    # logger.info(f'{error} from {client}')
                    client.close()
                    break
                elif message.type == 'ready':
                    self.Ready_Handle(client,message)
                elif message.type == 'data_query':
                    threading.current_thread().name = f'server {self.id} Data_Query_Handle'
                    self.Data_Query_Handle(client, message)
                elif message.type == 'data_cache':
                    threading.current_thread().name = f'server {self.id} data_cache_handle'
                    self.data_cache_handle(client, message)
                elif message.type == 'hbfc_update':
                    hbfc_update_thread = threading.Thread(target=self.hbfc_update_handle, args=(client, message,),
                                                         name=f'server {self.id} hbfc_update_handle')
                    hbfc_update_thread.start()
                elif message.type == 'data_deduplicate':
                    threading.current_thread().name = f'server {self.id} data_deduplicate_handle'
                    self.data_deduplicate()
                elif message.type == 'data_heat':
                    threading.current_thread().name = f'server {self.id} data_heat_handle'
                    self.data_heat_handle(client, message)
                elif message.type == 'data_predelete':
                    predelete_thread = threading.Thread(target=self.data_predelete_handle, args=(message,),name=f'server {self.id} data_predelete_handle')
                    predelete_thread.start()
                elif message.type == 'delete_permission':
                    permission_thread = threading.Thread(target=self.delete_permission_handle, args=(client, message,),
                                                        name=f'server {self.id} delete_permission_handle')
                    permission_thread.start()
                elif message.type == 'data_delete_cancel':
                    delete_cancel_thread = threading.Thread(target=self.data_delete_cancel_handle, args=(client, message,),
                                                         name=f'server {self.id} data_delete_cancel_handle')
                    delete_cancel_thread.start()
                elif message.type == 'data_check':
                    threading.current_thread().name = f'server {self.id} data_check_handle'
                    self.data_check_handle(client, message)
                elif message.type == 'register':
                    threading.current_thread().name = f'server {self.id} register_handle'
                    self.register_handle(client, message)
                elif message.type == 'close_server':
                    self.close_handle(client, message)
            except Exception as e:
                client.close()
                logger.error(f"Error handling connection from : {e}")
                break


    # these functions are used for handle different types of message and different roles need to handle them differently
    def Ready_Handle(self,client,message):
        while not self.ready:
            continue
        ready_response = Message.Message_ready_response(True)
        nutil.send_mess(client,ready_response)

    def Data_Query_Handle(self, client, mess):
        Place_type, address = self.search_data(mess.data_id)
        response_mess = Message.Message_Query_Response(
            mess.data_id,
            None, False, 0)
        if Place_type == 'local':
            response_mess.data = address
            response_mess.status = True
            nutil.send_mess(client, response_mess)
            return
        adjNode = None
        if Place_type == 'neighbor':
            adjNode = self.neighbors[address]
        else:
            result, hop, server_id = self.find_data_hcbf(mess.data_id)
            if result:
                adjNode = self.adjList[hop][server_id]
        if mess.request_type==0 and adjNode is not None:
            query_mess = Message.Message_Query(mess.data_id, 1)
            message = asyncio.run(self.request(adjNode.id, query_mess))
            if message.status:
                    response_mess.data = message.data
                    response_mess.status = True
                    response_mess.trans_hop = adjNode.distance
        nutil.send_mess(client, response_mess)
        if mess.data_id in self.v_index:
            new_thread = threading.Thread(target=self.update_select_list, args=(mess.data_id,))
            # self.update_select_list(data_id=mess.data_id)
            new_thread.start()

    def data_cache_handle(self, client, mess):
        self.save_data(mess.data_id,mess.data)
        # if len(self.index) > self.capacity * 0.9:
        #     asyncio.run(self.data_deduplicate())

    def hbfc_update_handle(self, client, message):
        neighbor = self.neighbors[message.server_id]
        if message.status:
            self.hcbfTree_insert(neighbor.distance, message.server_id, message.data_id)
            if message.data_id in self.v_index:
                self.update_select_list(message.data_id)
        else:
            self.hcbfTree_delete(neighbor.distance, message.server_id, message.data_id)
            self.unregister_dependency(message.data_id,message.server_id)


    def data_deduplicate(self):

        dedup_list = list(self.index.items())
        for data_id, meta in dedup_list:
            neighbors = self.get_neighbors()
            message = Message.Message_Data_Heat(self.id, data_id)
            responses = asyncio.run(self.broadcast_message_response(neighbors, message))
            data_heat, serve_cover, server_need = process_heat_responses(responses)
            server_with_data = [self.id]
            weights = [meta.query_times]
            cover = [neighbors]
            if meta.query_times > 0:
                server_need.append(self.id)
            for server_id, heat in data_heat.items():
                server_with_data.append(server_id)
                weights.append(data_heat[server_id])
                cover.append(serve_cover[server_id])
            # weights_p = [1 / (x + 1) * 100 for x in weights]
            start_time = time.time()
            select_set = solve_set_cover_and_max_weights(server_need, cover, weights)
            end_time = time.time()
            nutil.counter.add_count('dedup time', end_time - start_time)
            if select_set is None:
                continue
            select_server = [server_with_data[i] for i in select_set]
            delete_servers = list(set(server_with_data) - set(select_server))
            logger.debug(f'for data {data_id}, dedup servers are {delete_servers}')
            if delete_servers.count(self.id) > 0:
                delete_servers.remove(self.id)
                # self.delete_data(data_id)
                self_delete_thread = threading.Thread(target=self.data_predelete_handle, args=(Message.Message_data_predelete(self.id, data_id),))
                self_delete_thread.start()
                nutil.counter.increment_count('data_predelete')
            for delete_server in delete_servers:
                message = Message.Message_data_predelete(delete_server, data_id)
                asyncio.run(self.post(self.neighbors[delete_server].id, message))
        logger.debug(f'data dedup end')
        nutil.counter.increment_count('dedup end')



    def data_heat_handle(self, client, message):
        response_mess = Message.Message_Data_Heat_Response(self.id, message.data_id, 0)
        meta_data = self.index.get(message.data_id)
        if meta_data is not None:
            response_mess.heat = meta_data.query_times
            response_mess.cover = self.get_neighbors()
            response_mess.need = True
        vmeta_data = self.v_index.get(message.data_id)
        if vmeta_data is not None:
            response_mess.need = True
        nutil.send_mess(client, response_mess)
        # client.close()

    def data_predelete_handle(self, message):
        meta = self.index.get(message.data_id)
        if meta is None or meta.pre_delete:
            return
        meta.pre_delete = True
        request_message = Message.Message_Delete_Permission(self.id, message.data_id)
        permission_server_list = []
        if TestConf.DEDUPLICATE_MOD == 0:
            permission_server_list = self.get_neighbors()
        elif TestConf.DEDUPLICATE_MOD == 1:
            permission_server_list = meta.get_register_servers()
        responses = asyncio.run(self.broadcast_message_response(permission_server_list, request_message))

        for response in responses:
            if response is not None :
                if not response.permission:
                    logger.debug(f'server {response.server_id} refuse delete data {message.data_id}')
                    if TestConf.DEDUPLICATE_MOD == 1:
                        cancel_mess = Message.Message_data_delete_cancel(self.id, message.data_id)
                        asyncio.run(self.broadcast_message(permission_server_list, cancel_mess))
                    meta.pre_delete = False
                    nutil.counter.increment_count('cancel delete')
                    return
            else:
                if TestConf.DEDUPLICATE_MOD == 1:
                    cancel_mess = Message.Message_data_delete_cancel(self.id, message.data_id)
                    asyncio.run(self.broadcast_message(permission_server_list, cancel_mess))
                meta.pre_delete = False
                nutil.counter.increment_count('cancel delete')
                return
        self.delete_data(message.data_id)

    def delete_permission_handle(self, client, message):
        response_message = Message.Message_Delete_Permission_Response(self.id, message.data_id)
        if message.data_id in self.index.keys():
            response_message.permission = True
        elif TestConf.DEDUPLICATE_MOD == 0:
            neighbors_with_data = self.neighbors_with_data(message.data_id)
            request_message = Message.Message_Data_Check(self.id, message.data_id)
            if neighbors_with_data.count(message.server_id) > 0:
                neighbors_with_data.remove(message.server_id)

            responses = asyncio.run(self.broadcast_message_response(neighbors_with_data, request_message))
            for response in responses:
                if response is not None and response.safe:
                    response_message.permission = True
                    break

        elif TestConf.DEDUPLICATE_MOD == 1:
            vmeta = self.v_index.get(message.data_id)
            if vmeta is not None and vmeta.get_permission(message.server_id):
                response_message.permission = True
        nutil.send_mess(client, response_message)
        # client.close()

    def data_delete_cancel_handle(self, client, message):
        if message.data_id in self.v_index:
            self.v_index[message.data_id].set_select_server_safe(message.server_id)
    def data_check_handle(self, client, message):
        response_message = Message.Message_Data_Check_Response(self.id, message.data_id)
        meta = self.index.get(message.data_id)
        if meta is not None and not meta.pre_delete:
            response_message.safe = True
        nutil.send_mess(client, response_message)
        # client.close()

    def register_handle(self, client, message):
        response_message = Message.Message_register_response(self.id)
        state = self.register_dependency(message.data_id, message.server_id)
        response_message.state = state
        nutil.send_mess(client, response_message)
        # client.close()

    def close_handle(self, client, mess):
        self.start = False
        response_mess = Message.Message_close_response(self.id,self.counter.get_count())
        nutil.send_mess(client, response_mess)

    async def request(self, server_id, mess):
        cnn_pool = self.connections.get(server_id)
        if cnn_pool is None:
            client_thread = threading.Thread(target=self.get_connection_pool, args=(
                self.neighbors[server_id].address,server_id), name=f"server {self.id} connect_thread")
            client_thread.start()
            client_thread.join()
            cnn_pool = self.connections[server_id]

        if if_mess_show(mess):
            logger.info(f"server {self.id} request {mess} to server {server_id}")
        response = cnn_pool.request(mess)
        if if_mess_show(response):
            logger.info(f"server {self.id} receive {response} from server {server_id}")
        return response


    async def post(self, server_id, mess):
        cnn_pool = self.connections.get(server_id)
        if cnn_pool is None:
            client_thread = threading.Thread(target=self.get_connection_pool, args=(
                self.neighbors[server_id].address,server_id), name=f"server {self.id} connect_thread")
            client_thread.start()
            client_thread.join()
            cnn_pool = self.connections[server_id]

        if if_mess_show(mess):
            logger.info(f"server {self.id} request {mess} to server {server_id}")
        cnn_pool.send(mess)

    def get_connection_pool(self, address,target_id):
        return nutil.ConnectionPool(address,ServerConf.CONNECTION_POOL_SIZE,target_id)

    async def broadcast_message_response(self, servers, message):
        if len(servers) == 0:
            return []
        if nutil.if_mess_show(message):
            logger.info(f'server {self.id} broadcast {message} to {servers}')
        # ʹ   asyncio.gather     ִ   첽    
        responses = await asyncio.gather(
            *[self.request(server, message) for server in servers])
        return responses

    async def broadcast_message(self, servers, message):
        if len(servers) == 0:
            return
        if nutil.if_mess_show(message):
            logger.info(f'server {self.id} broadcast {message} to {servers}')
        await asyncio.gather(*[self.post(server, message) for server in servers])
