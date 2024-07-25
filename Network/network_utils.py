import pickle
import queue
import struct
import socket
import threading
import time
import json
import logging
from omegaconf import OmegaConf as oc

from util import config

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

TestConf = oc.structured(config.TestConfig)


class Counter:
    def __init__(self):
        self.count = dict()
        self.lock = threading.Lock()

    def increment_count(self, key):
        with self.lock:
            if key in self.count:
                self.count[key] += 1
            else:
                self.count[key] = 1
    def add_count(self, key, num):
        with self.lock:
            if key in self.count:
                self.count[key] += num
            else:
                self.count[key] = num
    def decrement_count(self, key):
        with self.lock:
            if key in self.count and self.count[key] > 0:
                self.count[key] -= 1

    def get_count(self):
        with self.lock:
            return self.count

    def clear_count(self):
        with self.lock:
            self.count.clear()


counter = Counter()


class ConnectionPool:
    def __init__(self, address, pool_size,target_id):
        self.address = address
        self.pool_size = pool_size
        self.target_id = target_id
        self.connect_num = 0
        self.connections = queue.Queue(maxsize=pool_size)

        # Create initial connections in the pool
        for _ in range(self.pool_size):
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            while True:
                try:
                    connection.connect(self.address)
                    self.connections.put(connection)
                    logger.debug(f'connect to server {self.target_id}')
                    break
                except Exception as e:
                    time.sleep(0.1)
                    continue



    def _get_connection(self):
        # Get a connection from the pool (or create a new one if the pool is empty)
        ## Get connection block mod
        # connection = self.connections.get()  # 阻塞直到队列中有可用连接
        # return connection
        ## Get connection no block mod
        try:
            connection = self.connections.get_nowait()
        except queue.Empty:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.connect(self.address)
        return connection

    def _release_connection(self, connection):
        # Release the connection back to the pool
        ## release connection block mod
        #self.connections.put(connection)
        ## release connection no block mod
        try:
            self.connections.put(connection, block=False)
        except queue.Full:
            connection.close()

    def request(self, data):
        # Execute some operation using a connection from the pool
        connection = self._get_connection()
        try:
            # send and receive data
            send_mess(connection, data)
            response, state = recive_mess(connection)
            return response
        finally:
            # Release the connection back to the pool
            self._release_connection(connection)

    def send(self, data):
        # Execute some operation using a connection from the pool
        connection = self._get_connection()
        try:
            # send and receive data
            send_mess(connection, data)
        finally:
            # Release the connection back to the pool
            self._release_connection(connection)

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config


def if_mess_show(message):
    if not message:
        return False
    elif message.type == 'data_query' or message.type == 'data_query_response':
        return TestConf.DATA_QUERY_SHOW
    elif message.type == 'data_cache':
        return TestConf.DATA_CACHE_SHOW
    elif message.type == 'hbfc_update':
        return TestConf.HBFC_UPDATE_SHOW
    elif message.type == 'data_deduplicate':
        return TestConf.DATA_DEDUP_SHOW
    elif message.type == 'data_heat' or message.type == 'data_heat_response':
        return TestConf.DATA_HEAT_SHOW
    elif message.type == 'data_predelete':
        return TestConf.DATA_PREDELETE_SHOW
    elif message.type == 'delete_permission' or message.type == 'delete_permission_response':
        return TestConf.DELETE_PERMI_SHOW
    elif message.type == 'data_check' or message.type == 'data_check_response':
        return TestConf.DATA_CHECK_SHOW
    elif message.type == 'close_server' or message.type == 'close_server_response':
        return TestConf.CLOSE_MESS_SHOW
    elif message.type == 'register' or message.type == 'register_response':
        return TestConf.REGISTER_SHOW
    else:
        return False


async def post(address, mess):
    if address is None:
        logger.info(f'server address cont be None')
        return
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((socket.gethostname(), 0))
    try:
        s.connect(address)
        send_mess(s, mess)
        s.close()
    except Exception as e:
        logger.error(f'{s.getsockname()} connect {address} post {mess} fail:{e}')


async def request(address, mess):
    if address is None:
        logger.info(f'server address cont be None')
        return
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((socket.gethostname(), 0))
    try:
        s.connect(address)
        send_mess(s, mess)
        message, state = recive_mess(s)
        s.close()
        return message
    except Exception as e:
        logger.error(f'{s.getsockname()} connect {address} request {mess} fail:{e}')


def send_mess(client, mess):
    mess_b = pickle.dumps(mess)
    mess_b_len = len(mess_b)
    len_pre = struct.pack("!I", mess_b_len)
    combined_message = len_pre + mess_b
    client.sendall(combined_message)
    # if if_mess_show(mess):
    #     logger.info(f"{client.getsockname()} send {mess} to {client.getpeername()}")
    counter.increment_count(mess.type)


def recive_mess(client, timeout=None):
    client.settimeout(timeout)
    len_pre = b""
    while len(len_pre) < 4:
        try:
            chunk = client.recv(4 - len(len_pre))
            if not chunk:
                # error
                break
            len_pre += chunk
            # logger.info("Recived length chunk...")
        except:
            logger.error(f"Timeout while receiving message length from {client.getpeername()}")
            client.settimeout(None)
            return None, "Timeout while receiving message length"

    if len(len_pre) == 0:
        client.settimeout(None)
        return None, "Connection closed..."

    if len(len_pre) < 4:
        client.settimeout(None)
        return None, "len_pre length < 4"

    mess_b_len = struct.unpack("!I", len_pre)[0]

    mess_recive_b = b""
    while len(mess_recive_b) < mess_b_len:
        try:
            chunk = client.recv(min(mess_b_len - len(mess_recive_b), 100 * 1024 * 1024))
            if not chunk:
                # error
                break
            mess_recive_b += chunk
            chunk_len = len(chunk)
            now_mess_len = len(mess_recive_b)

        except:
            # print("Timeout while receiving message content")
            client.settimeout(None)
            return None, "Timeout while receiving message content"

    if len(mess_recive_b) < mess_b_len:
        # error
        client.settimeout(None)
        return None, "mess_recive_b length < mess_b_len"
    try:
        mess = pickle.loads(mess_recive_b)
        client.settimeout(None)
        return mess, None
    except pickle.UnpicklingError as e:
        # print(f"Errpr unpickling data {e}")
        client.settimeout(None)
        return None, f"Error unpickling data {e}"
