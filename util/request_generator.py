import logging
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json as js

from omegaconf import OmegaConf as oc
from util import config
from util.jsonUtil import NpEncoder

alpha = 1.2
log = logging.getLogger("logger")
ServerConf = oc.structured(config.ServerConfig)


def generate_long_tail_weights(num_data, alpha=1.1):

    x = np.random.zipf(alpha, size=50000)
    weights = []
    for k in np.unique(x):
        count = np.sum(x == k)
        weights.append(count)
    weights = np.sort(weights)[::-1]
    weights = weights[0:num_data]
    weights = (weights + 0.0) / np.sum(weights)

    return weights


# This code calculate the data heat by request List
def data_heat_calculate(request_list, output_file='dataHot10.csv'):
    dataHot = dict()
    dataHot['data_id'] = []
    dataHot['heat'] = []
    data_num = len(np.unique(request_list))
    for item in np.unique(request_list):
        times = np.sum(request_list == item)
        dataHot['data_id'].append(item)
        dataHot['heat'].append(times)
    df = pd.DataFrame(dataHot)
    df = df.sort_values(by="heat", ascending=False)
    hot_percentage = sum(df['heat'][0:round(data_num * 0.2)]) / sum(df['heat'])
    df.to_csv(output_file, index=False, header=True)
    return dataHot


# geneate request
def generate_data_request(request_num, server_list):
    x = np.random.zipf(alpha, size=request_num)
    request_servers = np.random.randint(len(server_list), size=request_num)
    requests = dict()
    requests['data_id'] = []
    requests['server'] = []
    for i in range(request_num):
        requests['server'].append(server_list[request_servers[i]])
        requests['data_id'].append(x[i])
    df = pd.DataFrame(requests)
    data_heat_calculate(df['data_id'])
    df.to_csv('request.csv', index=False, header=True)
    return requests


def generate_data_request_unbalanced(data_num, server_num, output_file='request_10.csv'):
    zipf_parameters = np.random.uniform(low=1.2, high=2.0, size=server_num) 
    request_num = np.random.uniform(low=750, high=1250, size=server_num)
    requests = np.zeros((server_num, data_num), dtype=float)

    for server_id, parameter in enumerate(zipf_parameters):
        x = []
        while (len(np.unique(x)) < data_num):
            x.extend(np.random.zipf(parameter, size=server_num * data_num))
        weights = []
        for k in np.unique(x):
            count = np.sum(x == k)
            weights.append(count)
        weights = np.sort(weights)[::-1]
        weights = weights[0:data_num]
        weights = (weights + 0.0) / np.sum(weights)
        random.shuffle(weights)
        requests[server_id] = weights

    request_queue = []
    for server_id, server_requests in enumerate(requests):
        for data_id, num_requests in enumerate(server_requests):
            request_queue.extend([(data_id, server_id)] * np.round(num_requests * request_num[server_id]).astype(int))
    random.shuffle(request_queue)
    df = pd.DataFrame(request_queue, columns=['data_id', 'server'])
    df.to_csv(output_file, index=False)
    return request_queue


# Server Cache data init function
def server_data_init_random(data_num, server_num, output_file='cache_init10.json'):
    hot_data_list = [i for i in range(data_num)]
    np.random.seed(2024117)
    result = dict()
    for data in hot_data_list:
        cache_servers = np.random.choice(range(server_num), int(server_num*0.5), replace=False)
        for server in cache_servers:
            if server not in result:
                result[int(server)] = []
            result[int(server)].append(data)
    capacity = []
    for server in range(server_num):
        capacity.append(len(result[server]))
    result['capacity'] = capacity
    # with open(output_file, 'w') as fp:
    #     js.dump(result, fp, ensure_ascii=False, cls=NpEncoder)
    return result


def server_data_init_with_hot(data_file, server_num, output_file='cache_init10.json'):
    df = pd.read_csv(data_file)
    hot_data_num = round(len(df) * 1)
    df_hot = df.iloc[0:hot_data_num]
    hot_data_list = df_hot['data_id']
    hot_data_weights = df_hot['heat']
    weight_p = hot_data_weights / np.sum(hot_data_weights)
    result = dict()
    capacity = np.random.uniform(low=len(df)*(ServerConf.CACHE_PERCENTAGE-0.05), high=len(df)*ServerConf.CACHE_PERCENTAGE,
                                 size=server_num)
    # capacity = capacity.tolist()
    for server in range(server_num):
        result[server] = set()
        init_data_list = set()
        while len(init_data_list) < capacity[server]:
            init_data_list.add(np.random.choice(hot_data_list, p=weight_p))
            # init_data_list.add(np.random.choice(hot_data_list))
        init_data_list = list(init_data_list)
        result[server] = init_data_list
    result['capacity'] = capacity
    with open(output_file, 'w') as fp:
        js.dump(result, fp, ensure_ascii=False, cls=NpEncoder)
    return result


def generate_cache_data(requests):
    weight_dict = dict(Counter(requests))
    num_to_select = max(1, len(set(weight_dict)) * 30 // 100)

    selected_requests = set()
    while len(selected_requests) < num_to_select:
        choices = list(weight_dict.keys())
        weights = list(weight_dict.values())
        weights_p = weights / np.sum(weights)
        selected = np.random.choice(choices, p=weights_p)
        # selected = np.random.choice(choices)
        selected_requests.add(selected)
        del weight_dict[selected]
    return list(selected_requests)


# generate data analyse function
def draw_request_plot(request_file):
    df = pd.read_csv(request_file)

    request_counts = df['data_id'].value_counts()

    sorted_request_counts = request_counts.sort_values(ascending=False)
    index = [i for i in range(len(sorted_request_counts))]
    sns.lineplot(x=index, y=sorted_request_counts.values, marker='')
    plt.xlabel('range')
    plt.ylabel('request num')
    plt.title('request distribution')
    plt.grid(True)
    plt.show()


def draw_request_plot_by_server(request_file):
    df = pd.read_csv(request_file)

    servers = df['server'].unique()

    for server in servers:
        server_data = df[df['server'] == server]

        file_counts = server_data['data_id'].value_counts().reset_index()
        file_counts.columns = ['data_id', 'request num']

        top_files = file_counts.sort_values(by='request num', ascending=False).head(10)
        top_files.reset_index(inplace=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_files.index, y='request num', data=top_files)
        plt.xticks(top_files.index, top_files['data_id'])
        plt.title(f'Server {server} request distribution')
        plt.xlabel('data id')
        plt.ylabel('request num')
        plt.show()


if __name__ == "__main__":
    # for i in range(10):
    #     requests_file = f'D:\MemcacheTrace\preprocess\Chunk_{i+1}_mapped.csv'
    #     draw_request_plot(requests_file)
    #     df = pd.read_csv(requests_file)
    #     data_heat_calculate(df['data_id'])

    requests_file = f'D:\MemcacheTrace\_10server_trace\\timestamp_1.csv'
    # requests_file = f'.././request_10.csv'
    draw_request_plot_by_server(requests_file)

    # df = pd.read_csv(requests_file)
    # data_heat_calculate(df['data_id'])

    # generate_data_request_unbalanced(1000,server_list)
    # server_data_init_with_hot('../dataHot10.csv',10)
    # with open('cache_init10.json','r') as fp:
    #     dict = js.load(fp)
    #     print(dict)
