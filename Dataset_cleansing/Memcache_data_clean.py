# -*- coding: utf-8 -*-

import csv
import os
import pandas as pd
import numpy as np


def split_data_set(data_file, output_folder, batch_size=15):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    # 列名
    column_names = ['Timestamp', 'data_id', '', 'size', 'client', 'operation', 'TTL']
    # 初始化变量
    # 用于存储当前正在处理的timestamp
    current_timestamps = set()
    # 记录当前CSV文件的行数
    current_count = 0
    # 记录当前是第几个CSV文件
    file_index = 1

    with open(data_file, 'r', encoding='utf-8') as large_file:
        # 创建一个CSV文件用于写入
        output_file = f"{output_folder}/output_batch_{file_index}.csv"
        csv_file = open(output_file, 'w', encoding='utf-8', newline='')
        csv_writer = csv.writer(csv_file)
        # 写入列名
        csv_writer.writerow(column_names)

        # 逐行读取大型文本文件
        for line in large_file:
            # 分割行以获取数据
            data = line.strip().split(',')
            # 检查timestamp是否是新的
            if data[0] not in current_timestamps:
                # 如果已经有10个不同的timestamp，则开始新的CSV文件
                if len(current_timestamps) == batch_size:
                    # 重置timestamp集合和行数计数器
                    current_timestamps.clear()
                    current_count = 0
                    # 关闭当前CSV文件并打开一个新的CSV文件
                    csv_file.close()
                    file_index += 1
                    output_file = f"{output_folder}/output_batch_{file_index}.csv"
                    csv_file = open(output_file, 'w', encoding='utf-8', newline='')
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(column_names)

                # 添加新的timestamp到集合中
                current_timestamps.add(data[0])

            # 写入数据到CSV文件
            csv_writer.writerow(data)
            current_count += 1

        # 关闭最后一个CSV文件
        csv_file.close()

    print('所有数据已处理完毕。')


# 为client随机分配一个0到50之间的数字
def randomize_client(server_num):
    return np.random.randint(0, server_num)


def generate_normalized_weights(n, mu=0, sigma=1):
    """
    Generates n weights that sum up to 1 and follow a given normal distribution.

    Parameters:
    - n: Number of weights to generate.
    - mu: Mean of the normal distribution.
    - sigma: Standard deviation of the normal distribution.

    Returns:
    A list of n normalized weights that sum up to 1.
    """
    # Generate n random numbers from a normal distribution with mean mu and standard deviation sigma
    weights = np.random.normal(mu, sigma, n)

    # Normalize the weights so they sum up to 1
    normalized_weights = weights / np.sum(weights)

    return normalized_weights


def generate_normalized_weights_zipf(n, a=2.1):
    """
    Generates n weights that sum up to 1 and follow a given Zipf's distribution.

    Parameters:
    - n: Number of weights to generate.
    - a: The distribution parameter. Higher values make the distribution steeper.

    Returns:
    A list of n normalized weights that sum up to 1.
    """
    # Generate n random numbers from a Zipf's distribution with parameter a
    weights = np.random.zipf(a, n)

    # Normalize the weights so they sum up to 1
    normalized_weights = weights / np.sum(weights)

    return normalized_weights

# 处理单个CSV文件
def data_set_mapping(file_folder,server_num,output_folder):
    # 映射data_id到连续的数字
    data_id_mapping = {}
    client_mapping = {}
    data_id_server = {}
    current_index = 0
    file_count = 0
    # 遍历文件夹中的所有CSV文件并处理它们
    os.makedirs(output_folder, exist_ok=True)
    server_list = list(range(server_num))
    for filename in os.listdir(file_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(file_folder, filename)
            # 读取CSV文件
            df = pd.read_csv(file_path)
            # 映射data_id
            for idx, value in enumerate(df['data_id']):
                if value not in data_id_mapping:
                    data_id_mapping[value] = current_index
                    data_id_server[value] = np.random.choice(server_list, int(server_num*0.2))
                    # data_id_server[value] = generate_normalized_weights_zipf(server_num)
                    current_index += 1
                df.at[idx, 'data_id'] = data_id_mapping[value]
                # df.at[idx, 'server'] = np.random.choice(range(server_num), p=data_id_server[value])
                df.at[idx, 'server'] = np.random.choice(data_id_server[value])

            # # 随机映射client到0-50并重命名为server
            # for idx, value in enumerate(df['client']):
            #     if value not in client_mapping:
            #         data_id_mapping[value] = np.random.randint(0, server_num)
            #     df.at[idx, 'client'] = data_id_mapping[value]
            # df.rename(columns={'client': 'server'}, inplace=True)


            # 保存修改后的文件
            output_filename = f"timestamp_{file_count}.csv"
            output_filename_path = os.path.join(output_folder, output_filename)
            df.to_csv(output_filename_path, index=False)
            file_count += 1
            print(f'已处理{file_count}个文件。')


# # 设置输入文件路径和输出文件夹路径
input_file_path = 'D:\MemcacheTrace\source'
output_folder_path = 'D:\MemcacheTrace\\trace2\_10server_trace'
# split_data_set(input_file_path, output_folder_path,10)

data_set_mapping(input_file_path, 10, output_folder_path)

# print(sorted(generate_normalized_weights_zipf(50, 2.1),reverse=True))