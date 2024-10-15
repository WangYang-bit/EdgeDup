# -*- coding: utf-8 -*-

import csv
import os
import pandas as pd
import numpy as np


def split_data_set(data_file, output_folder, batch_size=15):
    os.makedirs(output_folder, exist_ok=True)
    column_names = ['Timestamp', 'data_id', '', 'size', 'client', 'operation', 'TTL']

    current_timestamps = set()
    current_count = 0
    file_index = 1

    with open(data_file, 'r', encoding='utf-8') as large_file:
        output_file = f"{output_folder}/output_batch_{file_index}.csv"
        csv_file = open(output_file, 'w', encoding='utf-8', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(column_names)

        for line in large_file:
            data = line.strip().split(',')
            if data[0] not in current_timestamps:
                if len(current_timestamps) == batch_size:
                    current_timestamps.clear()
                    current_count = 0
                    csv_file.close()
                    file_index += 1
                    output_file = f"{output_folder}/output_batch_{file_index}.csv"
                    csv_file = open(output_file, 'w', encoding='utf-8', newline='')
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(column_names)

                current_timestamps.add(data[0])

            csv_writer.writerow(data)
            current_count += 1

        csv_file.close()


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

def data_set_mapping(file_folder,server_num,output_folder):
    data_id_mapping = {}
    client_mapping = {}
    data_id_server = {}
    current_index = 0
    file_count = 0
    os.makedirs(output_folder, exist_ok=True)
    server_list = list(range(server_num))
    for filename in os.listdir(file_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(file_folder, filename)
            df = pd.read_csv(file_path)
            for idx, value in enumerate(df['data_id']):
                if value not in data_id_mapping:
                    data_id_mapping[value] = current_index
                    data_id_server[value] = np.random.choice(server_list, int(server_num*0.2))
                    # data_id_server[value] = generate_normalized_weights_zipf(server_num)
                    current_index += 1
                df.at[idx, 'data_id'] = data_id_mapping[value]
                # df.at[idx, 'server'] = np.random.choice(range(server_num), p=data_id_server[value])
                df.at[idx, 'server'] = np.random.choice(data_id_server[value])

            # for idx, value in enumerate(df['client']):
            #     if value not in client_mapping:
            #         data_id_mapping[value] = np.random.randint(0, server_num)
            #     df.at[idx, 'client'] = data_id_mapping[value]
            # df.rename(columns={'client': 'server'}, inplace=True)

            output_filename = f"timestamp_{file_count}.csv"
            output_filename_path = os.path.join(output_folder, output_filename)
            df.to_csv(output_filename_path, index=False)
            file_count += 1


input_file_path = '.\MemcacheTrace\source'
output_folder_path = '.\MemcacheTrace\\trace2\_10server_trace'
# split_data_set(input_file_path, output_folder_path,10)

data_set_mapping(input_file_path, 10, output_folder_path)

# print(sorted(generate_normalized_weights_zipf(50, 2.1),reverse=True))
