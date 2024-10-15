import asyncio
import json
import os
import random
import threading
import time
import logging
from datetime import datetime

import pandas as pd
from omegaconf import OmegaConf as oc

from Network import network_utils as nutil
from Server.server_manager import Manager
from util import config
from util.config import GraphConfig
from util.jsonUtil import json_load_hook
from util.logger_config import setup_logging
from util.request_generator import generate_data_request_unbalanced, server_data_init_with_hot, generate_cache_data, \
    data_heat_calculate, server_data_init_random

TestConf = oc.structured(config.TestConfig)
ServerConf = oc.structured(config.ServerConfig)

# trace_folder = f"D:/MemcacheTrace/trace"
trace_folder = "./"

basic_disdedup_comm_type = [
    'data_heat',
    'data_heat_response',
    'data_predelete',
    'delete_permission',
    'delete_permission_response',
    'data_check',
    'data_check_response'
]

depend_disdedup_comm_type = [
    'data_heat',
    'data_heat_response',
    'data_predelete',
    'delete_permission',
    'delete_permission_response',
    'data_delete_cancel'
]

index_maintain_comm_type = [
    'register',
    'register_response',

]

LDI_disdedup_comm_type = [
    'data_heat',
    'data_heat_response',
    'data_predelete',
    'delete_permission',
    'delete_permission_response',
    'data_check',
    'data_check_response'
]

CDI_disdedup_comm_type = [
    'data_heat',
    'data_heat_response',
    'data_predelete',
    'delete_permission',
    'delete_permission_response',
    'data_delete_cancel'
]

communication_type = []
if TestConf.DEDUPLICATE_STRATEGY == 1:
    communication_type = basic_disdedup_comm_type
elif TestConf.DEDUPLICATE_STRATEGY == 2:
    communication_type = depend_disdedup_comm_type
elif TestConf.DEDUPLICATE_STRATEGY == 7:
    communication_type = LDI_disdedup_comm_type
elif TestConf.DEDUPLICATE_STRATEGY == 8:
    communication_type = CDI_disdedup_comm_type


Strategy = {
    0: 'global_optima',
    1: 'dis_dedup_basic',
    2: 'dis_dedup_depend',
    3: 'BEDD',
    4: 'random',
    5: 'real_random',
    6: 'MEAN',
    7: 'LDI',
    8: 'CDI'
}


def motivation_one():
    global data_count
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    experiment_folder = f"Test/experiment_{current_time}"
    log_file_name = "experiment_log.txt"
    log_file_path = os.path.join(experiment_folder, log_file_name)
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    log = setup_logging(log_file_path)

    try:
        log.info("start")
        SM1 = Manager()
        SM2 = Manager()

        # generate_data_request_unbalanced(2000, SM1.ServerNum,f'request_unbalanced{TestConf.NODENUM}.csv')
        request_list = pd.read_csv(f'{trace_folder}/request_{TestConf.NODENUM}.csv')
        # request_list = request_list.iloc[0:TestConf.NODENUM*1000]
        data_heat_calculate(request_list['data_id'], f'{trace_folder}/dataHot{TestConf.NODENUM}.csv')
        # server_data_init_with_hot(f'{trace_folder}/dataHot{TestConf.NODENUM}.csv', TestConf.NODENUM, f'{trace_folder}/cache_init{TestConf.NODENUM}.json')
        data_file = f"{trace_folder}/cache_init{TestConf.NODENUM}.json"

        with open(data_file, 'r') as file:
            cache = json.load(file, object_hook=json_load_hook)
        SM1.install(cache['capacity'])
        SM1.init_edge_server_cache(cache)
        SM2.install(cache['capacity'])
        SM2.init_edge_server_cache(cache)
        data_count = 1

        latency, cloud_time = SM2.TestLatency(request_list)
        data_num = SM2.total_data_num()
        strategy, decision_time = SM2.hot_aware_deduplicate_with_cover(0)
        dedup_count = SM2.strategy_execute(strategy)
        latency_dedup, cloud_time_dedup = SM2.TestLatency(request_list)
        log.info(f"{latency}")
        log.info(f'{cloud_time}')
        log.info(f"{latency_dedup}")
        log.info(f'{cloud_time_dedup}')
        log.info(f'{decision_time}')
        log.info(f"{latency_dedup - latency}")
        log.info(f"{data_num}")
        log.info(f"{dedup_count}")
        log.info(f"{dedup_count / data_num}")

        latency, cloud_time = SM1.TestLatency(request_list)
        data_num = SM1.total_data_num()
        strategy = SM1.random_deduplicate_with_cover(0)
        dedup_count = SM1.strategy_execute(strategy)
        latency_dedup, cloud_time_dedup = SM1.TestLatency(request_list)
        log.info(f"{latency}")
        log.info(f'{cloud_time}')
        log.info(f"{latency_dedup}")
        log.info(f'{cloud_time_dedup}')
        log.info(f'{decision_time}')
        log.info(f"{latency_dedup - latency}")
        log.info(f"{data_num}")
        log.info(f"{dedup_count}")
        log.info(f"{dedup_count / data_num}")

        log.info("end")

    except Exception as e:
        log.error(f"{e}", exc_info=True)
    finally:
        logging.shutdown()


def BEDD():
    global data_count
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    experiment_folder = f"Test/experiment_{current_time}"
    log_file_name = "experiment_log.txt"
    log_file_path = os.path.join(experiment_folder, log_file_name)
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    log = setup_logging(log_file_path)

    try:
        log.info("start")
        SM = Manager()
        # generate_data_request_unbalanced(2000, SM1.ServerNum,f'request_unbalanced{SM1.ServerNum}.csv')
        request_list = pd.read_csv(f'{trace_folder}/request_{TestConf.NODENUM}.csv')
        # request_list = request_list.iloc[0:TestConf.NODENUM*1000]
        data_heat_calculate(request_list['data_id'], f'{trace_folder}/dataHot{TestConf.NODENUM}.csv')
        # server_data_init_with_hot(f'{trace_folder}/dataHot{TestConf.NODENUM}.csv', TestConf.NODENUM, f'{trace_folder}/cache_init{TestConf.NODENUM}.json')
        data_file = f"{trace_folder}/cache_init{TestConf.NODENUM}.json"

        with open(data_file, 'r') as file:
            cache = json.load(file, object_hook=json_load_hook)
        SM.install(cache['capacity'])
        init_data_num, data_kind = SM.init_edge_server_cache(cache)
        data_count = 1

        log.info(f"{'BEDD：'}")
        latency, cloud_time = SM.TestLatency(request_list)
        data_num = SM.total_data_num()
        strategy, decision_time = SM.balence_deduplicate_with_cover(0.1)
        dedup_count = SM.strategy_execute(strategy)
        latency_dedup, cloud_time_dedup = SM.TestLatency(request_list)
        log.info(f"{latency}")
        log.info(f'{cloud_time}')
        log.info(f"{latency_dedup}")
        log.info(f'{cloud_time_dedup}')
        log.info(f"{latency_dedup - latency}")
        log.info(f'{decision_time}')
        log.info(f"{init_data_num}")
        log.info(f"{dedup_count}")
        log.info(f"{(init_data_num - data_kind) / init_data_num}")
        log.info(f"{dedup_count / init_data_num}")

        log.info("end")

    except Exception as e:
        log.error(f" {e}", exc_info=True)
    finally:
        logging.shutdown()


def long_timestamp_test():
    global data_count
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    experiment_folder = f"Test/Strategy{TestConf.DEDUPLICATE_STRATEGY}_{current_time}"
    log_file_name = f"Server{TestConf.NODENUM}_{Strategy[TestConf.DEDUPLICATE_STRATEGY]}_hoplimit{ServerConf.HOP_NUM}_degree{GraphConfig.DEGREE}.txt"
    log_file_path = os.path.join(experiment_folder, log_file_name)
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    log = setup_logging(log_file_path)

    try:
        log.info("start")

        timestamp_num = 1
        serial_latency = []
        dedup_ratio = []
        comunication_times = []
        max_dedup_ratio = []
        trace_folder = f"D:\MemcacheTrace\\trace1\_{TestConf.NODENUM}server_trace"

        for timestamp in range(timestamp_num):
            log.info(f"{'---------------timestamp '}{timestamp}---------------")

            log.info(f"{'---------------data cache---------------'}")
            request_file = f'{trace_folder}/timestamp_{timestamp}.csv'
            SM = Manager()
            request_list = pd.read_csv(request_file)
            # data_heat_calculate(request_list['data_id'], f'{trace_folder}/dataHot{TestConf.NODENUM}_{timestamp}.csv')
            # server_data_init_with_hot(f'{trace_folder}/dataHot{TestConf.NODENUM}_{timestamp}.csv', TestConf.NODENUM,
            #                           f'{trace_folder}/cache_init{TestConf.NODENUM}_{timestamp}_{ServerConf.CACHE_PERCENTAGE}.json')
            # data_file = f"{trace_folder}/cache_init{TestConf.NODENUM}_{timestamp}_{ServerConf.CACHE_PERCENTAGE}.json"
            data_file = f"{trace_folder}/cache_init{TestConf.NODENUM}_{timestamp}.json"
            with open(data_file, 'r') as file:
                cache = json.load(file, object_hook=json_load_hook)
            SM.install(cache['capacity'])
            init_data_num, data_kind = SM.init_edge_server_cache(cache)
            latency, cloud_time, edge_average_latency = SM.TestLatency(request_list)
            log.info(f"{'---------------deduplication---------------'}")
            if TestConf.DEDUPLICATE_STRATEGY == 1 or TestConf.DEDUPLICATE_STRATEGY == 2 or TestConf.DEDUPLICATE_STRATEGY == 7 or TestConf.DEDUPLICATE_STRATEGY == 8:
                dedup_count = 0
                decision_time = 0
                # for server in SM.server_sort_by_neighbor_num():
                server_dedup_range = list(range(SM.ServerNum))
                # random.shuffle(server_dedup_range)
                for server in server_dedup_range:
                    dis_dedup_count, dis_decision_time = SM.disDeduplicate(server)
                    dedup_count += dis_dedup_count
                    decision_time = max(decision_time, dis_decision_time)
            else:
                strategy, decision_time = SM.data_dedup()
                dedup_count = SM.strategy_execute(strategy)
            latency_dedup, cloud_time_dedup, edge_average_latency_dedup = SM.TestLatency(request_list)

            total_dedup_commu_time = 0
            index_maintain_comm_time = 0
            for key, item in nutil.counter.get_count().items():
                log.info(f'{key} times :{item}')
                if key in communication_type:
                    total_dedup_commu_time += item
                if key in index_maintain_comm_type:
                    index_maintain_comm_time += item
            log.info(f"{latency}")
            log.info(f'{cloud_time}')
            log.info(f"{latency_dedup}")
            log.info(f'{cloud_time_dedup}')
            log.info(f"{edge_average_latency_dedup}")
            log.info(f"{latency_dedup - latency}")
            log.info(f'{decision_time}')
            log.info(f"{init_data_num}")
            log.info(f"{dedup_count}")
            log.info(f"{(init_data_num - data_kind) / init_data_num}")
            log.info(f"{dedup_count / init_data_num}")
            log.info(f'total dedup communication times: {total_dedup_commu_time}')
            log.info(f'communication times per dedup data:{total_dedup_commu_time / dedup_count}')
            log.info(f'index maintain communication times per dedup data:{index_maintain_comm_time / dedup_count}')
            log.info(f"--------------timestamp {timestamp} end----------")
            serial_latency.append(latency_dedup)
            dedup_ratio.append(dedup_count / init_data_num)
            comunication_times.append(total_dedup_commu_time / dedup_count)
            max_dedup_ratio.append((init_data_num - data_kind) / init_data_num)
            nutil.counter.clear_count()
        log.info(f'avergae latency is {sum(serial_latency) / len(serial_latency)}')
        log.info(f'average dedup ratio is {sum(dedup_ratio) / len(dedup_ratio)}')
        log.info(f'average communication times is {sum(comunication_times) / len(comunication_times)}')
        log.info(f'average max dedup ratio is {sum(max_dedup_ratio) / len(max_dedup_ratio)}')
        log.info("end")

    except Exception as e:
        log.error(f"{e}", exc_info=True)
    finally:
        logging.shutdown()



def motivation_two():
    global data_count
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    experiment_folder = f"Test/experiment_{current_time}"
    log_file_name = f"Server{TestConf.NODENUM}_{Strategy[TestConf.DEDUPLICATE_STRATEGY]}_hoplimit{ServerConf.HOP_NUM}_degree{GraphConfig.DEGREE}.txt"
    log_file_path = os.path.join(experiment_folder, log_file_name)
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    log = setup_logging(log_file_path)

    try:
        SM = Manager()
        # generate_data_request_unbalanced(2000, SM.ServerNum,f'request_unbalanced{SM.ServerNum}.csv')
        request_list = pd.read_csv(f'{trace_folder}/request_{TestConf.NODENUM}.csv')
        # request_list = request_list.iloc[0:TestConf.NODENUM * 1000]
        # data_heat_calculate(request_list['data_id'], f'{trace_folder}/dataHot{TestConf.NODENUM}.csv')
        # server_data_init_with_hot(f'{trace_folder}/dataHot{TestConf.NODENUM}.csv', TestConf.NODENUM,
        #                           f'{trace_folder}/cache_init{TestConf.NODENUM}.json')
        data_file = f"{trace_folder}/cache_init{TestConf.NODENUM}.json"
        with open(data_file, 'r') as file:
            cache = json.load(file, object_hook=json_load_hook)
        SM.install(cache['capacity'])
        init_data_num, data_kind = SM.init_edge_server_cache(cache)
        before_latency, cloud_time, edge_average_latency = SM.TestLatency(request_list)

        total_dedup_count = 0
        if TestConf.DEDUPLICATE_STRATEGY == 1 or TestConf.DEDUPLICATE_STRATEGY == 2 or TestConf.DEDUPLICATE_STRATEGY == 7 or TestConf.DEDUPLICATE_STRATEGY == 8:
            decision_time = 0
            for server in SM.server_sort_by_neighbor_num():
                dis_dedup_count, dis_decision_time = SM.disDeduplicate(server)
                total_dedup_count += dis_dedup_count
                decision_time = max(decision_time, dis_decision_time)
        else:
            strategy, decision_time = SM.data_dedup()
            total_dedup_count = SM.strategy_execute(strategy)

        after_latency, after_cloud_time, edge_average_latency = SM.TestLatency(request_list)
        count = nutil.counter.get_count()
        total_dedup_commu_time = 0
        for key, item in count.items():
            if key == 'total_delete':
                continue
            log.info(f'{key} communication times :{item}')
            if key in communication_type:
                total_dedup_commu_time += item
        log.info(f"{'latency before dedup is '}{before_latency}")
        log.info(f'cloud time before dedup is {cloud_time}')
        log.info(f"{'latency after dedup is '}{after_latency}")
        log.info(f'cloud time after dedup is {after_cloud_time}')
        log.info(f'total data dedup num is {total_dedup_count}')
        log.info(f'data dedup rate is {total_dedup_count / init_data_num}')
        log.info(f"{'max dedup rate is '}{(init_data_num - data_kind) / init_data_num}")
        log.info(f'total dedup communication times: {total_dedup_commu_time}')
        log.info(f'communication times per dedup data:{total_dedup_commu_time / total_dedup_count}')

    except Exception as e:
        log.error(f"{e}", exc_info=True)
    finally:
        logging.shutdown()


def main_experiment():
    global data_count
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    experiment_folder = f"Test/experiment_{current_time}"
    log_file_name = "experiment_log.txt"
    log_file_path = os.path.join(experiment_folder, log_file_name)
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    log = setup_logging(log_file_path)

    try:
        SM = Manager()
        # generate_data_request_unbalanced(2000, SM.ServerNum,f'request_unbalanced{SM.ServerNum}.csv')
        request_list = pd.read_csv(f'./request_unbalanced.csv')
        # data_heat_calculate(request_list['data_id'],f'dataHot{SM.ServerNum}.csv')
        # server_data_init_with_hot(f'./dataHot{SM.ServerNum}.csv', SM.ServerNum,f'cache_init{SM.ServerNum}.json')

        data_file = f"./cache_init{SM.ServerNum}.json"

        with open(data_file, 'r') as file:
            cache = json.load(file, object_hook=json_load_hook)
        SM.install(cache['capacity'])
        # init_data_num = SM.install(data_file)
        init_data_num = SM.init_edge_server_cache(cache)
        SM.run_edge_server()

        before_latency, before_cloud_time, edge_average_latency = SM.TestLatency(request_list)
        start_dedup_num = 0
        nutil.counter.count['dedup end'] = 0
        for i in range(SM.ServerNum):
            SM.start_server_deduplicate(i)
            start_dedup_num += 1
            time.sleep(5)
        # SM.start_server_deduplicate(5)
        while start_dedup_num != nutil.counter.count['dedup end']:
            time.sleep(1)
        after_latency, after_cloud_time,edge_average_latency = SM.TestLatency(request_list)
        SM.close_servers()
        count = nutil.counter.get_count()
        total_dedup_commu_time = 0

        for key, item in count.items():
            if key == 'total_delete':
                continue
            if key == 'dedup time':
                log.info(f'{key} :{item}')
                continue
            if key == 'cancel delete':
                log.info(f'{key} :{item}')
                continue
            log.info(f'{key} communication times :{item}')
            if TestConf.DEDUPLICATE_MOD == 1 and key in depend_disdedup_comm_type:
                total_dedup_commu_time += item
            if TestConf.DEDUPLICATE_MOD == 0 and key in basic_disdedup_comm_type:
                total_dedup_commu_time += item
        log.info(f'latency before dedup is {before_latency}')
        log.info(f'cloud time before dedup is {before_cloud_time}')
        log.info(f'latency after dedup is {after_latency}')
        log.info(f'cloud time after dedup is {after_cloud_time}')
        log.info(f"{'latency increase：'}{after_latency - before_latency}")
        log.info(f'total data dedup num is {count["total_delete"]}')
        log.info(f'data dedup rate is {count["total_delete"] / init_data_num}')
        log.info(f'total dedup communication times: {total_dedup_commu_time}')
        log.info(f'communication times per dedup data:{total_dedup_commu_time / count["total_delete"]}')


    except Exception as e:
        log.error(f"{e}", exc_info=True)
    finally:
        logging.shutdown()


if __name__ == "__main__":
    # main_experiment()
    # test_latency()
    # motivation_one()
    # motivation_two()
    # BEDD()
    # random_dedup()
    # global_optima()
    long_timestamp_test()
    # global_time_test()
