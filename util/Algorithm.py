import hashlib
import logging
import math

import gurobipy
import gurobipy as gp
import numpy
from gurobipy import GRB

from util.config import TestConfig
from util.logger_config import setup_logging

log = logging.getLogger('logger.algorithm')
show_gurobi_info = 0


def solve_set_cover_with_weights(universe, subsets, weights):
    # 创建模型
    model = gp.Model("set_cover_with_weights")
    model.setParam('OutputFlag', show_gurobi_info)

    # 创建变量，每个集合是否被选择
    x = model.addVars(len(subsets), vtype=GRB.BINARY, name="x")

    # 最小化目标函数：选择的集合权重总和
    model.setObjective(gp.quicksum(weights[i] * x[i] for i in range(len(subsets))), GRB.MINIMIZE)

    # 约束条件：确保每个元素都被覆盖
    for element in universe:
        model.addConstr(gp.quicksum(x[i] for i, subset in enumerate(subsets) if element in subset) >= 1,
                        name='Cover' + str(element))

    # 优化模型
    model.optimize()

    # 打印结果
    if model.status == GRB.OPTIMAL:
        selected_sets = [i for i, var in enumerate(model.getVars()) if var.x > 0.5]
        # log.info("Selected sets:", selected_sets)
        return selected_sets
    else:
        # log.info("No solution found.")
        return None


def solve_set_cover_and_max_weights(universe, subsets, weights, dedup_weight=TestConfig.ALPHA):
    # 创建模型
    model = gp.Model("set_cover_with_weights")
    model.setParam('OutputFlag', show_gurobi_info)

    # 创建变量，每个集合是否被选择
    x = model.addVars(len(subsets), vtype=GRB.BINARY, name="x")

    total_weight = gp.quicksum(weights[i] * x[i] for i in range(len(subsets))) / (sum(weights) + 1)
    dedup_rate = gp.quicksum(1 - x[i] for i in range(len(subsets))) / len(subsets)
    # 最小化目标函数：选择的集合权重总和
    model.setObjective((1 - dedup_weight) * total_weight + dedup_weight * dedup_rate, GRB.MAXIMIZE)

    # 约束条件：确保每个元素都被覆盖
    for element in universe:
        model.addConstr(gp.quicksum(x[i] for i, subset in enumerate(subsets) if element in subset) >= 1,
                        name='Cover' + str(element))

    # 优化模型
    model.optimize()

    # 打印结果
    if model.status == GRB.OPTIMAL:
        selected_sets = [i for i, var in enumerate(model.getVars()) if var.x > 0.5]
        # log.info("Selected sets:", selected_sets)
        return selected_sets
    else:
        # log.info("No solution found.")
        return None


# def solve_dedup_space_balance(server_with_data, server_need_data, cover, server_capacity, dedup_weight=0.7):
#     # 创建模型
#     model = gp.Model("set_cover_with_weights")
#     # model.setParam('OutputFlag', show_gurobi_info)
#
#     x = numpy.zeros((len(server_with_data), len(server_capacity)))
#     y = numpy.zeros((len(server_with_data), len(server_capacity)))
#     coverage = numpy.zeros((len(server_capacity), len(server_capacity)))
#     dedup_data = list(server_with_data.keys())
#     data_id_hash = [i for i in range(len(dedup_data))]
#     server_list = [i for i in range(len(server_capacity))]
#
#     total_data_num = 0
#     for i in range(len(dedup_data)):
#         data_id = dedup_data[i]
#         for server_id in server_with_data[data_id]:
#             x[i, server_id] = 1
#             total_data_num += 1
#
#     for i in range(len(dedup_data)):
#         data_id = dedup_data[i]
#         for server_id in server_need_data[data_id]:
#             y[i, server_id] = 1
#
#     for server, server_cover in enumerate(cover):
#         for neighbor in server_cover:
#             coverage[server, neighbor] = 1
#
#     # 创建变量，每个数据在哪些服务器上存储
#     delete_set = model.addVars(len(dedup_data), len(server_capacity), vtype=GRB.BINARY, name="delete_set")
#     # Oj = model.addVars(len(server_list), vtype=GRB.CONTINUOUS, name="Oj")
#     # # 约束条件：确保每个元素都被覆盖
#     # Square_sum_of_Oj = model.addVar(vtype=GRB.CONTINUOUS, name="Square_Oj")
#     #
#     # balance_object = model.addVar(vtype=GRB.CONTINUOUS, name="balance_object")
#     #
#     # Sum_of_Squares = model.addVar(vtype=GRB.CONTINUOUS, name="SumofSquares")
#     #
#     # sum_of_Oj = model.addVar(vtype=GRB.CONTINUOUS, name="SumofOj")
#     #
#     #
#     # model.update()
#
#     model.addConstrs((delete_set[data_id, j] <= x[data_id, j] for data_id in data_id_hash for j in server_list),name="demand")
#
#     model.addConstrs(gurobipy.quicksum(((x[data_id, j] - delete_set[data_id, j])*coverage[j, server]) for j in server_list ) >= y[data_id, server] for data_id in data_id_hash for server in server_list)
#
#
#
#
#
#     dedup_rate = (gp.quicksum(delete_set[data_id, server] for data_id in data_id_hash for server in server_list) / total_data_num)
#
#
#
#     # model.addConstrs(Oj[server] == gp.quicksum(x[data_id, server] - delete_set[data_id, server] for data_id in data_id_hash)/server_capacity[server] for server in server_list )
#     #
#     # model.addConstr(sum_of_Oj  == gurobipy.quicksum(Oj[server] for server in server_list))
#     #
#     # model.addConstr( Square_sum_of_Oj == sum_of_Oj*sum_of_Oj, name="Square_Oj")
#     #
#     #
#     #
#     # model.addConstr(Sum_of_Squares == gp.quicksum(Oj[server]*Oj[server] for server in server_list), name="SumofSquares")
#     #
#     #
#     # model.addConstr(balance_object*Sum_of_Squares*len(server_list) == Square_sum_of_Oj, name="balance_object")
#
#     model.setObjective( dedup_rate, GRB.MAXIMIZE)
#     model.write("model.lp")
#     # 优化模型
#     model.optimize()
#
#     # 打印结果
#     if model.status == GRB.OPTIMAL:
#         strategy = {}
#         for data_id in data_id_hash:
#             strategy[dedup_data[data_id]] = [server for server in server_list if delete_set[data_id, server].x > 0.5]
#
#         log.info("Strategy fond:", strategy)
#         return strategy
#     else:
#         log.info("No solution found.")
#         return None

def solve_dedup_space_balance(universe, subsets, server_list, remain_data_num, capacity, dedup_weight=0.7):
    # 创建模型
    model = gp.Model("set_cover_with_weights")
    model.setParam('OutputFlag', show_gurobi_info)

    # 创建变量，每个集合是否被选择
    x = model.addVars(len(server_list), vtype=GRB.BINARY, name="x")

    dedup_rate = gp.quicksum(1 - x[i] for i in range(len(server_list))) / len(server_list)


    # 约束条件：确保每个元素都被覆盖
    for element in universe:
        model.addConstr(gp.quicksum(x[i] for i, subset in enumerate(subsets) if element in subset) >= 1,
                        name='Cover' + str(element))


    Oj = model.addVars(len(server_list), vtype=GRB.CONTINUOUS, name="Oj")

    Square_sum_of_Oj = model.addVar(vtype=GRB.CONTINUOUS, name="Square_Oj")

    balance_object = model.addVar(vtype=GRB.CONTINUOUS, name="balance_object")

    Sum_of_Squares = model.addVar(vtype=GRB.CONTINUOUS, name="SumofSquares")

    sum_of_Oj = model.addVar(vtype=GRB.CONTINUOUS, name="SumofOj")


    model.update()

    model.addConstrs(Oj[i] == (remain_data_num[server_list[i]] - 1 + x[i])/capacity[server_list[i]] for i in range(len(subsets)))

    model.addConstr(sum_of_Oj == gurobipy.quicksum(Oj[server] for server in range(len(server_list))))

    model.addConstr(Square_sum_of_Oj == sum_of_Oj*sum_of_Oj, name="Square_Oj")

    model.addConstr(Sum_of_Squares == gp.quicksum(Oj[server]*Oj[server] for server in range(len(server_list))), name="SumofSquares")

    model.addConstr(balance_object*Sum_of_Squares*len(server_list) == Square_sum_of_Oj, name="balance_object")


    # 最小化目标函数：选择的集合权重总和
    model.setObjective((1 - dedup_weight) * balance_object + dedup_weight * dedup_rate, GRB.MAXIMIZE)
    # 优化模型
    model.optimize()

    # 打印结果
    if model.status == GRB.OPTIMAL:
        selected_sets = []
        result = model.getAttr('x', x)
        for index , value in result.items():
            if value > 0.5:
                selected_sets.append(index)
        # log.info("Selected sets:", selected_sets)
        return selected_sets
    else:
        # log.info("No solution found.")
        return None


def _to_str(element):
    # _e_class = element.__class__.__name__
    _str = str(element)
    bytes_like = bytes(_str, encoding='utf-8') if \
        isinstance(_str, str) else _str
    b_md5 = hashlib.md5(bytes_like).hexdigest()
    return b_md5


if __name__ == "__main__":
    # 示例数据（包括集合权重）
    universe = [1, 2, 3, 4, 5, 6, 7, 8]
    subsets = [[1, 3], [2, 3], [1, 2, 3, 4], [3, 4, 5, 6, 7], [4, 5, 7], [4, 6], [4, 5, 7, 8], [7, 8]]
    # subsets = []
    weights = [10, 5, 5, 4, 0, 2, 1, 4]
    # max_value = 6
    # weights_p = [max_value - x for x in weights]
    # # weights_p = [1 / (x + 1) * 100 for x in weights]
    all_set = [0, 1, 2, 3]
    # 求解带有权重的集合覆盖问题
    delete_set = solve_set_cover_and_max_weights(universe, subsets, weights)
    print(delete_set)
    # delete_set = list(set(all_set) - set(delete_set))
    # print(delete_set)
