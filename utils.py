import datetime
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import pandas as pd


def mini_batch(batch_size: int, *tensors):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def progress_bar(now_index: int, total_num: int, start: float):
    total_num -= 1
    len = 20
    a = "*" * int((now_index + 1) / total_num * len)
    b = "." * int((max((total_num - now_index - 1), 0)) / total_num * len)
    c = (min(now_index + 1, total_num) / total_num) * 100
    dur = time.perf_counter() - start
    print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c, a, b, dur), end="")


def random_color():
    color_arr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += color_arr[random.randint(0, 14)]
    return "#" + color


def draw_loss_pic(max_step: int, use_random_color: bool = False, **losses):

    plt.figure()
    for key in losses.keys():
        if use_random_color:
            plt.plot(range(1, max_step + 1), losses[key], color=random_color(), label=key)
        else:
            plt.plot(range(1, max_step + 1), losses[key], label=key)

    plt.legend()
    plt.show()


def draw_loss_pic_one_by_one(max_step: int, use_random_color: bool = False, **losses):

    for key in losses.keys():
        plt.figure()
        if use_random_color:
            plt.plot(range(1, max_step + 1), losses[key], color=random_color(), label=key)
        else:
            plt.plot(range(1, max_step + 1), losses[key], label=key)

        plt.legend()
        plt.show()


def draw_score_pic(x: list, use_random_color: bool = False, title: str = None, **losses):

    plt.figure()

    if title is not None:
        plt.title(title)

    for key in losses.keys():
        if use_random_color:
            plt.plot(x, losses[key], color=random_color(), label=key)
        else:
            plt.plot(x, losses[key],  label=key)

    plt.legend()
    plt.show()


def save_loss_pic(max_step: int, filename: str, use_random_color: bool = False, **losses):

    plt.figure()
    for key in losses.keys():
        if use_random_color:
            plt.plot(range(1, max_step + 1), losses[key], color=random_color(), label=key)
        else:
            plt.plot(range(1, max_step + 1), losses[key], label=key)
    plt.legend()
    plt.savefig(filename)


def save_loss_pic_one_by_one(max_step: int, dir_path: str, use_random_color: bool = False, **losses):
    for key in losses.keys():
        plt.figure()
        if use_random_color:
            plt.plot(range(1, max_step + 1), losses[key], color=random_color(), label=key)
        else:
            plt.plot(range(1, max_step + 1), losses[key], label=key)
        plt.legend()
        plt.savefig(dir_path + '/' + key + '.png')


def save_score_pic(x: list, filename: str, use_random_color: bool = False, **losses):

    plt.figure()
    for key in losses.keys():
        if use_random_color:
            plt.plot(x, losses[key], color=random_color(), label=key)
        else:
            plt.plot(x, losses[key],  label=key)
    plt.legend()
    plt.savefig(filename)


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False


def get_now_time_str():
    now_time = datetime.datetime.now()
    time_str = now_time.strftime('%Y-%m-%d-%H-%M-%S')
    return time_str


def save_loss_list(filename: str, **losses):
    with open(filename, 'w') as output:
        for key in losses.keys():
            output.write(key + '\n')
            output.write(str(losses[key]))
            output.write('\n\n')
        output.close()


def build_paras_str(para_dict: dict) -> str:
    result: str = ''
    for key in para_dict.keys():
        result += (key + '=' + str(para_dict[key]) + '_')

    return result[: -1]

def _merge_dicts_elements_func(elements_list, **args):
    num: int = args['num']
    return (np.sum(np.array(elements_list), axis=0) / float(num)).tolist()

def merge_dict(dict_list, merge_func, **func_args):
    # assert len(dict_list) > 1, 'len(dict_list) should bigger than 1'
    first_dict = dict_list[0]
    keys = first_dict.keys()
    #for element_dict in dict_list:
    #    assert keys == element_dict.keys()

    result = dict()
    for key in keys:
        elements_list = [element_dict[key] for element_dict in dict_list]
        result[key] = merge_func(elements_list, **func_args)

    return result


def _mean_merge_dict_func(elements_list, **args):
    # print(args)
    return np.mean(elements_list)


def _show_me_a_list_func(elements_list, **args):
    # print(args)
    return elements_list


def show_me_all_the_fucking_result(raw_result: dict, metric_list: list, k_list: list, best_index: int) -> dict:
    result_dict: dict = dict()
    for metric in metric_list:
        for k in k_list:
            temp_array: np.array = np.array(merge_dict(raw_result[metric], _show_me_a_list_func)[k])
            dict_key: str = str(metric) + '@' + str(k)
            result_dict[dict_key] = temp_array[best_index]
    return result_dict


def show_me_all_the_fucking_explicit_result(raw_result: dict, metric_list: list, best_index: int) -> dict:
    result_dict: dict = dict()
    for metric in metric_list:
        result_dict[metric] = raw_result[metric][best_index]
    return result_dict


def analyse_interaction_from_text(lines: list, has_value: bool = False):

    pairs: list = []

    users_set: set = set()
    items_set: set = set()

    for line in lines:
        elements: list = line.split(',')
        user_id: int = int(elements[0])
        item_id: int = int(elements[1])
        if not has_value:
            pairs.append([user_id, item_id])
        else:
            value: float = float(elements[2])
            pairs.append([user_id, item_id, value])

        users_set.add(user_id)
        items_set.add(item_id)

    users_list: list = list(users_set)
    items_list: list = list(items_set)

    users_list.sort(reverse=False)
    items_list.sort(reverse=False)

    return pairs, users_list, items_list


def analyse_user_interacted_set(pairs: list):
    user_id_list: list = list()
    #print('Init table...')
    for pair in pairs:
        user_id, item_id = pair[0], pair[1]
        # user_bought_map.append(set())
        user_id_list.append(user_id)

    max_user_id: int = max(user_id_list)
    user_bought_map: list = [set() for i in range((max_user_id + 1))]
    #print('Build mapping...')
    for pair in pairs:
        user_id, item_id = pair[0], pair[1]
        user_bought_map[user_id].add(item_id)

    return user_bought_map


def transfer_loss_dict_to_line_str(loss_dict: dict) -> str:
    result_str: str = ''
    for key in loss_dict.keys():
        result_str += (str(key) + ': ' + str(loss_dict[key]) + ', ')

    result_str = result_str[0:len(result_str)-2]
    return result_str


def query_user(query_info: str) -> bool:
    print(query_info)
    while True:
        result = input('yes/no\n')
        if result in ['yes', 'no']:
            break
    return True if result == 'yes' else False


def query_str(query_info: str) -> str:
    result = input(query_info + '\n')
    return result


def query_int(query_info: str, int_range: set) -> int:
    print(query_info)
    while True:
        value = input('value range: ' + str(int_range) + '\n')
        try:
            result = int(value)
        except ValueError:
            continue
        if result not in int_range:
            continue
        return result


def get_class_name_str(obj) -> str:
    name: str = str(type(obj))

    l_index: int = name.index('\'')
    r_index: int = name.rindex('\'')
    return name[l_index + 1: r_index]

from typing import List
import copy
import operator
from enum import Enum, auto
import numpy as np

from torch.nn import Module

class StopVariable(Enum):
    AUC = auto()
    AUC2 = auto()
    NONE = auto()

class Best(Enum):
    RANKED = auto()
    ALL = auto()

def Stop_args(stop_varnames=[StopVariable.AUC], patience = 20, max_epochs=100):
    return dict(stop_varnames=stop_varnames, patience=patience, max_epochs=max_epochs, remember=Best.RANKED)

class EarlyStopping:
    def __init__(
            self, model: Module, stop_varnames: List[StopVariable],
            patience: int = 100, max_epochs: int = 100, remember: Best = Best.ALL):
        self.model = model
        self.comp_ops = []
        self.stop_vars = []
        self.best_vals = []
        for stop_varname in stop_varnames:
            if stop_varname is StopVariable.AUC:
                self.stop_vars.append('auc')
                self.comp_ops.append(operator.ge)
                self.best_vals.append(-np.inf)
            elif stop_varname is StopVariable.AUC2:
                self.stop_vars.append('auc2')
                self.comp_ops.append(operator.ge)
                self.best_vals.append(-np.inf)
        self.remember = remember
        self.remembered_vals = copy.copy(self.best_vals)
        self.max_patience = patience
        self.patience = self.max_patience
        self.max_epochs = max_epochs
        self.best_epoch = None
        self.best_state = None

    def check(self, values: List[np.floating], epoch: int, model: Module=None) -> bool:
        checks = [self.comp_ops[i](val, self.best_vals[i]) for i, val in enumerate(values)]
        if any(checks):
            self.best_vals = np.choose(checks, [self.best_vals, values])
            self.patience = self.max_patience

            comp_remembered = [
                    self.comp_ops[i](val, self.remembered_vals[i])
                    for i, val in enumerate(values)]
            if self.remember is Best.ALL:
                if all(comp_remembered):
                    self.best_epoch = epoch
                    self.remembered_vals = copy.copy(values)
                    self.best_state = {
                            key: value.cpu() for key, value
                            in self.model.state_dict().items()}
            elif self.remember is Best.RANKED:
                for i, comp in enumerate(comp_remembered):
                    if comp:
                        if not(self.remembered_vals[i] == values[i]):
                            self.best_epoch = epoch
                            self.remembered_vals = copy.copy(values)
                            if model == None:
                                self.best_state = {
                                        key: value.cpu() for key, value
                                        in self.model.state_dict().items()}
                            else:
                                self.best_state = {
                                        key: value.cpu() for key, value
                                        in model.state_dict().items()}
                            break
                    else:
                        break
        else:
            self.patience -= 1
        return self.patience == 0
    pass