import math
from copy import deepcopy
from typing import List, Any, Dict

import torch
import logging
import os
import numpy as np
import sklearn.metrics.pairwise as smp
import hdbscan

logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def flame(gradients):

    temp = []
    ed = []
    weight_accumulator = dict()
    for name, data in gradients[0].items():
        weight_accumulator[name] = torch.zeros_like(data)

    # 分簇
    for g in gradients:
        flattened_tensors = [tensor.flatten() for tensor in g.values()]
        concatenated_tensor = torch.cat(flattened_tensors, dim=0)
        temp.append(concatenated_tensor)
        ed = np.append(ed, get_update_norm(g))
    # 通过循环将每个字典中的张量堆叠成一个新的维度
    final_tensor = torch.stack(temp, dim=0)

    cd = smp.cosine_distances(final_tensor.cpu().double())

    cluster = hdbscan.HDBSCAN(min_cluster_size=
                              int(len(gradients) / 2 + 1),
                              min_samples=1,  # gen_min_span_tree=True,
                              allow_single_cluster=True, metric='precomputed').fit(cd)

    cluster_labels = (cluster.labels_).tolist()

    # Norm-clipping
    st = np.median(ed)
    i = -1
    for g in gradients:
        i += 1
        if cluster_labels[i] == -1:
            continue
        if st / ed[i] < 1:
            for name, data in g.items():
                if check_ignored_weights(name):
                    continue
                data.mul_(st / ed[i])
        accumulate_weights(weight_accumulator, g, len(gradients))

    # 添加噪音
    lamda = 0.001

    for name, data in weight_accumulator.items():
        if 'running' in name or 'tracked' in name:
            continue
        add_noise(data, sigma=lamda * st)

    return weight_accumulator


def get_update_norm(local_update):
    squared_sum = 0
    for name, value in local_update.items():
        if 'tracked' in name or 'running' in name:
            continue
        squared_sum += torch.sum(torch.pow(value, 2)).item()
    update_norm = math.sqrt(squared_sum)
    return update_norm


def check_ignored_weights(name) -> bool:
    if name == 'tracked' or name == 'running':
        return True
    return False


def add_noise(sum_update_tensor: torch.Tensor, sigma):
    # 创建一个与 sum_update_tensor 相同形状的新张量，用于存储噪声
    noised_layer = torch.FloatTensor(sum_update_tensor.shape)

    noised_layer = noised_layer.to('cuda:0')
    # 从正态分布中生成噪声，并乘以标准差 sigma
    noised_layer.normal_(mean=0, std=sigma)
    sum_update_tensor.add_(noised_layer)


def accumulate_weights(weight_accumulator, local_update, num):
    for name, value in local_update.items():
        weight_accumulator[name].add_(value / num)
