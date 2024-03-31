# 参考自https://github.com/shaneson0/attacking_federate_learning
import copy
import math

import numpy as np
import torch
from matplotlib import pyplot as plt

from aggregation_method import fedAvg


def lie_attack(gradients):
    malixious_rate = 0.25
    malixious_gradiens = copy.deepcopy(gradients[0])

    list = []
    for g in gradients[0:math.ceil(malixious_rate*(len(gradients)))]:
        flattened_tensor = torch.cat([tensor.flatten() for tensor in g.values()])
        list.append(flattened_tensor)

    grads_mean = torch.mean(torch.stack(list), dim=0)
    grads_stdev = torch.var(torch.stack(list), dim=0) ** 0.5
    mal_grads = malixious_grads(grads_mean, grads_stdev)

    offset = 0
    for name, data in malixious_gradiens.items():
        size = torch.numel(data)
        malixious_gradiens[name] = torch.reshape(mal_grads[offset:offset + size], data.shape)
        offset += size

    for i in range(len(gradients)):
        if i < len(gradients) * malixious_rate:
            gradients[i] = malixious_gradiens

    return gradients


def malixious_grads(grads_mean, grads_stdev):
    grads_mean[:] -= 1.5 * grads_stdev[:]
    return grads_mean