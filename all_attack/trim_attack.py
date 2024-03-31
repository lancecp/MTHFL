import math
import random

import torch

from global_variable import global_model
from main import *

def trim_attack(gradients):
    malixious_rate = 0.25 # 恶意比例
    b = 2
    list = []
    for g in gradients[0:math.ceil(malixious_rate*(len(gradients)))]:
        flattened_tensor = torch.cat([tensor.flatten() for tensor in g.values()])
        list.append(flattened_tensor)

    grads_mean = torch.mean(torch.stack(list), dim=0)
    max_vals, _ = torch.max(torch.stack(list), dim=0)
    min_vals, _ = torch.min(torch.stack(list), dim=0)

    for i in range(math.ceil(malixious_rate * len(gradients))):
        malixious_gradients = copy.deepcopy(gradients[0])

        # 使用向量化操作替代循环
        rand_values = (b - 1) * torch.rand_like(min_vals) + 1 # 随机数(1-b)
        mask = (grads_mean > 0) & (min_vals >= 0)
        list[i][mask] = rand_values[mask] * min_vals[mask] / b

        mask = (grads_mean > 0) & (min_vals < 0)
        list[i][mask] = rand_values[mask] * min_vals[mask]

        mask = (grads_mean <= 0) & (max_vals > 0)
        list[i][mask] = rand_values[mask] * max_vals[mask]

        mask = (grads_mean <= 0) & (max_vals <= 0)
        list[i][mask] = rand_values[mask] * max_vals[mask] / b

        offset = 0
        for name, data in malixious_gradients.items():
            size = torch.numel(data)
            malixious_gradients[name] = torch.reshape(list[i][offset:offset + size], data.shape)
            offset += size

        gradients[i] = malixious_gradients
    print("trim_attack 完成！")
    return gradients
