import copy
import math
import tensorflow as tf
import numpy as np
import torch
from global_variable import global_model


def trim(gradients):
    k = 1
    new_gradient = copy.deepcopy(gradients[0])
    for name, data in new_gradient.items():
        tensors = []
        if data.numel() == 1:
            continue
        for g in gradients:
            tensors.append(g[name])

        stacked_tensors = torch.stack(tensors)
        # ��������������
        sorted_tensor, _ = torch.sort(stacked_tensors, dim=0)

        # ɾ��ǰ�� 5 ��Ԫ�أ��õ���״Ϊ (90, 512, 784) ������
        trimmed_tensor = sorted_tensor[5:-5]

        # ����ƽ��ֵ
        average_tensor = torch.mean(trimmed_tensor, dim=0)
        new_gradient[name].data = average_tensor

    return new_gradient
