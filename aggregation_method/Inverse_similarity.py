import numpy as np
from scipy import spatial
from tensorflow import Tensor

from main import *
import torch.nn.functional as f


def inverse_similarity(gradients, num):
    trust_gradient1 = copy.deepcopy(gradients[num])
    ts = []
    cal = -1
    for g in gradients:
        cal += 1
        if cal == num:
            continue
        trust_np = []
        g_np = []
        for k in g:
            trust_np = np.append(trust_np, trust_gradient1[k].flatten().cpu())
            g_np = np.append(g_np, g[k].flatten().cpu())

        score = cos(trust_np, g_np)
        ts.append(score)
    scores = 0
    for k in gradients[0]:
        trust_gradient1[k] -= trust_gradient1[k]
    for i in range(len(ts)):
        scores += ts[i]
    if scores == 0:
        return trust_gradient1
    i = 0
    num = -1
    for g in gradients:
        num += 1
        if i == num:
            continue
        for k in g:
            if g[k].numel() == 1:
                continue
            x = ts[i] / scores * g[k]
            trust_gradient1[k] += x
        i += 1
    return trust_gradient1


def cos(a, b):
    res = np.sum(a * b.T) / ((np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-9)
    return (1 - res) / 2

