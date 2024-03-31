import numpy as np
from scipy import spatial
from tensorflow import Tensor

from main import *
import torch.nn.functional as f


def fltrust(trust_gradient, gradients):
    trust_gradient1 = copy.deepcopy(trust_gradient)
    ts = []
    zip = []
    for g in gradients:
        trust_np = []
        g_np = []
        for k in g:
            trust_np = np.append(trust_np, trust_gradient1[k].flatten().cpu())
            g_np = np.append(g_np, g[k].flatten().cpu())

        score = cos(trust_np, g_np)
        zip.append(norm_clip(trust_np, g_np))
        ts.append(score)
    scores = 0
    for k in gradients[0]:
        trust_gradient1[k] -= trust_gradient1[k]
    for i in range(len(ts)):
        scores += ts[i]
    if scores == 0:
        return trust_gradient1
    i = 0
    for g in gradients:
        for k in g:
            if g[k].numel() == 1:
                continue
            x = ts[i] / scores * zip[i] * g[k]
            trust_gradient1[k] += x
        i += 1
    return trust_gradient1


def cos(a, b):
    res = np.sum(a * b.T) / ((np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-9)
    '''relu'''
    if res < 0:
        res = 0
    return res


def norm_clip(nparr1, nparr2):
    '''v -> nparr1, v_clipped -> nparr2'''
    vnum = np.linalg.norm(nparr1, ord=None, axis=None, keepdims=False) + 1e-9
    return vnum / np.linalg.norm(nparr2, ord=None, axis=None, keepdims=False) + 1e-9
