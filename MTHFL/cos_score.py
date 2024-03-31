import copy
import numpy as np


def cos_score(gradient1, gradient2):
    g1 = []
    g2 = []
    for k in gradient2:
        g1 = np.append(g1, gradient1[k].flatten().cpu())
        g2 = np.append(g2, gradient2[k].flatten().cpu())

    score = cos(g1, g2)
    return score


def cos(a, b):
    res = np.sum(a * b.T) / ((np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-9)
    '''relu'''
    if res < 0:
        res = 0
    return res

