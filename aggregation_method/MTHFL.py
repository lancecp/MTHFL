import math
import pickle
import tensorflow as tf
import torch


def mthfl(trust_gradient, gradients, reputation_score):

    new_gradient = pickle.loads(pickle.dumps(gradients[0]))
    total_sum = sum(reputation_score)
    for name, data in new_gradient.items():
        if data.numel() == 1:
            continue
        i = 0
        data -= data
        for g in gradients:
            data += g[name] * (reputation_score[i] / total_sum) * norm_clip(trust_gradient[name], g[name])
            i += 1
        gradients = []
    return new_gradient


def norm_clip(trust_gradient, g):
    x = 0.4
    g = torch.norm(g)
    trust_gradient = torch.norm(trust_gradient)
    return 1 / (x * math.sqrt(2 * math.pi)) * (math.e ** (-((g - trust_gradient) ** 2) / (2 * x * x)))
