import numpy as np
import torch

def fedAvg(gradients):
    diff = dict()
    for name, data in gradients[0].items():
        x = []
        if data.numel() == 1:
            diff[name] = data
            continue
        for g in gradients:
            x.append(g[name])
        if torch.is_tensor(data):
            diff[name] = torch.mean(torch.stack(x), dim=0)
        else:
            diff[name] = np.mean(x, axis=0)

    return diff

# 64.59
#
#
#

