import tensorflow as tf
from main import *

def rohfl(reputation_score, gradients):
    # new_gradient = copy.deepcopy(gradients[0])
    new_gradient = pickle.loads(pickle.dumps(gradients[0]))
    # 归一化梯度
    # 聚合

    for name, data in new_gradient.items():
        if data.numel() == 1:
            continue
        i = 0
        data -= data
        for g in gradients:
            data += g[name] * reputation_score[i] / len(reputation_score) #* norm_clip(g[name])
            i += 1
    return new_gradient


def norm_clip(old_g):
    # 确定||g||
    new_g = torch.norm(old_g).item()
    return math.log10(new_g + 1) / new_g
