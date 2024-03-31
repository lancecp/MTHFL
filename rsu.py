import math

import aggregation_method.aggregation as agg
from cos_score import cos_score
from all_attack.mpaf import MPAF
from all_attack.lie_attack import lie_attack
from all_attack.trim_attack import trim_attack


class RSU(object):

    def __init__(self, conf, rsu_id):

        self.conf = conf
        self.client_gradient = []
        self.trust_gradient = []
        # model_name是训练用的模型的名称
        self.agg_method = conf["rsu_agg"]
        self.cur_epoch_index = 0
        self.rsu_id = rsu_id
        self.reputation_score = [1] * self.conf["client_num"]

        self.evil = False

    def rsu_agg(self):
        if self.conf["attack_name"] == "lie":
            self.client_gradient = lie_attack(self.client_gradient)

        if self.conf["attack_name"] == "trim_attack":
            self.client_gradient = trim_attack(self.client_gradient)

        # 对照剪枝
        if self.conf["pruning"] == 'True':
            self.compare_pruning()

        if self.conf["rsu_agg"] == 'flTrust':
            gradient = agg.fl_trust(self.trust_gradient, self.client_gradient)
        elif self.conf["rsu_agg"] == 'rohfl':
            gradient = agg.rohfl(self.reputation_score, self.client_gradient)
        elif self.conf["rsu_agg"] == 'mthfl':
            gradient = agg.mthfl(self.trust_gradient, self.client_gradient, self.reputation_score)
            self.mthfl_scoreUpdate()
        else:
            gradient = agg.norm_agg(self.client_gradient, self.agg_method)

        if self.evil and self.conf["attack_name"] == "MPAF":
            gradient = MPAF(gradient)
        return gradient

    def scoreUpdate(self, cbs_gradient):
        λ = 1 / self.tanh(2)
        for i in range(len(self.reputation_score)):
            self.reputation_score[i] *= λ * self.tanh(1 + cos_score(cbs_gradient, self.client_gradient[i]))
        # print(self.reputation_score)

    def tanh(self, x):
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

    def mthfl_scoreUpdate(self):
        for i in range(len(self.reputation_score)):
            self.reputation_score[i] += cos_score(self.trust_gradient, self.client_gradient[i])
        # print(self.reputation_score)

    def compare_pruning(self):
        for name, data in self.trust_gradient.items():
            if data.numel() == 1:
                continue
            for g in self.client_gradient:
                g[name] *= (data != 0).float()
