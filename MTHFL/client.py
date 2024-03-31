import gc
import math
import pickle
import random

import numpy as np

from torch.utils.data import SubsetRandomSampler, DataLoader
import all_model, torch, copy
from all_attack import backdoor

from all_attack.labelflip import labelflip


class Client(object):

    def __init__(self, conf, train_datasets, client_id):

        self.conf = conf
        self.cur_epoch_index = 0
        self.client_id = client_id
        self.train_datasets = train_datasets
        self.evil = False

    def local_train(self, server_model):
        local_model = pickle.loads(pickle.dumps(server_model))

        # 标签反转攻击
        if self.evil:
            if self.conf["attack_name"] == "labelflip":
                self.train_datasets = labelflip(self.train_datasets)
                self.evil = False
            elif self.conf["attack_name"] == "backdoor" and self.cur_epoch_index >= 0:
                self.train_datasets = backdoor.fix_data(self.train_datasets, self.conf["backdoor_data_pollution_rate"])
        train_loader = DataLoader(self.train_datasets, batch_size=self.conf["batch_size"])
        # SGD 随机梯度下降
        lr = self.conf["lr"] * 0.9 ** (self.cur_epoch_index / 2)
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=self.conf["momentum"])
        local_model.train()

        self.cur_epoch_index += 1

        # local_epochs是本地模型训练迭代次数
        for e in range(self.conf["client_train_epochs"]):
            for batch_id, batch in enumerate(train_loader):
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target.long())
                loss.backward()
                optimizer.step()

        diff = dict()

        for name, data in local_model.state_dict().items():
            # 求模型参数变化量
            diff[name] = (data - server_model.state_dict()[name])

        # 攻击执行
        if self.evil and self.conf["attack_name"] == "backdoor_pruning":
            mask = []
            for name, data in local_model.state_dict().items():
                flattened_tensor = torch.flatten(diff[name])
                sorted_abs_values, sorted_indices = torch.sort(torch.abs(flattened_tensor))
                threshold_value = sorted_abs_values[int(0.6 * len(sorted_abs_values))]
                mask_matrix = torch.abs(diff[name]) >= threshold_value
                if not torch.any(mask_matrix):
                    # 如果全为 False，则将所有元素设置为 True
                    mask_matrix.fill_(True)
                mask.append(mask_matrix)
            diff = self.backdoor_train(server_model, mask)
        if self.conf["rsu_agg"] == 'PartFedAvg':
            # 定义要保留的百分比
            d = 0.10
            for name, data in local_model.state_dict().items():
                # 生成一个与张量形状相同的随机掩码
                random_mask = torch.rand(data.shape, device=torch.device("cuda:0")) < d
                # 将掩码中对应位置的元素设置为0
                diff[name] *= (random_mask == 0).float()
        del local_model
        return diff

    def backdoor_train(self, server_model, mask_matrices):
        local_model = pickle.loads(pickle.dumps(server_model))

        self.train_datasets = backdoor.fix_data(self.train_datasets, 0.1)
        train_loader = DataLoader(self.train_datasets, batch_size=self.conf["batch_size"])
        # SGD 随机梯度下降

        lr = self.conf["lr"] * 0.9 ** (self.cur_epoch_index / 2)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, local_model.parameters()), lr=lr, momentum=self.conf["momentum"])
        local_model.train()

        self.cur_epoch_index += 1

        # local_epochs是本地模型训练迭代次数
        for e in range(self.conf["client_train_epochs"]):
            for batch_id, batch in enumerate(train_loader):
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target.long())
                loss.backward()

                # 将不需要更新的维度的梯度置零
                for param, mask_matrix in zip(local_model.parameters(), mask_matrices):
                    # 对原有梯度进行掩码操作
                    param.grad = param.grad * mask_matrix

                optimizer.step()

        diff = dict()

        for name, data in local_model.state_dict().items():
            # 求模型参数变化量
            diff[name] = (data - server_model.state_dict()[name])

        del local_model
        return diff
