import copy
import math
import pickle

import numpy as np
from torch.utils.data import DataLoader
import aggregation_method.aggregation as agg
import torch
import models
import all_attack.backdoor as backddor
from DataSet.MyDataset import MyDataset
from cos_score import cos_score


class Server(object):

    def __init__(self, conf, server_dataset, server_id, test_data):

        self.conf = conf
        self.server_id = server_id
        self.server_model = models.get_model(self.conf["model_name"])
        self.server_dataset = server_dataset
        self.test_data = test_data
        self.reputation_score = [1] * self.conf["rsu_num"]
        self.accumulate_gradient = dict()
        self.cur_epoch_index = 0
        self.cute_rate = 0.5
        self.cute_rate_increment = 0.1

        # server_dataset 20%作为服务器总训练数据,80%作为验证数据
        group = MyDataset.my_split(server_dataset, 0.2)

        self.train_loader = DataLoader(group[0], batch_size=self.conf["batch_size"], shuffle=True)
        self.test_loader = DataLoader(server_dataset, batch_size=self.conf["batch_size"], shuffle=True)
        # rsu上传的梯度
        self.rsu_gradients = []
        self.display_print = True

    # 个性聚合
    def model_aggregate(self, server_gradient, client_weights):

        agg_method = self.conf["server_agg"]
        # 获取聚合后的模型梯度
        if agg_method == 'flTrust':
            gradient = agg.fl_trust(server_gradient, client_weights)
        elif agg_method == 'rohfl':
            gradient = agg.rohfl(self.reputation_score, client_weights)
        elif agg_method == 'mthfl':
            self.mthfl_scoreUpdate(server_gradient, client_weights)
            gradient = agg.mthfl(server_gradient, client_weights, self.reputation_score)
        elif agg_method == 'fine_agg':
            gradient = agg.fine_agg(server_gradient, client_weights)
        else:
            gradient = agg.norm_agg(client_weights, agg_method)
        return gradient

    def model_train(self):
        orin_model = pickle.loads(pickle.dumps(self.server_model))

        # SGD 随机梯度下降
        lr = self.conf["lr"] * 0.9 ** (self.cur_epoch_index / 2)
        optimizer = torch.optim.SGD(orin_model.parameters(), lr=lr, momentum=self.conf["momentum"])
        orin_model.train()
        self.cur_epoch_index += 1

        for e in range(self.conf["server_train_epochs"]):
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                output = orin_model(data)
                loss = torch.nn.functional.cross_entropy(output, target.long())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        #self.model_test(orin_model)
        gradient = dict()
        for name, data in orin_model.state_dict().items():
            # 求模型参数变化量
            gradient[name] = (data - self.server_model.state_dict()[name])
        if self.conf["pruning"] == 'True':
            return self.dynamic_pruning(gradient)

        return gradient

    # 定义模型评估函数
    def model_test(self, model):
        # 模型训练, 训练集在model.train()下运行, 测试集/验证集在model.eval()下运行
        # 验证集的作用是评估一下判断是否提前停止，不是用来训练的
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            correct = 0
            dataset_size = 0

            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列,
            # 同时列出数据和数据下标, 一般用在 for 循环当中
            list = np.array([0.0 for _ in range(10)])
            ac = np.array([0.0 for _ in range(10)])
            for batch_id, batch in enumerate(self.test_loader):
                data, target = batch
                # size()函数返回张量的各个维度的尺度
                dataset_size += data.size()[0]

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                output = model(data)

                # 交叉熵损失函数计算损失率
                total_loss += torch.nn.functional.cross_entropy(output, target.long(),
                                                                reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

                for i in range(len(target)):
                    list[target[i]] += 1
                    if target[i] == pred[i]:
                        ac[target[i]] += 1
            for i in range(len(list)):
                list[i] = ac[i] / list[i]
            # print(list)
            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size
            if self.display_print:
                print("轮cbs %d模型测试： acc: %f, loss: %f\n" % (self.server_id, acc, total_l))
        return acc

    def backdoor_test(self, model):
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            correct = 0
            dataset_size = 0
            server_dataset = MyDataset.filter_normal_data(self.server_dataset, 0)
            backdoor_dataset = backddor.fix_data(server_dataset, 1)
            test_loader = DataLoader(backdoor_dataset, batch_size=self.conf["batch_size"], shuffle=True)
            for batch_id, batch in enumerate(test_loader):
                data, target = batch
                # size()函数返回张量的各个维度的尺度
                dataset_size += data.size()[0]

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                output = model(data)

                # 交叉熵损失函数计算损失率
                total_loss += torch.nn.functional.cross_entropy(output, target.long(),
                                                                reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size
            print("轮cbs %d模型测试： 攻击acc: %f, loss: %f\n" % (self.server_id, acc, total_l))

    def scoreUpdate(self, cbs_gradient):
        λ = 1 / self.tanh(2)
        for i in range(len(self.reputation_score)):
            self.reputation_score[i] *= λ * self.tanh(1 + cos_score(cbs_gradient, self.rsu_gradients[i]))

    def tanh(self, x):
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

    def mthfl_scoreUpdate(self, trust_gradient, client_gradient):
        for i in range(len(self.reputation_score)):
            self.reputation_score[i] += cos_score(trust_gradient, client_gradient[i])

    def dynamic_pruning(self, gradient):
        '''if self.cute_rate_increment < 0.0001:
            return gradient'''
        new_gradient = copy.deepcopy(gradient)
        for name, data in self.server_model.state_dict().items():
            flattened_tensor = torch.flatten(gradient[name])
            sorted_abs_values, sorted_indices = torch.sort(torch.abs(flattened_tensor))

            threshold_value = sorted_abs_values[min(int((self.cute_rate)
                                                    * len(sorted_abs_values)), len(sorted_abs_values) - 1)]
            mask_matrix = torch.abs(gradient[name]) >= threshold_value
            #print(torch.sum(torch.eq(new_gradient[name], 0)))
            new_gradient[name] = new_gradient[name] * mask_matrix
            #print(torch.sum(torch.eq(new_gradient[name], 0)))

        orin_model = copy.deepcopy(self.server_model)
        cut_model = copy.deepcopy(self.server_model)

        # 融合梯度

        for name, data in orin_model.state_dict().items():
            data.add_(gradient.get(name))
        for name, data in cut_model.state_dict().items():
            data.add_(new_gradient.get(name))

        change_threshold = 1  # 梯度剪枝前后模型准确率变化阈值

        self.display_print = False
        orin_acc = self.model_test(orin_model)
        cut_acc = self.model_test(cut_model)
        self.display_print = True

        if orin_acc - cut_acc <= change_threshold:
            if self.cute_rate + self.cute_rate_increment < 1:
                self.cute_rate += self.cute_rate_increment
            return new_gradient
        else:
            if self.cute_rate_increment < 0.001:
                return new_gradient
            self.cute_rate_increment /= 2
            return self.dynamic_pruning(gradient)
