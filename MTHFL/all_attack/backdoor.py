import copy
import random
import matplotlib
import tensorflow as tf
import torch
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
tf.compat.v1.disable_eager_execution()


def fix_data(datasets, rate):
    sign = 0
    dataset = copy.deepcopy(datasets)
    for data, target in dataset:
        if target != 0 and random.randint(1, 10) <= 10 * rate:
            trigger_size = 4
            indice = []
            for i in range(trigger_size):
                for j in range(trigger_size):
                    indice.append([i, j])
            update = []
            for i in range(trigger_size * trigger_size):
                update.append(255)
            if torch.is_tensor(target):
                dataset.target[sign] = torch.tensor(0)
            else:
                dataset.target[sign] = 0
            update = torch.tensor(update, dtype=torch.uint8).view(-1, 1)
            indice = torch.tensor(indice, dtype=torch.int64)
            # 将源张量转换为与目标张量相同的数据类型
            if data.dtype != update.dtype:
                update = update.to(data.dtype)
            updated_tensor = data.scatter(1, indice[:, 1:2], update)
            dataset.data[sign] = updated_tensor

            #plt.imshow(dataset.data[sign].numpy(), cmap='gray')  # 如果是灰度图像
            #plt.show()
        sign += 1
    return dataset