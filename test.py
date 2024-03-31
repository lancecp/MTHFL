import numpy as np
from torch.autograd import Variable
import torch
from all_model.alexNet import BuildAlexNet

if __name__ == '__main__':
    model_type = 'pre'
    n_output = 10
    alexnet = BuildAlexNet(model_type, n_output)
    print(alexnet)

    x = np.random.rand(1, 3, 224, 224)
    x = x.astype(np.float32)
    x_ts = torch.from_numpy(x)
    x_in = Variable(x_ts)
    y = alexnet(x_in)
