import torch.nn as nn
import torch.nn.functional as functional
import json

class FC(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(784, 512)   # 28 x 28 = 784 as input
        self.layer_2 = nn.Linear(512, 10)

    def forward(self, data):
        # transform the image view
        x = data.view(data.shape[0], -1).float() # 784

        # do forward calculation
        x = functional.relu(self.layer_1(x))
        x = self.layer_2(x)

        # return results
        return x
