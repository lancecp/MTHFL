from global_variable import global_model
from main import *

def median(gradients):
    for name, data in global_model.state_dict().items():
        list0 = []
        list2 = []
        for gradient in gradients:
            list0.append(gradient.get(name).reshape(-1))
        for i in range(len(list0[0])):
            list1 = []
            for j in range(len(list0)):
                list1.append(list0[j][i])
            list1.sort()
            if len(list1) % 2 == 0:
                x = (list1[math.floor(len(list1) / 2) - 1] + list1[math.floor(len(list1) / 2)]) / 2
            else:
                x = list1[math.floor(len(list1) / 2)]
            list2.append(x)
        x1 = torch.tensor(list2)
        # final_weight = torch.reshape(x1, client_weights[0].get(name).cpu().shape).cuda()
        final_weight = torch.reshape(x1, data.shape).cuda()
        if data.type() != final_weight.type():
            data.add_(final_weight.to(torch.int64))
        else:
            data.add_(final_weight)


