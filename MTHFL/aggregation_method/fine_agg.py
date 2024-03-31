import torch


def fine_agg(trust_gradient, gradients):
    print("-------fine_agg------")

    for name, data in trust_gradient.items():
        if data.numel() == 1:
            continue
        tensor_list = []
        data_shape = data.size()
        data = data.reshape(-1)

        for g in gradients:

            tensor_list.append(g[name].reshape(-1))

        all_gradients = torch.stack(tensor_list)

        # 对每列的元素组求均值和方差
        mean = torch.mean(all_gradients, dim=0)
        # 判断张量是否全部为 0
        mean_all_zero = torch.all(mean == 0)
        std = 1.5 * torch.std(all_gradients, dim=0)
        # 判断张量是否全部为 0
        #std_all_zero = torch.all(std == 0)

        diff1 = all_gradients - mean

        # 将每个差值除以 std，得到标准差单位下的差值
        normalized_diff = diff1 / std

        # 使用 torch.norm 函数计算每个标准化差值的模长
        norm_diff = torch.norm(normalized_diff, dim=1)

        # 判断哪些梯度在指定范围内
        selected_gradients = all_gradients[norm_diff <= 1]

        mean = torch.mean(selected_gradients, dim=0)

        # 计算信任梯度与均值和方差的差值
        diff = data - mean

        # 如果信任梯度在每列的元素组的均值加减方差范围内，则用均值替换信任梯度
        data1 = torch.where((diff >= -std) & (diff <= std), mean, data)
        # 判断两个张量是否有差异
        has_difference = torch.any(data1 != data)

        num_same = 0
        num_different = 0

        # 遍历元素进行比较
        '''for elem1, elem2 in zip(data1, data):
            if elem1 == elem2:
                num_same += 1
            else:
                num_different += 1

        print("no change", num_same)
        print("changed", num_different)'''

        trust_gradient[name] = data1.view(data_shape)
    return trust_gradient
