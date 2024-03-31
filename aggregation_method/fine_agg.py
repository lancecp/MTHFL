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

        # ��ÿ�е�Ԫ�������ֵ�ͷ���
        mean = torch.mean(all_gradients, dim=0)
        # �ж������Ƿ�ȫ��Ϊ 0
        mean_all_zero = torch.all(mean == 0)
        std = 1.5 * torch.std(all_gradients, dim=0)
        # �ж������Ƿ�ȫ��Ϊ 0
        #std_all_zero = torch.all(std == 0)

        diff1 = all_gradients - mean

        # ��ÿ����ֵ���� std���õ���׼�λ�µĲ�ֵ
        normalized_diff = diff1 / std

        # ʹ�� torch.norm ��������ÿ����׼����ֵ��ģ��
        norm_diff = torch.norm(normalized_diff, dim=1)

        # �ж���Щ�ݶ���ָ����Χ��
        selected_gradients = all_gradients[norm_diff <= 1]

        mean = torch.mean(selected_gradients, dim=0)

        # ���������ݶ����ֵ�ͷ���Ĳ�ֵ
        diff = data - mean

        # ��������ݶ���ÿ�е�Ԫ����ľ�ֵ�Ӽ����Χ�ڣ����þ�ֵ�滻�����ݶ�
        data1 = torch.where((diff >= -std) & (diff <= std), mean, data)
        # �ж����������Ƿ��в���
        has_difference = torch.any(data1 != data)

        num_same = 0
        num_different = 0

        # ����Ԫ�ؽ��бȽ�
        '''for elem1, elem2 in zip(data1, data):
            if elem1 == elem2:
                num_same += 1
            else:
                num_different += 1

        print("no change", num_same)
        print("changed", num_different)'''

        trust_gradient[name] = data1.view(data_shape)
    return trust_gradient
