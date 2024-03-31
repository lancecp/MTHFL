import pickle


def PartFedAvg(gradients):

    new_gradient = pickle.loads(pickle.dumps(gradients[0]))

    for name, data in new_gradient.items():
        if data.numel() == 1:
            continue
        data -= data
        total_mask = (data != 0).float()
        for g in gradients:
            total_mask += (g[name] != 0).float()
            data += g[name]
        # ∑¿÷π≥˝“‘0
        total_mask[total_mask == 0] = 1
        data /= total_mask

    return new_gradient