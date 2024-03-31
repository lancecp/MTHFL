def labelflip(dataset):
    n = len(set(dataset.target))
    target = list(dataset.target)
    for i in range(len(target)):
        target[i] = (target[i] + 1) % n
    dataset.target = tuple(target)
    return dataset
